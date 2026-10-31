"""KhetRakshak Phase-0 smoke test.

Calls every external API once with a REAL request and prints PASS / WARN / FAIL / SKIP.
No mock data: every number printed comes from the live response.

Usage (from repo root):
    pip install -r backend/requirements.txt
    python scripts/smoke_test.py

Exit code 1 if any REQUIRED check fails.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import httpx
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

LAT = float(os.getenv("SMOKE_LAT", "30.85"))
LON = float(os.getenv("SMOKE_LON", "75.70"))
DEST_LAT = float(os.getenv("SMOKE_DEST_LAT", "30.705"))
DEST_LON = float(os.getenv("SMOKE_DEST_LON", "76.222"))
STATE = os.getenv("SMOKE_STATE", "Punjab")
MAX_CLOUD = float(os.getenv("SMOKE_MAX_CLOUD", "20"))

TIMEOUT = httpx.Timeout(25.0, connect=10.0)
SECRET_ENV_KEYS = [
    "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DATA_GOV_IN_API_KEY", "CEDA_API_KEY",
    "SARVAM_API_KEY", "ORS_API_KEY", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_ROLE_KEY",
    "TELEGRAM_BOT_TOKEN",
]


class Skip(Exception):
    """Check not run (missing config)."""


class Warn(Exception):
    """API reachable but result is degraded (e.g. empty, cloudy)."""


@dataclass
class Result:
    name: str
    status: str
    latency_ms: int
    detail: str
    required: bool


def env(key: str) -> str:
    value = os.getenv(key, "").strip()
    if not value:
        raise Skip(f"{key} not set in .env")
    return value


def redact(text: str) -> str:
    for key in SECRET_ENV_KEYS:
        secret = os.getenv(key, "").strip()
        if secret:
            text = text.replace(secret, f"<{key}>")
    return text


def request(client: httpx.Client, method: str, url: str, *, retries: int = 3,
            backoff: float = 1.5, **kwargs) -> httpx.Response:
    """HTTP call with retries + exponential backoff on network errors, 429 and 5xx."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = client.request(method, url, **kwargs)
            if response.status_code != 429 and response.status_code < 500:
                response.raise_for_status()
                return response
            last_error = httpx.HTTPStatusError(
                f"HTTP {response.status_code}", request=response.request, response=response)
        except httpx.TransportError as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(backoff ** attempt)
    assert last_error is not None
    raise last_error


# ---------------------------------------------------------------- weather

def check_open_meteo_forecast(c: httpx.Client) -> str:
    r = request(c, "GET", "https://api.open-meteo.com/v1/forecast", params={
        "latitude": LAT, "longitude": LON, "timezone": "Asia/Kolkata", "forecast_days": 7,
        "daily": "temperature_2m_max,precipitation_sum",
    })
    daily = r.json()["daily"]
    rain_7d = sum(v for v in daily["precipitation_sum"] if v is not None)
    return (f"{len(daily['time'])} days from {daily['time'][0]}; "
            f"Tmax today {daily['temperature_2m_max'][0]} °C; 7-day rain {rain_7d:.1f} mm")


def check_open_meteo_archive(c: httpx.Client) -> str:
    end = date.today() - timedelta(days=7)  # archive lags a few days
    start = end - timedelta(days=30)
    r = request(c, "GET", "https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": LAT, "longitude": LON, "timezone": "Asia/Kolkata",
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "daily": "precipitation_sum,temperature_2m_max",
    })
    daily = r.json()["daily"]
    rain = [v for v in daily["precipitation_sum"] if v is not None]
    if not rain:
        raise Warn(f"archive returned no values for {start}..{end}")
    return f"{len(rain)} days {start}..{end}; total rain {sum(rain):.1f} mm"


# ---------------------------------------------------------------- satellite

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
PC_TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"


def check_sentinel2_stac(c: httpx.Client) -> str:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=90)
    r = request(c, "POST", PC_STAC, json={
        "collections": ["sentinel-2-l2a"],
        "intersects": {"type": "Point", "coordinates": [LON, LAT]},
        "datetime": f"{start:%Y-%m-%dT%H:%M:%SZ}/{end:%Y-%m-%dT%H:%M:%SZ}",
        "limit": 100,
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
    })
    scenes = r.json().get("features", [])
    if not scenes:
        raise Warn("no Sentinel-2 scenes in last 90 days for this point")
    clear = [s for s in scenes if s["properties"].get("eo:cloud_cover", 100) < MAX_CLOUD]
    summary = f"{len(scenes)} scenes in 90d, {len(clear)} with cloud < {MAX_CLOUD:g}%"
    if not clear:
        raise Warn(summary + " -> confidence must be lowered for this field")

    latest = clear[0]
    token = request(c, "GET", PC_TOKEN).json()["token"]
    href = latest["assets"]["B08"]["href"]
    head = request(c, "HEAD", f"{href}?{token}")
    size_mb = int(head.headers.get("content-length", 0)) / 1e6
    return (f"{summary}; latest clear {latest['properties']['datetime'][:10]} "
            f"({latest['properties']['eo:cloud_cover']:.1f}% cloud); "
            f"signed B08 asset readable ({size_mb:.0f} MB)")


# ---------------------------------------------------------------- market

def check_agmarknet(c: httpx.Client) -> str:
    key = env("DATA_GOV_IN_API_KEY")
    resource = os.getenv("AGMARKNET_RESOURCE_ID", "9ef84268-d588-465a-a308-a864a43d0070")
    url = f"https://api.data.gov.in/resource/{resource}"
    base = {"api-key": key, "format": "json", "limit": 5}
    slow = httpx.Timeout(60.0, connect=10.0)  # data.gov.in is slow, worst on filtered queries

    def fetch(extra: dict, retries: int) -> dict:
        r = request(c, "GET", url, params={**base, **extra}, timeout=slow, retries=retries)
        try:
            return r.json()
        except json.JSONDecodeError:
            raise Warn(f"non-JSON response: {r.text[:120]!r}")

    payload = fetch({}, retries=2)  # unfiltered: proves the key works
    if not payload.get("records"):
        raise Warn("key accepted but 0 records today -> labelled cache / history fallback")
    summary = f"key OK, {payload.get('total')} records today (all states)"

    # data.gov.in has used both filter spellings; first one that returns rows wins.
    for field in ("state.keyword", "state"):
        try:
            records = fetch({f"filters[{field}]": STATE}, retries=1).get("records") or []
        except (httpx.HTTPError, Warn):
            continue
        if records:
            rec = records[0]
            return (f"{summary}; filters[{field}]={STATE} works, e.g. {rec.get('commodity')} "
                    f"@ {rec.get('market')}: modal ₹{rec.get('modal_price')}/qtl "
                    f"on {rec.get('arrival_date')}")
    raise Warn(f"{summary}; but {STATE} filter timed out / empty -> filter client-side")


def check_ceda(c: httpx.Client) -> str:
    base = env("CEDA_API_BASE")
    key = env("CEDA_API_KEY")
    r = request(c, "GET", base, headers={"Authorization": f"Bearer {key}"})
    return f"HTTP {r.status_code}, {len(r.content)} bytes"


# ---------------------------------------------------------------- voice / language

SARVAM = "https://api.sarvam.ai"
PROBE_TEXT = "Irrigate your field before Thursday."  # a test sentence, not data


def sarvam_headers() -> dict:
    return {"api-subscription-key": env("SARVAM_API_KEY")}


def check_sarvam_chain(c: httpx.Client) -> str:
    """translate (en->hi) -> TTS (hi audio) -> STT (audio -> hi text). Proves the full voice loop."""
    headers = sarvam_headers()
    tr = request(c, "POST", f"{SARVAM}/translate", headers=headers, json={
        "input": PROBE_TEXT, "source_language_code": "en-IN", "target_language_code": "hi-IN",
        "model": os.getenv("SARVAM_TRANSLATE_MODEL", "mayura:v1"),
    }).json()
    hindi = tr["translated_text"]

    tts_body = {"text": hindi, "target_language_code": "hi-IN",
                "model": os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")}
    if speaker := os.getenv("SARVAM_TTS_SPEAKER", "").strip():
        tts_body["speaker"] = speaker  # empty -> model's default voice
    tts = request(c, "POST", f"{SARVAM}/text-to-speech", headers=headers, json=tts_body).json()
    audio = base64.b64decode(tts["audios"][0])

    stt = request(c, "POST", f"{SARVAM}/speech-to-text", headers=headers,
                  files={"file": ("probe.wav", audio, "audio/wav")},
                  data={"model": os.getenv("SARVAM_STT_MODEL", "saarika:v2.5"),
                        "language_code": "hi-IN"}).json()
    return (f"translate: {hindi!r} | TTS: {len(audio) / 1024:.0f} KB wav | "
            f"STT heard: {stt.get('transcript')!r}")


# ---------------------------------------------------------------- routing / geo

def check_osrm(c: httpx.Client) -> str:
    base = os.getenv("OSRM_BASE_URL", "https://router.project-osrm.org").rstrip("/")
    r = request(c, "GET", f"{base}/route/v1/driving/{LON},{LAT};{DEST_LON},{DEST_LAT}",
                params={"overview": "false"})
    route = r.json()["routes"][0]
    return f"road distance {route['distance'] / 1000:.1f} km, {route['duration'] / 60:.0f} min"


def check_ors(c: httpx.Client) -> str:
    key = env("ORS_API_KEY")
    r = request(c, "POST", "https://api.openrouteservice.org/v2/directions/driving-car",
                headers={"Authorization": key},
                json={"coordinates": [[LON, LAT], [DEST_LON, DEST_LAT]]})
    summary = r.json()["routes"][0]["summary"]
    return f"road distance {summary['distance'] / 1000:.1f} km (backup router OK)"


def check_nominatim(c: httpx.Client) -> str:
    r = request(c, "GET", "https://nominatim.openstreetmap.org/reverse", params={
        "lat": LAT, "lon": LON, "format": "jsonv2", "zoom": 10,
    }, headers={"User-Agent": os.getenv("NOMINATIM_USER_AGENT", "KhetRakshak-hackathon/0.1")})
    addr = r.json().get("address", {})
    if addr.get("country_code") != "in":
        raise Warn(f"point resolves outside India: {addr.get('country')}")
    return f"{addr.get('county') or addr.get('city') or addr.get('town')}, {addr.get('state')}, India"


# ---------------------------------------------------------------- LLM

WEATHER_TOOL_DESC = "Get the weather forecast for a latitude/longitude."


def openai_compatible_tool_call(c: httpx.Client, base: str, key: str, model: str,
                                prompt: str, schema: dict) -> str:
    """Tool-calling probe for any OpenAI-compatible endpoint (OpenAI, Gemini, Groq, ...)."""
    base = base.rstrip("/")
    headers = {"Authorization": f"Bearer {key}"}
    available = [m["id"].removeprefix("models/")
                 for m in request(c, "GET", f"{base}/models", headers=headers).json().get("data", [])]
    if available and model not in available:
        suggestions = ", ".join(sorted(m for m in available if "gemini" in m or "gpt" in m)[:12])
        raise Warn(f"model {model!r} not available for this key. Available: {suggestions}")

    r = request(c, "POST", f"{base}/chat/completions", headers=headers, json={
        "model": model, "tool_choice": "required",
        "tools": [{"type": "function", "function": {
            "name": "get_weather", "description": WEATHER_TOOL_DESC, "parameters": schema}}],
        "messages": [{"role": "user", "content": prompt}],
    }).json()
    calls = r["choices"][0]["message"].get("tool_calls") or []
    if not calls:
        raise Warn(f"{model} answered without calling the tool")
    fn = calls[0]["function"]
    return f"{model} called {fn['name']}({fn['arguments']})"


def check_llm_tool_calling(c: httpx.Client) -> str:
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    prompt = f"What is the weather forecast for the field at latitude {LAT}, longitude {LON}?"
    schema = {"type": "object", "required": ["lat", "lon"],
              "properties": {"lat": {"type": "number"}, "lon": {"type": "number"}}}

    if provider == "anthropic":
        model = env("ANTHROPIC_MODEL")
        r = request(c, "POST", "https://api.anthropic.com/v1/messages", headers={
            "x-api-key": env("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01",
        }, json={
            "model": model, "max_tokens": 256,
            "tools": [{"name": "get_weather", "description": WEATHER_TOOL_DESC,
                       "input_schema": schema}],
            "tool_choice": {"type": "any"},
            "messages": [{"role": "user", "content": prompt}],
        }).json()
        calls = [b for b in r["content"] if b["type"] == "tool_use"]
        if not calls:
            raise Warn(f"{model} answered without calling the tool")
        return f"{model} called {calls[0]['name']}({json.dumps(calls[0]['input'])})"

    if provider == "gemini":
        return openai_compatible_tool_call(
            c, os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"),
            env("GEMINI_API_KEY"), env("GEMINI_MODEL"), prompt, schema)

    if provider == "openai":
        return openai_compatible_tool_call(
            c, os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            env("OPENAI_API_KEY"), env("OPENAI_MODEL"), prompt, schema)

    raise Skip(f"unknown LLM_PROVIDER={provider!r}")


# ---------------------------------------------------------------- infra

def check_supabase(c: httpx.Client) -> str:
    url = env("SUPABASE_URL").rstrip("/")
    anon = env("SUPABASE_ANON_KEY")
    service = env("SUPABASE_SERVICE_ROLE_KEY")
    # anon key: public auth settings endpoint accepts any valid project key
    request(c, "GET", f"{url}/auth/v1/settings", headers={"apikey": anon})
    # service_role key: only it may read the REST schema root
    request(c, "GET", f"{url}/rest/v1/",
            headers={"apikey": service, "Authorization": f"Bearer {service}"})
    return "anon key and service_role key both accepted"


def check_telegram(c: httpx.Client) -> str:
    token = env("TELEGRAM_BOT_TOKEN")
    me = request(c, "GET", f"https://api.telegram.org/bot{token}/getMe").json()["result"]
    detail = f"bot @{me['username']} OK"
    chat_id = os.getenv("TELEGRAM_TEST_CHAT_ID", "").strip()
    if chat_id:
        request(c, "POST", f"https://api.telegram.org/bot{token}/sendMessage", json={
            "chat_id": chat_id, "text": f"KhetRakshak smoke test {datetime.now():%H:%M:%S}",
        })
        detail += f"; test message sent to {chat_id}"
    return detail


# ---------------------------------------------------------------- runner

CHECKS: list[tuple[str, Callable[[httpx.Client], str], bool]] = [
    ("Open-Meteo forecast", check_open_meteo_forecast, True),
    ("Open-Meteo archive", check_open_meteo_archive, True),
    ("Sentinel-2 (Planetary Computer)", check_sentinel2_stac, True),
    ("LLM tool calling", check_llm_tool_calling, True),
    ("Sarvam translate->TTS->STT", check_sarvam_chain, True),
    ("Agmarknet (data.gov.in)", check_agmarknet, True),
    ("CEDA Agri Market", check_ceda, False),
    ("OSRM routing", check_osrm, True),
    ("OpenRouteService (backup)", check_ors, False),
    ("Nominatim reverse geocode", check_nominatim, False),
    ("Supabase", check_supabase, True),
    ("Telegram bot", check_telegram, True),
]


def run_check(client: httpx.Client, name: str, fn: Callable, required: bool) -> Result:
    start = time.perf_counter()
    try:
        status, detail = "PASS", fn(client)
    except Skip as exc:
        status, detail = "SKIP", str(exc)
    except Warn as exc:
        status, detail = "WARN", str(exc)
    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:200].replace("\n", " ")
        status, detail = "FAIL", f"HTTP {exc.response.status_code}: {body}"
    except Exception as exc:  # noqa: BLE001 - smoke test must never crash
        status, detail = "FAIL", f"{type(exc).__name__}: {exc}"
    latency = int((time.perf_counter() - start) * 1000)
    return Result(name, status, latency, redact(detail), required)


def main() -> int:
    print(f"KhetRakshak smoke test  |  point ({LAT}, {LON}) -> dest ({DEST_LAT}, {DEST_LON})"
          f"  |  {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    results = []
    # data.gov.in stalls requests carrying httpx's default User-Agent; always send our own.
    with httpx.Client(timeout=TIMEOUT, follow_redirects=True,
                      headers={"User-Agent": "KhetRakshak/0.1"}) as client:
        for name, fn, required in CHECKS:
            res = run_check(client, name, fn, required)
            results.append(res)
            tag = "" if required else " (optional)"
            print(f"[{res.status:4}] {res.name}{tag}  {res.latency_ms} ms\n       {res.detail}")

    report = ROOT / "scripts" / "smoke_report.json"
    report.write_text(json.dumps({
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "results": [asdict(r) for r in results],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    counts = {s: sum(r.status == s for r in results) for s in ("PASS", "WARN", "FAIL", "SKIP")}
    blocking = [r.name for r in results if r.required and r.status in ("FAIL", "SKIP")]
    print(f"\n{counts}  ->  report: {report}")
    if blocking:
        print("BLOCKING (required, not passing): " + ", ".join(blocking))
        return 1
    print("All required checks passing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
