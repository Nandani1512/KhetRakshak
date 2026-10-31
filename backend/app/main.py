"""KhetRakshak API + dashboard (single deployable: FastAPI serves the static frontend)."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from typing import Literal

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import config, db, llm, telegram_bot
from app.agents import events
from app.agents.graph import SUPPORTED_CROPS, start_run
from app.rag.retrieve import index_stats
from app.risk.engine import assess
from app.tools import geo, satellite, voice, weather
from app.tools.base import DISABLED_SOURCES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# httpx logs full request URLs, and Telegram puts the bot token in the URL: keep those out of logs.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger("khetrakshak")
IST = timezone(timedelta(hours=5, minutes=30))
state = {"telegram": "not started", "started_at": datetime.now(timezone.utc).isoformat()}


async def _daily_monitoring() -> None:
    """Every day at 06:30 IST, run monitoring for every field linked to Telegram."""
    while True:
        now = datetime.now(IST)
        nxt = now.replace(hour=6, minute=30, second=0, microsecond=0)
        if nxt <= now:
            nxt += timedelta(days=1)
        await asyncio.sleep((nxt - now).total_seconds())
        for f in await db.list_fields():
            if f.get("telegram_chat_id"):
                try:
                    await start_run(f, "monitor", background=False)
                except Exception as exc:  # noqa: BLE001
                    log.warning("scheduled run failed for %s: %s", f["id"], exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(db.init)
    state["telegram"] = await telegram_bot.start()
    task = asyncio.create_task(_daily_monitoring())
    yield
    task.cancel()


app = FastAPI(title="KhetRakshak API", version="1.0.0", lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"error": "internal error", "detail": type(exc).__name__})


# ------------------------------------------------------------------ health

@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat(), "started_at": state["started_at"],
            "database": db.status, "rag_index": index_stats(), "telegram": state["telegram"],
            "telegram_bot": telegram_bot.bot_username, "llm": {"model": config.GEMINI_MODEL, **llm.stats},
            "disabled_sources": sorted(DISABLED_SOURCES), "crops": SUPPORTED_CROPS,
            "languages": voice.LANGUAGES}


# ------------------------------------------------------------------ fields

class FieldIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    farmer_name: str | None = Field(default=None, max_length=80)
    polygon: dict
    crop: str
    sowing_date: date | None = None
    language: Literal["hi-IN", "ta-IN", "pa-IN", "en-IN"] = "hi-IN"
    transport_rate_per_qtl_km: float | None = Field(default=None, ge=0, le=100)


async def _prewarm(field: dict) -> None:
    """Fetch satellite history in the background so the first run is fast."""
    try:
        start = date.fromisoformat(field["sowing_date"]) if field.get("sowing_date") else date.today() - timedelta(days=150)
        await satellite.get_ndvi(field["polygon"], start)
    except Exception as exc:  # noqa: BLE001
        log.warning("prewarm failed: %s", exc)


@app.post("/api/fields", status_code=201)
async def create_field(body: FieldIn) -> dict:
    crop = body.crop.strip().lower()
    if crop not in SUPPORTED_CROPS:
        raise HTTPException(422, f"Crop '{body.crop}' is not supported yet (supported: {', '.join(SUPPORTED_CROPS)}). "
                                 "Please contact your local KVK for advice on this crop.")
    today = date.today()
    if body.sowing_date and body.sowing_date > today:
        raise HTTPException(422, "Sowing date is in the future. Please enter the date the crop was actually sown.")
    if body.sowing_date and body.sowing_date < today - timedelta(days=270):
        raise HTTPException(422, "Sowing date is more than 9 months ago - that crop is already harvested. "
                                 "Please enter the current season's sowing date.")
    try:
        place = await geo.validate_field(body.polygon)
    except geo.FieldValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    field = {"id": uuid.uuid4().hex[:10], "name": body.name.strip(), "farmer_name": body.farmer_name,
             "polygon": body.polygon, "centroid_lat": place["lat"], "centroid_lon": place["lon"],
             "area_ha": place["area_ha"], "crop": crop,
             "sowing_date": body.sowing_date.isoformat() if body.sowing_date else None,
             "language": body.language, "state": place["state"], "district": place["district"],
             "transport_rate_per_qtl_km": body.transport_rate_per_qtl_km}
    saved = await db.create_field(field)
    asyncio.create_task(_prewarm(saved))
    return {**saved, "validation_note": place["note"]}


@app.get("/api/fields")
async def list_fields() -> list[dict]:
    return await db.list_fields()


async def _field_or_404(field_id: str) -> dict:
    f = await db.get_field(field_id)
    if not f:
        raise HTTPException(404, "field not found")
    return f


@app.get("/api/fields/{field_id}")
async def get_field(field_id: str) -> dict:
    f = await _field_or_404(field_id)
    return {**f, "runs": await db.list_runs(field_id)}


@app.delete("/api/fields/{field_id}", status_code=204)
async def delete_field(field_id: str) -> Response:
    await db.delete_field(field_id)
    return Response(status_code=204)


@app.get("/api/fields/{field_id}/overview")
async def overview(field_id: str, as_of: date | None = None) -> dict:
    """Charts for the dashboard: weather, NDVI, stage and rule-based risks (no LLM)."""
    f = await _field_or_404(field_id)
    ref = as_of or date.today()
    replay = ref != date.today()
    sowing = date.fromisoformat(f["sowing_date"]) if f.get("sowing_date") else None
    w, c, n = await asyncio.gather(
        weather.get_weather(f["centroid_lat"], f["centroid_lon"], ref if replay else None),
        weather.get_climatology(f["centroid_lat"], f["centroid_lon"], ref),
        satellite.get_ndvi(f["polygon"], sowing or ref - timedelta(days=150), ref))
    a = assess(w.data, c.data, n.data, sowing, ref)
    return {"as_of": ref.isoformat(), "replay": replay, "weather": w.model_dump(), "ndvi": n.model_dump(),
            "climatology": {k: v for k, v in c.model_dump().items() if k != "data"}, "assessment": a}


@app.get("/api/fields/{field_id}/telegram-link")
async def telegram_link(field_id: str) -> dict:
    await _field_or_404(field_id)
    if not telegram_bot.bot_username:
        raise HTTPException(503, "Telegram bot not available")
    return {"url": f"https://t.me/{telegram_bot.bot_username}?start={field_id}"}


# ------------------------------------------------------------------ agent runs

class RunIn(BaseModel):
    field_id: str
    mode: Literal["monitor", "ask"] = "monitor"
    question: str | None = Field(default=None, max_length=1500)
    as_of: date | None = None
    language: Literal["hi-IN", "ta-IN", "pa-IN", "en-IN"] | None = None


@app.post("/api/runs", status_code=202)
async def create_run(body: RunIn) -> dict:
    f = await _field_or_404(body.field_id)
    if body.mode == "ask" and not (body.question or "").strip():
        raise HTTPException(422, "Please type or speak a question.")
    if body.as_of and body.as_of > date.today():
        raise HTTPException(422, "Replay date cannot be in the future.")
    if body.as_of and body.as_of < date(2018, 1, 1):
        raise HTTPException(422, "Replay is available from 2018 onward (Sentinel-2 L2A archive).")
    rid = await start_run(f, body.mode, body.as_of, body.question, body.language)
    return {"run_id": rid}


@app.post("/api/fields/{field_id}/ask-voice", status_code=202)
async def ask_voice(field_id: str, audio: UploadFile = File(...), language: str | None = Form(None)) -> dict:
    f = await _field_or_404(field_id)
    blob = await audio.read()
    if len(blob) < 1000:
        raise HTTPException(422, "Recording too short. Please hold the button and speak.")
    if len(blob) > 10_000_000:
        raise HTTPException(422, "Recording too long. Please keep questions under a minute.")
    try:
        heard = await voice.stt(blob)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"Speech recognition unavailable right now ({type(exc).__name__}). "
                                 "Please type your question.") from exc
    if not heard["transcript"]:
        raise HTTPException(422, "Sorry, I could not hear a question. Please speak again, closer to the mic.")
    lang = language if language in voice.LANGUAGES else (heard["language"] if heard["language"] in voice.LANGUAGES else None)
    rid = await start_run(f, "ask", question=heard["transcript"], language=lang)
    return {"run_id": rid, "transcript": heard["transcript"], "detected_language": heard["language"]}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    run = await db.get_run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    return run


@app.get("/api/runs/{run_id}/stream")
async def stream_run(run_id: str):
    if not events.is_known(run_id):
        run = await db.get_run(run_id)
        if not run:
            raise HTTPException(404, "run not found")

        async def replay():
            for ev in run.get("events", []):
                yield f"data: {json.dumps(ev, default=str, ensure_ascii=False)}\n\n"
            yield "event: end\ndata: {}\n\n"
        return StreamingResponse(replay(), media_type="text/event-stream")

    async def live():
        async for ev in events.stream(run_id):
            yield f"data: {json.dumps(ev, default=str, ensure_ascii=False)}\n\n"
        yield "event: end\ndata: {}\n\n"
    return StreamingResponse(live(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/runs/{run_id}/audio")
async def run_audio(run_id: str) -> Response:
    audio = events.artifacts.get(run_id, {}).get("audio")
    if not audio:
        raise HTTPException(404, "no audio for this run (generated audio is kept in memory for the session)")
    return Response(audio, media_type="audio/wav")


# ------------------------------------------------------------------ telegram + admin

@app.post("/api/telegram/webhook")
async def telegram_webhook(request: Request,
                           x_telegram_bot_api_secret_token: str | None = Header(default=None)) -> dict:
    if config.TELEGRAM_WEBHOOK_SECRET and x_telegram_bot_api_secret_token != config.TELEGRAM_WEBHOOK_SECRET:
        raise HTTPException(403, "bad secret")
    asyncio.create_task(telegram_bot.handle_update(await request.json()))
    return {"ok": True}


class SourceToggle(BaseModel):
    source: Literal["Open-Meteo Forecast", "Open-Meteo Archive", "Planetary Computer", "Agmarknet", "OSRM",
                    "Gemini", "Sarvam"]
    enabled: bool


@app.post("/api/admin/sources")
async def toggle_source(body: SourceToggle, x_admin_token: str | None = Header(default=None)) -> dict:
    """Demo control: switch a live data source off to show graceful degradation."""
    if config.ADMIN_TOKEN and x_admin_token != config.ADMIN_TOKEN:
        raise HTTPException(403, "admin token required")
    (DISABLED_SOURCES.discard if body.enabled else DISABLED_SOURCES.add)(body.source)
    return {"disabled_sources": sorted(DISABLED_SOURCES)}


# ------------------------------------------------------------------ frontend

app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(config.STATIC_DIR / "index.html")
