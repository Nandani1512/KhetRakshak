"""Weather tool: Open-Meteo forecast (16 days + 30 past days) and a 10-year climatology baseline.

Live mode   : forecast API (past_days=30 gives recent observed-model data with no archive lag).
Replay mode : archive API; the 16 days after as_of_date are *observed* history, labelled as such.
"""
from __future__ import annotations

from datetime import date, timedelta

from app.tools.base import ToolResult, http, run_with_fallbacks

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY = "temperature_2m_max,temperature_2m_min,precipitation_sum,et0_fao_evapotranspiration,wind_speed_10m_max"
BASELINE_YEARS = 10
HORIZON_DAYS = 16


def _rows(daily: dict) -> list[dict]:
    keys = [k for k in daily if k != "time"]
    return [{"date": d, **{k: daily[k][i] for k in keys}} for i, d in enumerate(daily["time"])]


async def get_weather(lat: float, lon: float, as_of: date | None = None) -> ToolResult:
    """Past 30 days + next 16 days of daily weather for the point."""
    today = date.today()
    replay = as_of is not None and as_of < today - timedelta(days=HORIZON_DAYS + 5)

    async def live() -> ToolResult:
        params = {"latitude": lat, "longitude": lon, "timezone": "Asia/Kolkata",
                  "past_days": 30, "forecast_days": HORIZON_DAYS,
                  "daily": DAILY + ",precipitation_probability_max"}
        resp = await http("GET", FORECAST_URL, source="Open-Meteo Forecast", params=params)
        rows = _rows(resp.json()["daily"])
        ref = today.isoformat()
        return ToolResult(
            data={"mode": "forecast", "as_of": ref,
                  "past": [r for r in rows if r["date"] < ref],
                  "forecast": [r for r in rows if r["date"] >= ref]},
            source="Open-Meteo Forecast API", source_url=str(resp.url), confidence=0.85)

    async def replay_attempt() -> ToolResult:
        start, end = as_of - timedelta(days=30), as_of + timedelta(days=HORIZON_DAYS - 1)
        params = {"latitude": lat, "longitude": lon, "daily": DAILY, "timezone": "Asia/Kolkata",
                  "start_date": start.isoformat(), "end_date": end.isoformat()}
        resp = await http("GET", ARCHIVE_URL, source="Open-Meteo Archive", params=params)
        rows = _rows(resp.json()["daily"])
        ref = as_of.isoformat()
        return ToolResult(
            data={"mode": "replay", "as_of": ref,
                  "past": [r for r in rows if r["date"] < ref],
                  "forecast": [r for r in rows if r["date"] >= ref]},
            source="Open-Meteo Historical Archive (ERA5)", source_url=str(resp.url), confidence=0.95,
            notes=["Historical replay: the 'forecast' window is observed reanalysis data for those dates"])

    key = f"weather:{lat:.3f}:{lon:.3f}:" + (as_of.isoformat() if replay else "live")
    attempts = [("Open-Meteo Archive", replay_attempt)] if replay else [("Open-Meteo Forecast", live)]
    return await run_with_fallbacks("Open-Meteo", attempts, cache_key=key)


async def get_climatology(lat: float, lon: float, as_of: date | None = None) -> ToolResult:
    """Same calendar window (-30..+16 days around as_of) for each of the previous 10 years.

    This is the field's own baseline: risk is judged against what is normal *for this location*.
    """
    ref = as_of or date.today()

    async def attempt() -> ToolResult:
        start = date(ref.year - BASELINE_YEARS, 1, 1)
        end = date(ref.year - 1, 12, 31)
        params = {"latitude": lat, "longitude": lon, "timezone": "Asia/Kolkata",
                  "start_date": start.isoformat(), "end_date": end.isoformat(),
                  "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum"}
        resp = await http("GET", ARCHIVE_URL, source="Open-Meteo Archive", params=params,
                          timeout=60.0)
        rows = _rows(resp.json()["daily"])
        windows: list[list[dict]] = []
        for year in range(ref.year - BASELINE_YEARS, ref.year):
            try:
                centre = ref.replace(year=year)
            except ValueError:  # 29 Feb
                centre = ref.replace(year=year, day=28)
            lo = (centre - timedelta(days=30)).isoformat()
            hi = (centre + timedelta(days=HORIZON_DAYS - 1)).isoformat()
            windows.append([r for r in rows if lo <= r["date"] <= hi])
        return ToolResult(
            data={"years": BASELINE_YEARS, "windows": windows},
            source="Open-Meteo Historical Archive (ERA5)", source_url=str(resp.url), confidence=0.95)

    key = f"climatology:{lat:.2f}:{lon:.2f}:{ref.strftime('%m-%d')}:{ref.year}"
    return await run_with_fallbacks("Open-Meteo Archive", [("Open-Meteo Archive", attempt)],
                                    cache_key=key)
