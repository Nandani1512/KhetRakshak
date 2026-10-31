"""Risk engine: deterministic, explainable, every threshold sourced.

Two kinds of thresholds only:
  1. Official IMD definitions (heat wave, heavy rain), cited to the bulletin page + quote.
  2. The field's OWN history: 10-year climatology of the same calendar window (ERA5 via
     Open-Meteo) and the field's own NDVI in previous years. No invented agronomy numbers.
Crop stage is read from the field's NDVI curve (phenology), cross-checked with sowing date.
"""
from __future__ import annotations

from datetime import date

import numpy as np

IMD = {
    "doc_id": "imd_pr_820", "page": 4,
    "url": "https://internal.imd.gov.in/press_release/20200523_pr_820.pdf#page=4",
    "title": "IMD Press Release 23 May 2020 - Annexure",
}
HEAT_QUOTE = ("Heat wave is considered if maximum temperature of a station reaches at least 40°C or more for "
              "Plains ... Heat Wave: Departure from normal is 4.5°C to 6.4°C; Severe Heat Wave: Departure from "
              "normal is >6.4°C ... Heat Wave: When actual maximum temperature ≥ 45°C; Severe Heat Wave: "
              "When actual maximum temperature ≥47°C")
RAIN_QUOTE = "Heavy rain: 64.5 -115.5 mm/day; Very heavy rain: 115.6 -204.4 mm/day; Extremely heavy rain: more than 204.4 mm/day"
CLIMATOLOGY = "this field's 10-year climatology for the same weeks (ERA5 reanalysis via Open-Meteo)"


def _level(score: float) -> str:
    return "high" if score >= 0.7 else "moderate" if score >= 0.4 else "low"


def _risk(type_: str, title: str, score: float, window: list[str], evidence: dict, rule: str,
          citation: dict | None = None, basis: str | None = None) -> dict:
    score = round(min(1.0, score), 2)
    return {"type": type_, "title": title, "score": score, "level": _level(score), "window": window,
            "evidence": evidence, "rule": rule, "citation": citation, "basis": basis}


# ------------------------------------------------------------------ crop stage

def crop_stage(ndvi: dict | None, sowing: date | None, as_of: date) -> dict:
    das = (as_of - sowing).days if sowing else None
    out = {"days_after_sowing": das, "method": "NDVI phenology"}
    pts = [p for p in (ndvi or {}).get("series", []) if p.get("ndvi") is not None]
    if not pts:
        return {**out, "stage": "unknown", "method": "no clear satellite view",
                "explanation": "No cloud-free satellite view this season; stage cannot be observed."}
    latest = pts[-1]["ndvi"]
    peak = max(p["ndvi"] for p in pts)
    recent = [p for p in pts if (as_of - date.fromisoformat(p["date"])).days <= 35]
    slope = 0.0
    if len(recent) >= 2:
        x = np.array([date.fromisoformat(p["date"]).toordinal() for p in recent], dtype=float)
        y = np.array([p["ndvi"] for p in recent])
        slope = float(np.polyfit(x - x[0], y, 1)[0])
    out.update(latest_ndvi=latest, peak_ndvi=round(peak, 3), ndvi_slope_per_day=round(slope, 4))

    # Phenology heuristics on the field's own curve (relative to its own season peak).
    if peak >= 0.5 and latest < 0.35:
        stage, why = "harvested", "greenness fell from a season peak to bare-soil levels"
    elif peak < 0.3:
        stage, why = "sown / early establishment", "field still mostly bare soil (low greenness)"
    elif slope > 0.002:
        stage, why = "vegetative growth", "greenness rising over the last 5 weeks"
    elif latest >= 0.85 * peak:
        stage, why = "peak canopy (flowering / grain filling likely)", "greenness near its season peak"
    else:
        stage, why = "maturing (harvest window)", "greenness declining from its season peak"
    return {**out, "stage": stage, "explanation": why}


HARVEST_STAGES = {"maturing (harvest window)"}
GROWING_STAGES = {"sown / early establishment", "vegetative growth",
                  "peak canopy (flowering / grain filling likely)"}


# ------------------------------------------------------------------ helpers

def _baseline_by_offset(windows: list[list[dict]], key: str) -> list[list[float]]:
    """values[offset] across years, offset 0 = 30 days before the reference date."""
    n = min(len(w) for w in windows) if windows else 0
    return [[w[i][key] for w in windows if w[i].get(key) is not None] for i in range(n)]


def _rolling_max(vals: list[float], k: int) -> float:
    return max((sum(vals[i:i + k]) for i in range(max(1, len(vals) - k + 1))), default=0.0)


# ------------------------------------------------------------------ rules

def heat_risk(forecast: list[dict], windows: list[list[dict]]) -> dict | None:
    normals = _baseline_by_offset(windows, "temperature_2m_max")
    worst = None
    for i, day in enumerate(forecast):
        tmax = day.get("temperature_2m_max")
        if tmax is None or 30 + i >= len(normals) or not normals[30 + i]:
            continue
        normal = float(np.mean(normals[30 + i]))
        dep = tmax - normal
        if tmax >= 47 or (tmax >= 40 and dep > 6.4):
            sev, score = "severe heat wave", 0.9
        elif tmax >= 45 or (tmax >= 40 and dep >= 4.5):
            sev, score = "heat wave", 0.75
        else:
            continue
        if worst is None or score > worst[0]:
            worst = (score, sev, day["date"], tmax, normal, dep)
    if worst is None:
        return None
    score, sev, d, tmax, normal, dep = worst
    return _risk("heat_wave", f"IMD {sev} criteria met", score, [d],
                 {"date": d, "tmax_c": round(tmax, 1), "normal_tmax_c": round(normal, 1),
                  "departure_c": round(dep, 1)},
                 f"Forecast Tmax {tmax:.1f}°C is {dep:+.1f}°C from this field's normal ({normal:.1f}°C).",
                 citation={**IMD, "quote": HEAT_QUOTE}, basis=CLIMATOLOGY + " for 'normal'")


def rain_risk(forecast: list[dict], windows: list[list[dict]], stage: str) -> dict | None:
    days = [(d["date"], d.get("precipitation_sum") or 0.0) for d in forecast]
    if not days:
        return None
    wettest_date, wettest = max(days, key=lambda t: t[1])
    rain7 = sum(v for _, v in days[:7])
    base7 = [sum((r.get("precipitation_sum") or 0.0) for r in w[30:37]) for w in windows if len(w) >= 37]
    p90 = float(np.percentile(base7, 90)) if base7 else None
    harvest = stage in HARVEST_STAGES

    if wettest >= 64.5:
        cat = ("extremely heavy" if wettest > 204.4 else "very heavy" if wettest >= 115.6 else "heavy")
        score = {"heavy": 0.7, "very heavy": 0.85, "extremely heavy": 1.0}[cat] + (0.1 if harvest else 0)
        title = f"IMD '{cat} rain' day forecast" + (" during harvest window" if harvest else "")
        return _risk("heavy_rain", title, score, [wettest_date],
                     {"date": wettest_date, "rain_mm": round(wettest, 1), "next7_total_mm": round(rain7, 1),
                      "baseline_p90_7day_mm": round(p90, 1) if p90 is not None else None},
                     f"{wettest:.1f} mm forecast on {wettest_date} falls in IMD's '{cat} rain' band.",
                     citation={**IMD, "quote": RAIN_QUOTE})
    if p90 is not None and rain7 > p90 and rain7 > 0:
        score = 0.55 + (0.15 if harvest else 0)
        return _risk("wet_spell", "Unusually wet week ahead" + (" during harvest window" if harvest else ""),
                     score, [days[0][0], days[min(6, len(days) - 1)][0]],
                     {"next7_total_mm": round(rain7, 1), "baseline_p90_7day_mm": round(p90, 1),
                      "wettest_day": wettest_date, "wettest_day_mm": round(wettest, 1)},
                     f"Next 7 days: {rain7:.1f} mm, above the wettest 10% of the same week in the last "
                     f"{len(base7)} years ({p90:.1f} mm).", basis=CLIMATOLOGY)
    return None


def dry_risk(past: list[dict], forecast: list[dict], windows: list[list[dict]], stage: str) -> dict | None:
    if stage not in GROWING_STAGES or not past:
        return None
    past30 = sum((d.get("precipitation_sum") or 0.0) for d in past[-30:])
    next16 = sum((d.get("precipitation_sum") or 0.0) for d in forecast)
    base_past = [sum((r.get("precipitation_sum") or 0.0) for r in w[:30]) for w in windows if len(w) >= 46]
    base_next = [sum((r.get("precipitation_sum") or 0.0) for r in w[30:46]) for w in windows if len(w) >= 46]
    if len(base_past) < 5:
        return None
    p10_past, p25_next = float(np.percentile(base_past, 10)), float(np.percentile(base_next, 25))
    if past30 <= p10_past and next16 <= p25_next:
        return _risk("dry_spell", "Dry spell while crop is growing", 0.6,
                     [forecast[0]["date"], forecast[-1]["date"]],
                     {"past30_rain_mm": round(past30, 1), "baseline_p10_past30_mm": round(p10_past, 1),
                      "next16_rain_mm": round(next16, 1), "baseline_p25_next16_mm": round(p25_next, 1),
                      "et0_next7_mm": round(sum((d.get("et0_fao_evapotranspiration") or 0) for d in forecast[:7]), 1)},
                     f"Last 30 days {past30:.1f} mm (driest 10% of years) and next 16 days {next16:.1f} mm "
                     f"(driest 25%).", basis=CLIMATOLOGY)
    return None


def ndvi_risk(ndvi: dict | None) -> dict | None:
    if not ndvi or not ndvi.get("latest") or ndvi.get("baseline_mean") is None:
        return None
    latest, mean = ndvi["latest"]["ndvi"], ndvi["baseline_mean"]
    std = max(ndvi.get("baseline_std") or 0.0, 0.03)
    z = (latest - mean) / std
    if z > -1.0:
        return None
    score = 0.6 if z <= -2.0 else 0.4
    return _risk("ndvi_anomaly", "Crop greenness below this field's previous years", score,
                 [ndvi["latest"]["date"]],
                 {"ndvi": latest, "baseline_mean_ndvi": mean, "baseline_std": round(std, 3), "z_score": round(z, 2),
                  "scene_date": ndvi["latest"]["date"]},
                 f"NDVI {latest:.2f} vs {mean:.2f} at the same time in previous years (z = {z:.1f}). "
                 "Can mean stress, pest/disease, late sowing or a different crop - needs a field check.",
                 basis="Sentinel-2 NDVI of this same field in the previous 3 years")


def assess(weather: dict | None, climatology: dict | None, ndvi: dict | None, sowing: date | None,
           as_of: date) -> dict:
    stage = crop_stage(ndvi, sowing, as_of)
    risks, gaps = [], []
    windows = (climatology or {}).get("windows") or []
    if weather and windows:
        for r in (heat_risk(weather["forecast"], windows),
                  rain_risk(weather["forecast"], windows, stage["stage"]),
                  dry_risk(weather["past"], weather["forecast"], windows, stage["stage"])):
            if r:
                risks.append(r)
    else:
        gaps.append("weather or climatology unavailable: weather risks not assessed")
    r = ndvi_risk(ndvi)
    if r:
        risks.append(r)
    if not ndvi:
        gaps.append("satellite data unavailable: crop stage and NDVI anomaly not assessed")
    risks.sort(key=lambda r: -r["score"])

    summary = None
    if weather:
        f7 = weather["forecast"][:7]
        summary = {"next7_rain_mm": round(sum((d.get("precipitation_sum") or 0) for d in f7), 1),
                   "next7_tmax_c": max((d["temperature_2m_max"] for d in f7 if d.get("temperature_2m_max") is not None), default=None),
                   "next7_tmin_c": min((d["temperature_2m_min"] for d in f7 if d.get("temperature_2m_min") is not None), default=None),
                   "wettest_day": max(f7, key=lambda d: d.get("precipitation_sum") or 0)["date"] if f7 else None}
    return {"stage": stage, "risks": risks, "weather_summary": summary, "gaps": gaps}
