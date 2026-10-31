"""Market tool: today's Agmarknet mandi prices near the field, with road distance and net price.

Prices   : data.gov.in Agmarknet "Current Daily Price" (official, Ministry of Agriculture).
Location : mandi names geocoded with Open-Meteo Geocoding (GeoNames); district HQ if not found.
Distance : OSRM road distance -> OpenRouteService -> straight-line (labelled) if routing is down.
Net price: modal price minus the farmer's OWN transport rate (Rs/quintal/km). Never assumed.
"""
from __future__ import annotations

import asyncio
import math
import re

from app import config
from app.tools.base import ToolResult, describe_error, http, run_with_fallbacks

AGMARKNET_URL = f"https://api.data.gov.in/resource/{config.AGMARKNET_RESOURCE_ID}"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
COMMODITY = {"paddy": "Paddy(Dhan)(Common)", "wheat": "Wheat", "mustard": "Mustard"}
MAX_MANDIS = 8

_geo_cache: dict[str, tuple[float, float, str] | None] = {}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


async def _agmarknet(state: str, commodity: str) -> tuple[list[dict], str]:
    base = {"api-key": config.DATA_GOV_IN_API_KEY, "format": "json", "limit": 500,
            "filters[state.keyword]": state}
    resp = await http("GET", AGMARKNET_URL, source="Agmarknet",
                      params={**base, "filters[commodity]": commodity}, timeout=60.0, retries=2)
    records = resp.json().get("records") or []
    if not records:  # commodity filter spelling varies; filter client-side instead
        resp = await http("GET", AGMARKNET_URL, source="Agmarknet", params=base, timeout=60.0, retries=2)
        records = [r for r in resp.json().get("records") or []
                   if r.get("commodity", "").lower().startswith(commodity.split("(")[0].lower())]
    url = str(resp.url).replace(config.DATA_GOV_IN_API_KEY, "<key>")
    return records, url


def _clean_market_name(name: str) -> str:
    name = re.sub(r"\(.*?\)", " ", name)
    name = re.sub(r"\b(APMC|Mandi|Grain Market|New Grain Market|Market Yard|Sabzi Mandi)\b", " ", name, flags=re.I)
    return re.sub(r"\s+", " ", name).strip()


async def _geocode(place: str, state: str) -> tuple[float, float, str] | None:
    key = f"{place}|{state}"
    if key in _geo_cache:
        return _geo_cache[key]
    result = None
    try:
        resp = await http("GET", GEOCODE_URL, source="Open-Meteo Geocoding",
                          params={"name": place, "count": 10, "countryCode": "IN", "format": "json"}, retries=2)
        for hit in resp.json().get("results") or []:
            if hit.get("admin1", "").lower() == state.lower():
                result = (hit["latitude"], hit["longitude"], hit["name"])
                break
    except Exception:  # noqa: BLE001 - geocoding is best-effort per mandi
        result = None
    _geo_cache[key] = result
    return result


async def _road_distances(origin: tuple[float, float], dests: list[tuple[float, float]]) -> tuple[list, str]:
    coords = ";".join(f"{lon},{lat}" for lat, lon in [origin, *dests])
    try:
        resp = await http("GET", f"{config.OSRM_BASE_URL}/table/v1/driving/{coords}", source="OSRM",
                          params={"sources": "0", "annotations": "distance,duration"}, retries=2)
        row = resp.json()["distances"][0][1:]
        return [d / 1000 if d is not None else None for d in row], "OSRM road network (OpenStreetMap)"
    except Exception as osrm_exc:  # noqa: BLE001
        if config.ORS_API_KEY:
            try:
                resp = await http("POST", "https://api.openrouteservice.org/v2/matrix/driving-car",
                                  source="OpenRouteService", headers={"Authorization": config.ORS_API_KEY},
                                  json={"locations": [[lon, lat] for lat, lon in [origin, *dests]],
                                        "sources": [0], "metrics": ["distance"], "units": "km"}, retries=2)
                return resp.json()["distances"][0][1:], "OpenRouteService road network"
            except Exception:  # noqa: BLE001
                pass
        straight = [haversine_km(*origin, *d) for d in dests]
        return straight, f"straight-line distance (routing unavailable: {describe_error(osrm_exc)})"


async def get_market(lat: float, lon: float, state: str, crop: str,
                     transport_rate: float | None = None) -> ToolResult:
    commodity = COMMODITY.get(crop)
    if commodity is None:
        return ToolResult(source="Agmarknet", error=f"no mandi commodity mapping for crop '{crop}'", confidence=0)
    if not config.DATA_GOV_IN_API_KEY:
        return ToolResult(source="Agmarknet", error="DATA_GOV_IN_API_KEY not configured", confidence=0)

    async def attempt() -> ToolResult:
        records, url = await _agmarknet(state, commodity)
        if not records:
            raise RuntimeError(f"no {commodity} arrivals reported in {state} today")
        by_market: dict[str, dict] = {}
        for r in records:
            by_market.setdefault(r["market"], r)
        located = await asyncio.gather(*(
            _geocode(_clean_market_name(m), state) for m in by_market))
        mandis = []
        for (market, r), loc in zip(by_market.items(), located):
            try:
                prices = {k: float(r[k]) for k in ("modal_price", "min_price", "max_price")}
            except (KeyError, TypeError, ValueError):
                continue  # e.g. "NR" (not reported) in a price column
            approx = False
            if loc is None:
                loc = await _geocode(r["district"], state)
                approx = True
            if loc is None:
                continue
            mandis.append({"market": market, "district": r["district"], "variety": r.get("variety"),
                           **prices, "arrival_date": r["arrival_date"],
                           "lat": loc[0], "lon": loc[1], "location_approx": approx})
        if not mandis:
            raise RuntimeError("reporting mandis could not be located")
        mandis.sort(key=lambda m: haversine_km(lat, lon, m["lat"], m["lon"]))
        mandis = mandis[:MAX_MANDIS]
        distances, dist_source = await _road_distances((lat, lon), [(m["lat"], m["lon"]) for m in mandis])
        for m, km in zip(mandis, distances):
            m["distance_km"] = round(km, 1) if km is not None else None
            if transport_rate is not None and km is not None:
                m["transport_cost_per_qtl"] = round(transport_rate * km, 0)
                m["net_price_per_qtl"] = round(m["modal_price"] - transport_rate * km, 0)
        rank_key = "net_price_per_qtl" if transport_rate is not None else "modal_price"
        best = max((m for m in mandis if m.get(rank_key) is not None), key=lambda m: m[rank_key], default=None)
        notes = [f"distances: {dist_source}"]
        if transport_rate is None:
            notes.append("No transport rate given by farmer: ranked by modal price, net price not computed.")
        if any(m["location_approx"] for m in mandis):
            notes.append("Some mandis located at district level (exact market not geocoded).")
        return ToolResult(
            data={"commodity": commodity, "state": state, "mandis": mandis, "best": best,
                  "ranked_by": rank_key, "transport_rate": transport_rate},
            source="Agmarknet daily mandi prices (data.gov.in, Ministry of Agriculture & Farmers Welfare)",
            source_url=url, confidence=0.8 if "straight-line" not in dist_source else 0.6, notes=notes)

    return await run_with_fallbacks("Agmarknet", [("Agmarknet", attempt)],
                                    cache_key=f"market:{state}:{crop}")
