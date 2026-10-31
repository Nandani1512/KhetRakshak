"""Satellite tool: Sentinel-2 L2A NDVI over the farmer's own field boundary.

- STAC search on Microsoft Planetary Computer for scenes intersecting the polygon.
- Windowed Cloud-Optimised GeoTIFF reads of only the field's pixels (B04 red, B08 NIR, SCL).
- Pixel-level cloud masking with the Scene Classification Layer; a scene counts only if most
  of the field is clear. Cloudy scenes are reported, not hidden.
- Baseline: the same field, same calendar window, in the previous 3 years -> NDVI anomaly.
"""
from __future__ import annotations

import asyncio
import math
import os
import time
from datetime import date, timedelta

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "1")
os.environ.setdefault("VSI_CACHE", "TRUE")

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from pyproj import Transformer  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.features import geometry_mask  # noqa: E402
from rasterio.warp import reproject  # noqa: E402
from rasterio.windows import Window, from_bounds  # noqa: E402
from shapely.geometry import mapping, shape  # noqa: E402
from shapely.ops import transform as shp_transform  # noqa: E402  (pyproj callable)

from app import db  # noqa: E402
from app.tools.base import (DISABLED_SOURCES, ToolResult, describe_error, http,  # noqa: E402
                            run_with_fallbacks)

STAC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS_TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"
SOURCE = "Sentinel-2 L2A (ESA Copernicus via Microsoft Planetary Computer)"
# SCL classes treated as a clear view of the ground: vegetation, bare soil, water, unclassified.
CLEAR_SCL = (4, 5, 6, 7)
MIN_CLEAR_FRACTION = 0.6
BASELINE_YEARS = 3

_token: dict = {"value": None, "expires": 0.0}
_read_slots = asyncio.Semaphore(8)


async def _sas_token() -> str:
    if _token["value"] and time.time() < _token["expires"]:
        return _token["value"]
    resp = await http("GET", SAS_TOKEN, source="Planetary Computer")
    _token.update(value=resp.json()["token"], expires=time.time() + 40 * 60)
    return _token["value"]


async def _search(geometry: dict, start: date, end: date, max_cloud: int, limit: int = 100) -> list[dict]:
    body = {
        "collections": ["sentinel-2-l2a"],
        "intersects": geometry,
        "datetime": f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
        "limit": limit,
    }
    resp = await http("POST", STAC_SEARCH, source="Planetary Computer", json=body)
    scenes = resp.json().get("features", [])
    # Overlapping tiles give the same acquisition twice: keep the least cloudy per date.
    best: dict[str, dict] = {}
    for s in scenes:
        d = s["properties"]["datetime"][:10]
        if d not in best or s["properties"]["eo:cloud_cover"] < best[d]["properties"]["eo:cloud_cover"]:
            best[d] = s
    return [best[d] for d in sorted(best)]


def _read_scene(scene: dict, geometry: dict, token: str) -> dict:
    """Blocking: mean NDVI over clear pixels inside the polygon for one scene."""
    props = scene["properties"]
    # Processing baseline >= 04.00 (Jan 2022 onward) adds a -1000 radiometric offset to L2A DNs.
    offset = 1000 if float(props.get("s2:processing_baseline", "0") or 0) >= 4.0 else 0
    out = {"date": props["datetime"][:10], "scene_id": scene["id"],
           "scene_cloud_pct": round(props["eo:cloud_cover"], 1)}
    with rasterio.Env():
        with rasterio.open(f"{scene['assets']['B04']['href']}?{token}") as red_ds:
            to_crs = Transformer.from_crs("EPSG:4326", red_ds.crs, always_xy=True).transform
            poly = shp_transform(to_crs, shape(geometry))
            w = from_bounds(*poly.bounds, transform=red_ds.transform)
            window = Window(math.floor(w.col_off), math.floor(w.row_off),
                            math.ceil(w.width) + 1, math.ceil(w.height) + 1)
            red = red_ds.read(1, window=window, boundless=True, fill_value=0).astype("float32")
            win_transform = red_ds.window_transform(window)
            red_ds_crs = red_ds.crs
        with rasterio.open(f"{scene['assets']['B08']['href']}?{token}") as nir_ds:
            nir = nir_ds.read(1, window=window, boundless=True, fill_value=0).astype("float32")
        with rasterio.open(f"{scene['assets']['SCL']['href']}?{token}") as scl_ds:
            # SCL is 20 m: read a padded window, then reproject onto the 10 m field grid.
            sw = from_bounds(*poly.bounds, transform=scl_ds.transform)
            scl_window = Window(math.floor(sw.col_off) - 1, math.floor(sw.row_off) - 1,
                                math.ceil(sw.width) + 3, math.ceil(sw.height) + 3)
            scl_src = scl_ds.read(1, window=scl_window, boundless=True, fill_value=0)
            scl = np.zeros(red.shape, dtype=scl_src.dtype)
            reproject(scl_src, scl, src_transform=scl_ds.window_transform(scl_window),
                      src_crs=scl_ds.crs, dst_transform=win_transform, dst_crs=red_ds_crs,
                      resampling=Resampling.nearest)

    inside = geometry_mask([mapping(poly)], out_shape=red.shape, transform=win_transform, invert=True)
    n_inside = int(inside.sum())
    if n_inside == 0:
        return {**out, "ndvi": None, "clear_fraction": 0.0, "pixels": 0, "status": "no pixels"}
    sky_clear = inside & np.isin(scl, CLEAR_SCL)
    # Valid surface reflectance must sit above the radiometric offset; some freshly published
    # scenes carry placeholder values (DN <= 255) and must not be mistaken for real data.
    clear = sky_clear & (red > offset + 100) & (nir > offset + 100)
    clear_fraction = float(clear.sum()) / n_inside
    if clear_fraction < MIN_CLEAR_FRACTION:
        bad_data = float(sky_clear.sum()) / n_inside >= MIN_CLEAR_FRACTION
        return {**out, "ndvi": None, "clear_fraction": round(clear_fraction, 2), "pixels": n_inside,
                "status": "invalid sensor data" if bad_data else "cloudy"}
    r = (red[clear] - offset) / 10000.0
    n = (nir[clear] - offset) / 10000.0
    ndvi = (n - r) / np.clip(n + r, 1e-6, None)
    return {**out, "ndvi": round(float(np.mean(ndvi)), 3), "ndvi_std": round(float(np.std(ndvi)), 3),
            "clear_fraction": round(clear_fraction, 2), "pixels": n_inside, "status": "clear"}


async def _scene_ndvi(scene: dict, geometry: dict, field_key: str, token: str) -> dict:
    """Per-scene result, cached forever: a past acquisition never changes."""
    key = f"ndvi:v3:{field_key}:{scene['id']}"
    cached = await db.cache_get(key)
    if cached is not None and cached.data:
        return cached.data
    async with _read_slots:
        try:
            result = await asyncio.to_thread(_read_scene, scene, geometry, token)
        except Exception as exc:  # noqa: BLE001
            return {"date": scene["properties"]["datetime"][:10], "scene_id": scene["id"],
                    "ndvi": None, "status": f"read failed: {describe_error(exc)}"}
    if result["status"] in ("clear", "cloudy"):
        await db.cache_put(key, ToolResult(data=result, source=SOURCE))
    return result


def field_key(geometry: dict) -> str:
    c = shape(geometry).centroid
    return f"{c.y:.5f},{c.x:.5f}:{len(geometry['coordinates'][0])}"


async def get_ndvi(geometry: dict, season_start: date, as_of: date | None = None) -> ToolResult:
    """Season NDVI time series for the polygon + same-window baseline from previous years."""
    ref = as_of or date.today()
    start = min(season_start, ref - timedelta(days=30))
    start = max(start, ref - timedelta(days=200))
    fkey = field_key(geometry)

    async def attempt() -> ToolResult:
        if "Planetary Computer" in DISABLED_SOURCES:
            raise RuntimeError("Planetary Computer disabled by operator")
        token = await _sas_token()
        season = await _search(geometry, start, ref, max_cloud=80)
        season_pts = await asyncio.gather(*(_scene_ndvi(s, geometry, fkey, token) for s in season))

        baseline: list[dict] = []
        for back in range(1, BASELINE_YEARS + 1):
            try:
                centre = ref.replace(year=ref.year - back)
            except ValueError:
                centre = ref.replace(year=ref.year - back, day=28)
            scenes = await _search(geometry, centre - timedelta(days=20), centre + timedelta(days=20),
                                   max_cloud=40)
            scenes = sorted(scenes, key=lambda s: s["properties"]["eo:cloud_cover"])[:3]
            pts = await asyncio.gather(*(_scene_ndvi(s, geometry, fkey, token) for s in scenes))
            baseline.extend({**p, "year": centre.year} for p in pts)

        clear = [p for p in season_pts if p.get("ndvi") is not None]
        latest = clear[-1] if clear else None
        notes, confidence = [], 0.9
        if latest is None:
            confidence = 0.0
            notes.append("No clear view of this field this season (cloud cover). NDVI unavailable.")
        else:
            age = (ref - date.fromisoformat(latest["date"])).days
            confidence = round(max(0.3, 0.95 - 0.02 * age) * latest["clear_fraction"], 2)
            cloudy_since = [p for p in season_pts if p["date"] > latest["date"] and p.get("status") != "clear"]
            if cloudy_since:
                notes.append(f"{len(cloudy_since)} newer scene(s) were cloudy or unusable over the field; "
                             f"using last clear view from {latest['date']} ({age} days old).")
        base_vals = [p["ndvi"] for p in baseline if p.get("ndvi") is not None]
        return ToolResult(
            data={"series": season_pts, "latest": latest, "baseline": baseline,
                  "baseline_mean": round(float(np.mean(base_vals)), 3) if base_vals else None,
                  "baseline_std": round(float(np.std(base_vals)), 3) if len(base_vals) > 1 else None,
                  "scenes_total": len(season_pts), "scenes_clear": len(clear)},
            source=SOURCE, source_url=STAC_SEARCH, confidence=confidence, notes=notes)

    return await run_with_fallbacks("Sentinel-2", [("Planetary Computer", attempt)],
                                    cache_key=f"ndvi-series:{fkey}:{ref.isoformat()}")
