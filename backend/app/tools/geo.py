"""Field boundary validation: geometry sanity, size limits, and 'is this farmland in India?'."""
from __future__ import annotations

from pyproj import Geod
from shapely.geometry import shape
from shapely.validation import explain_validity

from app import config
from app.tools.base import http

GEOD = Geod(ellps="WGS84")
MIN_HA = 0.05   # ~5 Sentinel-2 pixels (10 m): below this NDVI is mostly edge pixels
MAX_HA = 50.0   # a single smallholder field, not a whole village
INDIA_BBOX = (68.0, 6.0, 97.5, 37.6)  # lon_min, lat_min, lon_max, lat_max


class FieldValidationError(ValueError):
    pass


def area_ha(geometry: dict) -> float:
    area, _ = GEOD.geometry_area_perimeter(shape(geometry))
    return abs(area) / 10_000


async def reverse_geocode(lat: float, lon: float) -> dict:
    """Nominatim first, BigDataCloud as fallback. Returns {country_code, state, district, source}."""
    try:
        resp = await http("GET", "https://nominatim.openstreetmap.org/reverse", source="Nominatim",
                          params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 10},
                          headers={"User-Agent": config.NOMINATIM_USER_AGENT}, retries=2)
        body = resp.json()
        if "error" in body:
            return {"country_code": None, "state": None, "district": None, "source": "Nominatim"}
        addr = body.get("address", {})
        return {"country_code": addr.get("country_code"), "state": addr.get("state"),
                "district": addr.get("state_district") or addr.get("county") or addr.get("city"),
                "source": "Nominatim (OpenStreetMap)"}
    except Exception:  # noqa: BLE001
        resp = await http("GET", "https://api.bigdatacloud.net/data/reverse-geocode-client",
                          source="BigDataCloud", params={"latitude": lat, "longitude": lon,
                                                         "localityLanguage": "en"}, retries=2)
        body = resp.json()
        admin = body.get("localityInfo", {}).get("administrative", [])
        district = next((a["name"] for a in admin if a.get("adminLevel") == 5), None)
        return {"country_code": (body.get("countryCode") or "").lower() or None,
                "state": body.get("principalSubdivision") or None, "district": district,
                "source": "BigDataCloud reverse geocoding"}


async def validate_field(geometry: dict) -> dict:
    """Raise FieldValidationError with a farmer-readable reason, or return location facts."""
    if geometry.get("type") != "Polygon" or not geometry.get("coordinates"):
        raise FieldValidationError("Please draw the field boundary as a closed shape.")
    poly = shape(geometry)
    if not poly.is_valid:
        raise FieldValidationError(f"The boundary crosses itself ({explain_validity(poly)}). Please redraw it.")
    ha = area_ha(geometry)
    if ha < MIN_HA:
        raise FieldValidationError(f"Field is {ha:.3f} ha. Satellites see 10 m squares, so fields under "
                                   f"{MIN_HA} ha are too small to monitor reliably.")
    if ha > MAX_HA:
        raise FieldValidationError(f"Field is {ha:.1f} ha. Please draw a single field (max {MAX_HA:.0f} ha).")
    c = poly.centroid
    lon_min, lat_min, lon_max, lat_max = INDIA_BBOX
    if not (lon_min <= c.x <= lon_max and lat_min <= c.y <= lat_max):
        raise FieldValidationError("This location is outside India. KhetRakshak currently covers India only.")
    try:
        place = await reverse_geocode(c.y, c.x)
    except Exception as exc:  # noqa: BLE001 - geocoders down: allow, but say so
        return {"area_ha": round(ha, 3), "lat": c.y, "lon": c.x, "state": None, "district": None,
                "note": f"location lookup unavailable ({type(exc).__name__}); state not verified"}
    if place["country_code"] is None:
        raise FieldValidationError("This point is not on land (sea or unmapped area). Please place it on your field.")
    if place["country_code"] != "in":
        raise FieldValidationError("This location is outside India. KhetRakshak currently covers India only.")
    return {"area_ha": round(ha, 3), "lat": c.y, "lon": c.x, "state": place["state"],
            "district": place["district"], "note": f"location verified via {place['source']}"}
