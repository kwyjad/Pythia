# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Country boundaries + point-to-territory distances for hazard detection.

Geometry choice (documented per the Phase 1 design):

- Boundaries are a vendored, slimmed Natural Earth 1:50m Admin-0
  countries layer (``data/ne_50m_admin_0_countries.slim.geojson.gz``,
  public domain).  1:50m — not 1:110m — because the coarser layer drops
  the small island states (Tonga, Tuvalu, Kiribati, …) that matter most
  for tropical-cyclone detection.  Vendoring — not runtime download —
  because CI tests must be network-free and detection must not depend on
  a CDN.  Regenerate with ``data/build_boundaries.py``.
- Distances are computed by re-projecting the (locally clipped) country
  geometry into an azimuthal equidistant (AEQD) projection centred on
  the track point and measuring to the origin.  AEQD preserves true
  distance from its centre, so the result is the geodesic distance from
  the point to the nearest point of the country's territory — including
  0 for points over land.  This is exact where it matters (near the
  buffer threshold) and immune to the lon/lat-degree anisotropy that a
  naive degree-based buffer would suffer.
- The antimeridian is handled twice over: the clip window is split at
  ±180°, and the AEQD projection itself is periodic in longitude.

The buffer test itself ("within ``cyclone.buffer_km`` of territory")
lives in ``detect.py``; this module only answers distance questions.
"""

from __future__ import annotations

import gzip
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger(__name__)

BOUNDARIES_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "ne_50m_admin_0_countries.slim.geojson.gz"
)

# Metres-per-degree constants used ONLY for conservative pre-filter
# windows (never for the trigger decision itself, which is always the
# exact AEQD distance).
_KM_PER_DEG_LAT = 110.574
_KM_PER_DEG_LON_EQUATOR = 111.320
# The clip/prefilter window is this factor larger than the distance cap
# it must contain.  Any factor >= 1 yields identical trigger decisions;
# the slack only buys evidence about near misses, so this is an
# implementation constant, not a rulebook threshold.
_WINDOW_SLACK = 1.35


@dataclass(frozen=True)
class Country:
    """A country's territory geometry, ready for distance queries."""

    iso3: str
    name: str
    geom: object  # shapely BaseGeometry
    bounds: tuple[float, float, float, float]  # (minx, miny, maxx, maxy)


def load_country_geometries(path: Path | str | None = None) -> dict[str, "Country"]:
    """Load the vendored boundary layer as ``{iso3: Country}``.

    ``path`` may point at an alternative GeoJSON (optionally .gz) — used
    by tests to inject synthetic geometries.
    """
    from shapely.geometry import shape

    src = Path(path) if path else BOUNDARIES_PATH
    opener = gzip.open if src.suffix == ".gz" else open
    with opener(src, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
        gj = json.load(fh)

    countries: dict[str, Country] = {}
    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        iso3 = str(props.get("iso3", "")).strip().upper()
        if len(iso3) != 3:
            continue
        geom = shape(feat["geometry"])
        if not geom.is_valid:
            geom = geom.buffer(0)
        countries[iso3] = Country(
            iso3=iso3,
            name=str(props.get("name", iso3)),
            geom=geom,
            bounds=geom.bounds,
        )
    LOG.info("Loaded %d country geometries from %s", len(countries), src)
    return countries


def _window_deg(lat: float, max_km: float) -> tuple[float, float]:
    """Half-window sizes (dlat, dlon) in degrees covering ``max_km``."""
    dlat = max_km / _KM_PER_DEG_LAT * _WINDOW_SLACK
    coslat = max(math.cos(math.radians(lat)), 0.05)
    dlon = max_km / (_KM_PER_DEG_LON_EQUATOR * coslat) * _WINDOW_SLACK
    return dlat, dlon


def point_near_bounds(country: Country, lat: float, lon: float, max_km: float) -> bool:
    """Cheap conservative pre-filter: could the point be within ``max_km``?

    Tests the point (and its ±360° longitude copies, for the
    antimeridian) against the country's bounding box expanded by a
    conservative degree conversion of ``max_km``.  False positives are
    fine (the exact test follows); false negatives are not, so the
    window errs large.
    """
    minx, miny, maxx, maxy = country.bounds
    dlat, dlon = _window_deg(lat, max_km)
    if not (miny - dlat <= lat <= maxy + dlat):
        return False
    for shifted in (lon, lon - 360.0, lon + 360.0):
        if minx - dlon <= shifted <= maxx + dlon:
            return True
    return False


def _clip_windows(lat: float, lon: float, max_km: float) -> list[tuple[float, float, float, float]]:
    """Clip rectangles around the point, split at the antimeridian."""
    dlat, dlon = _window_deg(lat, max_km)
    miny = max(lat - dlat, -90.0)
    maxy = min(lat + dlat, 90.0)
    minx = lon - dlon
    maxx = lon + dlon
    if maxx - minx >= 360.0:
        return [(-180.0, miny, 180.0, maxy)]
    windows: list[tuple[float, float, float, float]] = []
    if minx < -180.0:
        windows.append((minx + 360.0, miny, 180.0, maxy))
        minx = -180.0
    if maxx > 180.0:
        windows.append((-180.0, miny, maxx - 360.0, maxy))
        maxx = 180.0
    windows.append((minx, miny, maxx, maxy))
    return windows


def distance_km(country: Country, lat: float, lon: float, max_km: float) -> float | None:
    """Distance in km from a track point to the country's territory.

    Returns 0.0 for points over the territory, the geodesic distance to
    the nearest territory point when it is within ``max_km`` (plus
    window slack), and ``None`` when the territory is definitely
    further than ``max_km`` away.
    """
    import pyproj
    from shapely import clip_by_rect
    from shapely.geometry import Point
    from shapely.ops import transform as shp_transform

    pieces = []
    for minx, miny, maxx, maxy in _clip_windows(lat, lon, max_km):
        piece = clip_by_rect(country.geom, minx, miny, maxx, maxy)
        if not piece.is_empty:
            pieces.append(piece)
    if not pieces:
        return None

    aeqd = pyproj.Proj(proj="aeqd", lat_0=lat, lon_0=lon, datum="WGS84", units="m")
    transformer = pyproj.Transformer.from_proj(
        pyproj.Proj("EPSG:4326"), aeqd, always_xy=True
    )
    origin = Point(0.0, 0.0)
    best_m = math.inf
    for piece in pieces:
        projected = shp_transform(transformer.transform, piece)
        d = projected.distance(origin)
        if d < best_m:
            best_m = d
    if not math.isfinite(best_m):
        return None
    return best_m / 1000.0
