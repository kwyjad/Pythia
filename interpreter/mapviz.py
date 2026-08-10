# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The attention map, as a self-contained SVG.

The dashboard draws this map with a JavaScript component. A PDF cannot run
JavaScript, so the printed report needs its own drawing. It reads the same
vendored Natural Earth boundaries the resolution machine uses, projects them
with a plain equirectangular transform, and fills each country by how far its
forecasts sit from their usual pattern.

No plotting library, no network, no runtime download. The map is a few hundred
kilobytes of inline SVG and travels inside the PDF.
"""

from __future__ import annotations

import functools
import gzip
import html
import json
import logging
from pathlib import Path
from typing import Any, Iterable

from interpreter import names

LOGGER = logging.getLogger(__name__)

# Vendored by the PA resolution machine (data/build_boundaries.py). 1:50m,
# because 1:110m drops the small island states that matter most for cyclones.
BOUNDARIES = (
    Path(__file__).resolve().parents[1]
    / "resolver" / "hazard_resolution" / "data"
    / "ne_50m_admin_0_countries.slim.geojson.gz"
)

# Fred's palette, light to dark. A country with no forecast is left in the
# "no data" grey, which must not read as "nothing is happening there".
NO_DATA = "#E8E8E8"
LAND_STROKE = "#FFFFFF"
SCALE = ("#DCE9F0", "#A9C9DA", "#6FA5C0", "#3B7F9F", "#156082")
SCALE_BREAKS = (0.1, 0.25, 0.4, 0.6)

# Web Mercator is wrong for a world choropleth (it triples Greenland), and the
# report is about people, not ice. Equirectangular, clipped to the latitudes
# where the countries are, keeps areas honest enough to read.
_LAT_MIN, _LAT_MAX = -58.0, 84.0
_WIDTH = 960.0


@functools.lru_cache(maxsize=1)
def _features() -> list[dict[str, Any]]:
    try:
        with gzip.open(BOUNDARIES, "rt", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:  # noqa: BLE001 - a missing map is not a failed report
        LOGGER.warning("[interpreter.map] boundaries unavailable: %s", exc)
        return []
    return list(data.get("features") or [])


def colour_for(value: float | None) -> str:
    """Attention value in [0, 1] to a fill."""
    if value is None:
        return NO_DATA
    try:
        v = float(value)
    except (TypeError, ValueError):
        return NO_DATA
    for i, edge in enumerate(SCALE_BREAKS):
        if v < edge:
            return SCALE[i]
    return SCALE[-1]


def _project(lon: float, lat: float, height: float) -> tuple[float, float]:
    x = (lon + 180.0) / 360.0 * _WIDTH
    y = (_LAT_MAX - lat) / (_LAT_MAX - _LAT_MIN) * height
    return x, y


def _rings(geometry: dict[str, Any]) -> Iterable[list[list[float]]]:
    kind = geometry.get("type")
    coords = geometry.get("coordinates") or []
    if kind == "Polygon":
        yield from coords
    elif kind == "MultiPolygon":
        for polygon in coords:
            yield from polygon


def _path_for(geometry: dict[str, Any], height: float) -> str:
    """One SVG path per country.

    Coordinates are rounded to the pixel and consecutive duplicates dropped.
    The 1:50m source carries far more detail than a 960px-wide map can show,
    and keeping it would put megabytes of invisible precision into every PDF.
    """
    parts: list[str] = []
    for ring in _rings(geometry):
        if len(ring) < 3:
            continue
        points: list[str] = []
        last: tuple[int, int] | None = None
        for point in ring:
            try:
                x, y = _project(float(point[0]), float(point[1]), height)
            except (TypeError, ValueError, IndexError):
                continue
            key = (round(x), round(y))
            if key == last:
                continue
            last = key
            points.append(f"{key[0]},{key[1]}")
        if len(points) < 3:
            continue
        parts.append("M" + "L".join(points) + "Z")
    return "".join(parts)


def attention_map_svg(
    values_by_iso3: dict[str, float],
    *,
    height: float = 420.0,
    title: str = "",
) -> str:
    """A world choropleth of this month's attention values.

    Returns an empty string when the boundaries cannot be read, so a caller
    can append it unconditionally and a missing map degrades the report rather
    than failing it.
    """
    features = _features()
    if not features:
        return ""
    lookup = {}
    for iso3, value in (values_by_iso3 or {}).items():
        try:
            lookup[str(iso3).upper()] = float(value)
        except (TypeError, ValueError):
            continue

    header = 18 if title else 0
    total_height = height + header + 34  # room for the legend
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_WIDTH:.0f}" '
        f'height="{total_height:.0f}" viewBox="0 0 {_WIDTH:.0f} {total_height:.0f}" '
        f'role="img" aria-label="Map of where the forecasts depart most from '
        f'their usual pattern">'
    ]
    if title:
        parts.append(
            f'<text x="0" y="13" font-family="sans-serif" font-size="12" '
            f'fill="#3A3A3A">{html.escape(title)}</text>'
        )
    parts.append(f'<g transform="translate(0,{header})">')
    for feature in features:
        iso3 = str((feature.get("properties") or {}).get("iso3") or "").upper()
        path = _path_for(feature.get("geometry") or {}, height)
        if not path:
            continue
        fill = colour_for(lookup.get(iso3))
        parts.append(
            f'<path d="{path}" fill="{fill}" stroke="{LAND_STROKE}" '
            f'stroke-width="0.3" />'
        )
    parts.append("</g>")

    # Legend. The bands are the same ones the fills use, so the reader can
    # match a colour on the map to a phrase in the text.
    labels = ("close to usual", "", "well away", "", "a long way from usual")
    x = 0.0
    y = header + height + 12
    for i, colour in enumerate(SCALE):
        parts.append(
            f'<rect x="{x:.0f}" y="{y:.0f}" width="26" height="10" fill="{colour}" />'
        )
        if labels[i]:
            parts.append(
                f'<text x="{x:.0f}" y="{y + 22:.0f}" font-family="sans-serif" '
                f'font-size="9" fill="#6B7280">{labels[i]}</text>'
            )
        x += 28
    parts.append(
        f'<rect x="{x + 12:.0f}" y="{y:.0f}" width="26" height="10" fill="{NO_DATA}" />'
        f'<text x="{x + 12:.0f}" y="{y + 22:.0f}" font-family="sans-serif" '
        f'font-size="9" fill="#6B7280">no forecast</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def values_from_deviation(con, run_id: str | None, *, include_test: bool) -> dict[str, float]:
    """Per-country attention: the largest divergence any of its forecasts
    shows this run, scaled to [0, 1].

    Mirrors ``/v1/interpreter/attention_map`` (same table, same preference
    order, same ln 2 scaling) so the printed map and the dashboard map cannot
    disagree.
    """
    if con is None:
        return {}
    test_clause = "" if include_test else " AND COALESCE(is_test, FALSE) = FALSE"
    where = ["js_vs_baserate IS NOT NULL"]
    params: list[Any] = []
    if run_id:
        where.append("run_id = ?")
        params.append(run_id)
    sql = f"""
        SELECT iso3, MAX(js_vs_baserate) AS m
        FROM forecast_deviation
        WHERE {' AND '.join(where)}{test_clause}
        GROUP BY iso3
    """
    try:
        rows = con.execute(sql, params).fetchall()
    except Exception as exc:  # noqa: BLE001 - a missing table is not a failure
        LOGGER.warning("[interpreter.map] deviation values unavailable: %s", exc)
        return {}
    import math

    out: dict[str, float] = {}
    for iso3, value in rows:
        if iso3 is None or value is None:
            continue
        out[str(iso3).upper()] = max(0.0, min(float(value) / math.log(2.0), 1.0))
    return out


def country_labels(values_by_iso3: dict[str, float], *, top: int = 8) -> list[str]:
    """The most-diverging countries, named, for the caption under the map."""
    ordered = sorted(
        ((k, v) for k, v in (values_by_iso3 or {}).items() if v is not None),
        key=lambda kv: (-float(kv[1]), kv[0]),
    )
    return [names.country_name(iso3) for iso3, _ in ordered[:top]]
