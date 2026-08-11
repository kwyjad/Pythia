# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The attention map, as a self-contained SVG.

The dashboard draws this map with a JavaScript component. A PDF cannot run
JavaScript, so the printed report needs its own drawing. It reads the same
vendored Natural Earth boundaries the resolution machine uses, projects them
with a plain equirectangular transform, and fills each country by how far the
figure a planner would act on has moved from its historical level, ABOVE or
BELOW. The direction is the point: an unsigned scale shaded a country whose
forecast had fallen exactly as it shaded one whose forecast had risen.

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

# A DIVERGING scale, because the quantity has a sign and the reader needs it.
# The map used to shade unsigned distance from the anchor, so Uganda appeared
# among the countries furthest from usual on the same page as text saying
# Uganda's forecast had moved DOWN. That is an error, not a styling choice.
#
# One hue above the anchor, another below, a pale neutral for close to it, and
# grey for no forecast at all — which must never read as "nothing is happening
# there".
NO_DATA = "#E8E8E8"
LAND_STROKE = "#FFFFFF"

# Above the anchor: more people expected than usual. Warm, darkening.
SCALE_ABOVE = ("#F6D6C4", "#EFAF8C", "#DE7F53", "#B85527", "#8A3410")
# Below the anchor: fewer. Cool, darkening.
SCALE_BELOW = ("#D6E3EE", "#A9C4DA", "#7398B8", "#456F92", "#274C68")
# Close enough to the anchor that the direction is not worth colouring.
NEUTRAL = "#F2F0EC"

# Break points on the SIGNED movement, expressed as multiples of the metric's
# own action threshold (see `values_from_deviation`). One threshold of
# movement is by definition the size worth mobilising against, so it sits in
# the middle of the ramp rather than at its end.
SCALE_BREAKS = (0.1, 0.4, 1.0, 2.0)

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
    """A SIGNED movement, in multiples of the metric's threshold, to a fill.

    Positive is above the anchor, negative below, and a movement inside the
    first break is neutral in either direction. None is no forecast at all.
    """
    if value is None:
        return NO_DATA
    try:
        v = float(value)
    except (TypeError, ValueError):
        return NO_DATA
    magnitude = abs(v)
    if magnitude < SCALE_BREAKS[0]:
        return NEUTRAL
    ramp = SCALE_ABOVE if v > 0 else SCALE_BELOW
    for i, edge in enumerate(SCALE_BREAKS):
        if magnitude < edge:
            return ramp[i]
    return ramp[-1]


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
    total_height = height + header + 48  # room for the legend and its note
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_WIDTH:.0f}" '
        f'height="{total_height:.0f}" viewBox="0 0 {_WIDTH:.0f} {total_height:.0f}" '
        f'role="img" aria-label="Map of where forecasts sit against their '
        f'usual level, above or below">'
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

    # Legend. Diverging, and it names the DIRECTION: a reader looking at a
    # shaded country has to be able to tell "more people than usual" from
    # "fewer" without reading the text beside it.
    x = 0.0
    y = header + height + 12
    swatches = (
        [(c, "") for c in reversed(SCALE_BELOW)]
        + [(NEUTRAL, "")]
        + [(c, "") for c in SCALE_ABOVE]
        + [(NO_DATA, "")]
    )
    for colour, _ in swatches:
        parts.append(
            f'<rect x="{x:.0f}" y="{y:.0f}" width="18" height="10" '
            f'fill="{colour}" />'
        )
        x += 19
    captions = (
        (0.0, "well below its usual level"),
        (5 * 19.0, "near it"),
        (6 * 19.0, "above its usual level"),
        (11 * 19.0 + 6, "no forecast"),
    )
    for cx, label in captions:
        parts.append(
            f'<text x="{cx:.0f}" y="{y + 22:.0f}" font-family="sans-serif" '
            f'font-size="9" fill="#6B7280">{html.escape(label)}</text>'
        )
    parts.append(
        f'<text x="0" y="{y + 34:.0f}" font-family="sans-serif" font-size="8" '
        f'fill="#9CA3AF">Shading is how far the figure a planner would act on '
        f'sits from its historical level, as a multiple of the size worth '
        f'mobilising against. Where a country carries several hazards, the '
        f'largest movement is shown.</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


# One aggregate per question, best-available, then the largest movement per
# country. Shared verbatim with /v1/interpreter/attention_map so the printed
# map and the dashboard map cannot disagree; the printed one exists only
# because a PDF cannot run the dashboard's JavaScript.
#
# Scaled by the metric's own action threshold, so deaths and people in crisis
# land on one ramp: 1.0 means "moved by the size worth mobilising against".
# The old ln 2 scaling of an UNDIRECTED divergence is what shaded a country
# whose forecast had fallen as though it had risen.
SIGNED_MOVEMENT_SQL = """
    WITH ranked AS (
        SELECT iso3, question_id,
               delta_p50, delta_p90, movement_threshold,
               ROW_NUMBER() OVER (
                   PARTITION BY question_id
                   ORDER BY CASE {order} ELSE 99 END, model_name
               ) AS rn
        FROM forecast_deviation
        WHERE {where}
    ),
    scaled AS (
        SELECT iso3,
               -- GREATEST(a, b) ignores NULLs and is NULL only when both are,
               -- which is exactly gating.material_movement. The map is a
               -- picture of the gate's own input, so it colours by the same
               -- quantity the gate tests.
               --
               -- The asymmetry is deliberate. For a rise this takes the more
               -- alarming of the two figures; for a fall it takes the less
               -- negative, so an improvement is shaded conservatively. A map
               -- that understated a deterioration would be the worse fault.
               CASE WHEN GREATEST(delta_p50, delta_p90) IS NULL
                         OR movement_threshold IS NULL
                         OR movement_threshold = 0
                    THEN NULL
                    ELSE GREATEST(delta_p50, delta_p90) / movement_threshold
               END AS signed_move
        FROM ranked WHERE rn = 1
    )
    SELECT iso3, signed_move
    FROM scaled
    WHERE signed_move IS NOT NULL
"""


def _largest_signed(rows) -> dict[str, float]:
    """Per country, the movement with the largest MAGNITUDE, sign kept.

    A country carrying several hazards gets the one that moved most, in
    either direction, and the legend says so. Taking the maximum would hide a
    large fall behind a small rise.
    """
    out: dict[str, float] = {}
    for iso3, value in rows:
        if iso3 is None or value is None:
            continue
        code = str(iso3).upper()
        v = max(-3.0, min(float(value), 3.0))
        if code not in out or abs(v) > abs(out[code]):
            out[code] = v
    return out


def values_from_deviation(con, run_id: str | None, *, include_test: bool) -> dict[str, float]:
    """Per-country SIGNED movement, in multiples of the action threshold."""
    if con is None:
        return {}
    test_clause = "" if include_test else " AND COALESCE(is_test, FALSE) = FALSE"
    where = ["1 = 1"]
    params: list[Any] = []
    if run_id:
        where.append("run_id = ?")
        params.append(run_id)
    order = " ".join(
        f"WHEN model_name = '{m}' THEN {i}"
        for i, m in enumerate(names.AGGREGATE_PREFERENCE)
    )
    sql = SIGNED_MOVEMENT_SQL.format(
        order=order, where=" AND ".join(where) + test_clause
    )
    try:
        rows = con.execute(sql, params).fetchall()
    except Exception as exc:  # noqa: BLE001 - a missing column is not a failure
        LOGGER.warning("[interpreter.map] movement values unavailable: %s", exc)
        return {}
    return _largest_signed(rows)


def country_labels(values_by_iso3: dict[str, float], *, top: int = 8) -> list[str]:
    """The countries that moved most, named, for the caption under the map.

    Ordered by MAGNITUDE: a country whose forecast fell a long way is as
    notable as one whose forecast rose, and the caption says which by naming
    it beside a map that colours the direction.
    """
    ordered = sorted(
        ((k, v) for k, v in (values_by_iso3 or {}).items() if v is not None),
        key=lambda kv: (-abs(float(kv[1])), kv[0]),
    )
    return [names.country_name(iso3) for iso3, _ in ordered[:top]]
