# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Regenerate the vendored country-boundary file for the resolution machine.

The resolution machine's cyclone detection needs country territories to
measure track-point proximity against (see ``geometry.py``).  We vendor a
slimmed Natural Earth 1:50m Admin-0 countries layer instead of downloading
at runtime because (a) CI tests must run network-free, and (b) the 1:110m
layer drops the small island states (Tonga, Tuvalu, Kiribati, …) that
matter most for tropical-cyclone detection.

Input:  the full ``ne_50m_admin_0_countries.geojson`` from the
        nvkelso/natural-earth-vector GitHub mirror (public domain):
        https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson

Output: ``ne_50m_admin_0_countries.slim.geojson.gz`` next to this script —
        one feature per ISO3 with properties reduced to ``{iso3, name}``,
        coordinates rounded to 4 decimals (~11 m, negligible against the
        rulebook's 200 km buffer), and geometries merged where Natural
        Earth splits a country we treat as one (Somaliland → SOM,
        Northern Cyprus → CYP).

Usage:
    python resolver/resolution_machine/data/build_boundaries.py path/to/ne_50m_admin_0_countries.geojson
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

# Natural Earth's ISO_A3 is famously "-99" for a handful of features; we
# prefer ISO_A3_EH and fall back to ADM0_A3, then remap to the codes the
# resolver universe uses (resolver/data/countries.csv).
_ISO_OVERRIDES = {
    "KOS": "RKS",  # Kosovo — repo convention (see countries.csv)
    "SOL": "SOM",  # Somaliland — merged into Somalia
    "CYN": "CYP",  # Northern Cyprus — merged into Cyprus
}
# Disputed slivers with no place in the country universe.
_DROP = {"KAS"}  # Siachen Glacier


def _feature_iso3(props: dict) -> str | None:
    for key in ("ISO_A3_EH", "ISO_A3", "ADM0_A3"):
        code = str(props.get(key, "") or "").strip().upper()
        if len(code) == 3 and code.isalpha():
            return _ISO_OVERRIDES.get(code, code)
    return None


def _round_coords(obj, ndigits: int = 4):
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, (list, tuple)):
        return [_round_coords(x, ndigits) for x in obj]
    return obj


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2

    src = Path(argv[1])
    out = Path(__file__).resolve().parent / "ne_50m_admin_0_countries.slim.geojson.gz"

    with open(src, encoding="utf-8") as fh:
        gj = json.load(fh)

    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union

    by_iso3: dict[str, list] = {}
    names: dict[str, str] = {}
    dropped: list[str] = []
    for feat in gj["features"]:
        props = feat.get("properties", {})
        iso3 = _feature_iso3(props)
        if iso3 is None or iso3 in _DROP:
            dropped.append(str(props.get("NAME", "?")))
            continue
        geom = shape(feat["geometry"])
        if not geom.is_valid:
            geom = geom.buffer(0)
        by_iso3.setdefault(iso3, []).append(geom)
        # First (primary) feature's name wins; merged slivers keep it.
        names.setdefault(iso3, str(props.get("NAME", iso3)))

    features = []
    for iso3 in sorted(by_iso3):
        geoms = by_iso3[iso3]
        merged = geoms[0] if len(geoms) == 1 else unary_union(geoms)
        geom_json = mapping(merged)
        geom_json["coordinates"] = _round_coords(geom_json["coordinates"])
        features.append(
            {
                "type": "Feature",
                "properties": {"iso3": iso3, "name": names[iso3]},
                "geometry": geom_json,
            }
        )

    slim = {
        "type": "FeatureCollection",
        "name": "ne_50m_admin_0_countries_slim",
        "features": features,
    }
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        json.dump(slim, fh, separators=(",", ":"))

    print(f"wrote {out} — {len(features)} countries, dropped: {dropped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
