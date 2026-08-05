# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Geometry tests: vendored boundary layer + AEQD distance accuracy."""

from __future__ import annotations

import pytest

from resolver.resolution_machine.geometry import (
    distance_km,
    load_country_geometries,
    point_near_bounds,
)
from resolver.tests.resolution_machine_utils import SYNTHETIC_COUNTRIES_GEOJSON

# 1 degree of longitude at the equator (WGS84).
_KM_PER_DEG_EQUATOR = 111.3195


@pytest.fixture(scope="module")
def vendored():
    return load_country_geometries()


@pytest.fixture(scope="module")
def synthetic():
    return load_country_geometries(SYNTHETIC_COUNTRIES_GEOJSON)


def test_vendored_layer_covers_small_island_states(vendored):
    # The whole point of vendoring 1:50m instead of 1:110m: the island
    # states most exposed to tropical cyclones must be present.
    for iso3 in ("PHL", "TON", "FJI", "VUT", "WSM", "KIR", "TUV", "MDG", "BGD"):
        assert iso3 in vendored, f"{iso3} missing from vendored boundaries"
    # Codes remapped to the resolver universe.
    assert "RKS" in vendored  # Kosovo
    assert "KOS" not in vendored
    assert "SOL" not in vendored  # Somaliland merged into SOM


def test_point_over_land_measures_zero(vendored):
    # Tacloban City, Philippines (Haiyan's landfall region).
    assert distance_km(vendored["PHL"], 11.24, 125.00, 400) == 0.0


def test_synthetic_distance_matches_geodesy(synthetic):
    # Squareland's western edge is at lon 10; a point on the equator
    # 2.24576 degrees west is 250.0 km away by great circle.
    aaa = synthetic["AAA"]
    d = distance_km(aaa, 0.0, 10.0 - 250.0 / _KM_PER_DEG_EQUATOR, 400)
    assert d == pytest.approx(250.0, abs=1.5)
    # And inside the square is 0.
    assert distance_km(aaa, 0.0, 12.5, 400) == 0.0


def test_distance_returns_none_beyond_window(synthetic):
    # BBB is thousands of km from the equator point — outside any
    # 400 km window, the exact test must short-circuit to None.
    assert distance_km(synthetic["BBB"], 0.0, 8.0, 400) is None


def test_antimeridian_distance(vendored):
    # Fiji straddles 180°.  A point at 179.9°W must measure a small
    # distance, not a half-world wraparound.
    d = distance_km(vendored["FJI"], -17.8, -179.9, 400)
    assert d is not None and d < 150


def test_prefilter_never_excludes_near_points(synthetic):
    aaa = synthetic["AAA"]
    near_lon = 10.0 - 250.0 / _KM_PER_DEG_EQUATOR
    assert point_near_bounds(aaa, 0.0, near_lon, 300)
    # A point on the other side of the world fails the prefilter.
    assert not point_near_bounds(aaa, 0.0, -170.0, 300)
