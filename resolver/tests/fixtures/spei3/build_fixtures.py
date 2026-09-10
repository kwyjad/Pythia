# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Regenerate the committed SPEI-3 test fixtures.

Committed beside the fixtures it writes so the next reader can see what the
NetCDF holds without reverse-engineering it from ``ncdump``. Nothing in CI
runs this: run it from the repository root when the fixture has to change.

    python resolver/tests/fixtures/spei3/build_fixtures.py

The grid is deliberately tiny and its values are a readable arithmetic
pattern in plausible sigma units. It carries one NaN cell, because the
difference between a missing cell and a cell whose value is zero is the
distinction the reduction exists to keep, and a fixture with no missing
cell cannot test it.
"""

from __future__ import annotations

import pathlib

import numpy as np
import xarray as xr

HERE = pathlib.Path(__file__).resolve().parent


def main() -> None:
    lats = np.array([2.5, 1.5, 0.5, -0.5], dtype="float64")
    lons = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], dtype="float64")
    times = np.array(["2016-01-01", "2016-02-01"], dtype="datetime64[ns]")

    values = np.empty((2, 4, 8), dtype="float32")
    for t in range(2):
        for i in range(4):
            for j in range(8):
                values[t, i, j] = round(-2.0 + 0.1 * (i * 8 + j) + t, 2)
    values[0, 0, 0] = np.nan

    xr.Dataset(
        {"SPEI3": (("time", "latitude", "longitude"), values)},
        coords={"time": times, "latitude": lats, "longitude": lons},
    ).to_netcdf(HERE / "spei3_2016.nc", format="NETCDF4_CLASSIC")


if __name__ == "__main__":
    main()
