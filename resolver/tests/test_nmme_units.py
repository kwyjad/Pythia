# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.
"""NMME anomalies are stored in a unit that survives rounding (Oct 2026).

CPC's ENSMEAN anomaly files carry ``prate`` in mm/s (a typical anomaly is
1e-5) and ``tmp2m`` in kelvin. The ingest rounded the raw figure to four
decimals, so all 21,294 precipitation rows of the 1 Oct 2026 release sat
within 0.0002 of zero and every prompt read them as sigma.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")
xr = pytest.importorskip("xarray")

from resolver.ingestion import nmme
from resolver.tools import ingest_nmme


def _nc(tmp_path, value):
    da = xr.DataArray(
        np.full((2, 3, 4), value, dtype="float32"),
        dims=("target", "lat", "lon"),
        coords={"target": [800.0, 801.0], "lat": [10.0, 11.0, 12.0], "lon": [0.0, 1.0, 2.0, 3.0]},
        name="fcst",
        attrs={"units": "mm/s"},
    )
    path = tmp_path / "NMME.prate.202609.ENSMEAN.anom.nc"
    da.to_dataset().to_netcdf(path)
    return path


@pytest.fixture()
def no_regions(monkeypatch):
    class _Regions:
        def mask(self, da):
            return None

    monkeypatch.setattr(nmme, "_get_country_regions", lambda: _Regions())
    monkeypatch.setattr(
        nmme, "_aggregate_2d_field_to_countries",
        lambda da, countries, mask: pd.DataFrame(
            [{"iso3": "SOM", "anomaly_value": round(float(da.mean()), 4)}]
        ),
    )


def test_precipitation_is_stored_in_mm_per_day(tmp_path, no_regions):
    # -2e-5 mm/s is -1.728 mm/day; the old code stored -0.0.
    out = nmme._aggregate_multi_lead_nc(_nc(tmp_path, -2e-5), "prate")
    assert [lead for lead, _ in out] == [1, 2]
    assert out[0][1]["anomaly_value"].iloc[0] == pytest.approx(-1.728, abs=1e-3)


def test_category_reads_the_variable_threshold():
    assert nmme.UNITS == {"tmp2m": "degC", "prate": "mm/day", "prate_prob_below": "probability"}
    assert nmme._classify_tercile(-1.7, "prate") == "below_normal"
    assert nmme._classify_tercile(0.3, "prate") == "near_normal"
    assert nmme._classify_tercile(0.8, "tmp2m") == "above_normal"


def test_near_zero_share():
    assert nmme.near_zero_share([0.0, 0.0001, -0.0002, 1.2]) == pytest.approx(0.75)
    assert nmme.near_zero_share([]) == 0.0


def test_rows_without_units_are_purged_and_refetched():
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE seasonal_forecasts (iso3 TEXT, variable TEXT, lead_months INTEGER, "
        "anomaly_value DOUBLE, tercile_category TEXT, forecast_issue_date DATE, units TEXT)"
    )
    con.execute(
        "INSERT INTO seasonal_forecasts VALUES "
        "('SOM','prate',1,0.0,'near_normal',DATE '2026-08-08',NULL),"
        "('SOM','tmp2m',1,0.4,'near_normal',DATE '2026-08-08',NULL),"
        "('SOM','prate',1,-1.2,'below_normal',DATE '2026-09-08','mm/day'),"
        "('SOM','tmp2m',1,0.4,'near_normal',DATE '2026-09-08','degC'),"
        "('SOM','prate_prob_below',1,0.55,'below_normal',DATE '2026-09-08','probability')"
    )
    assert ingest_nmme._vintage_is_held(con, "202608") is False
    assert ingest_nmme._vintage_is_held(con, "202609") is True
    assert ingest_nmme.purge_unitless_rows(con) == 2
    assert ingest_nmme.purge_unitless_rows(con) == 0
    assert con.execute("SELECT COUNT(*) FROM seasonal_forecasts").fetchone()[0] == 3


# --- CPC's probability of below-normal precipitation (Oct 2026) -----------


def _prob_nc(tmp_path, below, scale=1.0):
    shape = (1, 2, 3, 4)
    coords = {"initial_time": [800.0], "target": [800.0, 801.0],
              "lat": [10.0, 11.0, 12.0], "lon": [0.0, 1.0, 2.0, 3.0]}
    dims = ("initial_time", "target", "lat", "lon")
    ds = xr.Dataset({
        "prob_above": (dims, np.full(shape, 0.2 * scale, dtype="float32")),
        "prob_below": (dims, np.full(shape, below * scale, dtype="float32")),
        "prob_norm": (dims, np.full(shape, (0.8 - below) * scale, dtype="float32")),
    }, coords=coords)
    path = tmp_path / "prate.202609.prob.adj.mon.nc"
    ds.to_netcdf(path)
    return path


def test_probability_reads_prob_below_not_the_first_variable(tmp_path, no_regions):
    # prob_above is the first data variable; the old picker would take it.
    out = nmme._aggregate_prob_nc(_prob_nc(tmp_path, 0.6))
    assert [lead for lead, _ in out] == [1, 2]
    assert out[0][1]["anomaly_value"].iloc[0] == pytest.approx(0.6, abs=1e-4)


def test_probability_in_percent_is_read_as_a_fraction(tmp_path, no_regions):
    out = nmme._aggregate_prob_nc(_prob_nc(tmp_path, 0.6, scale=100.0))
    assert out[0][1]["anomaly_value"].iloc[0] == pytest.approx(0.6, abs=1e-4)


def test_probability_category_is_relative_to_one_third():
    assert nmme._classify_tercile(0.5, "prate_prob_below") == "below_normal"
    assert nmme._classify_tercile(0.34, "prate_prob_below") == "near_normal"


def test_a_vintage_without_probability_rows_is_fetched_again():
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE seasonal_forecasts (iso3 TEXT, variable TEXT, lead_months INTEGER, "
        "anomaly_value DOUBLE, tercile_category TEXT, forecast_issue_date DATE, units TEXT)"
    )
    con.execute(
        "INSERT INTO seasonal_forecasts VALUES "
        "('SOM','prate',1,-1.2,'below_normal',DATE '2026-08-08','mm/day'),"
        "('SOM','tmp2m',1,0.4,'near_normal',DATE '2026-08-08','degC')"
    )
    assert ingest_nmme._vintage_is_held(con, "202608") is False


def test_crossing_table_counts_country_months_and_countries():
    from scripts.ci.nmme_prob_crossing import crossing_table

    values = {"EGY": {"202601": 0.3, "202602": 0.6}, "GTM": {"202601": 0.2, "202602": 0.25}}
    row = crossing_table(values, (0.5,), below=False)[0]
    assert row["share_crossing"] == 0.25
    assert row["countries_ever_crossing"] == 1
    assert row["per_country"] == {"EGY": 1, "GTM": 0}
