# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.
"""The NMME outlook names its units and says when rain is missing (Oct 2026)."""

from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")

from horizon_scanner import seasonal_context as sc


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "p.duckdb"
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE seasonal_forecasts (iso3 TEXT, variable TEXT, lead_months INTEGER, "
        "anomaly_value DOUBLE, tercile_category TEXT, forecast_issue_date DATE, units TEXT)"
    )
    con.execute(
        "INSERT INTO seasonal_forecasts VALUES "
        "('SOM','tmp2m',1,0.8,'above_normal',DATE '2026-09-08','degC'),"
        "('SOM','prate',1,-1.2,'below_normal',DATE '2026-09-08','mm/day'),"
        "('KEN','tmp2m',1,0.2,'near_normal',DATE '2026-09-08','degC'),"
        "('ETH','prate',1,0.0,'near_normal',DATE '2026-09-08',NULL)"
    )
    con.close()
    return f"duckdb:///{path}"


def test_units_are_printed_and_sigma_is_not(db):
    out = sc.load_seasonal_forecasts("SOM", db_url=db)
    assert "(+0.80 °C)" in out["nmme_temp_outlook"]
    assert "(-1.20 mm/day)" in out["nmme_precip_outlook"]
    assert "σ" not in "".join(str(v) for v in out.values())


def test_missing_precipitation_says_unavailable(db):
    out = sc.load_seasonal_forecasts("KEN", db_url=db)
    assert out["nmme_precip_outlook"].startswith("unavailable")
    assert "near-normal" not in out["nmme_precip_outlook"]


def test_rows_without_units_are_not_printed(db):
    assert sc.load_seasonal_forecasts("ETH", db_url=db) is None
