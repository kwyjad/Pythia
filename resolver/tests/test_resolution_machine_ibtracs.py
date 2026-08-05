# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""IBTrACS connector tests: parsing, normalisation, idempotent store."""

from __future__ import annotations

import duckdb
import pytest

from resolver.resolution_machine.ibtracs import parse_ibtracs_csv, store_ibtracs, store_summary
from resolver.tests.resolution_machine_utils import IBTRACS_SAMPLE_CSV


@pytest.fixture()
def frame():
    return parse_ibtracs_csv(IBTRACS_SAMPLE_CSV)


def test_parse_skips_units_row_and_extra_columns(frame):
    # The fixture has 11 data rows, one of which duplicates (sid, iso_time).
    assert len(frame) == 10
    assert set(frame.columns) == {
        "sid", "season", "basin", "storm_name", "iso_time", "lat", "lon",
        "usa_wind_kt", "wmo_wind_kt", "nature",
    }
    # The units row must not survive as data.
    assert not (frame["sid"] == "").any()


def test_parse_normalises_longitude_to_180(frame):
    row = frame[frame["sid"] == "TEST04LON360"].iloc[0]
    assert row["lon"] == pytest.approx(-124.0)
    # Untouched longitudes stay untouched.
    haiyan = frame[frame["sid"] == "2013306N07162"]
    assert haiyan["lon"].between(121, 129).all()


def test_parse_keeps_wind_columns_separately(frame):
    # TEST03INSIDE has an empty USA_WIND but a WMO_WIND — the parser must
    # keep both raw columns (which one wins is a rulebook decision).
    row = frame[frame["sid"] == "TEST03INSIDE"].iloc[0]
    assert row["wmo_wind_kt"] == pytest.approx(60.0)
    assert row["usa_wind_kt"] != row["usa_wind_kt"]  # NaN


def test_parse_deduplicates_primary_key(frame):
    near = frame[frame["sid"] == "TEST01NEAR250"]
    assert len(near) == 1


def test_store_is_idempotent(frame):
    con = duckdb.connect(":memory:")
    url = "https://example.test/ibtracs.csv"
    first = store_ibtracs(con, frame, "last3years", url)
    second = store_ibtracs(con, frame, "last3years", url)
    assert first == second == len(frame)
    # PK intact: one row per (sid, iso_time).
    n_keys = con.execute(
        "SELECT COUNT(*) FROM (SELECT DISTINCT sid, iso_time FROM haz_raw_ibtracs)"
    ).fetchone()[0]
    assert n_keys == len(frame)


def test_store_summary_reports_provenance(frame):
    con = duckdb.connect(":memory:")
    store_ibtracs(con, frame, "last3years", "https://example.test/ibtracs.csv")
    summary = store_summary(con)
    assert summary["total_points"] == len(frame)
    assert summary["max_iso_time"].startswith("2013-12-15")
    assert summary["source_scopes"] == ["last3years"]
    assert summary["source_urls"] == ["https://example.test/ibtracs.csv"]
    assert summary["last_fetched_at"]
