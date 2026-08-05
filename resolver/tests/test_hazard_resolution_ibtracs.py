# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""IBTrACS connector tests: parsing, normalisation, idempotent raw-cache store."""

from __future__ import annotations

import json

import duckdb
import pytest

from resolver.hazard_resolution.ibtracs import (
    load_track_points,
    parse_ibtracs_csv,
    store_ibtracs,
    store_summary,
)
from resolver.tests.hazard_resolution_utils import IBTRACS_SAMPLE_CSV


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


def test_parse_deduplicates_track_points(frame):
    near = frame[frame["sid"] == "TEST01NEAR250"]
    assert len(near) == 1


def test_store_is_idempotent_and_revisions_append(frame):
    con = duckdb.connect(":memory:")
    url = "https://example.test/ibtracs.csv"
    first = store_ibtracs(con, frame, "last3years", url)
    second = store_ibtracs(con, frame, "last3years", url)
    # One row per storm; an identical re-store is a content-hash no-op.
    assert first["storms"] == second["storms"] == frame["sid"].nunique()
    assert first["inserted"] == frame["sid"].nunique()
    assert second["inserted"] == 0
    assert second["rows_total"] == first["rows_total"]

    # A revised best track (changed content) appends a revision row for
    # that storm — the newest revision is what detection reads.
    revised = frame.copy()
    revised.loc[revised["sid"] == "2013306N07162", "usa_wind_kt"] = 170.0
    third = store_ibtracs(con, revised, "last3years", url)
    assert third["inserted"] == 1
    assert third["rows_total"] == first["rows_total"] + 1


def test_load_track_points_reads_newest_revision(frame):
    con = duckdb.connect(":memory:")
    store_ibtracs(con, frame, "last3years", "https://example.test/ibtracs.csv")
    revised = frame.copy()
    revised.loc[revised["sid"] == "2013306N07162", "usa_wind_kt"] = 170.0
    # Ensure a strictly later retrieved_at for the revision row.
    con.execute("SELECT 1")
    store_ibtracs(con, revised, "last3years", "https://example.test/ibtracs.csv")

    points = load_track_points(con, "2013-11-01", "2013-11-30")
    haiyan = points[points["sid"] == "2013306N07162"]
    assert len(haiyan) == 5
    assert (haiyan["usa_wind_kt"] == 170.0).all()
    # Storms outside the window are not loaded.
    assert set(points["sid"]) == {"2013306N07162"}


def test_store_summary_reports_provenance(frame):
    con = duckdb.connect(":memory:")
    store_ibtracs(con, frame, "last3years", "https://example.test/ibtracs.csv")
    summary = store_summary(con)
    assert summary["total_storms"] == frame["sid"].nunique()
    assert summary["max_point_time"].startswith("2013-12-15")
    assert summary["source_scopes"] == ["last3years"]
    assert summary["source_urls"] == ["https://example.test/ibtracs.csv"]
    assert summary["last_retrieved_at"]


def test_payload_carries_point_level_provenance(frame):
    con = duckdb.connect(":memory:")
    store_ibtracs(con, frame, "last3years", "https://example.test/ibtracs.csv")
    payload_json, ym = con.execute(
        "SELECT payload_json, ym FROM haz_raw_ibtracs WHERE record_id = '2013306N07162'"
    ).fetchone()
    payload = json.loads(payload_json)
    assert payload["name"] == "HAIYAN"
    assert payload["first_point_time"] == "2013-11-07 12:00:00"
    assert payload["last_point_time"] == "2013-11-08 12:00:00"
    assert payload["n_points"] == 5
    assert ym == "2013-11"
