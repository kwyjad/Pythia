# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The ACLED monthly writer never stores the month still in progress.

The ingest ran on the 28th and the forecast on the 1st (now the 11th and the
13th), so a row for the
month in progress was what the forecast read as "last month": on 1 August
2026 a median 28% of the settled count, on 1 September 75%.
"""

import pandas as pd
import pytest

from resolver.cli import acled_to_duckdb
from resolver.db import duckdb_io


@pytest.mark.duckdb
def test_the_month_in_progress_is_not_written(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    if not duckdb_io.DUCKDB_AVAILABLE:
        pytest.skip("DuckDB module not available")

    frame = pd.DataFrame(
        {
            "iso3": ["SOM", "SOM", "SOM"],
            "month": ["2026-07-01", "2026-08-01", "2026-09-01"],
            "fatalities": [473, 583, 300],
            "source": ["ACLED"] * 3,
        }
    )
    frame["month"] = pd.to_datetime(frame["month"])

    class _StubClient:
        def __init__(self, *_, **__):
            pass

        def monthly_fatalities(self, *_args, **_kwargs):
            return frame.copy()

    monkeypatch.setattr(acled_to_duckdb, "ACLEDClient", _StubClient)
    monkeypatch.setenv("ACLED_TODAY", "2026-09-28")
    db_path = tmp_path / "acled.duckdb"
    args = ["--start", "2026-07-01", "--end", "2026-09-30", "--db", str(db_path)]
    assert acled_to_duckdb.run(args) == 0

    conn = duckdb_io.get_db(f"duckdb:///{db_path.as_posix()}")
    try:
        months = [
            str(r[0])[:7]
            for r in conn.execute(
                "SELECT month FROM acled_monthly_fatalities ORDER BY month"
            ).fetchall()
        ]
    finally:
        duckdb_io.close_db(conn)
    # August has ended and is rewritten whole; September is still running.
    assert months == ["2026-07", "2026-08"]


def test_current_month_start_honours_the_override(monkeypatch):
    monkeypatch.setenv("ACLED_TODAY", "2026-10-01")
    assert acled_to_duckdb._current_month_start() == pd.Timestamp("2026-10-01")
    monkeypatch.setenv("ACLED_TODAY", "2026-09-30")
    assert acled_to_duckdb._current_month_start() == pd.Timestamp("2026-09-01")


def _run_with_latest_event(tmp_path, monkeypatch, latest_event, today, countries=None):
    frame = pd.DataFrame(
        {
            "iso3": ["SOM", "SOM", "SOM"],
            "month": ["2026-08-01", "2026-09-01", "2026-10-01"],
            "fatalities": [583, 300, 120],
            "source": ["ACLED"] * 3,
        }
    )
    frame["month"] = pd.to_datetime(frame["month"])

    class _StubClient:
        def __init__(self, *_, **__):
            self.latest_event_date = None

        def monthly_fatalities(self, *_args, **_kwargs):
            self.latest_event_date = pd.Timestamp(latest_event) if latest_event else None
            return frame.copy()

    monkeypatch.setattr(acled_to_duckdb, "ACLEDClient", _StubClient)
    monkeypatch.setenv("ACLED_TODAY", today)
    db_path = tmp_path / "acled.duckdb"
    args = ["--start", "2026-08-01", "--end", "2026-11-10", "--db", str(db_path)]
    if countries:
        args += ["--countries", countries]
    assert acled_to_duckdb.run(args) == 0
    conn = duckdb_io.get_db(f"duckdb:///{db_path.as_posix()}")
    try:
        return [
            str(r[0])[:7]
            for r in conn.execute(
                "SELECT month FROM acled_monthly_fatalities ORDER BY month"
            ).fetchall()
        ]
    finally:
        duckdb_io.close_db(conn)


@pytest.mark.duckdb
def test_on_the_11th_the_month_just_ended_is_written(tmp_path, monkeypatch):
    """The 11 November ingest: ACLED's release of Tuesday 10 November covers
    to Friday 6 November, so October is whole and is written; November is in
    progress and is not."""
    if not duckdb_io.DUCKDB_AVAILABLE:
        pytest.skip("DuckDB module not available")
    months = _run_with_latest_event(tmp_path, monkeypatch, "2026-11-06", "2026-11-11")
    assert months == ["2026-08", "2026-09", "2026-10"]


@pytest.mark.duckdb
def test_a_month_whose_last_week_is_unreleased_is_held_back(tmp_path, monkeypatch):
    """A late ACLED release: the newest event fetched is 28 October, so
    October's last days are missing and October is not stored as complete."""
    if not duckdb_io.DUCKDB_AVAILABLE:
        pytest.skip("DuckDB module not available")
    months = _run_with_latest_event(tmp_path, monkeypatch, "2026-10-28", "2026-11-11")
    assert months == ["2026-08", "2026-09"]


@pytest.mark.duckdb
def test_a_country_restricted_run_is_not_held_back(tmp_path, monkeypatch):
    """One country's newest event says nothing about the release date."""
    if not duckdb_io.DUCKDB_AVAILABLE:
        pytest.skip("DuckDB module not available")
    months = _run_with_latest_event(
        tmp_path, monkeypatch, "2026-10-20", "2026-11-11", countries="SOM"
    )
    assert months == ["2026-08", "2026-09", "2026-10"]
