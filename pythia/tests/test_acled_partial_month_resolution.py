# Pythia / Copyright (c) 2025 Kevin Wyjad
"""A partial ACLED month is not a resolution and does not cover a month.

Until Sept 2026 ``acled_to_duckdb`` wrote the month in progress, and the
prompt readers learned to skip such rows (``base_rate_spd.ACLED_COMPLETE_MONTH_SQL``).
``compute_resolutions`` and ``source_coverage`` read the same table and did
not: a September row written on 28 September would resolve a September
horizon to a fraction of the month, and it would make September "covered",
so every country with no row would zero-default. Both now apply the
prompt readers' rule.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pytest

from pythia.tools import compute_resolutions as cr
from pythia.tools.source_coverage import (
    months_with_source_data,
    refresh_source_coverage,
)

TODAY = date(2026, 10, 5)


def _db(path: Path):
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE acled_monthly_fatalities (iso3 TEXT, month DATE, fatalities BIGINT, "
        "source TEXT, updated_at TIMESTAMP)"
    )
    con.execute("CREATE TABLE hs_runs (hs_run_id TEXT PRIMARY KEY)")
    con.execute("INSERT INTO hs_runs VALUES ('run1')")
    con.execute(
        "CREATE TABLE questions (question_id TEXT, hs_run_id TEXT, iso3 TEXT, hazard_code TEXT, "
        "metric TEXT, target_month TEXT, window_start_date DATE, status TEXT, "
        "is_test BOOLEAN DEFAULT FALSE)"
    )
    return con


def _seed(con) -> None:
    # August: complete rows (written in September) for SOM and KEN.
    for iso, n in [("SOM", 120), ("KEN", 7)]:
        con.execute(
            "INSERT INTO acled_monthly_fatalities VALUES (?, DATE '2026-08-01', ?, 'ACLED', "
            "TIMESTAMP '2026-09-28 03:00:00')",
            [iso, n],
        )
    # September: a row for SOM written on 28 September, before the month ended.
    con.execute(
        "INSERT INTO acled_monthly_fatalities VALUES ('SOM', DATE '2026-09-01', 80, 'ACLED', "
        "TIMESTAMP '2026-09-28 03:00:00')"
    )
    for iso in ("SOM", "ETH"):
        con.execute(
            "INSERT INTO questions VALUES (?, 'run1', ?, 'ACE', 'FATALITIES', '2027-01', "
            "DATE '2026-08-01', 'active', FALSE)",
            [f"{iso}_ACE_FATALITIES_2026-08", iso],
        )


def test_partial_month_does_not_cover_the_month(tmp_path: Path) -> None:
    con = _db(tmp_path / "c.duckdb")
    _seed(con)
    refresh_source_coverage(con)
    assert months_with_source_data(con, "FATALITIES") == {"2026-08"}
    assert cr._data_freshness_cutoff(con, "FATALITIES") == "2026-08"


@pytest.mark.db
def test_a_partial_september_row_resolves_and_zero_defaults_nothing(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    db = tmp_path / "r.duckdb"
    db_url = f"duckdb:///{db}"
    monkeypatch.setattr(cr, "load_cfg", lambda: {"app": {"db_url": db_url}})
    con = _db(db)
    _seed(con)
    # ETH appears in the universe through an older complete month.
    con.execute(
        "INSERT INTO acled_monthly_fatalities VALUES ('ETH', DATE '2026-07-01', 3, 'ACLED', "
        "TIMESTAMP '2026-08-28 03:00:00')"
    )
    con.close()

    with caplog.at_level("INFO"):
        cr.compute_resolutions(db_url=db_url, today=TODAY)

    con = duckdb.connect(str(db))
    rows = con.execute(
        "SELECT question_id, horizon_m, observed_month, value, source_desc "
        "FROM resolutions ORDER BY question_id, horizon_m"
    ).fetchall()
    con.close()

    # August resolves for both: SOM from its complete row, ETH zero-defaults
    # because August is a covered month and ETH is in ACLED's universe.
    assert rows == [
        ("ETH_ACE_FATALITIES_2026-08", 1, "2026-08", 0.0, "zero_default"),
        ("SOM_ACE_FATALITIES_2026-08", 1, "2026-08", 120.0, cr.ACE_FATALITIES_SERIES),
    ]
    # September: nothing resolved, nothing zero-defaulted.
    assert not [r for r in rows if r[2] == "2026-09"]
    assert "1 horizon-months skipped (only a partial-month ACLED row)" in caplog.text


@pytest.mark.db
def test_a_complete_september_row_resolves(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "ok.duckdb"
    db_url = f"duckdb:///{db}"
    monkeypatch.setattr(cr, "load_cfg", lambda: {"app": {"db_url": db_url}})
    con = _db(db)
    _seed(con)
    # The October ingest rewrites September after it ended.
    con.execute(
        "UPDATE acled_monthly_fatalities SET fatalities = 140, "
        "updated_at = TIMESTAMP '2026-10-03 03:00:00' "
        "WHERE iso3 = 'SOM' AND month = DATE '2026-09-01'"
    )
    con.close()

    cr.compute_resolutions(db_url=db_url, today=TODAY)

    con = duckdb.connect(str(db))
    rows = con.execute(
        "SELECT question_id, horizon_m, value, source_desc FROM resolutions "
        "WHERE observed_month = '2026-09' ORDER BY question_id"
    ).fetchall()
    con.close()
    assert ("SOM_ACE_FATALITIES_2026-08", 2, 140.0, cr.ACE_FATALITIES_SERIES) in rows
