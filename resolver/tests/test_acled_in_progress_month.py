# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The ACLED monthly writer never stores the month still in progress.

The ingest runs on the 28th and the forecast on the 1st, so a row for the
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
