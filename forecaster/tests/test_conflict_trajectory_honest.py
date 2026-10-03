# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.
"""The conflict base-rate block states what it has (Oct 2026).

The 1 October 2026 Afghanistan ACE/PA prompt read "Last month (2026-08):
-1,791 new displacements ... 3-month avg -53,865 ... -131.2%". A monthly
flow cannot be negative, August was two months before the forecast, the
"3-month" average spanned five months of irregular reports, and a percentage
off a negative base means nothing.
"""

from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")

from forecaster.history_loaders import _format_base_rate_for_prompt
from pythia.tools.base_rate_spd import conflict_trajectory, idmc_flow_rows


def _summary(disp_rows, as_of="2026-10", n_negative=0):
    disp = conflict_trajectory(disp_rows, "IDMC")
    if n_negative:
        disp["n_negative_dropped"] = n_negative
    return {
        "type": "conflict_trajectory",
        "as_of_ym": as_of,
        "fatalities": conflict_trajectory([("2026-07", 10), ("2026-08", 12), ("2026-09", 14)], "ACLED"),
        "displacements": disp,
    }


def test_latest_month_is_named_with_its_age():
    text = _format_base_rate_for_prompt(
        _summary([("2026-04", 10758), ("2026-07", 1881), ("2026-08", 90)]), [], metric="PA"
    )
    assert "Latest month reported (2026-08, 2 months before this forecast): 90" in text
    assert "Average of the last 3 months reported (2026-04 to 2026-08)" in text
    # The fatalities series ends last month: the plain wording stands.
    assert "Last month (2026-09): 14" in text
    assert "3-month avg: 12/month" in text


def test_a_negative_base_prints_no_percentage():
    traj = conflict_trajectory(
        [("2026-01", -5), ("2026-02", -5), ("2026-03", -5), ("2026-04", 1), ("2026-05", 2), ("2026-06", 3)],
        "IDMC",
    )
    assert traj["trend_pct"] is None
    assert traj["trend_note"].startswith("no trend")


def test_a_zero_base_still_reads_as_new_activity():
    traj = conflict_trajectory(
        [("2026-01", 0), ("2026-02", 0), ("2026-03", 0), ("2026-04", 1), ("2026-05", 2), ("2026-06", 3)],
        "ACLED",
    )
    assert traj["trend_pct"] == "new_activity"


def test_the_reader_drops_and_counts_a_negative_flow():
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE facts_deltas (ym TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT, "
        "value_new DOUBLE, series_semantics TEXT, source_id TEXT)"
    )
    con.execute(
        "INSERT INTO facts_deltas VALUES "
        "('2026-07','AFG','IDU','new_displacements',1881,'new','idmc'),"
        "('2026-08','AFG','IDU','new_displacements',-1791,'new','idmc')"
    )
    rows, dropped = idmc_flow_rows(con, "AFG", "ACE", "2026-10")
    assert rows == [("2026-07", 1881.0)]
    assert dropped == 1
    text = _format_base_rate_for_prompt(_summary(rows, n_negative=dropped), [], metric="PA")
    assert "-1,791" not in text
    assert "1 negative monthly value(s) left out" in text
