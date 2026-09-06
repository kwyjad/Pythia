# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Group D of the run-33946954189 repairs: tropical cyclone is missing months.

The cell ledger held TC rows for 310 distinct months from 2000-01 to
2026-09 with a clean hole from 2025-07 to 2026-05 inclusive. Every other
month carried exactly 237 cells; the months either side of the hole were
present and full.

Two things caused it and neither was a crash. The backcast walks its window
OLDEST FIRST under a nightly time budget, and cyclone starts in 2000 — so
by September 2026 its ledger had reached 2025-06 and simply had not got to
the rest. The months after the hole are the live trailing window, not the
backcast. Meanwhile the acceptance report presented the ONE month of TC it
had as a twelve-month window and computed 57.4% on it.

Network-free: the month runner is injected.
"""

from __future__ import annotations

import datetime as dt

import duckdb
import pytest

from resolver.hazard_resolution import acceptance as acc
from resolver.hazard_resolution import backcast as bc
from resolver.hazard_resolution.schema import ensure_haz_schema
from resolver.tests.hazard_resolution_utils import make_rulebook, seed_trigger

TODAY = dt.date(2026, 8, 5)


@pytest.fixture()
def con(tmp_path):
    connection = duckdb.connect(str(tmp_path / "haz.duckdb"))
    ensure_haz_schema(connection)
    yield connection
    connection.close()


def _rulebook():
    """A short window ending at the last frozen month.

    Flood rather than cyclone: the ordering is hazard-agnostic and the
    cyclone path prefetches the 200 MB IBTrACS archive before walking, which
    a unit test has no business doing.
    """

    return make_rulebook({"backcast": {"flood": 2026}})


class TestWalkOrder:
    def _walk(self, con, order):
        seen: list[str] = []

        def runner(*, ym, **_kwargs):
            seen.append(ym)
            return 0

        bc.run_backcast(
            hazard_name="flood",
            rulebook=_rulebook(),
            con=con,
            today=TODAY,
            runner=runner,
            order=order,
        )
        return seen

    def test_the_default_walk_is_newest_first(self, con):
        seen = self._walk(con, "newest")

        assert seen == sorted(seen, reverse=True), (
            "the months every consumer reads — the acceptance window, the "
            "severity quantiles, the SPD prompt block — are the recent ones, "
            "and a time-budgeted oldest-first walk reaches them last"
        )
        assert seen[0] == "2026-05"

    def test_oldest_first_is_still_available(self, con):
        seen = self._walk(con, "oldest")

        assert seen == sorted(seen)

    def test_both_orders_walk_the_same_months(self, con, tmp_path):
        newest = self._walk(con, "newest")
        other = duckdb.connect(str(tmp_path / "other.duckdb"))
        ensure_haz_schema(other)
        try:
            oldest = self._walk(other, "oldest")
        finally:
            other.close()

        assert sorted(newest) == sorted(oldest)


class TestLedgerIsCheckedAgainstTheDatabase:
    def test_a_month_the_ledger_calls_done_with_no_rows_is_rewalked(self, con):
        bc.record_month(
            con, hazard="TC", ym="2026-04", status="ok", counts={"cells": 237}
        )

        assert "2026-04" not in bc.completed_months(con, "TC"), (
            "the ledger says complete and the database holds nothing for the "
            "month — the database wins, or the hole is permanent"
        )

    def test_a_month_with_rows_stays_complete(self, con):
        seed_trigger(con, iso3="FJI", ym="2026-04", hazard="TC")
        bc.record_month(
            con, hazard="TC", ym="2026-04", status="ok", counts={"cells": 237}
        )

        assert "2026-04" in bc.completed_months(con, "TC")

    def test_a_month_with_nothing_to_do_stays_complete(self, con):
        """An off-season cyclone month assessed no cells. There was nothing to
        do, so re-walking it every night would be noise, not repair."""

        bc.record_month(
            con, hazard="TC", ym="2026-04", status="ok", counts={"cells": 0}
        )

        assert "2026-04" in bc.completed_months(con, "TC")

    def test_another_hazards_rows_do_not_vouch_for_this_one(self, con):
        seed_trigger(con, iso3="FJI", ym="2026-04", hazard="FL")
        bc.record_month(
            con, hazard="TC", ym="2026-04", status="ok", counts={"cells": 237}
        )

        assert "2026-04" not in bc.completed_months(con, "TC")

    def test_a_rewalked_month_is_actually_visited(self, con):
        bc.record_month(
            con, hazard="FL", ym="2026-05", status="ok", counts={"cells": 237}
        )
        seen: list[str] = []

        def runner(*, ym, **_kwargs):
            seen.append(ym)
            return 0

        bc.run_backcast(
            hazard_name="flood", rulebook=_rulebook(), con=con,
            today=TODAY, runner=runner,
        )

        assert "2026-05" in seen


class TestAcceptanceStatesItsMonths:
    def test_a_hazard_short_of_the_window_says_so(self, con):
        # One month of TC against a twelve-month window.
        for iso3 in ("FJI", "TON", "VUT"):
            seed_trigger(con, iso3=iso3, ym="2026-05", hazard="TC")

        result = acc.build_result(
            con, make_rulebook(), months=12, today=TODAY, hazards=["TC"]
        )
        report = acc.render_report(result)

        assert result.rates["TC"].months_present == 1
        assert "1 of 12" in report
        assert "do not cover the window they are printed under" in report

    def test_a_hazard_covering_the_window_is_not_flagged(self, con):
        for month in range(1, 13):
            seed_trigger(
                con, iso3="FJI", ym=f"{2025 if month > 5 else 2026}-{month:02d}",
                hazard="TC",
            )

        result = acc.build_result(
            con, make_rulebook(), months=12, today=TODAY, hazards=["TC"]
        )

        assert result.rates["TC"].months_present == 12
        assert "do not cover the window" not in acc.render_report(result)

    def test_the_rate_itself_is_never_adjusted_to_hide_the_gap(self, con):
        for iso3 in ("FJI", "TON"):
            seed_trigger(con, iso3=iso3, ym="2026-05", hazard="TC")

        result = acc.build_result(
            con, make_rulebook(), months=12, today=TODAY, hazards=["TC"]
        )

        # Two cells assessed, none resolved: 0.0%, over the months present.
        assert result.rates["TC"].cells_assessed == 2
        assert result.rates["TC"].rate_pct == 0.0
