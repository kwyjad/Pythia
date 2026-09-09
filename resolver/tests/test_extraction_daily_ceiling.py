# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The backcast's extraction share is spread over the month, not raced.

A monthly cap alone lets one night take the whole month, and one did. On 1
September 2026 a single backcast run made 1,998 calls of a 2,000-call
share. By the 8th, 2,526 of the monthly 4,000 were spent, leaving 1,474 --
below the 1,500-call live reserve -- so the backcast made no extraction
calls at all in run 34222175003 and 443 cells sat deferred with three weeks
of the month still to run.

Nothing was misconfigured. `reliefweb_extracted` is the second rung of the
impact ladder, and the machine had spent its access to it in one night.
"""

from __future__ import annotations

import datetime as dt

import pytest

from resolver.hazard_resolution.extract import (
    ExtractionBudget,
    SHARE_OVERRIDE_ENV,
    daily_backcast_ceiling,
)


class TestTheDerivation:
    def test_a_fresh_share_is_spread_over_the_whole_month(self):
        assert daily_backcast_ceiling(2000, 0, dt.date(2026, 9, 1)) == 66

    def test_the_night_that_caused_this_would_now_be_stopped(self):
        """1,998 calls on 1 September against a ceiling of 66."""

        ceiling = daily_backcast_ceiling(2000, 0, dt.date(2026, 9, 1))
        assert 1998 > ceiling

    def test_a_quiet_night_raises_tomorrows_ceiling_rather_than_forfeiting(self):
        """Unused headroom rolls forward. Otherwise a slow month wastes it."""

        first = daily_backcast_ceiling(2000, 0, dt.date(2026, 9, 1))
        second = daily_backcast_ceiling(2000, 0, dt.date(2026, 9, 2))
        assert second > first

    def test_a_share_already_spent_gives_nothing(self):
        """The honest answer, not a floor invented to keep the job busy."""

        assert daily_backcast_ceiling(2000, 1998, dt.date(2026, 9, 8)) == 0
        assert daily_backcast_ceiling(2000, 5000, dt.date(2026, 9, 8)) == 0

    def test_the_last_day_of_the_month_may_spend_what_is_left(self):
        assert daily_backcast_ceiling(2000, 0, dt.date(2026, 9, 30)) == 2000

    def test_february_is_shorter_and_the_ceiling_knows_it(self):
        assert daily_backcast_ceiling(2000, 0, dt.date(2026, 2, 1)) == 71
        assert daily_backcast_ceiling(2000, 0, dt.date(2024, 2, 1)) == 68

    def test_no_share_means_no_daily_ceiling(self):
        """A live run has none: the reserve exists for its benefit."""

        assert daily_backcast_ceiling(None, 0, dt.date(2026, 9, 1)) is None


class TestTheBudgetHonoursIt:
    def _budget(self, **kwargs) -> ExtractionBudget:
        defaults = dict(
            max_calls_per_month=4000, used_this_month=0, run_type="backcast",
            backcast_max_calls_per_month=2000, backcast_used_this_month=0,
            live_reserve_calls=1500, backcast_max_calls_per_day=66,
            backcast_used_today=0,
        )
        defaults.update(kwargs)
        return ExtractionBudget(**defaults)

    def test_the_daily_ceiling_binds_before_the_monthly_share(self):
        budget = self._budget()
        assert budget.remaining == 66
        assert budget.binding_limit == "backcast daily share (66)"

    def test_the_monthly_cap_is_still_the_hard_bound(self):
        """A daily ceiling never licenses more than the month allows.

        On the last day of the month the derived ceiling is the whole
        remaining share, and the monthly total must still stop it.
        """

        budget = self._budget(
            used_this_month=3990, backcast_max_calls_per_day=2000,
            live_reserve_calls=0,
        )
        assert budget.remaining == 10
        assert budget.binding_limit == "monthly total (4000)"

    def test_the_live_reserve_still_outranks_a_generous_daily_ceiling(self):
        budget = self._budget(
            used_this_month=2400, backcast_max_calls_per_day=2000,
        )
        assert budget.remaining == 100
        assert "live reserve" in budget.binding_limit

    def test_calls_made_today_count_against_it(self):
        assert self._budget(backcast_used_today=60).remaining == 6
        assert self._budget(backcast_used_today=66).exhausted is True

    def test_a_live_run_has_no_daily_ceiling(self):
        budget = ExtractionBudget(
            max_calls_per_month=4000, used_this_month=0, run_type="live",
            live_reserve_calls=1500,
        )
        assert budget.remaining == 4000
        assert "daily" not in budget.binding_limit

    def test_the_provenance_names_it(self):
        """A capped cell must say which limit capped it."""

        provenance = self._budget(backcast_used_today=66).as_provenance()
        assert provenance["backcast_max_calls_per_day"] == 66
        assert provenance["backcast_used_today_before_run"] == 66
        assert provenance["binding_limit"] == "backcast daily share (66)"

    def test_headroom_reports_both_scales(self):
        headroom = self._budget(
            used_this_month=1000, backcast_used_this_month=900,
            backcast_used_today=10,
        ).headroom()
        assert headroom["monthly_headroom"] == 3000
        assert headroom["backcast_share_headroom"] == 1100
        assert headroom["backcast_daily_headroom"] == 56


class TestTheOneDispatchOverride:
    """Raising the share for one dispatch must still work.

    The daily ceiling exists to stop the STANDING nightly job racing its
    month away. The override exists so an operator can clear one named
    backlog in one dispatch, and a ceiling that bound it would leave the
    override doing nothing at all -- which is the fault in a different coat.
    """

    def test_the_override_lifts_the_daily_ceiling_for_that_dispatch(
        self, tmp_path, monkeypatch
    ):
        duckdb = pytest.importorskip("duckdb")
        from resolver.hazard_resolution.extract import load_budget
        from resolver.hazard_resolution.rulebook import load_rulebook

        con = duckdb.connect(str(tmp_path / "t.duckdb"))
        rulebook = load_rulebook()

        monkeypatch.delenv(SHARE_OVERRIDE_ENV, raising=False)
        standing = load_budget(con, rulebook, today=dt.date(2026, 9, 1),
                               run_type="backcast")
        assert standing.backcast_max_calls_per_day is not None

        monkeypatch.setenv(SHARE_OVERRIDE_ENV, "2500")
        dispatched = load_budget(con, rulebook, today=dt.date(2026, 9, 1),
                                 run_type="backcast")
        assert dispatched.backcast_max_calls_per_day is None
        assert dispatched.backcast_max_calls_per_month == 2500
        con.close()


class TestItReachesTheRegister:
    """A budget nobody reads is a budget that races again next month."""

    def _bundle_register(self, tmp_path, rows):
        duckdb = pytest.importorskip("duckdb")
        from scripts import build_resolver_debug_bundle as bundle

        db = tmp_path / "resolver.duckdb"
        con = duckdb.connect(str(db))
        con.execute(
            "CREATE TABLE haz_doc_extractions (doc_id TEXT, model TEXT, "
            "prompt_version TEXT, iso3 TEXT, hazard TEXT, year INTEGER, "
            "month INTEGER, status TEXT, run_type TEXT, prompt_tokens INTEGER, "
            "completion_tokens INTEGER, cost_usd DOUBLE, created_at TIMESTAMP)"
        )
        for i in range(rows):
            con.execute(
                "INSERT INTO haz_doc_extractions VALUES "
                "(?, 'haiku', 'v1', 'PHL', 'FL', 2026, 9, 'ok', 'backcast', "
                "100, 100, 0.009, CURRENT_TIMESTAMP)",
                [f"rw-{i}"],
            )
        con.close()
        diagnostics = tmp_path / "diagnostics"
        diagnostics.mkdir()
        bundle.build_bundle(
            out_path=tmp_path / "b.zip", db_path=db, diagnostics_dir=diagnostics,
            run_log_dir=None, staging=tmp_path / "stg", environ={},
        )
        return bundle.read_register_from(tmp_path / "stg")

    def test_the_headroom_is_carried_at_info(self, tmp_path):
        register = self._bundle_register(tmp_path, rows=5)
        issue = next(i for i in register.issues if i.id == "extraction_budget_headroom")
        assert issue.severity == "info"
        assert "backcast share 5 of 2000" in issue.evidence

    def test_a_raced_share_is_carried_at_degraded(self, tmp_path, monkeypatch):
        """Exactly the September shape: the share gone, the month not."""

        import datetime as real_dt

        from scripts import build_resolver_debug_bundle as bundle

        class _Mid(real_dt.date):
            @classmethod
            def today(cls):
                return cls(2026, 9, 8)

        monkeypatch.setattr(bundle.dt, "date", _Mid)
        register = self._bundle_register(tmp_path, rows=2000)
        issue = next(
            i for i in register.issues if i.id == "backcast_extraction_share_raced"
        )
        assert issue.severity == "degraded"
        assert issue.cost == 22  # days of September left with no extraction
