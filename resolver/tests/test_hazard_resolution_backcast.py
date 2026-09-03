# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Phase-5 tests for the backcast driver.

What matters here is not that the loop runs, but that the backcast is the
SAME machine replaying history rather than a second one:

* every row it writes is stamped ``run_type='backcast'``, and nothing else
  about the write path changes;
* it FILLS GAPS and never rewrites — the freeze guard is untouched, so an
  answer that already exists survives and the attempt is audited;
* the resume ledger stops a re-run from re-deciding completed months, which
  would bury the genuine post-freeze revisions ``haz_revisions`` exists for;
* the window comes from ``rulebook.backcast.*`` and the freeze deadline, and
  ``--from``/``--to`` may narrow it but never widen it;
* a drought backcast that cannot work announces itself before it starts.

Network-free: the month runner is injected.
"""

from __future__ import annotations

import datetime as dt

import duckdb
import pytest

from resolver.hazard_resolution import backcast as bc
from resolver.hazard_resolution import resolutions as res_mod
from resolver.hazard_resolution.detect import CountryTrigger, DetectionResult, write_triggers
from resolver.hazard_resolution.reconcile import Reconciliation
from resolver.hazard_resolution.schema import (
    RUN_TYPE_BACKCAST,
    RUN_TYPE_LIVE,
    ensure_haz_schema,
)
from resolver.tests.hazard_resolution_utils import (
    make_rulebook,
    seed_resolution,
    silent_sweep_evidence,
)

TODAY = dt.date(2026, 8, 5)


def _absence(iso3: str, ym: str) -> dict:
    """Evidence of absence shaped as the CLI assembles it for a zero."""
    sweep = silent_sweep_evidence(iso3, ym)
    return {"reliefweb": sweep, "retrieved_at": sweep["retrieved_at"]}


@pytest.fixture()
def rulebook():
    return make_rulebook()


@pytest.fixture()
def con():
    con = duckdb.connect(":memory:")
    ensure_haz_schema(con)
    return con


def _writing_runner(status: str = "RESOLVED_ZERO", iso3: str = "PHL", hazard: str = "FL"):
    """A month runner that writes one resolution, recording the calls made."""

    calls: list[str] = []

    def runner(*, ym: str, run_type: str, dry_run: bool = False, **_: object) -> int:
        calls.append(ym)
        if dry_run:
            return 0
        year, month = (int(p) for p in ym.split("-"))
        con = _writing_runner.con  # bound by the test before use
        result = DetectionResult(
            hazard=hazard, ym=ym, coverage_ok=True, coverage_note="test",
            n_points_month=0, n_points_qualifying=0, max_point_time=None,
        )
        triggered = status != "RESOLVED_ZERO"
        result.rows.append(
            CountryTrigger(
                iso3=iso3,
                triggered=triggered,
                trigger_source="gdacs" if triggered else "none",
            )
        )
        write_triggers(con, result, _writing_runner.rulebook, run_type=run_type)
        if status == "RESOLVED_ZERO":
            res_mod.write_zero_resolution(
                con, iso3=iso3, year=year, month=month, hazard=hazard,
                evidence_of_absence=_absence(iso3, ym),
                rulebook=_writing_runner.rulebook, today=TODAY, run_type=run_type,
            )
        else:
            res_mod.write_reconciliation(
                con,
                Reconciliation(
                    iso3=iso3, ym=ym, hazard=hazard, status=status, value=123.0,
                    rule_fired="ladder:emdat", flagged=False, provisional=False,
                    provenance={"source": "emdat", "rule_fired": "ladder:emdat"},
                ),
                _writing_runner.rulebook, today=TODAY, run_type=run_type,
            )
        return 0

    return runner, calls


# ---------------------------------------------------------------------------
# Month planning
# ---------------------------------------------------------------------------


def test_months_between_is_inclusive_at_both_ends():
    assert bc.months_between("2020-11", "2021-02") == [
        "2020-11", "2020-12", "2021-01", "2021-02",
    ]
    assert bc.months_between("2020-11", "2020-11") == ["2020-11"]
    assert bc.months_between("2021-02", "2020-11") == []


def test_plan_runs_from_the_rulebook_year_to_the_last_frozen_month(rulebook):
    months = bc.plan_months("flood", rulebook, today=TODAY)
    assert months[0] == "2010-01"
    assert months[-1] == "2026-05"
    assert len(months) == 197


def test_plan_honours_each_hazards_own_depth(rulebook):
    assert bc.plan_months("cyclone", rulebook, today=TODAY)[0] == "2000-01"
    assert bc.plan_months("flood", rulebook, today=TODAY)[0] == "2010-01"
    assert bc.plan_months("drought", rulebook, today=TODAY)[0] == "2017-01"


def test_plan_narrows_but_never_widens_the_window(rulebook):
    """A --from before the detector's record, or a --to past the freeze
    boundary, must be clamped: the first would produce months whose zeros
    the coverage gate suppresses anyway, and the second months that can
    still change."""

    early = bc.plan_months("flood", rulebook, from_ym="1990-01", today=TODAY)
    assert early[0] == "2010-01"

    late = bc.plan_months("flood", rulebook, to_ym="2030-01", today=TODAY)
    assert late[-1] == "2026-05"

    narrowed = bc.plan_months(
        "flood", rulebook, from_ym="2015-03", to_ym="2015-06", today=TODAY
    )
    assert narrowed == ["2015-03", "2015-04", "2015-05", "2015-06"]


def test_plan_follows_a_changed_rulebook_without_a_code_change():
    moved = make_rulebook({"backcast": {"flood": 2020}})
    assert bc.plan_months("flood", moved, today=TODAY)[0] == "2020-01"


# ---------------------------------------------------------------------------
# run_type stamping
# ---------------------------------------------------------------------------


def test_backcast_stamps_every_row_it_writes(con, rulebook):
    runner, calls = _writing_runner()
    _writing_runner.con, _writing_runner.rulebook = con, rulebook

    run = bc.run_backcast(
        hazard_name="flood", rulebook=rulebook, con=con, today=TODAY,
        from_ym="2015-01", to_ym="2015-03", runner=runner,
    )

    assert calls == ["2015-01", "2015-02", "2015-03"]
    assert run.months_run == 3
    assert run.resolved_zero == 3

    run_types = {
        row[0]
        for row in con.execute("SELECT DISTINCT run_type FROM haz_resolutions").fetchall()
    }
    assert run_types == {RUN_TYPE_BACKCAST}
    trigger_types = {
        row[0] for row in con.execute("SELECT DISTINCT run_type FROM haz_triggers").fetchall()
    }
    assert trigger_types == {RUN_TYPE_BACKCAST}


def test_live_runs_are_still_stamped_live(con, rulebook):
    """The default must not move: only the backcast opts in."""

    res_mod.write_zero_resolution(
        con, iso3="PHL", year=2015, month=1, hazard="FL",
        evidence_of_absence=_absence("PHL", "2015-01"), rulebook=rulebook, today=TODAY,
    )
    assert con.execute("SELECT run_type FROM haz_resolutions").fetchone()[0] == RUN_TYPE_LIVE


def test_an_unknown_run_type_is_refused_at_the_writer(con, rulebook):
    """The ALTER that migrates an existing DB cannot carry the DDL's CHECK,
    so a live DB and a fresh one must reject the same value in Python."""

    with pytest.raises(ValueError, match="run_type"):
        res_mod.write_zero_resolution(
            con, iso3="PHL", year=2015, month=1, hazard="FL",
            evidence_of_absence=_absence("PHL", "2015-01"), rulebook=rulebook,
            today=TODAY, run_type="historical",
        )


# ---------------------------------------------------------------------------
# The freeze guard still owns history
# ---------------------------------------------------------------------------


def test_backcast_never_overwrites_an_existing_frozen_answer(con, rulebook):
    """A backcast fills gaps. An answer that already exists survives it."""

    seed_resolution(
        con, iso3="PHL", ym="2015-02", hazard="FL", status="RESOLVED_VALUE",
        value=50_000.0, source="emdat", run_type=RUN_TYPE_LIVE,
    )
    runner, _ = _writing_runner()
    _writing_runner.con, _writing_runner.rulebook = con, rulebook

    bc.run_backcast(
        hazard_name="flood", rulebook=rulebook, con=con, today=TODAY,
        from_ym="2015-02", to_ym="2015-02", runner=runner,
    )

    status, value, run_type = con.execute(
        """
        SELECT status, value, run_type FROM haz_resolutions
        WHERE iso3 = 'PHL' AND year = 2015 AND month = 2
        """
    ).fetchone()
    assert (status, value, run_type) == ("RESOLVED_VALUE", 50_000.0, RUN_TYPE_LIVE)

    # ...and the attempt is audited rather than lost.
    assert con.execute("SELECT COUNT(*) FROM haz_revisions").fetchone()[0] == 1


# ---------------------------------------------------------------------------
# Resume ledger
# ---------------------------------------------------------------------------


def test_completed_months_are_skipped_on_a_second_run(con, rulebook):
    runner, calls = _writing_runner()
    _writing_runner.con, _writing_runner.rulebook = con, rulebook
    kwargs = dict(
        hazard_name="flood", rulebook=rulebook, con=con, today=TODAY,
        from_ym="2015-01", to_ym="2015-03", runner=runner,
    )

    bc.run_backcast(**kwargs)
    calls.clear()
    second = bc.run_backcast(**kwargs)

    assert calls == []
    assert second.months_run == 0
    assert second.months_skipped_done == 3


def test_no_resume_rewalks_completed_months(con, rulebook):
    runner, calls = _writing_runner()
    _writing_runner.con, _writing_runner.rulebook = con, rulebook
    kwargs = dict(
        hazard_name="flood", rulebook=rulebook, con=con, today=TODAY,
        from_ym="2015-01", to_ym="2015-02", runner=runner,
    )

    bc.run_backcast(**kwargs)
    calls.clear()
    bc.run_backcast(resume=False, **kwargs)

    assert calls == ["2015-01", "2015-02"]


def test_a_failed_month_is_not_marked_complete_and_is_retried(con, rulebook):
    attempts: list[str] = []
    fail_on = {"2015-02"}

    def runner(*, ym: str, **_: object) -> int:
        attempts.append(ym)
        return 1 if ym in fail_on else 0

    kwargs = dict(
        hazard_name="flood", rulebook=rulebook, con=con, today=TODAY,
        from_ym="2015-01", to_ym="2015-03", runner=runner,
    )
    first = bc.run_backcast(**kwargs)
    assert first.months_failed == 1
    assert first.failures and "2015-02" in first.failures[0]

    attempts.clear()
    fail_on.clear()
    second = bc.run_backcast(**kwargs)

    assert attempts == ["2015-02"]  # only the failure is retried
    assert second.months_run == 1
    assert second.months_skipped_done == 2


def test_a_raising_month_does_not_end_the_run(con, rulebook):
    seen: list[str] = []

    def runner(*, ym: str, **_: object) -> int:
        seen.append(ym)
        if ym == "2015-01":
            raise RuntimeError("upstream exploded")
        return 0

    run = bc.run_backcast(
        hazard_name="flood", rulebook=rulebook, con=con, today=TODAY,
        from_ym="2015-01", to_ym="2015-03", runner=runner,
    )

    assert seen == ["2015-01", "2015-02", "2015-03"]
    assert run.months_failed == 1
    assert run.months_run == 2
    assert "RuntimeError" in run.failures[0]


def test_dry_run_writes_nothing_and_claims_no_month_complete(con, rulebook):
    runner, calls = _writing_runner()
    _writing_runner.con, _writing_runner.rulebook = con, rulebook

    bc.run_backcast(
        hazard_name="flood", rulebook=rulebook, con=con, today=TODAY,
        from_ym="2015-01", to_ym="2015-02", runner=runner, dry_run=True,
    )

    assert calls == ["2015-01", "2015-02"]
    assert con.execute("SELECT COUNT(*) FROM haz_resolutions").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM haz_backcast_progress").fetchone()[0] == 0


def test_the_ledger_records_what_was_actually_written(con, rulebook):
    runner, _ = _writing_runner(status="RESOLVED_VALUE")
    _writing_runner.con, _writing_runner.rulebook = con, rulebook

    bc.run_backcast(
        hazard_name="flood", rulebook=rulebook, con=con, today=TODAY,
        from_ym="2015-01", to_ym="2015-01", runner=runner,
    )

    status, cells, values, zeros = con.execute(
        """
        SELECT status, cells, resolved_value, resolved_zero
        FROM haz_backcast_progress WHERE hazard = 'FL' AND ym = '2015-01'
        """
    ).fetchone()
    assert (status, cells, values, zeros) == ("ok", 1, 1, 0)


def test_progress_is_kept_per_hazard(con, rulebook):
    runner, _ = _writing_runner()
    _writing_runner.con, _writing_runner.rulebook = con, rulebook

    bc.run_backcast(
        hazard_name="flood", rulebook=rulebook, con=con, today=TODAY,
        from_ym="2015-01", to_ym="2015-01", runner=runner,
    )

    assert bc.completed_months(con, "FL") == {"2015-01"}
    assert bc.completed_months(con, "TC") == set()


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------


def test_drought_backcast_warns_that_latest_only_indicators_cannot_backcast(rulebook):
    """The shipped ASAP feed publishes a LATEST snapshot with no archive, so
    it cannot speak for a backcast month and those cells fail the
    observation-age test. That is correct (fail-closed), and it has to be
    announced before a multi-hour run, not discovered after it."""

    check = bc.check_backcastable("drought", rulebook)

    assert check.ok  # never blocks
    joined = " | ".join(check.warnings)
    assert "asap" in joined
    assert "backcast" in joined


def test_a_dated_table_indicator_is_reported_as_what_the_backcast_rests_on(rulebook):
    """The pythia_table entries carry per-row dates, so they CAN backcast.

    That is the difference between "every month will be inconclusive" and
    "months before the ingest will be" — and an operator about to spend hours
    should be told which of the two they are in.
    """

    check = bc.check_backcastable("drought", rulebook)
    joined = " | ".join(check.warnings)

    assert "hdx_agricultural_stress" in joined
    assert "as far as those tables were ingested" in joined
    # The all-inconclusive alarm is for the case where NOTHING is dated.
    assert "EVERY \nbackcast month" not in joined


def test_a_per_month_indicator_archive_clears_the_warning():
    dated = make_rulebook(
        {
            "drought": {
                "indicators": {
                    "entries": [
                        {
                            "name": "asap",
                            "provider": "asap",
                            "url": "https://example.test/asap/{year}/{month}.json",
                            "required": True,
                            "request_timeout_sec": 60,
                            "drought_classes": ["3"],
                        }
                    ]
                }
            }
        }
    )
    assert bc.check_backcastable("drought", dated).warnings == []


def test_flood_and_cyclone_have_no_preflight_warnings(rulebook):
    assert bc.check_backcastable("flood", rulebook).warnings == []
    assert bc.check_backcastable("cyclone", rulebook).warnings == []


def test_run_with_no_months_in_range_is_not_a_failure(con, rulebook):
    run = bc.run_backcast(
        hazard_name="flood", rulebook=rulebook, con=con, today=TODAY,
        from_ym="2026-05", to_ym="2010-01", runner=lambda **_: 0,
    )
    assert run.months_planned == 0
    assert run.months_failed == 0


def test_the_ledger_records_extraction_spend_and_frozen_skips(con, rulebook):
    """The extraction/frozen columns are read back, not left at zero.

    Regression: ``record_month`` wrote extraction_calls / extraction_cost_usd
    / frozen_skipped from keys ``month_counts`` never produced, so the
    backcast's only durable cost record was always 0.
    """

    runner, _ = _writing_runner(status="RESOLVED_VALUE")
    _writing_runner.con, _writing_runner.rulebook = con, rulebook

    # Ledger rows the extraction step would have written for this target month.
    con.execute(
        """
        INSERT INTO haz_doc_extractions
            (doc_id, iso3, year, month, hazard, model, prompt_version, status,
             figures_json, prompt_tokens, completion_tokens, cost_usd)
        VALUES
            ('d1', 'PHL', 2015, 1, 'FL', 'm', 'v2', 'ok', '{}', 6000, 200, 0.01),
            ('d2', 'PHL', 2015, 1, 'FL', 'm', 'v2', 'ok', '{}', 6000, 200, 0.02),
            ('d3', 'PHL', 2015, 1, 'FL', 'm', 'v2', 'error', '{}', 0, 0, 0.0)
        """
    )

    bc.run_backcast(
        hazard_name="flood", rulebook=rulebook, con=con, today=TODAY,
        from_ym="2015-01", to_ym="2015-01", runner=runner,
    )

    calls, cost = con.execute(
        """
        SELECT extraction_calls, extraction_cost_usd
        FROM haz_backcast_progress WHERE hazard = 'FL' AND ym = '2015-01'
        """
    ).fetchone()
    assert calls == 2, "billed rows count; the zero-token error row does not"
    assert cost == pytest.approx(0.03)


def test_time_budget_defers_cleanly_and_resumes(con, rulebook):
    """A spent time budget defers months (exit-0 semantics), never fails them."""

    runner, calls = _writing_runner(status="RESOLVED_ZERO")
    _writing_runner.con, _writing_runner.rulebook = con, rulebook

    # A budget that is already spent when the loop starts: every month defers.
    run = bc.run_backcast(
        hazard_name="flood", rulebook=rulebook, con=con, today=TODAY,
        from_ym="2015-01", to_ym="2015-03", runner=runner,
        time_budget_min=1e-9,
    )
    assert run.months_run == 0
    assert run.months_failed == 0
    assert run.months_deferred == 3
    assert calls == [], "no month may start after the budget is spent"

    # The next (unbudgeted) run picks up exactly where the last stopped.
    run2 = bc.run_backcast(
        hazard_name="flood", rulebook=rulebook, con=con, today=TODAY,
        from_ym="2015-01", to_ym="2015-03", runner=runner,
    )
    assert run2.months_run == 3
    assert run2.months_deferred == 0


# ---------------------------------------------------------------------------
# A month that did not crash is not the same as a month that finished
# ---------------------------------------------------------------------------


def test_a_month_that_assessed_cells_and_wrote_nothing_is_not_complete():
    """The resume ledger must retry it, or a source outage becomes permanent.

    The runner exits 0 when it walks every cell without raising — which it
    does even when a required source was unreadable and every cell came back
    INCONCLUSIVE. Recording that as complete is worse than a failure: the
    month is never re-attempted, so an outage that lasted a week costs a year
    of history. The August 2026 ASAP outage did exactly this to twelve months
    of drought.
    """

    complete, reason = bc.month_is_complete(
        {"cells": 187, "resolved_value": 0, "resolved_zero": 0, "no_data": 0}
    )
    assert complete is False
    assert "no row written" in reason


def test_a_month_that_wrote_something_is_complete():
    assert bc.month_is_complete(
        {"cells": 187, "resolved_value": 3, "resolved_zero": 120, "no_data": 4}
    ) == (True, "")


def test_a_month_with_nothing_to_assess_is_complete():
    """No cells is not a failure — there was nothing to do."""

    assert bc.month_is_complete({"cells": 0}) == (True, "")


def test_a_month_whose_only_outcome_was_frozen_skips_is_complete():
    """Frozen cells were looked at and deliberately left alone."""

    assert bc.month_is_complete({"cells": 12, "frozen_skipped": 12}) == (True, "")
