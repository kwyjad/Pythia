# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Group E of the run-34124705852 repairs: a backlog with no lever and no price.

The backcast's share of the monthly extraction allowance bound mid-month
and 510 cyclone cells were checkpointed. That is the policy working: the
cells are owed, not lost, and the ledger resumes on exactly those. But the
only way to clear them was to raise the rulebook default, which raises the
share every night thereafter — and the backcast spends whatever it is
given. So the lever is a one-dispatch override that leaves the default
alone.

Beside it, the message. "510 cell(s) owing an answer" leaves the operator
to guess what clearing them costs, and the guess available to them is the
rulebook's per-cell DOCUMENT CAP, which is an upper bound rather than a
rate. Both factors are now measured from the run itself: dollars per call,
and calls per cell.

The reserve is the one thing the override may not touch. It is calls the
live pass keeps whatever the backcast's share says, and an override large
enough to swallow it would remove it by the back door — which is the
implicit-reserve fault that making it explicit was meant to end.

Network-free, no model calls.
"""

from __future__ import annotations

import pytest

from resolver.hazard_resolution import backcast as bc
from resolver.hazard_resolution.rulebook import load_rulebook

#: The env var the workflow input reaches load_budget through. Written out
#: here rather than imported so a pre-fix run FAILS on an assertion in a
#: test body instead of erroring at collection, where the reader cannot see
#: which behaviour was missing.
SHARE_OVERRIDE_ENV = "PYTHIA_HAZ_BACKCAST_EXTRACTION_SHARE"


def _share_override(configured, *, monthly_total, reserve):
    from resolver.hazard_resolution.extract import _share_override as impl

    return impl(configured, monthly_total=monthly_total, reserve=reserve)


# ---------------------------------------------------------------------------
# E1: a lever that lasts one dispatch
# ---------------------------------------------------------------------------


def test_no_override_keeps_the_rulebook_share(monkeypatch):
    monkeypatch.delenv(SHARE_OVERRIDE_ENV, raising=False)
    assert _share_override(2000, monthly_total=4000, reserve=1500) == 2000


def test_an_override_raises_the_share_for_this_run(monkeypatch):
    monkeypatch.setenv(SHARE_OVERRIDE_ENV, "2400")
    assert _share_override(2000, monthly_total=4000, reserve=1500) == 2400


def test_the_override_can_never_eat_the_live_reserve(monkeypatch):
    """The reserve is calls the backcast may never take, whatever it asks."""

    monkeypatch.setenv(SHARE_OVERRIDE_ENV, "4000")
    granted = _share_override(2000, monthly_total=4000, reserve=1500)
    assert granted == 2500, (
        "an override may spend up to the monthly total less the reserve and "
        "not one call more — otherwise the reserve is removed by the back door"
    )


def test_a_lower_override_needs_no_guard(monkeypatch):
    """Spending LESS for one night is always allowed."""

    monkeypatch.setenv(SHARE_OVERRIDE_ENV, "500")
    assert _share_override(2000, monthly_total=4000, reserve=1500) == 500


@pytest.mark.parametrize("raw", ["", "   ", "lots", "0", "-1", "2.5"])
def test_an_unusable_override_keeps_the_rulebook_share(monkeypatch, raw):
    """A typo must not silently change what a night spends."""

    monkeypatch.setenv(SHARE_OVERRIDE_ENV, raw)
    assert _share_override(2000, monthly_total=4000, reserve=1500) == 2000


def test_the_rulebook_default_is_not_raised():
    """The whole point: the standing policy is untouched."""

    rb = load_rulebook()
    total = int(rb.get("extraction.max_calls_per_month"))
    share = int(rb.get("extraction.backcast_max_calls_per_month"))
    reserve = int(rb.get("extraction.live_reserve_calls"))
    assert share == 2000, "the backcast's standing share must stay where it was"
    assert reserve + share <= total, "the reserve must still fit beside the share"


def test_load_budget_honours_the_override(monkeypatch, tmp_path):
    """End to end through the function the ladder actually calls."""

    import duckdb

    from resolver.hazard_resolution import extract as ex
    from resolver.hazard_resolution.schema import ensure_haz_schema

    con = duckdb.connect(str(tmp_path / "haz.duckdb"))
    ensure_haz_schema(con)
    rb = load_rulebook()

    monkeypatch.delenv(SHARE_OVERRIDE_ENV, raising=False)
    plain = ex.load_budget(con, rb, run_type="backcast")

    monkeypatch.setenv(SHARE_OVERRIDE_ENV, "2400")
    raised = ex.load_budget(con, rb, run_type="backcast")

    assert raised.backcast_max_calls_per_month == 2400
    assert plain.backcast_max_calls_per_month == 2000
    assert raised.remaining > plain.remaining, "the raise must buy calls"


def test_a_live_run_ignores_the_override(monkeypatch, tmp_path):
    """It is the BACKCAST's share; a live run has no share to raise."""

    import duckdb

    from resolver.hazard_resolution import extract as ex
    from resolver.hazard_resolution.schema import ensure_haz_schema

    con = duckdb.connect(str(tmp_path / "haz.duckdb"))
    ensure_haz_schema(con)
    monkeypatch.setenv(SHARE_OVERRIDE_ENV, "9999")
    budget = ex.load_budget(con, load_rulebook(), run_type="live")
    assert budget.backcast_max_calls_per_month is None


# ---------------------------------------------------------------------------
# E2: the deferral message states a price
# ---------------------------------------------------------------------------


def _run(**kw) -> bc.BackcastRun:
    run = bc.BackcastRun(hazard="TC", hazard_name="cyclone")
    for key, value in kw.items():
        setattr(run, key, value)
    return run


def test_the_rates_are_observed_not_assumed():
    run = _run(extraction_calls=600, extraction_cost_usd=5.22, extraction_cells=50)
    assert run.cost_per_call_usd == pytest.approx(0.0087)
    assert run.calls_per_cell == pytest.approx(12.0)


def test_the_owed_cells_are_priced_at_those_rates():
    run = _run(
        extraction_calls=600, extraction_cost_usd=5.22, extraction_cells=50,
        cells_deferred_for_budget=510,
    )
    # 510 cells x 12 calls x $0.0087
    assert run.deferred_cost_estimate_usd == pytest.approx(53.244)
    note = bc.describe_deferred_cost(run)
    assert "$0.0087 per call" in note
    assert "12.0 call(s) per cell" in note
    assert "510 owed cell(s)" in note
    assert "$53.24" in note


def test_a_run_that_billed_nothing_says_so_rather_than_printing_zero():
    """A zero here would read as 'clearing the backlog is free'."""

    run = _run(cells_deferred_for_budget=510)
    assert run.deferred_cost_estimate_usd is None
    note = bc.describe_deferred_cost(run)
    assert "no observed rate" in note
    assert "$0.00" not in note


def test_the_warning_carries_the_price():
    """The message an operator actually reads."""

    run = _run(
        extraction_budget_bound=True,
        extraction_binding_limit="backcast share (2000)",
        months_budget_deferred=3,
        cells_deferred_for_budget=510,
        extraction_calls=600, extraction_cost_usd=5.22, extraction_cells=50,
    )
    warning = (
        f"extraction budget bound at its {run.extraction_binding_limit}: "
        f"{run.months_budget_deferred} month(s) recorded deferred with "
        f"{run.cells_deferred_for_budget} cell(s) owing an answer; policy is "
        f"{run.budget_policy} — no cap was raised, the cells resume once the "
        f"calendar month's allowance resets. {bc.describe_deferred_cost(run)}"
    )
    assert "costing about $53.24" in warning


def test_the_month_counts_measure_cells_not_only_calls(tmp_path):
    """calls-per-cell needs a cell count, and it is COUNT(DISTINCT iso3)."""

    import duckdb

    from resolver.hazard_resolution.schema import ensure_haz_schema

    con = duckdb.connect(str(tmp_path / "haz.duckdb"))
    ensure_haz_schema(con)
    rows = [
        ("doc1", "SOM"), ("doc2", "SOM"), ("doc3", "SOM"),
        ("doc4", "ETH"), ("doc5", "ETH"),
    ]
    for doc, iso3 in rows:
        con.execute(
            "INSERT INTO haz_doc_extractions "
            "(doc_id, model, prompt_version, iso3, hazard, year, month, "
            " status, figures_json, cost_usd, prompt_tokens, completion_tokens) "
            "VALUES (?, 'm', 'v1', ?, 'TC', 2026, 8, 'ok', '[]', 0.01, 100, 50)",
            [doc, iso3],
        )
    counts = bc.month_counts(con, hazard="TC", ym="2026-08")
    assert counts["extraction_calls"] == 5
    assert counts["extraction_cells"] == 2, (
        "two countries were read; without this the calls-per-cell rate has "
        "no denominator and the price cannot be stated"
    )
    assert counts["extraction_cost_usd"] == pytest.approx(0.05)
