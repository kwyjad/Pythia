# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Phase-5 tests for the acceptance report.

The report is the thing the whole machine is judged by, so what these tests
pin is that it cannot flatter itself:

* the denominator is CELLS ASSESSED — dividing resolutions by resolutions
  would score 100% against a machine that answered one cell;
* a cell that produced no row at all (fail-closed, coverage-gated) counts
  against the rate rather than vanishing from it;
* the shortfall against the 80% target is stated, per hazard, and the
  report says in as many words that it is not to be closed by tuning;
* the sample is reproducible from its seed, because a reviewer who checks a
  row must be able to regenerate the same page.

Network-free.
"""

from __future__ import annotations

import datetime as dt

import duckdb
import pytest

from resolver.hazard_resolution import acceptance as acc
from resolver.hazard_resolution.schema import ensure_haz_schema
from resolver.tests.hazard_resolution_utils import (
    make_rulebook,
    seed_candidate,
    seed_resolution,
    seed_trigger,
)

TODAY = dt.date(2026, 8, 5)
# The window build_result measures on 2026-08-05 with months=12.
IN_WINDOW = "2026-01"
OUT_OF_WINDOW = "2024-01"


@pytest.fixture()
def rulebook():
    return make_rulebook()


@pytest.fixture()
def con():
    con = duckdb.connect(":memory:")
    ensure_haz_schema(con)
    return con


def _result(con, rulebook, **kwargs):
    kwargs.setdefault("with_cross_check", False)
    return acc.build_result(con, rulebook, today=TODAY, **kwargs)


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------


def test_the_window_is_the_last_twelve_fully_frozen_months(rulebook):
    assert acc.months_window(rulebook, 12, today=TODAY) == ("2025-06", "2026-05")
    assert acc.months_window(rulebook, 1, today=TODAY) == ("2026-05", "2026-05")


def test_rows_outside_the_window_are_not_counted(con, rulebook):
    seed_resolution(con, iso3="PHL", ym=IN_WINDOW, status="RESOLVED_VALUE", value=10.0)
    seed_resolution(con, iso3="PHL", ym=OUT_OF_WINDOW, status="RESOLVED_VALUE", value=10.0)

    result = _result(con, rulebook)

    assert result.rates["FL"].cells_assessed == 1
    assert result.rates["FL"].resolved_value == 1


# ---------------------------------------------------------------------------
# The rate
# ---------------------------------------------------------------------------


def test_rate_divides_by_cells_assessed_not_by_rows_written(con, rulebook):
    """Ten cells looked at, two answered, is 20% — not 100%."""

    for index in range(10):
        seed_trigger(con, iso3=f"C{index:02d}", ym=IN_WINDOW, hazard="FL", triggered=True)
    seed_resolution(con, iso3="C00", ym=IN_WINDOW, status="RESOLVED_VALUE", value=5.0)
    seed_resolution(con, iso3="C01", ym=IN_WINDOW, status="RESOLVED_ZERO", value=0.0)

    rates = _result(con, rulebook).rates["FL"]

    assert rates.cells_assessed == 10
    assert rates.resolved == 2
    assert rates.rate_pct == pytest.approx(20.0)


def test_a_cell_that_produced_no_row_counts_against_the_rate(con, rulebook):
    """A fail-closed INCONCLUSIVE or a coverage-gated zero writes nothing.

    Those cells were assessed and not answered, so they belong in the
    denominator — and they are reported separately so the reason is visible.
    """

    for index in range(4):
        seed_trigger(con, iso3=f"C{index:02d}", ym=IN_WINDOW, hazard="DR", triggered=True)
    seed_resolution(con, iso3="C00", ym=IN_WINDOW, hazard="DR",
                    status="RESOLVED_VALUE", value=5.0)

    rates = _result(con, rulebook).rates["DR"]

    assert rates.cells_assessed == 4
    assert rates.no_row == 3
    assert rates.rate_pct == pytest.approx(25.0)


def test_no_data_is_not_a_resolution(con, rulebook):
    seed_resolution(con, iso3="PHL", ym=IN_WINDOW, status="NO_DATA", value=None, flagged=True)

    rates = _result(con, rulebook).rates["FL"]

    assert rates.no_data == 1
    assert rates.resolved == 0
    assert rates.rate_pct == pytest.approx(0.0)


def test_zeros_count_as_resolved(con, rulebook):
    """A credible zero IS an answer — that is the machine's whole premise."""

    seed_resolution(con, iso3="PHL", ym=IN_WINDOW, status="RESOLVED_ZERO", value=0.0)

    rates = _result(con, rulebook).rates["FL"]
    assert rates.resolved == 1
    assert rates.rate_pct == pytest.approx(100.0)


def test_hazards_are_reported_separately_and_summed_correctly(con, rulebook):
    seed_resolution(con, iso3="PHL", ym=IN_WINDOW, hazard="FL",
                    status="RESOLVED_VALUE", value=10.0)
    seed_resolution(con, iso3="PHL", ym=IN_WINDOW, hazard="TC",
                    status="NO_DATA", value=None)

    result = _result(con, rulebook)

    assert result.rates["FL"].rate_pct == pytest.approx(100.0)
    assert result.rates["TC"].rate_pct == pytest.approx(0.0)
    assert result.overall.cells_assessed == 2
    assert result.overall.rate_pct == pytest.approx(50.0)


def test_the_ifrc_go_baseline_is_recomputed_from_this_database(con, rulebook):
    """The ~4% figure is quoted for context; the comparison that can be
    audited is the one recomputed from the machine's own candidates."""

    for index in range(4):
        seed_trigger(con, iso3=f"C{index:02d}", ym=IN_WINDOW, hazard="FL", triggered=True)
        seed_resolution(con, iso3=f"C{index:02d}", ym=IN_WINDOW,
                        status="RESOLVED_VALUE", value=100.0, source="emdat")
    # Only one of the four cells had an IFRC GO figure to resolve from.
    seed_candidate(con, iso3="C00", ym=IN_WINDOW, source="ifrc_go")

    rates = _result(con, rulebook).rates["FL"]

    assert rates.rate_pct == pytest.approx(100.0)
    assert rates.baseline_pct == pytest.approx(25.0)


def test_lower_bound_values_are_counted_without_a_json_substring_probe(con, rulebook):
    seed_resolution(con, iso3="SOM", ym=IN_WINDOW, status="RESOLVED_VALUE",
                    value=1000.0, source="idmc_idu", lower_bound=True)
    seed_resolution(con, iso3="KEN", ym=IN_WINDOW, status="RESOLVED_VALUE",
                    value=2000.0, source="emdat", lower_bound=False)

    assert _result(con, rulebook).rates["FL"].lower_bound == 1


# ---------------------------------------------------------------------------
# The gap
# ---------------------------------------------------------------------------


def test_the_gap_section_states_the_shortfall_and_refuses_to_close_it(con, rulebook):
    for index in range(10):
        seed_trigger(con, iso3=f"C{index:02d}", ym=IN_WINDOW, hazard="FL", triggered=True)
    seed_resolution(con, iso3="C00", ym=IN_WINDOW, status="RESOLVED_VALUE", value=5.0)

    result = _result(con, rulebook)
    report = acc.render_report(result)

    assert result.rates["FL"].gap_pct == pytest.approx(70.0)
    assert "Gap against the target" in report
    assert "70.0 pp" in report
    # A hazard nobody ran is absent, not failing.
    assert "Not assessed in this window" in report
    assert "DR, TC" in report
    assert "would change the number without changing what the machine knows" in report


def test_a_hazard_that_meets_the_target_reports_no_gap(con, rulebook):
    for index in range(10):
        seed_resolution(con, iso3=f"C{index:02d}", ym=IN_WINDOW,
                        status="RESOLVED_ZERO", value=0.0)

    result = _result(con, rulebook)
    report = acc.render_report(result)

    assert result.rates["FL"].gap_pct == pytest.approx(0.0)
    assert result.rates["FL"].rate_pct >= acc.TARGET_RESOLUTION_PCT
    assert "Every hazard that ran meets the 80% target" in report


def test_a_hazard_never_run_is_not_reported_as_falling_short(con, rulebook):
    """0 of 0 cells is an absence of evidence, not a failure to resolve.

    Listing it at "0.0% — short by 80 points" would invent a failure, which
    is the same error the rate's denominator is careful to avoid.
    """

    seed_resolution(con, iso3="PHL", ym=IN_WINDOW, hazard="FL",
                    status="RESOLVED_ZERO", value=0.0)

    report = acc.render_report(_result(con, rulebook))

    assert "Not assessed in this window" in report
    assert "DR, TC" in report
    assert "80.0 pp" not in report


def test_an_empty_database_reports_no_rate_rather_than_a_failure(con, rulebook):
    report = acc.render_report(_result(con, rulebook))

    assert "No cells were assessed" in report
    assert "0.0%" in report


# ---------------------------------------------------------------------------
# Flags and provenance
# ---------------------------------------------------------------------------


def test_flags_are_counted_by_reason_not_just_as_a_boolean(con, rulebook):
    """'37 rows flagged' tells a reviewer nothing about where to look."""

    seed_resolution(con, iso3="PHL", ym=IN_WINDOW, status="RESOLVED_VALUE", value=1e9,
                    flagged=True, flags=["ceiling_exceeded", "population_cap_exceeded"])
    seed_resolution(con, iso3="VNM", ym=IN_WINDOW, status="NO_DATA", value=None,
                    flagged=True, flags=["no_candidate_past_freeze"])

    result = _result(con, rulebook)

    assert result.overall.flagged == 2
    assert result.flag_counts["ceiling_exceeded"] == 1
    assert result.flag_counts["population_cap_exceeded"] == 1
    assert result.flag_counts["no_candidate_past_freeze"] == 1


def test_provenance_mix_counts_answers_by_source(con, rulebook):
    seed_resolution(con, iso3="PHL", ym=IN_WINDOW, status="RESOLVED_VALUE",
                    value=10.0, source="emdat")
    seed_resolution(con, iso3="VNM", ym=IN_WINDOW, status="RESOLVED_VALUE",
                    value=20.0, source="emdat")
    seed_resolution(con, iso3="KEN", ym=IN_WINDOW, status="RESOLVED_VALUE",
                    value=30.0, source="reliefweb_extracted")

    mix = _result(con, rulebook).provenance_mix["FL"]

    assert mix == {
        "RESOLVED_VALUE/emdat": 2,
        "RESOLVED_VALUE/reliefweb_extracted": 1,
    }


def test_provenance_mix_separates_values_from_zeros(con, rulebook):
    """A source that produced 40 zeros and one that produced 40 figures are
    not doing the same job; one pooled count per source cannot say which."""

    seed_resolution(con, iso3="PHL", ym=IN_WINDOW, status="RESOLVED_VALUE",
                    value=10.0, source="emdat")
    seed_resolution(con, iso3="VNM", ym=IN_WINDOW, status="RESOLVED_ZERO",
                    value=0.0, source="detection:absence")

    mix = _result(con, rulebook).provenance_mix["FL"]

    assert mix == {
        "RESOLVED_VALUE/emdat": 1,
        "RESOLVED_ZERO/detection:absence": 1,
    }


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------


def test_sampling_is_reproducible_from_its_seed(con, rulebook):
    for index in range(60):
        seed_resolution(con, iso3=f"C{index:02d}", ym=IN_WINDOW,
                        status="RESOLVED_VALUE", value=float(index))

    first = _result(con, rulebook, seed=99).value_samples
    again = _result(con, rulebook, seed=99).value_samples
    different = _result(con, rulebook, seed=100).value_samples

    keys = lambda rows: [(r["iso3"], r["ym"]) for r in rows]  # noqa: E731
    assert keys(first) == keys(again)
    assert keys(first) != keys(different)


def test_samples_are_capped_at_the_requested_size(con, rulebook):
    for index in range(50):
        seed_resolution(con, iso3=f"C{index:02d}", ym=IN_WINDOW,
                        status="RESOLVED_VALUE", value=float(index))
        seed_resolution(con, iso3=f"Z{index:02d}", ym=IN_WINDOW,
                        status="RESOLVED_ZERO", value=0.0)

    result = _result(con, rulebook, sample_size=20)

    assert len(result.value_samples) == 20
    assert len(result.zero_samples) == 20


def test_a_thin_database_samples_everything_it_has(con, rulebook):
    seed_resolution(con, iso3="PHL", ym=IN_WINDOW, status="RESOLVED_VALUE", value=1.0)

    result = _result(con, rulebook, sample_size=20)

    assert len(result.value_samples) == 1
    assert result.zero_samples == []


def test_samples_carry_the_evidence_a_reviewer_needs(con, rulebook):
    seed_resolution(con, iso3="PHL", ym=IN_WINDOW, status="RESOLVED_VALUE",
                    value=12345.0, source="emdat")
    seed_resolution(con, iso3="VNM", ym=IN_WINDOW, status="RESOLVED_ZERO", value=0.0)

    result = _result(con, rulebook)
    value = result.value_samples[0]
    zero = result.zero_samples[0]

    assert value["source"] == "emdat"
    assert value["source_record_ids"]
    assert value["source_urls"]
    assert value["retrieved_at"]
    # A zero has to show what was checked, not just that it is zero.
    assert zero["evidence_of_absence"]["reliefweb_silent"] is True
    assert zero["evidence_of_absence"]["reliefweb_hits"] == 0


def test_the_rendered_report_shows_the_samples(con, rulebook):
    seed_resolution(con, iso3="PHL", ym=IN_WINDOW, status="RESOLVED_VALUE",
                    value=12345.0, source="emdat")
    seed_resolution(con, iso3="VNM", ym=IN_WINDOW, status="RESOLVED_ZERO", value=0.0)

    report = acc.render_report(_result(con, rulebook))

    assert "PHL / FL / 2026-01" in report
    assert "12,345" in report
    assert "VNM / FL / 2026-01" in report
    assert "Sample: 1 zeros" in report
    assert "Sample: 1 values" in report


def test_a_lower_bound_is_labelled_in_the_sample(con, rulebook):
    """Losing the label turns a floor into an answer."""

    seed_resolution(con, iso3="SOM", ym=IN_WINDOW, status="RESOLVED_VALUE",
                    value=900.0, source="idmc_idu", lower_bound=True)

    report = acc.render_report(_result(con, rulebook))

    assert "LOWER BOUND" in report


def test_a_backcast_row_is_marked_as_one_in_the_sample(con, rulebook):
    seed_resolution(con, iso3="PHL", ym=IN_WINDOW, status="RESOLVED_VALUE",
                    value=10.0, run_type="backcast")

    report = acc.render_report(_result(con, rulebook))

    assert "backcast" in report


# ---------------------------------------------------------------------------
# Cross-check section
# ---------------------------------------------------------------------------


def test_an_unavailable_cross_check_says_so_rather_than_going_quiet(con, rulebook):
    result = acc.build_result(con, rulebook, today=TODAY, with_cross_check=True)
    report = acc.render_report(result)

    assert "Dartmouth Flood Observatory" in report
    assert "Unavailable" in report
    assert "affects no resolved row" in report


# ---------------------------------------------------------------------------
# The headline must say what it is a headline about
# ---------------------------------------------------------------------------


def _scope_result(**assessed):
    from resolver.hazard_resolution.acceptance import AcceptanceResult, HazardRates

    return AcceptanceResult(
        first_ym="2025-08",
        last_ym="2026-07",
        months=12,
        rates={
            hazard: HazardRates(hazard=hazard, cells_assessed=cells)
            for hazard, cells in assessed.items()
        },
    )


def test_the_headline_names_the_hazards_it_covers():
    """The August 2026 report printed 0.0% as a whole-machine score when only
    drought had assessed cells — a number about one hazard, read as a number
    about the machine."""

    from resolver.hazard_resolution.acceptance import _headline_scope

    scope = _headline_scope(_scope_result(DR=756, FL=0, TC=0))
    assert "DR" in scope and "only" in scope
    assert "FL, TC" in scope
    assert "not scored at 0%" in scope


def test_a_window_where_nothing_ran_says_so_outright():
    """A headline over no assessed cells is not a measurement of anything."""

    from resolver.hazard_resolution.acceptance import _headline_scope

    scope = _headline_scope(_scope_result(DR=0, FL=0, TC=0))
    assert "not a measurement of anything" in scope


def test_full_coverage_needs_no_caveat():
    from resolver.hazard_resolution.acceptance import _headline_scope

    scope = _headline_scope(_scope_result(DR=10, FL=20, TC=30))
    assert "DR, FL, TC" in scope
    assert "absent" not in scope


def test_the_scope_line_reaches_the_rendered_report():
    """A guard that exists only in a helper is a guard nobody reads."""

    from resolver.hazard_resolution.acceptance import render_report

    assert "Scope:" in render_report(_scope_result(DR=756, FL=0, TC=0))
