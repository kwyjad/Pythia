# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Phase-6 tests for the base-rate block the forecaster is shown.

What these tests hold onto, in order of how much damage the alternative
would do:

* **the rulebook is the text's only source** — every parameter a forecaster
  is told (buffer, trigger level, ladder order, freeze window, drought rule)
  moves when the YAML moves, with no code change. A prompt describing last
  quarter's rules is confidently wrong, which is worse than silent;
* **eligibility is narrow on purpose** — these are people-affected rates, and
  a PHASE3PLUS_IN_NEED or EVENT_OCCURRENCE question resolves a different
  quantity from a source that only looks the same;
* **never assessed is not the same as assessed and quiet** — a country-hazard
  the machine has never looked at gets no block, not a block of zeros;
* **the budget is real** — an over-long block is flagged, never truncated.

Network-free and model-free: every input is seeded straight into the
``haz_base_rates_*`` tables. This file imports no other package, so it runs
under resolver-ci-fast on its own; the assembled-prompt snapshots live in
``forecaster/tests/test_haz_base_rate_prompt.py`` because they import the
prompt builder.
"""

from __future__ import annotations

import logging

import duckdb
import pytest

from resolver.hazard_resolution import prompt_block as pb
from resolver.hazard_resolution.schema import ensure_haz_schema
from resolver.tests.hazard_resolution_utils import make_rulebook, seed_base_rates

FLOOD_OCCURRENCE = {
    6: (0.44, 16), 7: (0.69, 16), 8: (0.75, 16),
    9: (0.50, 16), 10: (0.19, 16), 11: (0.06, 16),
}
FLOOD_MONTHS = [6, 7, 8, 9, 10, 11]
FLOOD_MIX = {
    "overall": {"emdat": 0.48, "ifrc_go": 0.29, "reliefweb_extracted": 0.14,
                "idmc_idu": 0.09},
    "lower_bound_share": 0.09,
}


@pytest.fixture()
def rulebook():
    return make_rulebook()


@pytest.fixture()
def con():
    connection = duckdb.connect(":memory:")
    ensure_haz_schema(connection)
    return connection


def _seed_flood(con, **overrides):
    kwargs = dict(
        iso3="PAK",
        hazard="FL",
        occurrence=FLOOD_OCCURRENCE,
        quantiles=(12_400, 48_000, 210_000, 900_000, 3_100_000),
        n_events=21,
        provenance_mix=FLOOD_MIX,
    )
    kwargs.update(overrides)
    seed_base_rates(con, **kwargs)


def _flood_block(con, rulebook, **kwargs):
    return pb.build_base_rate_block(
        "PAK", "FL", "PA", FLOOD_MONTHS, con=con, rulebook=rulebook, **kwargs
    )


# ---------------------------------------------------------------------------
# Eligibility — which questions these rates may anchor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hazard", sorted(pb.MACHINE_HAZARDS))
def test_pa_questions_on_every_machine_hazard_are_eligible(hazard):
    assert pb.is_eligible(hazard, "PA")
    assert pb.is_eligible(hazard.lower(), "pa")


@pytest.mark.parametrize(
    "hazard,metric",
    [
        # A stock of people already in Phase 3+; the machine's drought value
        # is the monthly INCREASE in that stock. Same source, different
        # quantity — the exact confusion that put GDACS modelled exposure
        # into IFRC reported-impact history once already.
        ("DR", "PHASE3PLUS_IN_NEED"),
        # Resolved from GDACS alert levels for all three hazards, whereas
        # this machine detects cyclones from IBTrACS track geometry.
        ("TC", "EVENT_OCCURRENCE"),
        ("FL", "EVENT_OCCURRENCE"),
        # Not a hazard the machine resolves at all.
        ("ACE", "PA"),
        ("ACE", "FATALITIES"),
    ],
)
def test_other_hazard_metric_pairs_are_refused(hazard, metric, con, rulebook):
    assert not pb.is_eligible(hazard, metric)
    _seed_flood(con, iso3="PAK", hazard="FL")
    assert (
        pb.build_base_rate_block(
            "PAK", hazard, metric, FLOOD_MONTHS, con=con, rulebook=rulebook
        )
        == ""
    )


def test_machine_hazards_track_the_rulebook_hazard_map():
    # Derived, not restated: a fourth hazard added to the rulebook's map
    # cannot be silently missing from prompt eligibility.
    from resolver.hazard_resolution.rulebook import HAZARD_CODE_BY_RULEBOOK_NAME

    assert pb.MACHINE_HAZARDS == frozenset(HAZARD_CODE_BY_RULEBOOK_NAME.values())


# ---------------------------------------------------------------------------
# The rules text is generated from the rulebook, never restated in code
# ---------------------------------------------------------------------------


def test_freeze_days_change_propagates_into_the_rendered_text(con):
    _seed_flood(con)
    shipped = _flood_block(con, make_rulebook())
    retuned = _flood_block(con, make_rulebook({"freeze_days": 45}))

    assert "freeze 60d after month end" in shipped
    assert "freeze 45d after month end" in retuned
    assert "60d" not in retuned


def test_trigger_level_and_ladder_order_come_from_the_rulebook(con):
    _seed_flood(con)
    shipped = _flood_block(con, make_rulebook())
    assert "GDACS flood alert at orange+" in shipped
    assert (
        "EM-DAT > ReliefWeb docs > IFRC GO > IDMC displacement (lower bound)"
        in shipped
    )

    retuned = _flood_block(
        con,
        make_rulebook(
            {
                "flood": {"gdacs_trigger_level": "red"},
                "ladder": ["ifrc_go", "emdat"],
            }
        ),
    )
    assert "GDACS flood alert at red+" in retuned
    assert "IFRC GO > EM-DAT" in retuned
    assert "ReliefWeb docs" not in retuned


def test_cyclone_buffer_and_wind_come_from_the_rulebook(rulebook):
    shipped = pb.resolution_process_text(rulebook, "TC")
    assert "within 200 km of the country at 34 kt+" in shipped

    retuned = pb.resolution_process_text(
        make_rulebook({"cyclone": {"buffer_km": 300, "min_wind_kt": 64}}), "TC"
    )
    assert "within 300 km of the country at 64 kt+" in retuned


def test_drought_text_states_the_ipc_rule_and_its_phase_threshold(rulebook):
    text = pb.resolution_process_text(rulebook, "DR")
    assert "IPC/CH Phase 3+" in text
    # The machine's drought value is a DELTA. A forecaster reading it as the
    # caseload would be out by the entire pre-existing Phase 3+ population.
    assert "not the Phase 3+ caseload" in text
    assert "no rise resolves ZERO" in text
    # No ladder for drought.
    assert "ladder" not in text.lower()


def test_drought_delta_floor_is_rendered_when_the_rulebook_sets_one():
    retuned = pb.resolution_process_text(
        make_rulebook({"drought": {"min_delta_people": 50_000, "min_delta_pct": 5.0}}),
        "DR",
    )
    assert "a rise of 50k people or 5%+" in retuned


def test_lower_bound_rung_keeps_its_label_under_the_token_budget(rulebook):
    # The token budget may shorten any rung label except this one: a
    # displacement count read as a people-affected figure is the single
    # misreading this rung can cause.
    assert "lower bound" in pb.resolution_process_text(rulebook, "FL")


# ---------------------------------------------------------------------------
# The numbers
# ---------------------------------------------------------------------------


def test_block_carries_occurrence_severity_and_provenance(con, rulebook):
    _seed_flood(con)
    block = _flood_block(con, rulebook, country_name="Pakistan")

    assert "Pakistan (PAK)" in block
    # Occurrence for the forecast months, in the forecast window's own order.
    assert "Jun 44%" in block and "Aug 75%" in block and "Nov 6%" in block
    assert block.index("Jun 44%") < block.index("Nov 6%")
    # A shared denominator is stated once rather than six times.
    assert "16y assessed" in block
    # Severity quantiles with n and window.
    assert "n=21 events, 2015-2026" in block
    assert "q10 12k" in block and "q50 210k" in block and "q90 3.1M" in block
    # Provenance mix, ordered by share.
    assert "emdat 48%" in block and "idmc_idu 9%" in block
    assert block.index("emdat 48%") < block.index("idmc_idu 9%")
    # The lower-bound share is called out, not left for the reader to infer
    # from the source names.
    assert "LOWER BOUNDS" in block


def test_months_with_no_published_rate_render_as_n_a_not_as_zero(con, rulebook):
    # base_rates.occurrence.min_years withholds a rate on a thin denominator.
    # A withheld rate rendered as 0% would be read as "this never happens".
    thin = dict(FLOOD_OCCURRENCE)
    thin.pop(8)
    _seed_flood(con, occurrence=thin)
    block = _flood_block(con, rulebook)

    assert "Aug n/a" in block
    assert "Aug 0%" not in block


def test_per_month_denominators_are_kept_when_they_differ(con, rulebook):
    uneven = dict(FLOOD_OCCURRENCE)
    uneven[8] = (0.75, 9)
    _seed_flood(con, occurrence=uneven)
    block = _flood_block(con, rulebook)

    assert "assessed" not in block.split("\n")[1]
    assert "Aug 75%(n=9y)" in block
    assert "Jun 44%(n=16y)" in block


def test_a_small_but_nonzero_rate_never_rounds_to_zero(con, rulebook):
    _seed_flood(con, occurrence={**FLOOD_OCCURRENCE, 6: (0.004, 16)})
    block = _flood_block(con, rulebook)

    assert "Jun <1%" in block
    assert "Jun 0%" not in block


# ---------------------------------------------------------------------------
# Thin and absent records
# ---------------------------------------------------------------------------


def test_thin_severity_states_the_event_count_instead_of_quantiles(con, rulebook):
    # min_events_for_quantiles = 3: two events do not make a distribution,
    # and five quantiles read as one whatever the header says.
    _seed_flood(con, n_events=2)
    block = _flood_block(con, rulebook)

    assert "2 resolved event(s) on record — too few for quantiles" in block
    assert "q10" not in block
    # The occurrence half is still there — thin severity is not thin detection.
    assert "Jul 69%" in block


def test_prompt_quantile_minimum_is_read_from_the_rulebook(con):
    _seed_flood(con, n_events=4)
    assert "q10" in _flood_block(con, make_rulebook())
    assert "q10" not in _flood_block(
        con, make_rulebook({"base_rates": {"prompt": {"min_events_for_quantiles": 5}}})
    )


def test_assessed_but_never_resolved_says_no_historical_events(con, rulebook):
    # Occurrence rows exist (the machine looked across the backcast) but no
    # value ever resolved. That is a true statement about the country.
    _seed_flood(con, quantiles=None)
    block = _flood_block(con, rulebook)

    assert "no historical events in record" in block
    assert "Jul 69%" in block
    assert "HOW THIS RESOLVES" in block


def test_never_assessed_renders_no_block_at_all(con, rulebook):
    # No rows for this country-hazard: the machine has not looked at it.
    # Claiming "no historical events in record" here would state as fact
    # something nothing was ever checked for.
    assert _flood_block(con, rulebook) == ""


def test_missing_tables_degrade_to_no_block(rulebook):
    # A forecaster DB that predates the machine is the normal case today —
    # no workflow populates haz_* yet — and must cost a prompt nothing.
    bare = duckdb.connect(":memory:")
    assert (
        pb.build_base_rate_block(
            "PAK", "FL", "PA", FLOOD_MONTHS, con=bare, rulebook=rulebook
        )
        == ""
    )


def test_a_broken_connection_degrades_to_no_block(con, rulebook):
    _seed_flood(con)
    con.close()
    assert _flood_block(con, rulebook) == ""


# ---------------------------------------------------------------------------
# Drought caveats
# ---------------------------------------------------------------------------


def test_drought_block_carries_the_record_depth_caveat(con, rulebook):
    seed_base_rates(
        con,
        iso3="ETH",
        hazard="DR",
        occurrence={3: (0.33, 9), 4: (0.44, 9), 5: (0.44, 9)},
        quantiles=(210_000, 600_000, 1_500_000, 3_400_000, 7_800_000),
        n_events=11,
        window="2017-01..2026-05",
        window_start=2017,
        window_end=2026,
        provenance_mix={"overall": {"ipc": 1.0}, "lower_bound_share": 0.0},
    )
    block = pb.build_base_rate_block(
        "ETH", "DR", "PA", [3, 4, 5], con=con, rulebook=rulebook, country_name="Ethiopia"
    )

    # Depth is derived from backcast.drought, not asserted in prose.
    assert "the drought record starts 2017" in block
    assert "IPC coverage is partial by country" in block
    # analysis_window attribution repeats one deterioration across a window,
    # so these are stocks; summing them double-counts the same people.
    assert "never sum them" in block


def test_drought_caveat_depth_follows_the_backcast_year(con):
    seed_base_rates(
        con,
        iso3="ETH",
        hazard="DR",
        occurrence={3: (0.33, 9)},
        window_start=2017,
        window_end=2026,
    )
    block = pb.build_base_rate_block(
        "ETH", "DR", "PA", [3], con=con,
        rulebook=make_rulebook({"backcast": {"drought": 2011}}),
    )
    assert "starts 2011" in block


def test_start_month_attribution_drops_the_never_sum_warning(con):
    seed_base_rates(
        con,
        iso3="ETH",
        hazard="DR",
        occurrence={3: (0.33, 9)},
        quantiles=(1, 2, 3, 4, 5),
        n_events=11,
    )
    block = pb.build_base_rate_block(
        "ETH", "DR", "PA", [3], con=con,
        rulebook=make_rulebook({"drought": {"month_attribution": "start_month"}}),
    )
    # Under start_month a value lands in one month only, so it is summable
    # and the warning would be false.
    assert "never sum them" not in block
    assert "the drought record starts" in block


# ---------------------------------------------------------------------------
# Token discipline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "iso3,hazard,months",
    [
        ("PAK", "FL", FLOOD_MONTHS),
        ("PHL", "TC", [9, 10, 11, 12, 1, 2]),
        ("ETH", "DR", [3, 4, 5, 6, 7, 8]),
    ],
)
def test_rendered_blocks_stay_inside_the_declared_budget(
    iso3, hazard, months, con, rulebook
):
    seed_base_rates(
        con,
        iso3=iso3,
        hazard=hazard,
        occurrence={m: (0.75, 16) for m in months},
        quantiles=(12_400, 48_000, 210_000, 900_000, 3_100_000),
        n_events=21,
        provenance_mix=FLOOD_MIX,
    )
    block = pb.build_base_rate_block(
        iso3, hazard, "PA", months, con=con, rulebook=rulebook,
        country_name="A Reasonably Long Country Name",
    )

    budget = int(rulebook.get("base_rates.prompt.max_chars"))
    assert 0 < len(block) <= budget
    # The stated aim is well under 300 tokens; ~4 chars/token.
    assert len(block) / 4 < 300


def test_an_over_budget_block_is_flagged_and_rendered_in_full(con, rulebook, caplog):
    _seed_flood(con)
    tiny_budget = make_rulebook({"base_rates": {"prompt": {"max_chars": 200}}})
    with caplog.at_level(logging.WARNING, logger=pb.LOG.name):
        block = _flood_block(con, tiny_budget)

    # Flagged, never cut: a block truncated mid-table still reads as a
    # complete base rate, which is how a prompt bug becomes a forecast bug.
    assert len(block) > 200
    assert "HOW THIS RESOLVES" in block
    assert "over the base_rates.prompt.max_chars budget" in caplog.text


def test_terse_count_keeps_two_significant_figures_where_they_matter():
    assert pb.terse_count(0) == "0"
    assert pb.terse_count(940) == "940"
    assert pb.terse_count(8_000) == "8.0k"
    assert pb.terse_count(12_400) == "12k"
    assert pb.terse_count(3_100_000) == "3.1M"
    assert pb.terse_count(None) == "n/a"
