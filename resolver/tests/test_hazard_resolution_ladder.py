# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Phase-2 acceptance tests for the deterministic impact ladder.

The Phase 2 acceptance list, one test group each:

- ladder ORDER — the highest populated rung wins, and nothing below it can
  overturn that;
- LOWER-BOUND labelling — an IDU figure resolves as a floor and says so;
- CEILING enforcement — GDACS exposure and national population bound the
  answer, and breaching one flags rather than rewrites;
- ORDER-OF-MAGNITUDE conflict flagging between adjacent rungs;
- FREEZE behaviour — provisional before the deadline, immutable after;
- EVENT-TO-MONTH attribution — detection spans months, figures do not.

Everything here is network-free and calls :func:`reconcile.reconcile`
directly: it is a pure function, which is the property that makes hard
rule 2 ("reconciliation is deterministic, no LLM calls") testable.
"""

from __future__ import annotations

import datetime as dt
import json

import duckdb
import pytest

from resolver.hazard_resolution import reconcile as reconcile_mod
from resolver.hazard_resolution import resolutions as res_mod
from resolver.hazard_resolution.candidates import (
    VALUE_LOWER_BOUND,
    ceiling_candidates,
    ladder_candidates,
)
from resolver.hazard_resolution.rules import attribution_month, event_months
from resolver.hazard_resolution.schema import ensure_haz_schema
from resolver.tests.hazard_resolution_utils import make_candidate, make_rulebook

YM = "2024-03"
# Far enough past 2024-03 that the cell is frozen; individual tests that
# care about the boundary pass their own `today`.
AFTER_FREEZE = dt.date(2024, 8, 1)
BEFORE_FREEZE = dt.date(2024, 4, 1)


@pytest.fixture()
def rulebook():
    return make_rulebook()


@pytest.fixture()
def con():
    con = duckdb.connect(":memory:")
    ensure_haz_schema(con)
    return con


def _reconcile(candidates, rulebook, **kwargs):
    kwargs.setdefault("today", AFTER_FREEZE)
    return reconcile_mod.reconcile(
        iso3="PHL", ym=YM, hazard="FL", candidates=candidates,
        rulebook=rulebook, **kwargs,
    )


# ---------------------------------------------------------------------------
# Ladder order
# ---------------------------------------------------------------------------


def test_highest_populated_rung_wins(rulebook):
    """EM-DAT outranks GO outranks IDU, regardless of insertion order."""
    verdict = _reconcile(
        [
            make_candidate("idmc_idu", 1_000),
            make_candidate("ifrc_go", 5_000),
            make_candidate("emdat", 4_000),
        ],
        rulebook,
    )
    assert verdict.status == reconcile_mod.STATUS_RESOLVED_VALUE
    assert verdict.value == 4_000
    assert verdict.rule_fired == "ladder:emdat"
    assert verdict.provenance["winning_rung"] == "emdat"


def test_ladder_falls_through_empty_rungs(rulebook):
    """An absent top rung is skipped, not treated as a zero."""
    verdict = _reconcile(
        [make_candidate("ifrc_go", 5_000), make_candidate("idmc_idu", 1_000)],
        rulebook,
    )
    assert verdict.value == 5_000
    assert verdict.provenance["winning_rung"] == "ifrc_go"
    assert verdict.provenance["decision"]["rungs_empty"] == [
        "emdat",
        "reliefweb_extracted",
    ]


def test_a_lower_rung_never_overturns_a_higher_one(rulebook):
    """Even a wildly larger lower-rung figure cannot displace the winner."""
    verdict = _reconcile(
        [make_candidate("emdat", 100), make_candidate("ifrc_go", 900_000)],
        rulebook,
    )
    assert verdict.value == 100  # the ladder's answer stands
    assert verdict.flagged is True  # ... and the disagreement is flagged


def test_ladder_order_follows_the_rulebook_not_the_code(rulebook):
    """Reversing the rulebook's ladder reverses the winner, with no code change."""
    reversed_rb = make_rulebook({"ladder": ["idmc_idu", "ifrc_go", "reliefweb_extracted", "emdat"],
                                 "lower_bound_rungs": ["idmc_idu"]})
    candidates = [make_candidate("emdat", 4_000), make_candidate("idmc_idu", 1_000)]

    assert _reconcile(candidates, rulebook).provenance["winning_rung"] == "emdat"
    assert _reconcile(candidates, reversed_rb).provenance["winning_rung"] == "idmc_idu"


def test_largest_figure_wins_within_one_rung(rulebook):
    """Several records on one rung collapse to the largest, deterministically."""
    verdict = _reconcile(
        [
            make_candidate("ifrc_go", 2_000, source_ref="go-a"),
            make_candidate("ifrc_go", 7_500, source_ref="go-b"),
            make_candidate("ifrc_go", 500, source_ref="go-c"),
        ],
        rulebook,
    )
    assert verdict.value == 7_500
    assert verdict.provenance["source_record_ids"] == ["go-b"]


def test_gdacs_can_never_be_the_answer(rulebook):
    """GDACS exposure is a ceiling; alone it yields NO_DATA, never a value.

    This is the guard on the hard rule that modelled exposure must never
    be published as reported impact.
    """
    candidates = [make_candidate("gdacs", 800_000)]
    assert ladder_candidates(candidates) == []
    assert len(ceiling_candidates(candidates)) == 1

    verdict = _reconcile(candidates, rulebook)
    assert verdict.status == reconcile_mod.STATUS_NO_DATA
    assert verdict.value is None


# ---------------------------------------------------------------------------
# Lower-bound labelling
# ---------------------------------------------------------------------------


def test_idu_resolves_as_a_labelled_lower_bound(rulebook):
    """A displacement figure is a floor, and every layer must say so."""
    verdict = _reconcile([make_candidate("idmc_idu", 12_000)], rulebook)

    assert verdict.value == 12_000
    assert verdict.lower_bound is True
    assert verdict.rule_fired == "ladder:idmc_idu:lower_bound"
    assert verdict.provenance["value_is_lower_bound"] is True
    assert "LOWER BOUND" in verdict.provenance["note"]
    # The candidate itself carries the type from creation, not from here.
    assert verdict.winner.value_type == VALUE_LOWER_BOUND


def test_non_lower_bound_rungs_are_not_labelled(rulebook):
    verdict = _reconcile([make_candidate("emdat", 12_000)], rulebook)
    assert verdict.lower_bound is False
    assert verdict.rule_fired == "ladder:emdat"
    assert verdict.provenance["value_is_lower_bound"] is False
    assert "note" not in verdict.provenance


def test_lower_bound_membership_comes_from_the_rulebook(rulebook):
    """Promoting GO to a lower-bound rung changes labelling, not code."""
    rb = make_rulebook({"lower_bound_rungs": ["ifrc_go", "idmc_idu"]})
    verdict = _reconcile([make_candidate("ifrc_go", 3_000)], rb)
    assert verdict.lower_bound is True
    assert verdict.rule_fired == "ladder:ifrc_go:lower_bound"


# ---------------------------------------------------------------------------
# Ceiling enforcement
# ---------------------------------------------------------------------------


def test_figure_above_the_gdacs_ceiling_is_flagged_not_rewritten(rulebook):
    """conflict_rule = ladder_with_flag: keep the answer, raise the flag."""
    verdict = _reconcile(
        [make_candidate("emdat", 900_000), make_candidate("gdacs", 100_000)],
        rulebook,
    )
    assert verdict.value == 900_000  # NOT clamped to the ceiling
    assert verdict.flagged is True
    assert reconcile_mod.FLAG_CEILING_EXCEEDED in verdict.flags


def test_figure_within_the_ceiling_is_not_flagged(rulebook):
    verdict = _reconcile(
        [make_candidate("emdat", 90_000), make_candidate("gdacs", 100_000)],
        rulebook,
    )
    assert verdict.flagged is False
    assert verdict.flags == []


def test_the_largest_overlapping_exposure_is_the_binding_ceiling(rulebook):
    """Two events overlap the month; either could account for the figure."""
    verdict = _reconcile(
        [
            make_candidate("emdat", 400_000),
            make_candidate("gdacs", 100_000, source_ref="g1"),
            make_candidate("gdacs", 500_000, source_ref="g2"),
        ],
        rulebook,
    )
    assert verdict.flagged is False  # 400k <= max(100k, 500k)


def test_ceiling_multiplier_is_honoured_from_the_rulebook(rulebook):
    """Doubling the multiplier stops the same figure breaching the ceiling."""
    candidates = [make_candidate("emdat", 150_000), make_candidate("gdacs", 100_000)]
    assert reconcile_mod.FLAG_CEILING_EXCEEDED in _reconcile(candidates, rulebook).flags

    lenient = make_rulebook({"sanity": {"ceiling_multiplier": 2.0}})
    assert _reconcile(candidates, lenient).flags == []


def test_no_exposure_estimate_means_no_ceiling(rulebook):
    """An absent GDACS figure is not evidence against a reported one."""
    verdict = _reconcile([make_candidate("emdat", 9_000_000)], rulebook)
    assert reconcile_mod.FLAG_CEILING_EXCEEDED not in verdict.flags


def test_figure_above_national_population_is_flagged(rulebook):
    """More people cannot be affected than live in the country."""
    verdict = _reconcile(
        [make_candidate("emdat", 200_000_000)],
        rulebook,
        national_population=115_000_000,
    )
    assert verdict.value == 200_000_000  # kept
    assert reconcile_mod.FLAG_POPULATION_EXCEEDED in verdict.flags


def test_population_cap_can_be_disabled_from_the_rulebook(rulebook):
    rb = make_rulebook({"sanity": {"population_cap": False}})
    verdict = _reconcile(
        [make_candidate("emdat", 200_000_000)], rb, national_population=115_000_000
    )
    assert reconcile_mod.FLAG_POPULATION_EXCEEDED not in verdict.flags


# ---------------------------------------------------------------------------
# Order-of-magnitude conflict between adjacent rungs
# ---------------------------------------------------------------------------


def test_adjacent_rungs_an_order_of_magnitude_apart_flag(rulebook):
    verdict = _reconcile(
        [make_candidate("emdat", 1_000), make_candidate("ifrc_go", 50_000)],
        rulebook,
    )
    assert verdict.value == 1_000
    assert reconcile_mod.FLAG_RUNG_CONFLICT in verdict.flags
    conflict = verdict.provenance["decision"]["conflicts"][0]
    assert conflict["upper_rung"] == "emdat"
    assert conflict["lower_rung"] == "ifrc_go"


def test_rungs_within_the_factor_do_not_flag(rulebook):
    verdict = _reconcile(
        [make_candidate("emdat", 10_000), make_candidate("ifrc_go", 50_000)],
        rulebook,
    )
    assert verdict.flags == []
    assert verdict.flagged is False


def test_conflict_factor_is_honoured_from_the_rulebook(rulebook):
    """A factor of 2 flags a 5x gap the default factor of 10 tolerates."""
    candidates = [make_candidate("emdat", 10_000), make_candidate("ifrc_go", 50_000)]
    assert _reconcile(candidates, rulebook).flags == []

    strict = make_rulebook({"conflict_detection": {"order_of_magnitude_factor": 2.0}})
    assert reconcile_mod.FLAG_RUNG_CONFLICT in _reconcile(candidates, strict).flags


def test_conflict_is_measured_between_ADJACENT_populated_rungs(rulebook):
    """Empty rungs collapse: emdat and idmc_idu are adjacent when GO is absent."""
    verdict = _reconcile(
        [make_candidate("emdat", 1_000), make_candidate("idmc_idu", 90_000)],
        rulebook,
    )
    conflict = verdict.provenance["decision"]["conflicts"][0]
    assert (conflict["upper_rung"], conflict["lower_rung"]) == ("emdat", "idmc_idu")


# ---------------------------------------------------------------------------
# NO_DATA / PENDING
# ---------------------------------------------------------------------------


def test_no_candidate_past_freeze_is_flagged_no_data(rulebook):
    verdict = _reconcile([], rulebook, today=AFTER_FREEZE)
    assert verdict.status == reconcile_mod.STATUS_NO_DATA
    assert verdict.value is None
    assert verdict.flagged is True
    assert reconcile_mod.FLAG_NO_CANDIDATE in verdict.flags
    assert verdict.provenance["decision"]["rungs_populated"] == []


def test_no_candidate_before_freeze_is_pending_not_no_data(rulebook):
    """Impatience is not a finding: sources have not finished reporting."""
    verdict = _reconcile([], rulebook, today=BEFORE_FREEZE)
    assert verdict.status == reconcile_mod.STATUS_PENDING
    assert verdict.flagged is False


def test_pending_writes_no_row(con, rulebook):
    verdict = _reconcile([], rulebook, today=BEFORE_FREEZE)
    assert res_mod.write_reconciliation(con, verdict, rulebook, today=BEFORE_FREEZE) == (
        res_mod.WRITE_PENDING
    )
    assert con.execute("SELECT COUNT(*) FROM haz_resolutions").fetchone()[0] == 0


def test_pending_sentinel_matches_the_reconciler(rulebook):
    """The literal in resolutions.py must track reconcile.STATUS_PENDING."""
    assert res_mod.WRITE_PENDING_STATUS == reconcile_mod.STATUS_PENDING


# ---------------------------------------------------------------------------
# Freeze behaviour
# ---------------------------------------------------------------------------


def test_answers_before_the_deadline_are_provisional(con, rulebook):
    verdict = _reconcile([make_candidate("emdat", 4_000)], rulebook, today=BEFORE_FREEZE)
    assert verdict.provisional is True
    res_mod.write_reconciliation(con, verdict, rulebook, today=BEFORE_FREEZE)
    row = con.execute(
        "SELECT status, value, provisional FROM haz_resolutions"
    ).fetchone()
    assert row == ("RESOLVED_VALUE", 4_000, True)


def test_answers_from_the_deadline_on_are_final(con, rulebook):
    verdict = _reconcile([make_candidate("emdat", 4_000)], rulebook, today=AFTER_FREEZE)
    assert verdict.provisional is False
    res_mod.write_reconciliation(con, verdict, rulebook, today=AFTER_FREEZE)
    assert con.execute("SELECT provisional FROM haz_resolutions").fetchone()[0] is False


def test_a_provisional_row_may_still_be_revised(con, rulebook):
    """Before the freeze the machine may change its mind — that is the point."""
    first = _reconcile([make_candidate("ifrc_go", 1_000)], rulebook, today=BEFORE_FREEZE)
    res_mod.write_reconciliation(con, first, rulebook, today=BEFORE_FREEZE)

    later = _reconcile(
        [make_candidate("ifrc_go", 1_000), make_candidate("emdat", 8_000)],
        rulebook,
        today=BEFORE_FREEZE,
    )
    assert res_mod.write_reconciliation(con, later, rulebook, today=BEFORE_FREEZE) == (
        res_mod.WRITE_WRITTEN
    )
    assert con.execute("SELECT value FROM haz_resolutions").fetchone()[0] == 8_000


def test_a_frozen_row_is_never_altered_and_the_attempt_is_audited(con, rulebook):
    """Hard rule 4: post-freeze revisions log, they do not overwrite."""
    original = _reconcile([make_candidate("emdat", 4_000)], rulebook, today=AFTER_FREEZE)
    res_mod.write_reconciliation(con, original, rulebook, today=AFTER_FREEZE)

    revised = _reconcile([make_candidate("emdat", 99_000)], rulebook, today=AFTER_FREEZE)
    outcome = res_mod.write_reconciliation(
        con, revised, rulebook, today=dt.date(2025, 1, 1)
    )

    assert outcome == res_mod.WRITE_FROZEN_SKIP
    assert con.execute("SELECT value FROM haz_resolutions").fetchone()[0] == 4_000

    revision = con.execute(
        "SELECT old_value, new_value, detail_json FROM haz_revisions"
    ).fetchone()
    assert revision[0] == 4_000
    assert revision[1] == 99_000
    assert "not altered" in json.loads(revision[2])["note"]


def test_freeze_window_is_honoured_from_the_rulebook(rulebook):
    """A 5-day freeze window freezes a cell the 60-day default leaves open."""
    quick = make_rulebook({"freeze_days": 5})
    day = dt.date(2024, 4, 10)  # 2024-03 + 10 days
    assert _reconcile([make_candidate("emdat", 1)], rulebook, today=day).provisional
    assert not _reconcile([make_candidate("emdat", 1)], quick, today=day).provisional


# ---------------------------------------------------------------------------
# Event-to-month attribution
# ---------------------------------------------------------------------------


def test_an_event_is_detected_in_every_month_it_overlaps():
    assert event_months(dt.date(2024, 1, 28), dt.date(2024, 2, 4)) == ["2024-01", "2024-02"]
    assert event_months(dt.date(2024, 1, 5), dt.date(2024, 1, 9)) == ["2024-01"]
    assert event_months(dt.date(2023, 12, 20), dt.date(2024, 2, 3)) == [
        "2023-12", "2024-01", "2024-02",
    ]


def test_a_figure_is_attributed_wholly_to_the_start_month(rulebook):
    """Never split: a source states one number for the whole event."""
    assert attribution_month(dt.date(2024, 1, 28), dt.date(2024, 2, 4), rulebook) == "2024-01"
    assert attribution_month(dt.date(2023, 12, 20), dt.date(2024, 2, 3), rulebook) == "2023-12"


def test_the_event_span_travels_with_the_figure(rulebook):
    """A reader must be able to see the figure covers more than its month."""
    verdict = _reconcile(
        [make_candidate("emdat", 4_000, span_start="2024-02-27", span_end="2024-03-06")],
        rulebook,
    )
    assert verdict.provenance["event_span"] == {
        "start": "2024-02-27",
        "end": "2024-03-06",
    }


def test_event_months_rejects_a_reversed_range():
    with pytest.raises(ValueError, match="precedes start"):
        event_months(dt.date(2024, 3, 5), dt.date(2024, 3, 1))


# ---------------------------------------------------------------------------
# Provenance completeness (hard rule 1)
# ---------------------------------------------------------------------------


def test_every_resolution_carries_full_provenance(con, rulebook):
    verdict = _reconcile(
        [
            make_candidate("emdat", 4_000, source_ref="EMDAT-2024-0123"),
            make_candidate("gdacs", 500_000),
        ],
        rulebook,
    )
    res_mod.write_reconciliation(con, verdict, rulebook, today=AFTER_FREEZE)
    stored = json.loads(
        con.execute("SELECT provenance_json FROM haz_resolutions").fetchone()[0]
    )

    assert stored["source"] == "emdat"
    assert stored["source_record_ids"] == ["EMDAT-2024-0123"]
    assert stored["source_urls"]  # document URLs
    assert stored["retrieved_at"]  # retrieval timestamp
    assert stored["rule_fired"] == "ladder:emdat"
    # Every candidate considered, not just the winner.
    assert len(stored["decision"]["candidates"]) == 2
    assert stored["decision"]["ceiling"]["exposed_population"] == 500_000


def test_reconcile_is_deterministic(rulebook):
    """Hard rule 2: same inputs, same answer, every time."""
    candidates = [
        make_candidate("emdat", 4_000),
        make_candidate("ifrc_go", 60_000),
        make_candidate("idmc_idu", 900),
        make_candidate("gdacs", 1_000),
    ]
    runs = [_reconcile(candidates, rulebook) for _ in range(5)]
    assert len({(r.status, r.value, r.rule_fired, tuple(r.flags)) for r in runs}) == 1


def test_one_countrys_exception_does_not_kill_the_month(con, rulebook, monkeypatch):
    """A cell that raises is recorded and skipped; every other cell resolves.

    The invariant the CLI comment states — one bad month must not cost the
    others their run — holds one level down too: one bad COUNTRY must not
    cost the rest of the month their answers.
    """

    from resolver.hazard_resolution import impact as impact_mod
    from resolver.tests.hazard_resolution_utils import seed_trigger

    for iso3 in ("AAA", "BBB", "CCC"):
        seed_trigger(con, iso3=iso3, ym=YM, hazard="FL", triggered=True)

    real = impact_mod.national_population

    def explode_for_bbb(con_, iso3, year):
        if iso3.upper() == "BBB":
            raise RuntimeError("synthetic per-country failure")
        return real(con_, iso3, year)

    monkeypatch.setattr(impact_mod, "national_population", explode_for_bbb)

    run = impact_mod.resolve_triggered_cells(
        con,
        ym=YM,
        hazard="FL",
        iso3s=["AAA", "BBB", "CCC"],
        rulebook=rulebook,
        extract=False,
        today=AFTER_FREEZE,
    )

    assert run.cells == 2, "AAA and CCC must still be walked"
    assert len(run.failed_cells) == 1 and run.failed_cells[0].startswith("BBB:")
    resolved = {
        row[0]
        for row in con.execute("SELECT iso3 FROM haz_resolutions").fetchall()
    }
    assert resolved == {"AAA", "CCC"}, "the failed cell writes nothing; the others stand"
