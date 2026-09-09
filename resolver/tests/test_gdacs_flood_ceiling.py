# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The GDACS flood exposure ceiling is not a population figure.

Sixteen years of flood events in ``haz_impact_candidates`` — 200 months,
5,415 ``exposed_ceiling`` rows — and the largest exposure ever recorded is
5,300 people, with a median monthly maximum of 68. Cyclone, using the same
connector, the same parser and the same table, reports a maximum of
92,308,060 and a median monthly maximum of 968,338. Three to four orders of
magnitude apart is not a difference between hazards.

The consequence sits in the cell ledger of run 34222175003: 1,962 resolved
rows carry ``ceiling_exceeded``, 1,841 of them floods, and the median
ceiling on those rows is 0.0. Bangladesh's 7.2 million in June 2022 and the
Philippines' 6.5 million in November 2025 are correct figures wearing a
doubt the machine had no basis to raise.

These tests pin the three things that have to hold: an absent unit is not a
count of people; a ceiling of zero never bounds anything; and a ceiling two
orders of magnitude below the values it bounds is reported as the broken
thing it is.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from resolver.connectors.gdacs import parse_gdacs_population
from resolver.hazard_resolution.rulebook import load_rulebook
from resolver.tests.hazard_resolution_utils import make_candidate
from resolver.hazard_resolution.rules import usable_exposure, within_sanity_ceiling

duckdb = pytest.importorskip("duckdb")

from scripts import build_resolver_debug_bundle as bundle  # noqa: E402


@pytest.fixture()
def rulebook():
    return load_rulebook()


# ---------------------------------------------------------------------------
# A: an absent unit is not a measurement
# ---------------------------------------------------------------------------


class TestTheEmptyUnit:
    """``_POPULATION_PEOPLE_UNITS`` used to contain the empty string."""

    def test_an_unlabelled_value_is_unknown(self):
        assert parse_gdacs_population("5300", None)[0] is None
        assert parse_gdacs_population("5300", "")[0] is None
        assert parse_gdacs_population("5300", "   ")[0] is None

    def test_a_zero_with_no_unit_does_not_become_a_bound_of_zero(self, rulebook):
        """The whole chain, from the feed to the rule that applies it.

        An absent measurement read as zero, then read as a ceiling, then
        used to reject a real figure. Each step is small; the sequence
        rejected 147 of 199 extracted figures in the August 2026 run.
        """

        people, detail = parse_gdacs_population("0", None, "")
        assert people is None
        assert detail["outcome"] == "no_unit"
        assert usable_exposure(people, rulebook) is None
        assert within_sanity_ceiling(7_200_000.0, people, rulebook) is True

    def test_a_labelled_value_still_passes(self):
        """The fix must not cost the figures GDACS does label."""

        assert parse_gdacs_population("500000", "people")[0] == 500_000.0
        assert parse_gdacs_population("1.67", "Million")[0] == pytest.approx(1_670_000.0)
        assert parse_gdacs_population("74000", "Pop74")[0] == 74_000.0


# ---------------------------------------------------------------------------
# B: zero is never a ceiling, wherever the ceiling is applied
# ---------------------------------------------------------------------------


class TestZeroIsNeverACeiling:
    """Already true in every application site. These keep it true."""

    @pytest.mark.parametrize("exposure", [0.0, -1.0, None, 5.0, 68.0, 955.0])
    def test_a_non_positive_or_implausible_exposure_bounds_nothing(
        self, exposure, rulebook
    ):
        assert usable_exposure(exposure, rulebook) is None
        # The four values that are not None here are the three distinct
        # ceilings the September 2026 figures ledger carried across 310
        # rows, plus the flood series' median monthly maximum. None of them
        # may reject a national monthly flood total.
        assert within_sanity_ceiling(1_917.0, exposure, rulebook) is True
        assert within_sanity_ceiling(7_200_000.0, exposure, rulebook) is True

    def test_a_plausible_exposure_still_bounds(self, rulebook):
        """The ceiling must still catch a mis-transcription."""

        multiplier = float(rulebook.get("sanity.ceiling_multiplier"))
        assert within_sanity_ceiling(100_000.0 * multiplier, 100_000.0, rulebook) is True
        assert within_sanity_ceiling(
            100_000.0 * multiplier + 1, 100_000.0, rulebook
        ) is False

    def test_the_reconciler_and_the_extractor_agree_on_what_is_usable(self, rulebook):
        """Two ceiling readers disagreeing is a figure accepted here, rejected there."""

        from resolver.hazard_resolution.candidates import Candidate, SOURCE_GDACS
        from resolver.hazard_resolution.reconcile import gdacs_ceiling_detail

        candidates = [
            Candidate(iso3="BGD", ym="2022-06", hazard="FL", value=0.0,
                      value_type="exposed_ceiling", source=SOURCE_GDACS,
                      source_ref="FL-1"),
            Candidate(iso3="BGD", ym="2022-06", hazard="FL", value=68.0,
                      value_type="exposed_ceiling", source=SOURCE_GDACS,
                      source_ref="FL-2"),
        ]
        detail = gdacs_ceiling_detail(candidates, rulebook)
        assert detail["value"] is None
        assert detail["n_events"] == 2
        assert detail["n_events_with_exposure"] == 0


# ---------------------------------------------------------------------------
# D: the two contradiction checks
# ---------------------------------------------------------------------------


def _db_with(tmp_path: Path, resolutions=(), candidates=()) -> Path:
    db = tmp_path / "resolver.duckdb"
    con = duckdb.connect(str(db))
    con.execute(
        "CREATE TABLE haz_resolutions (iso3 TEXT, year INTEGER, month INTEGER, "
        "hazard TEXT, status TEXT, value DOUBLE, provenance_json TEXT, "
        "rule_fired TEXT, flagged BOOLEAN, provisional BOOLEAN, run_type TEXT, "
        "frozen_at TIMESTAMP)"
    )
    con.execute(
        "CREATE TABLE haz_impact_candidates (iso3 TEXT, year INTEGER, month INTEGER, "
        "hazard TEXT, source TEXT, value DOUBLE, value_type TEXT, source_ref TEXT)"
    )
    for row in resolutions:
        con.execute(
            "INSERT INTO haz_resolutions VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL)", row
        )
    for row in candidates:
        con.execute("INSERT INTO haz_impact_candidates VALUES (?,?,?,?,?,?,?,?)", row)
    con.close()
    return db


def _provenance(ceiling, *, modern: bool) -> str:
    inner = {"exposed_population": ceiling, "multiplier": 3.0}
    if modern:
        inner["basis"] = "gdacs_exposed" if ceiling else "no_usable_gdacs_exposure"
    return json.dumps({"decision": {"ceiling": inner, "flags": ["ceiling_exceeded"]}})


def _checks(tmp_path: Path, db: Path) -> dict:
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    manifest = bundle.build_bundle(
        out_path=tmp_path / "b.zip", db_path=db, diagnostics_dir=diagnostics,
        run_log_dir=None, staging=tmp_path / "stg", environ={},
    )
    return {c["name"]: c for c in manifest["checks"]}


CHECK_ZERO = "no_positive_resolution_was_measured_against_a_ceiling_of_zero"
CHECK_SCALE = "every_hazard_ceiling_is_the_same_scale_as_the_values_it_bounds"


class TestTheZeroCeilingCheck:
    def test_a_current_code_row_bounded_by_zero_fails(self, tmp_path):
        db = _db_with(tmp_path, resolutions=[
            ("BGD", 2022, 6, "FL", "RESOLVED_VALUE", 7_200_000.0,
             _provenance(0.0, modern=True), "ladder", True, False, "backcast"),
        ])
        check = _checks(tmp_path, db)[CHECK_ZERO]
        assert check["verdict"] == "FAIL"
        assert "BGD/FL/2022-06" in check["detail"]
        assert "7,200,000" in check["detail"]
        assert check["issues"][0]["id"] == "resolution_bounded_by_a_zero_ceiling"

    def test_history_the_code_cannot_reach_is_counted_not_blamed(self, tmp_path):
        """A check that cannot pass teaches the reader to skip the report.

        1,841 flood rows carry this fault and every one is frozen. The
        freeze guard owns them, so a check that fails on them fails
        forever. They are reported as their own number instead.
        """

        db = _db_with(tmp_path, resolutions=[
            ("BGD", 2022, 6, "FL", "RESOLVED_VALUE", 7_200_000.0,
             _provenance(0.0, modern=False), "ladder", True, False, "backcast"),
            ("PHL", 2025, 11, "FL", "RESOLVED_VALUE", 6_522_834.0,
             _provenance(0.0, modern=False), "ladder", True, False, "backcast"),
        ])
        check = _checks(tmp_path, db)[CHECK_ZERO]
        assert check["verdict"] == "PASS"
        assert "2 row(s) predate" in check["detail"]

    def test_a_real_ceiling_passes(self, tmp_path):
        db = _db_with(tmp_path, resolutions=[
            ("PHL", 2025, 11, "TC", "RESOLVED_VALUE", 120_000.0,
             _provenance(968_338.0, modern=True), "ladder", False, False, "live"),
        ])
        assert _checks(tmp_path, db)[CHECK_ZERO]["verdict"] == "PASS"

    def test_a_row_recording_no_ceiling_is_not_a_row_bounded_by_zero(self, tmp_path):
        """`no bound` and `a bound of nothing` are different claims."""

        db = _db_with(tmp_path, resolutions=[
            ("COD", 2025, 4, "FL", "RESOLVED_VALUE", 3_450_412.0,
             _provenance(None, modern=True), "ladder", False, False, "live"),
        ])
        check = _checks(tmp_path, db)[CHECK_ZERO]
        assert check["verdict"] == "SKIP"


class TestTheCeilingScaleCheck:
    def test_the_flood_series_as_it_actually_stands_fails(self, tmp_path):
        """The numbers are the run's own, rounded to the shape they arrive in."""

        candidates = (
            [("BGD", 2022, 6, "FL", "gdacs", 68.0, "exposed_ceiling", f"FL-{i}")
             for i in range(20)]
            + [("BGD", 2022, 6, "FL", "emdat", 500_000.0, "affected", f"E-{i}")
               for i in range(20)]
        )
        check = _checks(tmp_path, _db_with(tmp_path, candidates=candidates))[CHECK_SCALE]
        assert check["verdict"] == "FAIL"
        assert "FL:" in check["detail"]
        assert check["issues"][0]["id"] == "gdacs_ceiling_is_not_the_quantity_it_bounds"

    def test_the_cyclone_series_passes_on_the_same_machinery(self, tmp_path):
        """Same connector, same parser, same table. Only the feed differs."""

        candidates = (
            [("PHL", 2025, 11, "TC", "gdacs", 968_338.0, "exposed_ceiling", f"TC-{i}")
             for i in range(20)]
            + [("PHL", 2025, 11, "TC", "emdat", 500_000.0, "affected", f"E-{i}")
               for i in range(20)]
        )
        check = _checks(tmp_path, _db_with(tmp_path, candidates=candidates))[CHECK_SCALE]
        assert check["verdict"] == "PASS"
        assert "TC:" in check["detail"]

    def test_a_ceiling_below_the_values_by_a_factor_of_a_few_is_not_flagged(
        self, tmp_path
    ):
        """Reported impact legitimately exceeds a modelled footprint.

        That is why sanity.ceiling_multiplier is 3.0 rather than 1.0, and a
        check that fired at 3x would contradict the rulebook it is meant to
        be auditing.
        """

        candidates = (
            [("PHL", 2025, 11, "TC", "gdacs", 100_000.0, "exposed_ceiling", f"TC-{i}")
             for i in range(10)]
            + [("PHL", 2025, 11, "TC", "emdat", 300_000.0, "affected", f"E-{i}")
               for i in range(10)]
        )
        check = _checks(tmp_path, _db_with(tmp_path, candidates=candidates))[CHECK_SCALE]
        assert check["verdict"] == "PASS"

    def test_a_hazard_with_no_ceiling_at_all_is_not_a_failure(self, tmp_path):
        """An honestly absent ceiling is the right outcome, not a fault."""

        candidates = [
            ("BGD", 2022, 6, "FL", "emdat", 500_000.0, "affected", "E-1"),
        ]
        check = _checks(tmp_path, _db_with(tmp_path, candidates=candidates))[CHECK_SCALE]
        assert check["verdict"] == "SKIP"


# ---------------------------------------------------------------------------
# C: what the investigation concluded, and what it cost to record it
# ---------------------------------------------------------------------------


class TestTheFindingIsRecordedWhereItCostsNothing:
    def test_the_rulebook_states_what_the_flood_feed_supplies(self):
        """The conclusion belongs beside the threshold it explains.

        A finding that lives only in a commit message is a finding the next
        reader of `sanity:` does not have.
        """

        text = (
            Path(__file__).resolve().parents[1]
            / "hazard_resolution" / "rulebook.yaml"
        ).read_text(encoding="utf-8")
        assert "not a national population exposure" in text
        assert "sendai" in text
        assert "geteventdata" in text

    def test_a_comment_moves_no_hazard_fingerprint(self):
        """Which is the whole reason the finding is a comment.

        A rulebook VALUE under `sanity` changes the digest of all three
        hazards, and the resume ledger would then re-walk history that the
        freeze guard will not let a re-walk change: months of nightly runs
        to write revision rows and nothing else. A comment is free, so the
        finding is recorded now and the semantic change stays a decision
        rather than a side effect.
        """

        import yaml

        from resolver.hazard_resolution.rulebook import Rulebook, load_rulebook

        loaded = load_rulebook()
        stripped = Rulebook(
            yaml.safe_load(
                "\n".join(
                    line for line in loaded.path.read_text(encoding="utf-8").splitlines()
                    if not line.lstrip().startswith("#")
                )
            ),
            path=loaded.path,
        )
        for hazard in ("flood", "cyclone", "drought"):
            assert stripped.hazard_fingerprint(hazard) == loaded.hazard_fingerprint(hazard)


# ---------------------------------------------------------------------------
# The flood ceiling is closed rather than thresholded (Sept 2026)
# ---------------------------------------------------------------------------

from dataclasses import replace as dc_replace  # noqa: E402

from resolver.hazard_resolution import candidates as cand_mod  # noqa: E402
from resolver.hazard_resolution import reconcile as rc_mod  # noqa: E402
from resolver.hazard_resolution import rules as rules_mod  # noqa: E402


class TestFloodTakesNoGdacsCeilingAtAnySize:
    """A threshold inside a series of wrong numbers is still wrong numbers.

    Run 34222175003's cell ledger, on rows where GDACS supplied the ceiling:
    flood's median ceiling is 0 and its largest ever 5,300, against
    cyclone's 3,747,249 and 725,467,896 — same connector, same parser, same
    column. `min_plausible_exposure` discarded the small ones and left ten
    standing, and all five of those that ever bound a figure were wrong by
    three orders of magnitude: Libya 2023-09 rejected 1,600,000 against
    5,300, Pakistan 2022-06 rejected 7,245,490 against 1,061.

    So the question is not where to draw a line. It is whether that field
    bounds anything at all for floods, and until somebody establishes what
    it carries the answer is no.
    """

    def test_a_flood_exposure_far_above_the_floor_is_still_not_a_ceiling(self, rulebook):
        floor = float(rulebook.get("sanity.min_plausible_exposure"))
        assert rules_mod.usable_exposure(floor * 100, rulebook, "FL") is None

    def test_the_five_ceilings_that_survived_the_floor_no_longer_bind(self, rulebook):
        # The real ones, from the cell ledger of run 34222175003.
        for ceiling in (5_300.0, 1_377.0, 1_061.0):
            assert rules_mod.usable_exposure(ceiling, rulebook, "FL") is None

    def test_cyclone_keeps_its_ceiling(self, rulebook):
        # The same field on the same connector, and there it is a national
        # population exposure: a cyclone ceiling flags a figure twice its
        # size, which is a plausibility check working.
        assert rules_mod.usable_exposure(3_747_249.0, rulebook, "TC") == 3_747_249.0

    def test_an_unknown_hazard_is_judged_on_the_number_alone(self, rulebook):
        floor = float(rulebook.get("sanity.min_plausible_exposure"))
        assert rules_mod.usable_exposure(floor * 10, rulebook) == floor * 10
        assert rules_mod.usable_exposure(floor / 10, rulebook) is None

    def test_the_ladder_stops_flagging_derna_against_five_thousand(self, rulebook):
        """Libya 2023-09: 1,600,000 people, rejected against a bound of 5,300."""

        verdict = rc_mod.reconcile(
            iso3="LBY", ym="2023-09", hazard="FL",
            candidates=[
                dc_replace(make_candidate("emdat", 1_600_000), iso3="LBY",
                           ym="2023-09", hazard="FL"),
                dc_replace(make_candidate("gdacs", 5_300), iso3="LBY",
                           ym="2023-09", hazard="FL"),
            ],
            rulebook=rulebook, today=dt.date(2026, 9, 9),
        )
        assert verdict.value == 1_600_000
        assert rc_mod.FLAG_CEILING_EXCEEDED not in verdict.flags
        assert verdict.provenance["decision"]["ceiling"]["basis"] != "gdacs_exposed"

    def test_the_population_share_still_bounds_a_flood(self, rulebook):
        """The share bounds 771 resolved rows and is deliberately untouched.

        Closing the GDACS ceiling must not close this one: the rule is
        about what a GDACS figure carries, not about whether floods may be
        bounded at all.
        """

        share = float(rulebook.get("sanity.population_fallback_share"))
        multiplier = float(rulebook.get("sanity.ceiling_multiplier"))
        population = 20_000_000
        over = population * share * multiplier * 1.5
        verdict = rc_mod.reconcile(
            iso3="PHL", ym="2024-03", hazard="FL",
            candidates=[make_candidate("emdat", over)],
            rulebook=rulebook, national_population=population,
            today=dt.date(2026, 9, 9),
        )
        assert rc_mod.FLAG_CEILING_EXCEEDED in verdict.flags
        assert verdict.provenance["decision"]["ceiling"]["basis"] == "population_share"

    def test_the_extraction_path_offers_no_flood_ceiling_either(self, rulebook):
        """A figure rejected before it reaches the ladder is just as lost."""

        events = [
            dc_replace(make_candidate("gdacs", 5_300), hazard="FL"),
        ]
        keep = [
            c for c in events
            if rules_mod.usable_exposure(c.value, rulebook, c.hazard) is not None
        ]
        assert keep == []

    def test_the_record_says_why_the_ceiling_is_absent(self):
        """"GDACS described no event" and "its figure bounds nothing" want
        different repairs, and a blank ceiling column says neither."""

        assert "FL" in rules_mod.NO_POPULATION_EXPOSURE_HAZARDS
        assert "TC" not in rules_mod.NO_POPULATION_EXPOSURE_HAZARDS
        assert hasattr(cand_mod, "NO_POPULATION_EXPOSURE_HAZARDS")

    def test_closing_the_flood_ceiling_costs_no_re_walk(self, rulebook):
        """It must not be a rulebook value, or it re-walks frozen history.

        The resume ledger stamps each month with a digest of the rulebook
        that decided it, so a rulebook VALUE under `sanity` or `flood.gdacs`
        moves the digest for flood and cyclone alike and puts years of
        months back on the queue — to produce revision rows the freeze guard
        will not let change an answer. Cost with nothing on the other side.

        So the rule is a code constant, and this holds it there.
        """

        text = (
            Path(__file__).resolve().parents[1]
            / "hazard_resolution" / "rulebook.yaml"
        ).read_text(encoding="utf-8")
        keys = [
            line.split(":", 1)[0].strip()
            for line in text.splitlines()
            if ":" in line and not line.lstrip().startswith("#")
        ]
        assert not any("population_exposure" in k.lower() for k in keys), (
            "the flood-ceiling rule has moved into the rulebook and now "
            "re-walks history"
        )
        for hazard in ("flood", "cyclone", "drought"):
            # It has a fingerprint at all, i.e. the guard above is checking
            # something rather than an absent mechanism.
            assert rulebook.hazard_fingerprint(hazard)
