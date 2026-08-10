# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Part B: the dormant state, and the discipline that keeps it honest later.

The dormant tests matter now. The small-sample tests matter in six weeks,
when the first outcomes land and a naive report announces that Fred beat
climatology on eighty question-months. That announcement is the failure this
module exists to prevent, so it is pinned before it can happen.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from interpreter import performance

_GLOSSARY_TS = (
    Path(__file__).resolve().parents[2] / "web" / "src" / "lib" / "score_glossary.ts"
)


class TestUpcomingResolutions:
    def _questions(self):
        return [
            {"question_id": "a", "window_start_date": "2026-09-01",
             "hazard_code": "DR"},
            {"question_id": "b", "window_start_date": "2026-09-01",
             "hazard_code": "ACE"},
        ]

    def test_a_window_month_resolves_the_month_after_it(self):
        # The pipeline resolves the previous COMPLETE month, on the 28th.
        assert performance.resolution_date_for("2026-09") == "2026-10-28"

    def test_the_dormant_state_knows_what_is_coming_and_when(self):
        out = performance.upcoming_resolutions(
            self._questions(), as_of="2026-08-10"
        )
        assert out["n_question_horizons"] == 12  # two questions, six months
        assert out["first_resolution_date"] == "2026-10-28"
        assert out["first_resolution_window_month"] == "September 2026"
        assert out["n_first_resolution_horizons"] == 2
        assert out["by_hazard"] == {"ACE": 6, "DR": 6}

    def test_a_date_already_passed_is_reported_as_overdue_not_dropped(self):
        out = performance.upcoming_resolutions(
            self._questions(), as_of="2027-01-01"
        )
        assert out["n_overdue_slots"] > 0

    def test_the_dormant_sentences_say_more_than_nothing_to_score(self):
        out = performance.upcoming_resolutions(
            self._questions(), as_of="2026-08-10"
        )
        text = " ".join(performance.dormant_sentences(out))
        assert "2026-10-28" in text
        assert "September 2026" in text
        assert "shadow mode" in text


class TestSmallSampleDiscipline:
    def _paired(self, n, model, reference):
        return [(model, reference)] * n

    def test_below_the_floor_the_report_declines_the_claim(self):
        claim = performance.skill_claim(
            self._paired(40, 0.30, 0.60), min_resolved=100, samples=200,
        )
        assert claim["claim_allowed"] is False
        assert claim["verdict"] == performance.VERDICT_TOO_FEW
        # No number is offered at all. A printed figure beside a hedge is the
        # figure the reader keeps.
        assert claim["skill"] is None
        assert "40" in claim["reason"]

    def test_above_the_floor_the_interval_is_what_gets_printed(self):
        claim = performance.skill_claim(
            self._paired(150, 0.30, 0.60), min_resolved=100, samples=300,
        )
        assert claim["claim_allowed"] is True
        assert claim["skill"] == pytest.approx(0.5)
        assert claim["ci_low"] is not None and claim["ci_high"] is not None

    def test_an_interval_straddling_zero_declines_to_call_it(self):
        rows = [(0.5, 0.5)] * 60 + [(0.9, 0.1)] * 60 + [(0.1, 0.9)] * 60
        claim = performance.skill_claim(rows, min_resolved=100, samples=400)
        assert claim["verdict"] == performance.VERDICT_UNCLEAR

    def test_the_bootstrap_is_seeded_so_the_interval_does_not_move(self):
        rows = [(0.2 + i * 0.001, 0.5) for i in range(200)]
        first = performance.bootstrap_skill(rows, samples=300)
        second = performance.bootstrap_skill(rows, samples=300)
        assert first == second

    def test_a_reference_of_zero_supports_no_skill_figure(self):
        assert performance.bootstrap_skill([(0.1, 0.0)] * 200, samples=100) is None


class TestDiary:
    def test_only_flagged_questions_that_resolved_appear(self):
        reports = [{
            "month_label": "2026-08",
            "entries": [
                {"question_ids": ["resolved_q"], "iso3": "SOM",
                 "hazard_code": "DR", "metric": "PHASE3PLUS_IN_NEED",
                 "why_it_stands_out": "It looked bad."},
                {"question_ids": ["open_q"], "iso3": "TCD",
                 "hazard_code": "FL", "metric": "PA"},
            ],
        }]
        resolved = {"resolved_q": [
            {"horizon_m": 1, "observed_month": "2026-09", "value": 1234.0},
        ]}
        out = performance.diary(reports, resolved)
        assert [r["question_id"] for r in out] == ["resolved_q"]
        assert out[0]["report_month"] == "2026-08"
        assert out[0]["observed"][0]["value"] == 1234.0

    def test_a_null_resolution_is_not_an_outcome(self):
        reports = [{"month_label": "2026-08", "entries": [
            {"question_ids": ["q"], "iso3": "SOM"},
        ]}]
        resolved = {"q": [{"horizon_m": 1, "observed_month": "2026-09",
                           "value": None}]}
        assert performance.diary(reports, resolved) == []


class TestScoreCopyIsSingleSourced:
    """The report and the dashboard must not explain a score two ways.

    Python cannot import TypeScript, so the guarantee is a test: every
    sentence the report prints has to appear verbatim in the dashboard's
    shared glossary file.
    """

    def test_every_explanation_appears_verbatim_in_the_dashboard_glossary(self):
        assert _GLOSSARY_TS.exists(), _GLOSSARY_TS
        # The TS source wraps long strings across lines with " + ", so the
        # comparison is made on the concatenated literals.
        raw = _GLOSSARY_TS.read_text(encoding="utf-8")
        joined = re.sub(r'"\s*\+\s*\n?\s*"', "", raw)
        for term, text in performance.SCORE_EXPLANATIONS:
            assert term in joined, f"{term!r} missing from score_glossary.ts"
            assert text in joined, (
                f"the report's copy for {term!r} is not in score_glossary.ts — "
                "the two have drifted"
            )
