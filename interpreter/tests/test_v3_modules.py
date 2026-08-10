# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The v3 additions: decisions, the second reader, the sector, persistence.

Each of these modules is pure, so the policy they encode can be stated here
rather than inferred from a pack. What is pinned is the part a reader would
be misled by if it broke: a deadline that drifts, a "novel" fact the main
system had all along, a rank gap manufactured out of missing coverage, and a
persistence count that papers over a gap.
"""

from __future__ import annotations

import math

import pytest

from interpreter import decisions, persistence, secondopinion, sector


class TestDecisions:
    def test_month_arithmetic_crosses_the_year(self):
        assert decisions.add_months("2026-11", 3) == "2027-02"
        assert decisions.add_months("2026-02", -3) == "2025-11"
        assert decisions.add_months("nonsense", 1) is None

    def test_horizon_one_is_the_window_start_month(self):
        # The repo's anchoring convention everywhere: month index 1 IS the
        # window start, not the month after it.
        assert decisions.horizon_month("2026-09-01", 1) == "2026-09"
        assert decisions.horizon_month("2026-09-01", 3) == "2026-11"

    def test_the_deadline_is_the_peak_less_the_lead_time(self):
        point = decisions.decision_point(
            window_start="2026-09-01", peak_horizon=3, lead_months=3,
        )
        assert point["peak_month"] == "2026-11"
        assert point["deadline_month"] == "2026-08"
        assert point["deadline_month_label"] == "August 2026"

    def test_a_deadline_before_the_report_exists_reads_as_immediate(self):
        # Drought's three-month run-up subtracted from a horizon-1 peak lands
        # BEFORE the report is written. The first live report printed
        # "By June 2026" against three of its five entries, which is not a
        # deadline; it is a statement that the run-up has already begun.
        point = decisions.decision_point(
            window_start="2026-09-01", peak_horizon=1, lead_months=3,
            issued_month="2026-08",
        )
        assert point["deadline_month"] == "2026-06"       # the arithmetic stands
        assert point["derived_deadline_month_label"] == "June 2026"
        assert point["overdue"] is True
        assert point["deadline_month_label"] == decisions.OVERDUE_LABEL

    def test_a_future_deadline_keeps_its_month(self):
        point = decisions.decision_point(
            window_start="2026-09-01", peak_horizon=5, lead_months=1,
            issued_month="2026-08",
        )
        assert point["overdue"] is False
        assert point["deadline_month_label"] == "December 2026"

    def test_no_peak_horizon_means_no_derived_deadline(self):
        # The caller must treat this as "no deadline", never as licence to
        # invent one.
        assert decisions.decision_point(
            window_start="2026-09-01", peak_horizon=None, lead_months=1,
        ) == {}

    def test_the_calendar_is_ordered_by_deadline_and_undated_rows_sort_last(self):
        rows = [
            {"question_id": "late", "country_name": "Chad",
             "decision": {"deadline_month": "2026-12"}},
            {"question_id": "none", "country_name": "Angola", "decision": {}},
            {"question_id": "soon", "country_name": "Belize",
             "decision": {"deadline_month": "2026-08"}},
        ]
        out = decisions.calendar_rows(rows, {"soon": "Preposition stock."})
        assert [r["question_id"] for r in out] == ["soon", "late", "none"]
        assert out[0]["action"] == "Preposition stock."
        assert out[2]["deadline_label"] == "not dated"


class TestSecondOpinion:
    def test_direction_is_a_ratio_so_deaths_and_millions_share_one_rule(self):
        ratio = 1.5
        assert secondopinion.direction_tag(100, 400, ratio=ratio) == (
            secondopinion.TAG_MORE_ALARMED
        )
        assert secondopinion.direction_tag(400, 100, ratio=ratio) == (
            secondopinion.TAG_MORE_CAUTIOUS
        )
        assert secondopinion.direction_tag(100, 110, ratio=ratio) == (
            secondopinion.TAG_AGREES
        )

    def test_a_missing_side_is_uncovered_not_agreement(self):
        assert secondopinion.direction_tag(None, 5, ratio=1.5) == (
            secondopinion.TAG_NOT_COVERED
        )

    def test_unsettled_reads_the_share_of_the_divergence_maximum(self):
        assert secondopinion.unsettled(0.5 * math.log(2), share=0.25) is True
        assert secondopinion.unsettled(0.01, share=0.25) is False
        assert secondopinion.unsettled(None, share=0.25) is False

    def test_a_fact_the_main_system_already_had_is_not_novel(self):
        known = (
            "Heavy rainfall in the Zambezi basin displaced households in "
            "Sofala province during the second week, according to the "
            "national disaster agency."
        )
        trials = [
            "Heavy rainfall in the Zambezi basin displaced households in "
            "Sofala province during the second week, according to the "
            "national disaster agency.",
        ]
        assert secondopinion.novel_facts(trials, known) == []

    def test_a_specific_fact_absent_from_the_pack_is_surfaced(self):
        known = "Drought conditions persist across the country."
        trials = [
            "The Kariba dam reached 12 percent of usable storage on 4 July, "
            "forcing rationing across the southern grid interconnector.",
        ]
        out = secondopinion.novel_facts(trials, known)
        assert len(out) == 1
        assert "Kariba" in out[0]

    def test_a_vague_observation_never_counts_as_a_finding(self):
        known = "Nothing in particular."
        trials = [
            "The overall humanitarian situation continues to deteriorate and "
            "further deterioration cannot be ruled out at this stage.",
        ]
        assert secondopinion.novel_facts(trials, known) == []

    def test_sibyls_own_forecasting_talk_is_not_a_finding(self):
        # Every one of these is verbatim from the first live report, printed
        # under a heading promising things the main system did not know. Two
        # of them are a research to-do list.
        known = "Drought conditions persist across the country."
        process = [
            "Since Dec-Feb are always 0 and Nov is usually 0, ~2/3 of draws "
            "should be 0, so q0.1-q0.5 = 0.",
            "I therefore stay close to the pooled empirical quantiles, "
            "trimming only marginally at q0.9-q0.95.",
            "Any dated Phase 2 milestone or deadline falling within Sept "
            "2026-Feb 2027",
            "Fetching the dedicated war article should give a month-by-month "
            "casualty timeline before I submit.",
            "Nudged q0.99 up modestly to 150,000 to respect the fat "
            "conditional tail, so submitting.",
            "https://www.un.org/unispal/document/bulletin-april-2026-en-ar/",
        ]
        assert secondopinion.novel_facts(process, known, limit=10) == []

    def test_a_real_finding_still_survives_the_filter(self):
        known = "Conflict continues in several regions."
        finding = (
            "Indepaz recorded 48 massacres and 229 deaths between January and "
            "April 2026, the worst since 2016."
        )
        out = secondopinion.novel_facts([finding], known, limit=2)
        assert out == [finding]

    def test_trial_texts_walks_whatever_shape_the_traces_are_in(self):
        trials = {"trials": [{"belief_trace": [
            {"note": "x" * 50}, {"note": "short"},
        ]}]}
        out = secondopinion.trial_texts(trials)
        assert out == ["x" * 50]


class TestSector:
    def _fred(self):
        return [
            {"iso3": "SDN", "excess_nominal": 900_000.0},
            {"iso3": "SOM", "excess_nominal": 400_000.0},
            {"iso3": "TCD", "excess_nominal": 10_000.0},
            # A country expecting FEWER people than usual is not something
            # Fred is worried about, so it never enters the ranking.
            {"iso3": "UGA", "excess_nominal": -2_000_000.0},
        ]

    def test_only_positive_excess_counts_and_a_country_sums_its_hazards(self):
        scores = sector.fred_scores(
            self._fred() + [{"iso3": "SOM", "excess_nominal": 100_000.0}]
        )
        assert scores["SOM"] == pytest.approx(500_000.0)
        assert "UGA" not in scores

    def test_a_country_the_sector_does_not_cover_produces_no_gap(self):
        out = sector.compare(
            {"SDN": 900_000.0, "XXX": 800_000.0},
            {"SDN": 4.5},
            source="inform_severity", list_size=5, min_rank_gap=1,
        )
        assert out["n_compared"] == 1
        assert all(
            e["iso3"] != "XXX"
            for e in out["more_worried"] + out["less_worried"]
        )

    def test_the_gap_points_the_right_way(self):
        # Fred puts TCD first; INFORM puts it last.
        out = sector.compare(
            {"TCD": 900.0, "SDN": 500.0, "SOM": 100.0},
            {"TCD": 1.0, "SDN": 4.0, "SOM": 5.0},
            source="inform_severity", list_size=5, min_rank_gap=1,
        )
        assert out["more_worried"][0]["iso3"] == "TCD"
        assert out["less_worried"][0]["iso3"] == "SOM"

    def test_risk_radar_takes_the_worst_risk_on_file(self):
        scores = sector.risk_radar_scores([
            {"iso3": "MLI", "impact": 2},
            {"iso3": "MLI", "impact": 5},
            {"iso3": "NER", "risk_level": "Very high"},
        ])
        assert scores == {"MLI": 5.0, "NER": 5.0}

    def test_the_not_commensurable_framing_is_fixed_text(self):
        block = sector.build(
            self._fred(),
            inform_rows=[{"iso3": "SDN", "severity_score": 4.9}],
            list_size=5, min_rank_gap=1,
        )
        assert block["caveat"] == sector.NOT_COMMENSURABLE


class TestPersistence:
    def test_the_match_key_carries_the_metric(self):
        # Without it a country's drought DEATHS question matches its drought
        # PEOPLE AFFECTED question, which is what the old key did.
        a = persistence.match_key(
            {"iso3": "som", "hazard_code": "dr", "metric": "PA"}
        )
        b = persistence.match_key(
            {"iso3": "SOM", "hazard_code": "DR", "metric": "FATALITIES"}
        )
        assert a == ("SOM", "DR", "PA")
        assert a != b

    def test_a_gap_stops_the_count(self):
        key = ("SOM", "DR", "PA")
        other = ("TCD", "FL", "PA")
        # Newest first. Three runs in a row counts three.
        assert persistence.consecutive_runs(key, [[key], [key]]) == 3
        # Flagged last month, missing the month before, flagged again earlier:
        # the count stops at the gap. Calling that three would be a lie about
        # a month in which nobody flagged it.
        assert persistence.consecutive_runs(key, [[key], [other], [key]]) == 2
        assert persistence.consecutive_runs(key, [[other], [key]]) == 1
        assert persistence.consecutive_runs(key, []) == 1

    def test_persistence_reads_as_words(self):
        assert persistence.persistence_phrase(1) == "new this month"
        assert persistence.persistence_phrase(3) == "flagged for 3 consecutive runs"

    def test_movement_is_measured_only_on_the_shared_months(self):
        previous = {"2026-08": 100.0, "2026-09": 100.0, "2026-10": 100.0}
        current = {"2026-09": 200.0, "2026-10": 200.0, "2026-11": 1.0}
        out = persistence.movement(previous, current)
        assert out["n_overlapping_months"] == 2
        assert out["movement"] == persistence.MOVEMENT_RISEN
        # The month only the new run covers cannot drag the verdict down.
        assert out["current_mean"] == pytest.approx(200.0)

    def test_no_shared_month_is_no_answer_rather_than_a_guess(self):
        assert persistence.movement({"2026-01": 5.0}, {"2026-09": 5.0}) is None

    def test_a_small_move_holds(self):
        out = persistence.movement({"2026-09": 100.0}, {"2026-09": 105.0})
        assert out["movement"] == persistence.MOVEMENT_HELD

    def test_drop_reasons_name_the_cause(self):
        assert persistence.drop_reason(None) == persistence.DROP_NO_QUESTION
        assert persistence.drop_reason({"gate": None}) == persistence.DROP_BELOW_GATE
        assert persistence.drop_reason(
            {"gate": "heavy burden", "question_id": "q"},
            shown_question_ids=["q"],
        ) == ""
        assert persistence.drop_reason(
            {"gate": "heavy burden", "question_id": "q"},
        ) == persistence.DROP_NOT_SHOWN
