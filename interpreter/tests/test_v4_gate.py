# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The v4 materiality key: movement, not level.

The v3 gate tested the absolute level, so a country with a chronically large
caseload cleared it every month by construction and any positive excess then
read as worsening. Ethiopia led the September report on five expected excess
deaths. Indonesia's cyclone entry cleared on a tail whose planning figure
never moved.

Both cases are pinned here as data, because both were reported by a reader
looking at the artifact and neither was caught by the suite that existed.
"""

from __future__ import annotations

import pytest

from interpreter import config, gating, panels, selection


def _row(qid, **kw):
    row = {
        "question_id": qid,
        "iso3": kw.pop("iso3", "SOM"),
        "hazard_code": kw.pop("hazard_code", "DR"),
        "metric": kw.pop("metric", "PHASE3PLUS_IN_NEED"),
        "hazard_family": kw.pop("hazard_family", "climate"),
        "score_family": kw.pop("score_family", "spd"),
        "js_vs_baserate": kw.pop("js", 0.4),
        "exceedances": kw.pop("exceedances", [0.9]),
        "excess_nominal": kw.pop("excess", 1000.0),
        "baserate_n_obs": kw.pop("n_obs", 30),
        "delta_p50": kw.pop("delta_p50", None),
        "delta_p90": kw.pop("delta_p90", None),
        "movement_threshold": kw.pop("threshold", 1_000_000.0),
    }
    row.update(kw)
    return row


class TestTheKeyItself:
    def test_a_chronic_caseload_that_has_not_moved_is_a_burden_not_a_worsening(self):
        # Ethiopia in one row: the level test is cleared at essentially
        # certainty every month, and the planning figures have barely moved.
        row = _row(
            "eth", metric="FATALITIES", threshold=100.0,
            exceedances=[0.99, 0.99], excess=5.0,
            delta_p50=2.0, delta_p90=8.0,
        )
        counts = gating.gate_rows(
            [row], unusual_percentile=0.0, min_probability=0.25, thin_min_obs=12
        )
        assert row["passed_material"] is True      # it IS a heavy burden
        assert row["passed_worsening"] is False    # it is NOT worsening
        assert row["gate"] == gating.GATE_MAJOR
        assert counts["both"] == 0
        assert counts["major"] == 1

    def test_a_tail_that_never_moves_the_planning_figures_drops_out(self):
        # Indonesia: unusual, and a tail that clears the level test, but the
        # number a planner would write on a form does not move.
        row = _row(
            "idn", hazard_code="TC", metric="PA", threshold=50_000.0,
            exceedances=[0.3], excess=3_500.0,
            delta_p50=0.0, delta_p90=4_000.0, js=0.9,
        )
        gating.gate_rows(
            [row], unusual_percentile=0.0, min_probability=0.25, thin_min_obs=12
        )
        assert row["passed_worsening"] is False
        assert row["gate"] == gating.GATE_MAJOR

    def test_a_widened_tail_alone_is_worsening_and_says_which_figure_moved(self):
        # The case worth a reader's time: the middle sits still, the upper
        # end moves. The old key could not tell this from a caseload rising.
        row = _row(
            "sdn", threshold=1_000_000.0, js=0.9,
            delta_p50=10_000.0, delta_p90=3_000_000.0,
        )
        gating.gate_rows(
            [row], unusual_percentile=0.0, min_probability=0.25, thin_min_obs=12
        )
        assert row["gate"] == gating.GATE_WORSENING
        assert row["movement_shape"] == (
            "the contingency figure has risen while the planning figure has "
            "barely moved"
        )

    def test_a_forecast_below_its_anchor_can_never_be_worsening(self):
        # Direction is no longer a separate test: clearing a positive
        # threshold IS the direction, so Uganda's -2,069,829 cannot get in.
        row = _row(
            "uga", js=0.99, excess=-2_069_829.0,
            delta_p50=-1_500_000.0, delta_p90=-2_500_000.0,
        )
        gating.gate_rows(
            [row], unusual_percentile=0.0, min_probability=0.25, thin_min_obs=12
        )
        assert row["gate"] != gating.GATE_WORSENING

    def test_material_movement_takes_the_larger_of_the_two(self):
        assert gating.material_movement(10.0, 90.0) == 90.0
        assert gating.material_movement(90.0, 10.0) == 90.0
        assert gating.material_movement(None, 5.0) == 5.0
        assert gating.material_movement(None, None) is None

    def test_a_binary_question_is_gated_on_probability_points(self):
        # No size to plan against, so the movement is in its own units and
        # never mixed with people.
        row = _row(
            "nic", hazard_code="FL", metric="EVENT_OCCURRENCE",
            score_family="binary", threshold=0.25,
            delta_p50=0.30, delta_p90=None, exceedances=[0.55], js=0.9,
        )
        gating.gate_rows(
            [row], unusual_percentile=0.0, min_probability=0.25, thin_min_obs=12
        )
        assert row["gate"] == gating.GATE_WORSENING
        assert row["material_movement"] == pytest.approx(0.30)

    def test_level_mode_restores_the_pre_v4_behaviour(self):
        row = _row(
            "eth", metric="FATALITIES", threshold=100.0,
            exceedances=[0.99], excess=5.0, delta_p50=2.0, delta_p90=8.0,
        )
        counts = gating.gate_rows(
            [row], unusual_percentile=0.0, min_probability=0.25,
            thin_min_obs=12, mode=gating.MODE_LEVEL,
        )
        # The defect, reproduced on demand: five expected excess deaths, and
        # the row lands in the worsening section.
        assert row["gate"] == gating.GATE_WORSENING
        assert counts["mode"] == gating.MODE_LEVEL

    def test_a_row_with_no_movement_columns_is_never_worsening(self):
        # A DB predating the migration must degrade to "nothing worsened",
        # never to "everything worsened".
        row = _row("old", threshold=None, delta_p50=None, delta_p90=None, js=0.99)
        gating.gate_rows(
            [row], unusual_percentile=0.0, min_probability=0.25, thin_min_obs=12
        )
        assert row["gate"] in (gating.GATE_MAJOR, gating.GATE_WATCHLIST, None)
        assert row["gate"] != gating.GATE_WORSENING

    def test_near_misses_are_counted_so_a_recalibration_can_see_the_cut(self):
        rows = [
            _row("half", threshold=1_000_000.0, delta_p50=600_000.0, js=0.1,
                 exceedances=[0.0]),
            _row("nowhere", threshold=1_000_000.0, delta_p50=10.0, js=0.1,
                 exceedances=[0.0]),
        ]
        counts = gating.gate_rows(
            rows, unusual_percentile=0.99, min_probability=0.25, thin_min_obs=12
        )
        assert counts["near_miss"] == 1


class TestOrdering:
    def test_the_two_sections_are_ordered_differently_and_say_so(self):
        assert selection.CATEGORY_ORDERINGS[selection.CATEGORY_WORSENING] != (
            selection.CATEGORY_ORDERINGS[selection.CATEGORY_STABLE_MAJOR]
        )
        assert "planning figures" in (
            selection.CATEGORY_ORDERINGS[selection.CATEGORY_WORSENING]
        )

    def test_worsening_is_ordered_by_movement_not_by_excess(self):
        # A row with the larger excess but the smaller movement must not lead.
        rows = [
            _row("big_excess", gate=gating.GATE_WORSENING,
                 excess=9_000_000.0, material_movement=100_000.0),
            _row("big_move", gate=gating.GATE_WORSENING,
                 excess=1_000.0, material_movement=5_000_000.0),
        ]
        picked = selection.select_worsening(rows, max_entries=5)
        assert [r["question_id"] for r in picked] == ["big_move", "big_excess"]


class TestThePanelDescribesTheRuleInForce:
    def test_the_panel_states_the_movement_test(self):
        panel = panels.selection_panel({"considered": 10, "mode": "delta"})
        joined = " ".join(panel["tests"])
        assert "RISEN" in joined
        assert "heavy burden" in joined.lower()

    def test_the_panel_states_the_level_test_when_that_is_what_runs(self):
        panel = panels.selection_panel({"considered": 10, "mode": "level"})
        joined = " ".join(panel["tests"])
        assert "RISEN" not in joined


class TestThePlanningSentenceIsGenerated:
    def test_two_quantiles_in_one_open_ended_band_print_one_floor(self):
        # Sudan read "Plan against about 20,000,000 people. Hold contingency
        # for 20,000,000 people", because both quantiles land in the top band
        # and the renderer printed its centroid twice.
        text = panels.planning_sentence({
            "metric": "PHASE3PLUS_IN_NEED", "score_family": "spd",
            "p50_peak": 20_000_000.0, "p90_peak": 20_000_000.0,
            "peak_month": "November 2026",
        })
        assert "Plan against at least 15 million people" in text
        assert "Hold contingency" not in text

    def test_ordinary_figures_get_both_sentences_rounded_to_scale(self):
        text = panels.planning_sentence({
            "metric": "PA", "score_family": "spd",
            "p50_peak": 149_997.0, "p90_peak": 421_313.0,
            "peak_month": "December 2026",
        })
        assert "Plan against about 150,000 people in December 2026." in text
        assert "Hold contingency for about 421,000 people." in text

    def test_the_unit_appears_once_and_the_precision_is_honest(self):
        # "+9,577,500 people more people in crisis-level hunger" and
        # "+5 people more deaths" were both published.
        assert panels.humanise_count(9_577_500, "PHASE3PLUS_IN_NEED") == (
            "about 9.6 million people"
        )
        assert panels.humanise_count(5, "FATALITIES") == "about 5 deaths"
        assert "people" not in panels.humanise_count(5, "FATALITIES")

    def test_nothing_to_plan_against_prints_nothing(self):
        assert panels.planning_sentence(
            {"metric": "PA", "score_family": "spd"}
        ) is None


class TestConfig:
    def test_the_mode_is_delta_by_default_and_rolls_back_by_env(self, monkeypatch):
        monkeypatch.delenv("PYTHIA_INTERPRETER_GATE_MODE", raising=False)
        assert config.gate_mode() == "delta"
        monkeypatch.setenv("PYTHIA_INTERPRETER_GATE_MODE", "level")
        assert config.gate_mode() == "level"
        # An unrecognised value falls back rather than gating on nonsense.
        monkeypatch.setenv("PYTHIA_INTERPRETER_GATE_MODE", "sideways")
        assert config.gate_mode() == "delta"

    def test_the_movement_threshold_is_the_same_size_as_the_level_one(self):
        # One answer to "is this worth mobilising against", not two.
        assert config.movement_threshold("PA", None) == 50_000.0
        assert config.movement_threshold("FATALITIES", None) == 100.0
        # The population share still protects small states.
        assert config.movement_threshold("PA", 1_000_000) == 10_000.0
