# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""What the report covers, and what it calls things.

The selection rules are where a reader is most likely to be misled: a
forecast far BELOW its base rate is not a worsening situation, and an entry
that clears the worsening threshold must never be described as "roughly
stable" simply because a cap crowded it out. Both are pinned here.
"""

from __future__ import annotations

import math

import pytest

from interpreter import charts, names, selection


def _row(qid, family, *, log_ev=None, per_100k=None, nominal=0.0):
    return {
        "question_id": qid,
        "hazard_family": family,
        "log_ev_ratio": log_ev,
        "direction": selection.direction(log_ev),
        "eiv_per_100k": per_100k,
        "eiv_nominal": nominal,
    }


class TestDirection:
    def test_sign_decides_and_a_deadband_holds_noise_out(self):
        assert selection.direction(math.log(2.0)) == "above"
        assert selection.direction(math.log(0.5)) == "below"
        assert selection.direction(0.01) == "at"
        assert selection.direction(None) is None

    def test_ev_multiple_is_the_readable_form(self):
        assert selection.ev_multiple(math.log(14.11)) == pytest.approx(14.11)
        assert selection.ev_multiple(None) is None


class TestWorsening:
    def test_below_base_rate_can_never_enter(self):
        rows = [
            _row("a", "climate", log_ev=math.log(0.01)),   # 100x BELOW
            _row("b", "climate", log_ev=math.log(1.2)),
        ]
        picked = selection.select_worsening(
            rows, threshold_multiple=2.0, min_entries=3, max_entries=6
        )
        assert [r["question_id"] for r in picked] == ["b"]

    def test_minimum_shows_the_worst_few_even_below_threshold(self):
        rows = [
            _row("a", "conflict", log_ev=math.log(1.5)),
            _row("b", "conflict", log_ev=math.log(1.4)),
            _row("c", "conflict", log_ev=math.log(1.3)),
            _row("d", "conflict", log_ev=math.log(1.2)),
        ]
        picked = selection.select_worsening(
            rows, threshold_multiple=2.0, min_entries=3, max_entries=6
        )
        # Three always appear, ranked; the fourth clears nothing so it stops.
        assert [r["question_id"] for r in picked] == ["a", "b", "c"]

    def test_threshold_admits_extras_up_to_the_cap(self):
        rows = [_row(f"q{i}", "climate", log_ev=math.log(10 - i)) for i in range(8)]
        picked = selection.select_worsening(
            rows, threshold_multiple=2.0, min_entries=3, max_entries=6
        )
        assert len(picked) == 6
        assert [r["question_id"] for r in picked] == [f"q{i}" for i in range(6)]

    def test_ordering_is_stable_on_ties(self):
        rows = [
            _row("zzz", "climate", log_ev=0.5),
            _row("aaa", "climate", log_ev=0.5),
        ]
        picked = selection.select_worsening(
            rows, threshold_multiple=2.0, min_entries=2, max_entries=6
        )
        assert [r["question_id"] for r in picked] == ["aaa", "zzz"]


class TestStableMajor:
    def test_per_capita_floor_keeps_out_tiny_absolute_impacts(self):
        rows = [
            _row("small", "climate", per_100k=9000.0, nominal=50.0),
            _row("real", "climate", per_100k=400.0, nominal=90000.0),
        ]
        picked = selection.select_stable_major(
            rows,
            exclude_question_ids=set(),
            max_entries=6,
            per_capita_floor=10000.0,
            threshold_multiple=2.0,
        )
        assert [r["question_id"] for r in picked] == ["real"]

    def test_a_worsening_entry_is_never_relabelled_as_stable(self):
        # Clears the worsening threshold but was crowded out by the cap.
        rows = [
            _row("hot", "climate", log_ev=math.log(9.0), per_100k=800.0, nominal=99999.0),
        ]
        picked = selection.select_stable_major(
            rows,
            exclude_question_ids=set(),
            max_entries=6,
            per_capita_floor=10000.0,
            threshold_multiple=2.0,
        )
        assert picked == []

    def test_already_selected_never_appears_twice(self):
        rows = [_row("x", "conflict", per_100k=500.0, nominal=50000.0)]
        picked = selection.select_stable_major(
            rows,
            exclude_question_ids={"x"},
            max_entries=6,
            per_capita_floor=10000.0,
            threshold_multiple=2.0,
        )
        assert picked == []


class TestAssignCategories:
    def test_four_boxes_and_report_order(self):
        rows = [
            _row("c_w", "climate", log_ev=math.log(5.0)),
            _row("k_w", "conflict", log_ev=math.log(4.0)),
            _row("c_s", "climate", per_100k=900.0, nominal=80000.0),
            _row("k_s", "conflict", per_100k=700.0, nominal=80000.0),
            _row("ignored", "other", log_ev=math.log(50.0)),
        ]
        selection.assign_categories(
            rows,
            threshold_multiple=2.0,
            min_entries=3,
            max_entries=6,
            per_capita_floor=10000.0,
        )
        ordered = [r["question_id"] for r in selection.selected_rows(rows)]
        assert ordered == ["c_w", "k_w", "c_s", "k_s"]
        # A hazard outside the two families is carried in the index but never
        # categorised, so it cannot be filed under a heading that misdescribes it.
        assert [r for r in rows if r["question_id"] == "ignored"][0]["category"] is None

    def test_every_row_keeps_its_place_in_the_index(self):
        rows = [_row("a", "climate", log_ev=math.log(3.0)), _row("b", "climate")]
        out = selection.assign_categories(
            rows,
            threshold_multiple=2.0,
            min_entries=1,
            max_entries=6,
            per_capita_floor=10000.0,
        )
        assert len(out) == 2
        assert out[1]["category"] is None


class TestNames:
    def test_codes_become_words(self):
        assert names.country_name("NIC") == "Nicaragua"
        assert names.hazard_name("DR") == "Drought"
        assert names.metric_name("EVENT_OCCURRENCE") == "a major disaster alert"
        assert names.metric_name("PA", short=True) == "people affected"

    def test_unknown_code_degrades_to_the_code_not_a_crash(self):
        assert names.country_name("ZZZ") == "ZZZ"
        assert names.hazard_name("XX") == "XX"

    def test_families(self):
        assert names.hazard_family("FL") == "climate"
        assert names.hazard_family("ACE") == "conflict"
        assert names.hazard_family("PHE") == "other"

    def test_describe_pair_prints_no_code(self):
        text = names.describe_pair("NIC", "DR", "EVENT_OCCURRENCE")
        assert "NIC" not in text and "DR" not in text
        assert text.startswith("Nicaragua, drought")


class TestCharts:
    def test_bars_sum_to_the_forecast_and_mark_the_modal_band(self):
        svg = charts.probability_chart([0.1, 0.7, 0.2], ["0", "1-<10k", ">=10k"])
        assert svg.startswith("<svg")
        assert "70%" in svg and "1-&lt;10k" in svg
        assert charts.FRED_PRIMARY in svg

    def test_mapping_form_is_accepted(self):
        svg = charts.probability_chart({"1": 0.5, "2": 0.5})
        assert "50%" in svg

    def test_nothing_to_draw_returns_empty_so_callers_need_no_guard(self):
        assert charts.probability_chart(None) == ""
        assert charts.probability_chart([]) == ""
        assert charts.probability_chart([0.0, 0.0]) == ""
        assert charts.probability_bar(None) == ""

    def test_binary_bar_clamps(self):
        assert "100%" in charts.probability_bar(1.4)
        assert "0%" in charts.probability_bar(-0.2)


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


class TestPackRecordOrder:
    """The prompt's record budget must spend itself on the rows the report is
    required to cover, not on whatever ranked highest for attention."""

    def _pack(self):
        from interpreter import packs

        pack = packs.Pack(kind="current", path="x", manifest={}, files={})
        pack.attention_rows = [
            {"question_id": "loud", "category": "", "hazard_family": "climate",
             "attention_rank": 1},
            {"question_id": "covered", "category": "worsening",
             "hazard_family": "climate", "category_rank": 1, "attention_rank": 9},
        ]
        pack.records = {"loud": {"a": "x" * 4000}, "covered": {"a": "y" * 4000}}
        return pack

    def test_categorised_records_survive_a_tight_budget(self):
        from interpreter import packs

        _text, stats = packs.assemble_input_text(
            self._pack(), budget_tokens=1200, estimate_tokens=_tokens
        )
        assert stats["records_kept"] == ["covered"]
        assert stats["records_truncated"] == ["loud"]

    def test_a_dropped_covered_record_is_named_in_the_prompt(self):
        from interpreter import packs

        pack = self._pack()
        pack.attention_rows.append({
            "question_id": "also_covered", "category": "worsening",
            "hazard_family": "conflict", "category_rank": 1, "attention_rank": 5,
        })
        pack.records["also_covered"] = {"a": "z" * 4000}
        text, stats = packs.assemble_input_text(
            pack, budget_tokens=1200, estimate_tokens=_tokens
        )
        dropped = [q for q in stats["records_truncated"] if "covered" in q]
        assert dropped
        assert "Do not invent evidence" in text
        for qid in dropped:
            assert qid in text

    def test_pack_categories_reads_only_categorised_rows(self):
        from interpreter import packs

        got = packs.pack_categories(self._pack())
        assert got == {"covered": ("worsening", "climate")}


class TestTemplateFigureContract:
    """A template naming a figure key nothing produces is a broken report.

    This is the guard that was missing: interpreter/templates/v2/system.md
    told the model to write {{fig:ev_multiple}}, the selection module computed
    ev_multiple onto every attention row, and the figure map never exposed it.
    Every worsening entry's headline number rendered [figure unavailable] and
    the referential check failed the whole report.
    """

    def _template_keys(self) -> set[str]:
        import re
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "templates"
        keys: set[str] = set()
        for path in root.rglob("*.md"):
            keys |= set(re.findall(r"\{\{fig:([A-Za-z0-9_.-]+)\}\}", path.read_text()))
        return keys

    def test_template_figure_keys_are_all_produced(self):
        from interpreter import packs

        produced = (
            set(packs._ATTENTION_FIG_KEYS)
            | set(packs._RUN_SUMMARY_KEYS)
            | set(packs._PERFORMANCE_KEYS)
        )
        missing = sorted(self._template_keys() - produced)
        assert not missing, (
            f"templates instruct {{{{fig:...}}}} keys nothing produces: {missing}. "
            "Add them to packs._ATTENTION_FIG_KEYS (per-question), "
            "_RUN_SUMMARY_KEYS (run-level) or _PERFORMANCE_KEYS "
            "(scored), or stop naming them."
        )

    def test_attention_keys_exist_on_a_real_attention_row(self):
        """Every per-question figure key must be a column the pack carries."""
        from scripts.ai_bundle.build_current_run_bundle import ATTENTION_FIELDS
        from interpreter import packs

        # baserate_source and the rank columns are pack columns too; anything
        # here that ATTENTION_FIELDS does not carry can never resolve.
        missing = sorted(set(packs._ATTENTION_FIG_KEYS) - set(ATTENTION_FIELDS))
        assert not missing, f"figure keys absent from the attention index: {missing}"


class TestKindIsStated:
    def test_current_run_template_tells_the_model_its_kind(self):
        """The model wrote kind='current' on a combined run because the v2
        template dropped the {{KIND}} placeholder prompts.py substitutes."""
        from pathlib import Path

        text = (
            Path(__file__).resolve().parents[1]
            / "templates" / "v2" / "current_run.md"
        ).read_text()
        assert "{{KIND}}" in text

    def test_assembled_prompt_names_the_kind(self):
        from interpreter import prompts

        text = prompts.build_user_prompt(
            kind="combined", pack_text="PACK", version="v2"
        )
        assert "{{KIND}}" not in text
        assert "`combined`" in text
