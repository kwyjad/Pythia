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


class TestIdentityRepair:
    """The pack owns a question's country, hazard and metric.

    The August report printed "Mali, fatalities: deaths" because the model
    copied FATALITIES into hazard_code. Copying is transcription, not
    judgement, so it is repaired rather than published.
    """

    def test_miscopied_hazard_code_is_corrected(self):
        from interpreter.run import _repair_entry_identity

        content = {"attention": [{
            "question_ids": ["MLI_ACE_FATALITIES_2026-09"],
            "iso3": "MLI", "hazard_code": "FATALITIES", "metric": "FATALITIES",
            "category": "worsening", "hazard_family": "conflict",
        }]}
        identity = {"MLI_ACE_FATALITIES_2026-09": {
            "iso3": "MLI", "hazard_code": "ACE", "metric": "FATALITIES",
        }}
        assert _repair_entry_identity(content, identity) == 1
        assert content["attention"][0]["hazard_code"] == "ACE"

    def test_judgement_fields_are_never_touched(self):
        from interpreter.run import _repair_entry_identity

        content = {"attention": [{
            "question_ids": ["MLI_ACE_FATALITIES_2026-09"],
            "iso3": "MLI", "hazard_code": "ACE", "metric": "FATALITIES",
            "category": "worsening", "hazard_family": "conflict",
        }]}
        identity = {"MLI_ACE_FATALITIES_2026-09": {
            "iso3": "MLI", "hazard_code": "ACE", "metric": "FATALITIES",
        }}
        assert _repair_entry_identity(content, identity) == 0
        # The validator, not this repair, owns category and hazard_family.
        assert content["attention"][0]["category"] == "worsening"

    def test_an_unknown_question_is_left_alone(self):
        from interpreter.run import _repair_entry_identity

        content = {"attention": [{
            "question_ids": ["NOT_IN_PACK"], "iso3": "ZZZ",
            "hazard_code": "XX", "metric": "PA",
        }]}
        assert _repair_entry_identity(content, {"other": {}}) == 0
        assert content["attention"][0]["iso3"] == "ZZZ"


class TestEvMultipleComposes:
    """format_figure returns a noun phrase because the model writes the value
    inside a sentence of its own. Returning a clause published "about about
    14.1 times the usual level the usual number of people affected"."""

    def test_it_reads_inside_the_prompt_own_example(self):
        from interpreter.render import format_figure

        sentence = (
            "the system expects about "
            + format_figure("ev_multiple", 14.11)
            + " the usual number of people affected"
        )
        assert sentence == (
            "the system expects about 14.1 times the usual number of people affected"
        )
        assert "the usual level the usual" not in sentence

    def test_below_one_still_reads(self):
        from interpreter.render import format_figure

        assert format_figure("ev_multiple", 0.5) == "one half of"


class TestAggregatePreference:
    """One aggregate speaks for a question, declared once.

    It previously lived in three copies plus an implicit fourth: the printed
    map had no preference at all and took the largest divergence across every
    model row, Sibyl included, so it could disagree with the dashboard's map.
    """

    def test_the_mean_leads(self):
        assert names.AGGREGATE_PREFERENCE[0] == "ensemble_mean_v2"

    def test_every_consumer_reads_the_same_order(self):
        from interpreter import packs
        from scripts.ai_bundle.build_current_run_bundle import AGGREGATE_PREFERENCE
        import pythia.api.routes.interpreter as api_routes

        assert tuple(AGGREGATE_PREFERENCE) == tuple(names.AGGREGATE_PREFERENCE)
        assert tuple(api_routes._MODEL_PREFERENCE) == tuple(names.AGGREGATE_PREFERENCE)
        # packs picks its grid by the same order.
        record = {"ensemble": {
            "ensemble_bayesmc_v2": {"months": {"1": {"1": 1.0}}},
            "ensemble_mean_v2": {"months": {"1": {"2": 1.0}}},
        }}
        grid = packs._preferred_grid(record)
        assert grid == {"months": {"1": {"2": 1.0}}}

    def test_the_printed_map_picks_one_row_per_question(self, tmp_path):
        import duckdb
        from interpreter import mapviz

        con = duckdb.connect(str(tmp_path / "m.duckdb"))
        con.execute(
            "CREATE TABLE forecast_deviation (run_id TEXT, question_id TEXT, "
            "model_name TEXT, iso3 TEXT, js_vs_baserate DOUBLE, "
            "is_test BOOLEAN DEFAULT FALSE)"
        )
        con.executemany(
            "INSERT INTO forecast_deviation VALUES (?,?,?,?,?,FALSE)",
            [
                ("fc_1", "q1", "ensemble_mean_v2", "ETH", 0.20),
                ("fc_1", "q1", "ensemble_bayesmc_v2", "ETH", 0.60),
                ("fc_1", "q1", "sibyl", "ETH", 0.69),
            ],
        )
        values = mapviz.values_from_deviation(con, "fc_1", include_test=False)
        # The mean's 0.20, not bayesmc's 0.60 and not Sibyl's 0.69.
        import math as _m
        assert values["ETH"] == pytest.approx(0.20 / _m.log(2.0))
        con.close()


class TestGating:
    """The two-key gate and the excess ranking that replaced the ratio."""

    def test_exceedance_interpolates_within_the_straddled_bucket(self):
        from interpreter import gating

        # PA edges: 0, 1, 10k, 50k, 250k, 500k, inf
        probs = [0.5, 0.2, 0.15, 0.1, 0.03, 0.02]
        assert gating.exceedance(probs, "PA", 50_000) == pytest.approx(0.15)
        assert gating.exceedance(probs, "PA", 10_000) == pytest.approx(0.30)
        # 30k sits mid-way through the 10k-50k band: half of its 0.15.
        assert gating.exceedance(probs, "PA", 30_000) == pytest.approx(0.225)

    def test_quantiles_read_as_planning_figures(self):
        from interpreter import gating

        probs = [0.5, 0.2, 0.15, 0.1, 0.03, 0.02]
        assert gating.quantile(probs, "PA", 0.5) == pytest.approx(1.0)
        assert gating.quantile(probs, "PA", 0.9) == pytest.approx(150_000, rel=1e-6)
        assert gating.p_zero(probs) == pytest.approx(0.5)

    def test_the_population_share_lets_small_states_in(self):
        from interpreter import gating

        # A flat 50,000 floor would mean Vanuatu never appears in a cyclone
        # report, which would be absurd.
        vut = gating.materiality_threshold(
            "PA", 320_000, absolute=50_000, population_share=0.01
        )
        eth = gating.materiality_threshold(
            "PA", 120_000_000, absolute=50_000, population_share=0.01
        )
        assert vut == pytest.approx(3_200)
        assert eth == pytest.approx(50_000)

    def test_a_metric_without_a_threshold_gates_on_probability_alone(self):
        from interpreter import gating

        assert gating.materiality_threshold(
            "EVENT_OCCURRENCE", 1e6, absolute=None, population_share=None
        ) is None

    def test_any_horizon_clears_and_the_month_is_carried(self):
        from interpreter import gating

        cleared, horizon = gating.clears_materiality([0.05, 0.10, 0.40, 0.02], 0.25)
        assert cleared and horizon == 3
        cleared, horizon = gating.clears_materiality([0.05, 0.10], 0.25)
        assert not cleared and horizon == 2  # the best month, still reported

    def test_below_base_rate_is_never_called_worsening(self):
        from interpreter import gating

        # Unusual and material, but expecting FEWER people than usual. Uganda
        # did exactly this on the August run: excess -2,069,829, both keys
        # passed. "Worsening" is a claim about direction.
        rows = [{
            "question_id": "UGA", "js_vs_baserate": 0.9,
            "exceedances": [0.9], "excess_nominal": -2_069_829,
            "baserate_n_obs": 36,
        }]
        gating.gate_rows(rows, unusual_percentile=0.0, min_probability=0.25,
                         thin_min_obs=12)
        assert rows[0]["gate"] == gating.GATE_MAJOR
        assert rows[0]["gate"] != gating.GATE_WORSENING

    def test_unusual_but_immaterial_goes_to_the_watchlist_not_the_bin(self):
        from interpreter import gating

        rows = [{
            "question_id": "rare_storm", "js_vs_baserate": 0.9,
            "exceedances": [0.06], "excess_nominal": 400.0,
            "baserate_n_obs": 36,
        }]
        gating.gate_rows(rows, unusual_percentile=0.0, min_probability=0.25,
                         thin_min_obs=12)
        assert rows[0]["gate"] == gating.GATE_WATCHLIST

    def test_thin_anchors_are_demoted_not_dropped(self):
        from interpreter import gating

        thin = {"question_id": "thin", "excess_nominal": 999_999,
                "baserate_thin": True}
        clear = {"question_id": "clear", "excess_nominal": 10,
                 "baserate_thin": False}
        ordered = sorted([thin, clear], key=gating.rank_key)
        assert [r["question_id"] for r in ordered] == ["clear", "thin"]

    def test_counts_are_returned_so_the_panel_and_table_agree(self):
        from interpreter import gating

        rows = [
            {"question_id": "a", "js_vs_baserate": 0.9, "exceedances": [0.9],
             "excess_nominal": 100.0, "baserate_n_obs": 36},
            {"question_id": "b", "js_vs_baserate": 0.01, "exceedances": [0.9],
             "excess_nominal": 100.0, "baserate_n_obs": 36},
            {"question_id": "c", "js_vs_baserate": 0.9, "exceedances": [0.01],
             "excess_nominal": 1.0, "baserate_n_obs": 2},
        ]
        counts = gating.gate_rows(rows, unusual_percentile=0.5,
                                  min_probability=0.25, thin_min_obs=12)
        assert counts["considered"] == 3
        assert counts["both"] + counts["major"] + counts["watchlist"] <= 3
        assert counts["thin"] == 1

    def test_percentile_is_pinned_by_this_repo(self):
        from interpreter import gating

        assert gating.percentile([1, 2, 3, 4, 5], 0.5) == pytest.approx(3.0)
        assert gating.percentile([1, 2, 3, 4], 0.5) == pytest.approx(2.5)
        assert gating.percentile([], 0.5) is None


class TestPanels:
    """The report's account of itself is generated, never composed.

    A model asked to describe the selection rules in its own words will
    eventually describe them wrongly, and a reader has no way to check. So
    the panel, the per-entry tags and the appendix table all read the same
    counts, and these tests hold them to that.
    """

    def test_bucket_labels_are_readable_and_carry_their_unit(self):
        from interpreter import panels

        assert panels.humanise_bucket_label("0", "PA", 0) == "no people recorded"
        assert panels.humanise_bucket_label("1-<10k", "PA", 1) == "1 to 9,999 people"
        assert panels.humanise_bucket_label(">=500k", "PA", 5) == "500,000 people or more"
        # A fatality band counts deaths, not people affected. Calling the two
        # by the same noun is how a reader reads a hundred deaths as a
        # hundred people needing blankets.
        assert "deaths" in panels.humanise_bucket_label("0", "FATALITIES", 0)

    def test_panel_states_the_counts_the_gate_produced(self):
        from interpreter import panels

        panel = panels.selection_panel({
            "considered": 91, "both": 3, "major": 5, "watchlist": 7, "thin": 58,
        })
        assert panel["counts"]["considered"] == 91
        assert panel["counts"]["cleared_both"] == 3
        assert panel["counts"]["thin_anchor"] == 58
        for token in ("91", "3", "5", "7", "58"):
            assert token in panel["counts_sentence"]
        # It has to state the ordering rule, because that is the single thing
        # about this report a reader is most likely to get wrong.
        assert "ADDITIONAL" in panel["ordering"] or "additional" in panel["ordering"]

    def test_thresholds_never_hardcode_a_number_in_prose(self):
        from interpreter import config, panels

        panel = panels.selection_panel({"considered": 0})
        joined = " ".join(panel["thresholds"])
        assert f"{config.threshold_for_metric('PA'):,.0f}" in joined
        assert "whichever is lower" in joined

    def test_question_table_lists_everything_considered(self):
        from interpreter import gating, panels

        rows = [
            {"question_id": "a", "iso3": "SOM", "hazard_code": "DR",
             "metric": "PHASE3PLUS_IN_NEED", "excess_nominal": 2_900_000.0,
             "exceedances": [0.8, 0.9], "gate": gating.GATE_WORSENING},
            {"question_id": "b", "iso3": "IDN", "hazard_code": "TC",
             "metric": "PA", "excess_nominal": 3_524.0,
             "exceedances": [0.02], "gate": None, "baserate_thin": True},
        ]
        table = panels.question_table(rows)
        assert len(table) == 2
        assert table[0]["country"] == "Somalia"
        assert table[0]["excess"] == "+2,900,000"
        assert table[0]["chance"] == "90%"
        # Not selected is a verdict the reader is owed, not a blank.
        assert table[1]["gate"] == "not selected"
        assert table[1]["anchor"] == "thin"

    def test_planning_figures_report_the_peak_month_by_name(self):
        from interpreter import panels

        flat = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
        big = [0.1, 0.2, 0.3, 0.2, 0.1, 0.1]
        figs = panels.planning_figures(
            [flat, big, flat], "PA",
            ["September 2026", "October 2026", "November 2026"],
        )
        assert figs["peak_horizon"] == 2
        assert figs["peak_month"] == "October 2026"
        assert figs["p50_peak"] is not None
        assert figs["p90_peak"] >= figs["p50_peak"]
        assert figs["p_zero_peak"] == pytest.approx(0.1)

    def test_no_distributions_means_no_planning_figures_not_zeros(self):
        from interpreter import panels

        assert panels.planning_figures([], "PA") == {}


class TestReportSelfAccount:
    """The rendered report has to carry the panel, the tags and the table."""

    def _resolver(self):
        from interpreter import render

        return render.FigureResolver({}, {}, spd_by_question={})

    def _content(self):
        return {
            "kind": "current",
            "headline": "A headline.",
            "attention": [{
                "rank": 1, "iso3": "SOM", "hazard_code": "DR",
                "metric": "PHASE3PLUS_IN_NEED",
                "category": "worsening", "hazard_family": "climate",
                "question_ids": ["SOM_DR_PHASE3PLUS_IN_NEED_2026-09"],
                "why_it_stands_out": "Because it does.",
                "planning_sentence": "Plan against a large caseload.",
            }],
        }

    def test_panel_and_table_and_gate_tag_all_render(self):
        from interpreter import gating, panels, render

        extras = {
            "selection_panel": panels.selection_panel(
                {"considered": 91, "both": 3, "major": 5,
                 "watchlist": 7, "thin": 58}
            ),
            "question_table": panels.question_table([{
                "question_id": "SOM_DR_PHASE3PLUS_IN_NEED_2026-09",
                "iso3": "SOM", "hazard_code": "DR",
                "metric": "PHASE3PLUS_IN_NEED",
                "excess_nominal": 2_900_000.0, "exceedances": [0.9],
                "gate": gating.GATE_WORSENING,
            }]),
            "gates": {
                "SOM_DR_PHASE3PLUS_IN_NEED_2026-09": gating.GATE_WORSENING
            },
            "watchlist": [{"country": "Indonesia", "hazard": "tropical cyclone",
                           "metric": "people affected"}],
        }
        md = render.render_markdown(
            self._content(), self._resolver(), extras=extras
        )
        assert "How these entries were chosen" in md
        assert "91 forecasts were considered" in md
        assert f"Selected because: {gating.GATE_WORSENING}" in md
        assert "Every question considered" in md
        assert "Somalia" in md
        assert "Watchlist" in md
        assert "Indonesia" in md

    def test_report_still_renders_without_extras(self):
        from interpreter import render

        md = render.render_markdown(self._content(), self._resolver())
        assert "How these entries were chosen" not in md
        assert "A headline." in md

    def test_planning_sentence_leads_and_the_bucket_chart_left_the_entry(self):
        from interpreter import render

        md = render.render_markdown(self._content(), self._resolver())
        assert "What to plan against:" in md
        # The chart moved to the appendix. Inside the entry it crowded out
        # the two figures a planner can act on.
        body = md.split("## Appendix")[0]
        assert "<svg" not in body

    def test_bucket_tables_move_to_the_appendix_with_readable_bands(self):
        from interpreter import render

        resolver = render.FigureResolver({}, {}, spd_by_question={
            "SOM_DR_PHASE3PLUS_IN_NEED_2026-09": {
                "spd": [0.1, 0.2, 0.3, 0.2, 0.1, 0.1],
                "bucket_labels": None, "binary": False,
            }
        })
        md = render.render_markdown(self._content(), resolver, extras={})
        appendix = md.split("## Appendix")[1]
        assert "The full forecasts" in appendix
        assert "1 to 99,999 people" in appendix
