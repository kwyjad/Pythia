# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""What v4 put on the page, and what it took off.

The analytical fields are the reason a frontier model does this work rather
than a template. Everything numeric is still computed in SQL and rendered
deterministically; these fields are the part that cannot be.

The removals matter as much: the shape paragraph restated the two planning
figures beside it, and the report ran to eleven pages against a target of
seven.
"""

from __future__ import annotations

from interpreter import gating, validate
from interpreter.render import FigureResolver, render_markdown

QID = "SOM_DR_PHASE3PLUS_IN_NEED_2026-09"


def _content(**overrides) -> dict:
    entry = {
        "rank": 1,
        "reason_code": "large_impact_nominal",
        "iso3": "SOM",
        "hazard_code": "DR",
        "metric": "PHASE3PLUS_IN_NEED",
        "category": "worsening",
        "hazard_family": "climate",
        "question_ids": [QID],
        "why_it_stands_out": "The contingency figure has moved a long way.",
        "spd_shape": "Most of the weight sits in the middle band.",
        "what_the_model_was_reacting_to": "A government appeal and a poor season.",
        "falsifier": "Rain arriving on time through the season would show "
                     "this call to be wrong.",
        "tensions": [{
            "claim_a": "The government reported a sharp rise in need.",
            "claim_b": "The agency assessment recorded a modest fall.",
            "reconciliation": "They cover different districts over different "
                              "weeks; the government figure is the wider one.",
        }],
        "challenge": {
            "verdict": "weakened",
            "reasoning": "The objection that the coming rains carry flood "
                         "rather than drought risk is serious, and the entry "
                         "is written with less confidence for it.",
        },
        "second_opinion_explanation": "The second reader weighed the appeal "
                                      "more heavily than the season.",
        "decision_point": {
            "action": "Decide whether to open the pipeline early.",
            "deadline_month": "2026-08",
            "basis": "peak_horizon",
        },
    }
    entry.update(overrides.pop("entry", {}))
    content = {
        "schema_version": "1",
        "template_version": "v4",
        "kind": "combined",
        "headline": "Drought in Somalia is the month's largest departure.",
        "cross_cutting": "Four of the five entries are drought, and three "
                         "share one failed season.",
        "attention": [entry],
        "scan_forecast_disagreements": [{
            "question_ids": [QID],
            "explanation": "The scan read the appeal as a change of regime; "
                           "the ensemble sat on its anchor because the "
                           "caseload series has not turned.",
        }],
        "blind_spots": ["The full blind spot list."],
    }
    content.update(overrides)
    return content


def _extras(**overrides) -> dict:
    extras = {
        "issue_line": {"issued": "1 August 2026", "covering": "September 2026"},
        "gates": {QID: gating.GATE_WORSENING},
        "planning_sentences": {
            QID: "Plan against about 9.6 million people in November 2026. "
                 "Hold contingency for about 14 million people.",
        },
        "movement_notes": {
            QID: "the contingency figure has risen while the planning figure "
                 "has barely moved",
        },
        "question_table": [],
        "watchlist": [],
    }
    extras.update(overrides)
    return extras


def _md(content=None, extras=None) -> str:
    return render_markdown(
        content or _content(),
        FigureResolver(per_question={QID: {"p50_peak": 9_600_000.0}}),
        extras=extras if extras is not None else _extras(),
    )


class TestFrontMatter:
    def test_the_title_is_the_forecast_report(self):
        assert _md().startswith("# Fred's Monthly Forecast Report")

    def test_the_issue_line_and_the_stamp_still_sit_under_it(self):
        head = _md().split("## ")[0]
        assert "Forecasts issued 1 August 2026" in head
        assert "Automated output, unreviewed" in head


class TestTheAnalyticalFields:
    def test_a_reconciled_contradiction_is_printed_with_both_sides(self):
        md = _md()
        assert "Evidence that does not agree with itself" in md
        assert "The government reported a sharp rise in need." in md
        assert "Against that: The agency assessment recorded a modest fall." in md
        assert "different districts over different weeks" in md

    def test_the_challenge_carries_a_verdict_a_reader_can_see(self):
        md = _md()
        # Somalia is the live case: the challenge argued the coming rains
        # carry flood risk rather than drought, and the report gave the reader
        # no way to know whether it was taken seriously.
        assert "*The challenge to this reading* (the reading is weaker for it)" in md
        assert "flood" in md

    def test_the_falsifier_makes_the_call_testable_before_it_resolves(self):
        assert "What would show this call to be wrong:" in _md()

    def test_the_second_reader_difference_is_explained_not_labelled(self):
        assert "Why the second reader differs:" in _md()

    def test_the_scan_disagreement_gets_its_own_section(self):
        md = _md()
        assert "## Where the scan and the forecast disagree" in md
        assert "sat on its anchor" in md

    def test_the_cross_cutting_read_sits_near_the_front(self):
        md = _md()
        assert "## Reading the month as a whole" in md
        assert md.index("## Reading the month as a whole") < md.index(
            "## Potentially worsening situations"
        )

    def test_absent_analytical_fields_print_nothing_rather_than_a_hole(self):
        content = _content()
        entry = content["attention"][0]
        for field in ("tensions", "challenge", "second_opinion_explanation"):
            entry.pop(field)
        content.pop("cross_cutting")
        content.pop("scan_forecast_disagreements")
        md = _md(content)
        assert "Evidence that does not agree" not in md
        assert "The challenge to this reading" not in md
        assert "## Where the scan and the forecast disagree" not in md
        assert "## Reading the month as a whole" not in md


class TestWhatWasTakenOff:
    def test_the_shape_paragraph_is_gone(self):
        # It largely restated the two planning figures above it, and the full
        # distributions are in the appendix with their bands named in words.
        assert "The shape of the forecast" not in _md()

    def test_the_planning_sentence_is_the_generated_one(self):
        md = _md()
        assert "Plan against about 9.6 million people in November 2026." in md
        # The model's own version is used only when the pack has none.
        assert "+9,577,500" not in md

    def test_the_model_sentence_is_used_only_when_the_pack_has_none(self):
        content = _content()
        content["attention"][0]["planning_sentence"] = "Plan against {{fig:p50_peak}}."
        md = _md(content, _extras(planning_sentences={}))
        assert "**What to plan against:**" in md

    def test_a_count_carries_the_metric_s_own_unit(self):
        from interpreter.render import format_figure

        # "+5 people more deaths" was published. The model wrote "more
        # deaths"; the FORMATTER wrote "people", because it had no idea what
        # metric the figure belonged to and assumed one.
        assert format_figure("excess_nominal", 5, "FATALITIES") == "5 more deaths"
        assert format_figure("excess_nominal", 5, "PA") == "5 more people"
        assert format_figure("p50_peak", 120, "FATALITIES") == "120 deaths"
        # And it rounds by magnitude rather than printing an interpolation.
        assert format_figure("excess_nominal", 9_577_500, "PHASE3PLUS_IN_NEED") == (
            "9.6 million more people"
        )
        # A fall reads as a fall, not as a minus sign a reader has to parse.
        assert format_figure("material_movement", -4_200, "PA") == "4,200 fewer people"

    def test_the_metric_reaches_the_formatter_through_the_figure_map(self):
        from interpreter.render import METRIC_KEY, FigureResolver

        resolver = FigureResolver(
            per_question={QID: {METRIC_KEY: "FATALITIES", "excess_nominal": 5}}
        )
        assert resolver.resolve_text(
            "about {{fig:excess_nominal}}", [QID]
        ) == "about 5 more deaths"

    def test_the_entry_says_which_figure_moved(self):
        md = _md()
        assert "*What moved:* The contingency figure has risen" in md


class TestTheWatchlistIsCapped:
    def test_the_overflow_is_named_and_left_to_the_table(self):
        extras = _extras(
            watchlist=[
                {"country": f"Country {i}", "hazard": "Flood",
                 "metric": "people affected", "movement": "about 900 people"}
                for i in range(10)
            ],
            watchlist_total=31,
        )
        md = _md(extras=extras)
        assert "31 qualified this month; the largest 10 are shown" in md
        # Each line carries a figure: a list of names with no number attached
        # tells a reader nothing about which of them to look at.
        assert "(about 900 people)" in md


class TestGenericPhraseLint:
    def test_a_phrase_used_across_the_report_is_flagged(self):
        content = _content()
        content["attention"][0]["operational_challenges"] = [
            "Access constraints will slow any response.",
        ]
        content["headline"] = "Access constraints define the month."
        content["cross_cutting"] = "Access constraints are the common thread."
        result = validate.check_generic_phrases(content, max_entries_per_phrase=2)
        assert result.detail["counts"]["access constraints"] == 3
        assert "access constraints" in result.detail["flagged"]
        # It REPORTS: a proxy for writing quality must never stop a report
        # being published.
        assert result.passed is True

    def test_process_vocabulary_counts_too(self):
        content = _content()
        content["attention"][0]["why_it_stands_out"] = (
            "The left tail of the distribution carries the weight."
        )
        counts = validate.count_generic_phrases(content)
        assert counts.get("left tail") == 1
        assert counts.get("the distribution") == 1

    def test_clean_prose_flags_nothing(self):
        result = validate.check_generic_phrases(
            _content(), max_entries_per_phrase=3
        )
        assert result.detail["flagged"] == {}

    def test_a_phrase_used_twice_in_one_field_counts_once(self):
        # A writer repeating themselves in a paragraph is a different fault
        # from a writer with one sentence for every crisis.
        content = _content()
        content["headline"] = "Funding gap here, funding gap there."
        counts = validate.count_generic_phrases(content)
        assert counts["funding gap"] == 1


class TestTheAnalyticalFieldsAreLinted:
    def test_a_bare_numeral_in_a_tension_fails_like_any_other_prose(self):
        content = _content()
        content["attention"][0]["tensions"][0]["claim_a"] = (
            "The government reported 450,000 people in need."
        )
        result = validate.check_prose(
            content, per_question={}, global_figures={}
        )
        assert not result.passed
        assert any("tensions[0].claim_a" in e for e in result.errors)

    def test_a_code_in_a_falsifier_fails_the_style_check(self):
        content = _content()
        content["attention"][0]["falsifier"] = "SOM records a good season."
        result = validate.check_style(content)
        assert not result.passed
        assert any("falsifier" in e for e in result.errors)
