# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The v3 report: what a reader sees, and in what order.

Every section asserted here is GENERATED from the pack. The model contributes
one sentence to the decision calendar and nothing at all to the second
reader's section, the sector comparison or the dormant Part B, so the report's
account of itself cannot be wrong in the model's favour.
"""

from __future__ import annotations

from interpreter import gating, pdf
from interpreter.render import FigureResolver, render_markdown

QID = "SOM_DR_PHASE3PLUS_IN_NEED_2026-09"


def _content() -> dict:
    return {
        "schema_version": "1",
        "template_version": "v3",
        "kind": "combined",
        "headline": "Drought in Somalia is the month's largest departure.",
        "attention": [{
            "rank": 1,
            "reason_code": "large_impact_nominal",
            "iso3": "SOM",
            "hazard_code": "DR",
            "metric": "PHASE3PLUS_IN_NEED",
            "category": "worsening",
            "hazard_family": "climate",
            "question_ids": [QID],
            "why_it_stands_out": "The forecast sits well above its history.",
            "planning_sentence": "Plan against about {{fig:p50_peak}}.",
            "falsifier": "A good harvest through the season would show this "
                         "call to be wrong.",
            "decision_point": {
                "action": "Decide whether to open the pipeline early.",
                "deadline_month": "2026-08",
                "basis": "peak_horizon",
            },
        }],
        "changes_since_last_run": ["Somalia entered the list."],
        "blind_spots": ["The full blind spot list."],
    }


def _extras() -> dict:
    return {
        "issue_line": {
            "issued": "1 August 2026",
            "covering": "September 2026 to February 2027",
        },
        "blind_spots_short": [
            "People-affected ground truth is thin.",
            "Conflict inputs carry reporting bias.",
            "Blocked hazards produce no questions at all.",
        ],
        "selection_panel": {"title": "How these entries were chosen",
                            "counts_sentence": "3 forecasts were considered."},
        "question_table": [],
        "gates": {QID: gating.GATE_WORSENING},
        "sibyl_tags": {QID: "second opinion is more alarmed"},
        "persistence": {
            "entries": [{
                "question_id": QID,
                "country_name": "Somalia",
                "hazard_name": "Drought",
                "persistence": "flagged for 2 consecutive runs",
                "movement": {"movement": "risen", "n_overlapping_months": 5},
            }],
            "dropped": [{
                "country_name": "Chad", "hazard_name": "Flood",
                "reason": "it no longer passes the selection tests",
            }],
        },
        "decision_calendar": {"rows": [{
            "question_id": QID,
            "country": "Somalia",
            "hazard": "Drought",
            "deadline_month": "2026-08",
            "deadline_label": "August 2026",
            "peak_label": "November 2026",
            "action": "",
        }]},
        "sibyl": {
            "available": True,
            "caveat": "It is a second reader, not a tiebreaker.",
            "summary": {"n_covered": 1, "n_more_alarmed": 1,
                        "n_more_cautious": 0, "n_agrees": 0},
            "rows": [{
                "question_id": QID,
                "country_name": "Somalia",
                "hazard_name": "Drought",
                "fred_expected": 9_000_000.0,
                "sibyl_expected": 16_000_000.0,
                "tag": "second opinion is more alarmed",
                "unsettled": True,
                "novel_evidence": [
                    "The Baidoa reception centre logged new arrivals in July."
                ],
            }],
        },
        "sector": {
            "available": True,
            "caveat": "These three rankings measure different things.",
            "comparisons": [{
                "source": "inform_severity",
                "source_label": "INFORM Severity",
                "n_compared": 3,
                "more_worried": [{"iso3": "SOM", "country_name": "Somalia",
                                  "fred_rank": 1, "sector_rank": 8,
                                  "rank_gap": 7}],
                "less_worried": [],
            }],
        },
        "performance_outlook": {
            "dormant_sentences": [
                "No forecast window has closed yet.",
                "The first outcomes arrive on 2026-10-28.",
            ],
            "upcoming_resolutions": {"schedule": [{
                "resolution_date": "2026-10-28",
                "window_month_label": "September 2026",
                "n_question_horizons": 42,
            }]},
            "score_explanations": [
                {"term": "Brier score", "text": "How far the forecast sat."},
            ],
            "diary": [],
        },
        "watchlist": [],
    }


def _resolver() -> FigureResolver:
    return FigureResolver(per_question={QID: {"p50_peak": 1_200_000.0}})


class TestReportSections:
    def _md(self) -> str:
        return render_markdown(_content(), _resolver(), extras=_extras())

    def test_the_issue_line_and_the_stamp_sit_under_the_title(self):
        md = self._md()
        head = md.split("## ")[0]
        assert "Forecasts issued 1 August 2026" in head
        assert "covering September 2026 to February 2027" in head
        assert "Automated output, unreviewed" in head

    def test_what_we_cannot_see_is_on_page_one_and_in_full_at_the_back(self):
        md = self._md()
        front = md.index("## What this report cannot see")
        entries = md.index("## Potentially worsening situations")
        assert front < entries, "the blind spots must precede the entries"
        assert "### What we cannot see, in full" in md
        assert md.index("### What we cannot see, in full") > entries

    def test_the_calendar_comes_before_the_entries_and_is_dated(self):
        md = self._md()
        assert md.index("## The decision calendar") < md.index(
            "## Potentially worsening situations"
        )
        assert "| August 2026 | Somalia | Drought |" in md
        assert "Decide whether to open the pipeline early." in md

    def test_each_entry_carries_its_gate_persistence_and_second_opinion(self):
        md = self._md()
        line = [ln for ln in md.splitlines() if ln.startswith("*selected because")][0]
        assert gating.GATE_WORSENING in line
        assert "flagged for 2 consecutive runs" in line
        assert "second opinion is more alarmed" in line

    def test_the_entry_states_its_decision_and_its_deadline(self):
        md = self._md()
        assert "**Decision by August 2026:** Decide whether to open the pipeline" in md

    def test_the_second_reader_has_its_own_section_with_the_caveat(self):
        md = self._md()
        assert "## A second reader" in md
        assert "Baidoa" in md
        assert "did not settle it" in md
        assert "not a tiebreaker" in md
        assert "<svg" in md  # the dumbbell

    def test_the_sector_comparison_prints_both_directions_and_its_framing(self):
        md = self._md()
        assert "## Fred against the sector" in md
        assert "Compared with INFORM Severity" in md
        assert "a gap of 7 places" in md
        assert "measure different things" in md

    def test_what_changed_carries_movement_and_the_reason_for_a_drop(self):
        md = self._md()
        assert "## What changed since last month" in md
        assert "risen over 5 shared months" in md
        assert "it no longer passes the selection tests" in md

    def test_part_b_says_when_the_system_first_becomes_testable(self):
        md = self._md()
        assert "## How well did we do" in md
        assert "The first outcomes arrive on 2026-10-28." in md
        assert "| 2026-10-28 | September 2026 | 42 |" in md

    def test_no_bucket_table_in_the_body_and_the_score_copy_is_at_the_back(self):
        md = self._md()
        body, _, appendix = md.partition("## Appendix")
        assert "| How many | Chance |" not in body
        assert "### What the scores mean" in appendix

    def test_a_missing_section_simply_does_not_render(self):
        extras = _extras()
        extras["sibyl"] = {"available": False}
        extras["sector"] = {"available": False}
        md = render_markdown(_content(), _resolver(), extras=extras)
        assert "## A second reader" not in md
        assert "## Fred against the sector" not in md


class TestPdfLayout:
    def _row(self) -> dict:
        return {
            "interpretation_id": "int_1",
            "kind": "combined",
            "run_id": "fc_1",
            "hs_run_id": "hs_20260801T000000",
            "version": 3,
            "status": "ok",
            "created_at": "2026-08-10 12:00:00",
            "content_md": render_markdown(_content(), _resolver(), extras=_extras()),
        }

    def test_the_map_sits_after_the_title_not_before_it(self):
        html = pdf.build_report_html(
            self._row(), map_svg="<svg id='map'></svg>",
            map_captions=["Somalia"],
        )
        # Search for the DIV, not the bare word: the CSS block above the
        # body also mentions mapblock.
        assert html.index("<h1>") < html.index('<div class="mapblock">')
        # And the metadata line comes after the map, not between title and map.
        assert html.index('<div class="mapblock">') < html.index("class='meta'")

    def test_the_map_caption_reads_as_a_sentence(self):
        html = pdf.build_report_html(
            self._row(), map_svg="<svg id='map'></svg>",
            map_captions=["Somalia"],
        )
        assert "Darkest first" not in html
        assert "Warm shading means a country expects more people affected" in html
        # The caption must name the DIRECTION, not just the distance.
        assert "cool shading means fewer" in html
        assert "Moved most: Somalia." in html

    def test_a_contents_page_is_built_from_the_section_headings(self):
        html = pdf.build_report_html(self._row())
        assert 'class="toc"' in html
        assert "Contents" in html
        assert 'href="#sec-0"' in html
        # Every h2 the contents links to actually carries that id.
        assert 'id="sec-0"' in html

    def test_a_short_report_gets_no_contents_page(self):
        row = self._row()
        row["content_md"] = "# Title\n\n## Only section\n\nText."
        html = pdf.build_report_html(row)
        assert 'class="toc"' not in html

    def test_the_footer_clears_the_body_text(self):
        # The page number was colliding with the last line on the page.
        assert "margin: 16mm 18mm 24mm;" in pdf._CSS
        assert "padding-top: 6mm;" in pdf._CSS

    def test_list_markers_stay_with_their_content(self):
        # Indent with padding, not margin: with margin and zero padding the
        # marker is laid out outside the box and drifts off.
        assert "padding-left: 16pt" in pdf._CSS


class TestAQuietMonth:
    """Nothing cleared the tests. That is a finding, not a broken report."""

    def test_an_empty_attention_list_says_so_rather_than_leaving_a_gap(self):
        content = _content()
        content["attention"] = []
        md = render_markdown(content, _resolver(), extras=_extras())
        assert "## Nothing cleared both tests this month" in md
        assert "statement about the month" in md

    def test_the_schema_can_be_told_not_to_require_an_attention_list(self):
        from interpreter import schema

        content = _content()
        content["attention"] = []
        # Default: an empty list is a schema failure, which is right whenever
        # the pack DID categorise rows and the model dropped them.
        assert schema.validate_output(content, kind="combined")
        # Lowered by the runner when the pack itself categorised nothing.
        assert schema.validate_output(
            content, kind="combined", require_performance=False,
            require_attention=False,
        ) == []
