# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Renderer placeholder resolution + interpretations storage."""

from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")

from interpreter import store
from interpreter.render import UNAVAILABLE, FigureResolver, format_figure, render_markdown

QID = "ETH_ACE_FATALITIES_2026-08"


class TestFormatFigure:
    def test_key_aware_formatting(self):
        assert format_figure("eiv_nominal", 200000) == "200,000"
        assert format_figure("rc_level", 2) == "L2"
        assert format_figure("rc_score", 0.24) == "24%"
        assert format_figure("p_modal_bucket", 0.41) == "41%"
        assert format_figure("skill_brier_spd", 0.35) == "+35%"
        assert format_figure("skill_brier_spd", -0.1) == "-10%"
        assert format_figure("js_vs_baserate", 0.3466) == "50% of maximum"
        assert format_figure("log_ev_ratio", 0.693).startswith("2.0x")
        assert format_figure("baserate_source", "acled:6m") == "acled:6m"
        assert format_figure("anything", None) == UNAVAILABLE


class TestFigureResolver:
    def test_resolution_order_and_misses(self):
        r = FigureResolver(
            per_question={QID: {"js_vs_baserate": 0.3466}},
            global_figures={"skill_brier_spd": 0.2},
        )
        text = r.resolve_text(
            "moved {{fig:js_vs_baserate}}, skill {{fig:skill_brier_spd}}, "
            "and {{fig:nonexistent}}",
            [QID],
        )
        assert "50% of maximum" in text
        assert "+20%" in text
        assert UNAVAILABLE in text
        assert r.misses == ["nonexistent"]


def _content() -> dict:
    return {
        "schema_version": "1",
        "template_version": "v1",
        "kind": "combined",
        "headline": "One risk stands out.",
        "attention": [
            {
                "rank": 1,
                "reason_code": "base_rate_deviation",
                "iso3": "ETH",
                "hazard_code": "ACE",
                "metric": "FATALITIES",
                "question_ids": [QID],
                "why_it_stands_out": "The forecast moved {{fig:js_vs_baserate}} away.",
                "impacts": ["Impact prose."],
                "lead_time_months": 2,
            }
        ],
        "performance": {"plain_summary": "Skill was {{fig:skill_brier_spd}}."},
        "blind_spots": ["A blind spot."],
    }


class TestRenderMarkdown:
    def test_placeholders_resolved_and_sections_present(self):
        resolver = FigureResolver(
            per_question={QID: {"js_vs_baserate": 0.3466}},
            global_figures={"skill_brier_spd": 0.5},
        )
        md = render_markdown(_content(), resolver, provenance={"run_id": "fc_1"})
        assert "{{fig:" not in md  # every placeholder substituted
        assert "50% of maximum" in md
        assert "+50%" in md
        assert "## What to watch this month" in md
        assert "## How well did we do" in md
        assert "## What we cannot see" in md
        assert "### Probability words" in md  # appendix lexicon
        assert f"`{QID}`" in md  # claims carry their question ids
        assert "run_id: `fc_1`" in md

    def test_unresolved_figures_are_visible_never_silent(self):
        content = _content()
        content["attention"][0]["why_it_stands_out"] = "moved {{fig:mystery_key}}"
        resolver = FigureResolver()
        md = render_markdown(content, resolver)
        assert UNAVAILABLE in md
        assert "could not be" in md  # the appendix count line


class TestStore:
    def _con(self, tmp_path):
        return duckdb.connect(str(tmp_path / "t.duckdb"))

    def _save(self, con, kind="combined", run_id="fc_1", scored_run_id=None,
              status="ok", is_test=None):
        return store.save_interpretation(
            con, kind=kind, run_id=run_id, hs_run_id="hs_1",
            scored_run_id=scored_run_id, template_version="v1",
            model_name="claude-opus-5", thinking_level="high",
            prompt_hash="abc", pack_hash="def",
            content={"kind": kind}, content_md="# report",
            figures={"global": {"skill_brier_spd": 0.5}},
            status=status, validation={}, cost_usd=2.0,
            input_tokens=100, output_tokens=50, is_test=is_test,
        )

    def test_version_increments_per_kind_and_run(self, tmp_path):
        con = self._con(tmp_path)
        assert self._save(con)[1] == 1
        assert self._save(con)[1] == 2  # same (kind, run) → v2
        assert self._save(con, run_id="fc_2")[1] == 1  # other run → v1
        assert self._save(con, kind="scored", run_id=None, scored_run_id="2026-08")[1] == 1
        con.close()

    def test_existing_ok_version_skip_signal(self, tmp_path):
        con = self._con(tmp_path)
        assert store.existing_ok_version(con, "combined", "fc_1", None) is None
        self._save(con, status="failed_validation")
        # A failed row must NOT satisfy the skip check.
        assert store.existing_ok_version(con, "combined", "fc_1", None) is None
        self._save(con, status="ok")
        assert store.existing_ok_version(con, "combined", "fc_1", None) == 2
        con.close()

    def test_is_test_inherits_from_env(self, tmp_path, monkeypatch):
        con = self._con(tmp_path)
        monkeypatch.setenv("PYTHIA_TEST_MODE", "1")
        self._save(con)
        monkeypatch.delenv("PYTHIA_TEST_MODE")
        self._save(con)
        rows = con.execute(
            "SELECT version, is_test FROM interpretations ORDER BY version"
        ).fetchall()
        assert rows[0][1] is True and rows[1][1] is False
        con.close()

    def test_latest_scored_filters_test_rows(self, tmp_path, monkeypatch):
        con = self._con(tmp_path)
        monkeypatch.delenv("PYTHIA_TEST_MODE", raising=False)
        self._save(con, kind="scored", run_id=None, scored_run_id="2026-07")
        monkeypatch.setenv("PYTHIA_TEST_MODE", "1")
        self._save(con, kind="scored", run_id=None, scored_run_id="2026-08-test")
        monkeypatch.delenv("PYTHIA_TEST_MODE")

        latest = store.latest_scored(con)
        assert latest is not None
        assert latest["scored_run_id"] == "2026-07"  # the test row is excluded
        latest_incl = store.latest_scored(con, include_test=True)
        assert latest_incl["scored_run_id"] == "2026-08-test"
        assert latest["figures"]["global"]["skill_brier_spd"] == 0.5
        con.close()

    def test_pre_interpreter_db_returns_none(self, tmp_path):
        con = self._con(tmp_path)
        assert store.latest_scored(con) is None
        con.close()
