# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Phase 4 validator tests: schema, referential, numeric guard, prose lint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

import interpreter.validate as validate_mod
from interpreter.validate import (
    check_numeric,
    check_prose,
    check_referential,
    find_bare_numerals,
    validate_interpretation,
)

QID = "ETH_ACE_FATALITIES_2026-08"
PER_QUESTION = {
    QID: {
        "js_vs_baserate": 0.3466,
        "eiv_nominal": 200000.0,
        "rc_level": 2,
        "rc_score": 0.24,
        "p_modal_bucket": 0.41,
        "modal_bucket_label": "25-<100",
    }
}
GLOBAL = {"skill_brier_spd": 0.5}


def _content(**overrides) -> dict:
    content = {
        "schema_version": "1",
        "template_version": "v1",
        "kind": "combined",
        "headline": "One risk stands out.",
        "attention": [{
            "rank": 1,
            "reason_code": "base_rate_deviation",
            "iso3": "ETH",
            "hazard_code": "ACE",
            "metric": "FATALITIES",
            "question_ids": [QID],
            "figure_refs": ["js_vs_baserate"],
            "why_it_stands_out": "The ensemble moved {{fig:js_vs_baserate}} from its base rate.",
            "lead_time_months": 2,
        }],
        "performance": {"plain_summary": "Skill was {{fig:skill_brier_spd}}."},
    }
    content.update(overrides)
    return content


class TestProseLint:
    def test_bare_numeral_caught(self):
        content = _content()
        content["attention"][0]["why_it_stands_out"] = (
            "About 40,000 people could be affected."
        )
        result = check_prose(content, per_question=PER_QUESTION, global_figures=GLOBAL)
        assert not result.passed
        assert any("bare numeral" in e for e in result.errors)

    def test_placeholders_and_calendar_whitelist_pass(self):
        content = _content()
        content["attention"][0]["why_it_stands_out"] = (
            "By August 2026 the ensemble sat {{fig:js_vs_baserate}} from its "
            "base rate; early 2027 looks calmer."
        )
        result = check_prose(content, per_question=PER_QUESTION, global_figures=GLOBAL)
        assert result.passed, result.errors

    def test_find_bare_numerals_unit(self):
        assert find_bare_numerals("no digits here") == []
        assert find_bare_numerals("in March 2026 and 2027") == []
        assert find_bare_numerals("with {{fig:eiv_nominal}} affected") == []
        assert find_bare_numerals("roughly 50k people") == ["50k"]
        assert find_bare_numerals("a 3-month lead")  # caught, capture cosmetic

    def test_lexicon_band_mismatch_caught(self):
        content = _content()
        # p_modal_bucket = 0.41 → "about as likely as not"; "likely" is wrong.
        content["attention"][0]["why_it_stands_out"] = (
            "The modal outcome is likely, at {{fig:p_modal_bucket}}."
        )
        result = check_prose(content, per_question=PER_QUESTION, global_figures=GLOBAL)
        assert not result.passed
        assert any("does not match" in e and "p_modal_bucket" in e for e in result.errors)

    def test_lexicon_band_match_passes(self):
        content = _content()
        content["attention"][0]["why_it_stands_out"] = (
            "The modal outcome is about as likely as not, at {{fig:p_modal_bucket}}."
        )
        result = check_prose(content, per_question=PER_QUESTION, global_figures=GLOBAL)
        assert result.passed, result.errors

    def test_ambiguous_sentence_skipped(self):
        content = _content()
        # Two distinct lexicon words in one sentence → deliberately skipped.
        content["attention"][0]["why_it_stands_out"] = (
            "An escalation is likely while a collapse is very unlikely, "
            "at {{fig:p_modal_bucket}}."
        )
        result = check_prose(content, per_question=PER_QUESTION, global_figures=GLOBAL)
        assert result.passed, result.errors

    def test_length_cap(self):
        content = _content()
        content["attention"][0]["why_it_stands_out"] = "long prose " * 300
        result = check_prose(content, per_question=PER_QUESTION, global_figures=GLOBAL)
        assert any("exceeds" in e for e in result.errors)


class TestReferential:
    def test_valid_content_passes(self):
        result = check_referential(
            _content(), valid_question_ids={QID},
            per_question=PER_QUESTION, global_figures=GLOBAL,
        )
        assert result.passed, result.errors

    def test_unknown_question_id_fails(self):
        content = _content()
        content["attention"][0]["question_ids"] = ["SOM_FL_PA_2026-08"]
        result = check_referential(
            content, valid_question_ids={QID},
            per_question=PER_QUESTION, global_figures=GLOBAL,
        )
        assert any("not in the pack" in e for e in result.errors)

    def test_unresolvable_figure_ref_and_placeholder_fail(self):
        content = _content()
        content["attention"][0]["figure_refs"] = ["made_up_key"]
        content["performance"]["plain_summary"] = "Skill was {{fig:missing_global}}."
        result = check_referential(
            content, valid_question_ids={QID},
            per_question=PER_QUESTION, global_figures=GLOBAL,
        )
        assert sum("does not resolve" in e for e in result.errors) == 2


class TestNumericGuard:
    @pytest.fixture
    def con(self, tmp_path: Path):
        c = duckdb.connect(str(tmp_path / "n.duckdb"))
        c.execute(
            "CREATE TABLE forecast_deviation (run_id TEXT, question_id TEXT, "
            "model_name TEXT, js_vs_baserate DOUBLE, log_ev_ratio DOUBLE, "
            "eiv_nominal DOUBLE, eiv_per_100k DOUBLE)"
        )
        c.execute(
            "INSERT INTO forecast_deviation VALUES "
            f"('fc_1', '{QID}', 'ensemble_mean_v2', 0.3466, 1.2, 200000.0, 180.0)"
        )
        c.execute(
            "CREATE TABLE questions (question_id TEXT, hs_run_id TEXT, "
            "iso3 TEXT, hazard_code TEXT)"
        )
        c.execute(f"INSERT INTO questions VALUES ('{QID}', 'hs_1', 'ETH', 'ACE')")
        c.execute(
            "CREATE TABLE hs_triage (run_id TEXT, iso3 TEXT, hazard_code TEXT, "
            "regime_change_level INTEGER, regime_change_score DOUBLE, "
            "triage_score DOUBLE)"
        )
        c.execute("INSERT INTO hs_triage VALUES ('hs_1', 'ETH', 'ACE', 2, 0.24, 0.7)")
        yield c
        c.close()

    def test_matching_pack_passes(self, con):
        content = _content()
        content["attention"][0]["figure_refs"] = ["js_vs_baserate", "rc_score"]
        content["attention"][0]["why_it_stands_out"] = (
            "Moved {{fig:js_vs_baserate}} with {{fig:eiv_nominal}} affected."
        )
        result = check_numeric(
            content, con=con, run_id="fc_1",
            per_question=PER_QUESTION, global_figures=GLOBAL,
        )
        assert result.passed, result.errors
        assert result.detail["n_checked"] >= 3

    def test_doctored_pack_value_caught(self, con):
        doctored = {QID: dict(PER_QUESTION[QID], js_vs_baserate=0.9)}
        result = check_numeric(
            _content(), con=con, run_id="fc_1",
            per_question=doctored, global_figures=GLOBAL,
        )
        assert not result.passed
        assert any("pack and the DB disagree" in e for e in result.errors)

    def test_missing_db_row_is_an_error(self, con):
        con.execute("DELETE FROM forecast_deviation")
        result = check_numeric(
            _content(), con=con, run_id="fc_1",
            per_question=PER_QUESTION, global_figures=GLOBAL,
        )
        assert any("no DB row" in e for e in result.errors)

    def test_no_connection_is_skipped_not_passed_silently(self):
        result = check_numeric(
            _content(), con=None, run_id=None,
            per_question=PER_QUESTION, global_figures=GLOBAL,
        )
        assert result.passed and result.skipped

    def test_no_source_tables_is_skipped(self, tmp_path):
        c = duckdb.connect(str(tmp_path / "empty.duckdb"))
        try:
            result = check_numeric(
                _content(), con=c, run_id=None,
                per_question=PER_QUESTION, global_figures=GLOBAL,
            )
        finally:
            c.close()
        assert result.passed and result.skipped


class TestValidateInterpretation:
    def _run(self, content, con=None):
        return validate_interpretation(
            content, kind="combined", valid_question_ids={QID},
            per_question=PER_QUESTION, global_figures=GLOBAL,
            con=con, run_id="fc_1",
        )

    def test_all_checks_pass_on_clean_content(self):
        report = self._run(_content())
        assert report.passed, json.dumps(report.as_dict(), indent=1)
        assert set(report.checks) == {"schema", "referential", "numeric", "prose"}

    def test_schema_failure_flows_through(self):
        content = _content()
        del content["headline"]
        report = self._run(content)
        assert not report.passed
        assert not report.checks["schema"].passed

    def test_one_failed_check_fails_the_report(self):
        content = _content()
        content["blind_spots"] = ["We miss 12000 cells."]
        report = self._run(content)
        assert not report.passed
        assert not report.checks["prose"].passed
        assert report.checks["referential"].passed

    def test_crashed_check_is_a_failed_check(self, monkeypatch):
        def _boom(*args, **kwargs):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(validate_mod, "check_prose", _boom)
        report = self._run(_content())
        assert not report.passed
        assert any("check crashed" in e for e in report.checks["prose"].errors)
