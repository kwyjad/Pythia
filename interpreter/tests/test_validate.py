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
        "log_ev_ratio": 1.2,
        # exp(1.2) — the readable form the report leads with.
        "ev_multiple": 3.3201169227365472,
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
            "category": "worsening",
            "hazard_family": "conflict",
            "question_ids": [QID],
            "figure_refs": ["js_vs_baserate"],
            "why_it_stands_out": "The ensemble moved {{fig:js_vs_baserate}} from its base rate.",
            "decision_point": {"action": "Decide whether to preposition stock.", "deadline_month": "2026-09", "basis": "peak_horizon"},
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


class TestProperNounGuard:
    """The August report named "Typhoon Maysak". The pack does not.

    The existing guards protect numbers. The report's own footer tells the
    reader that every figure in it is machine-derived, which invites them to
    trust the names too, and nothing checked those.
    """

    EVIDENCE = (
        "gdacs recorded an orange alert for tropical cyclone activity "
        "affecting viet nam. the national committee issued a warning for "
        "quang ninh province. reliefweb carried an ocha situation report."
    )

    def _content_with(self, prose: str) -> dict:
        content = _content()
        content["attention"][0]["why_it_stands_out"] = prose
        return content

    def test_a_planted_storm_name_is_flagged(self):
        content = self._content_with(
            "The system reacted to Typhoon Maysak approaching the coast."
        )
        result = validate_mod.check_proper_nouns(
            content, evidence_text=self.EVIDENCE
        )
        violations = result.detail["violations"]
        assert any("Maysak" in v for v in violations), violations

    def test_flagging_does_not_fail_the_report(self):
        # It reports; it does not block. A validator that failed a whole
        # report on an unusual spelling would be switched off within months.
        content = self._content_with("Typhoon Maysak is approaching.")
        result = validate_mod.check_proper_nouns(
            content, evidence_text=self.EVIDENCE
        )
        assert result.passed is True

    def test_a_name_the_pack_carries_is_left_alone(self):
        content = self._content_with(
            "The warning covers Quang Ninh province on the coast."
        )
        result = validate_mod.check_proper_nouns(
            content, evidence_text=self.EVIDENCE
        )
        assert result.detail["violations"] == []

    def test_countries_and_hazards_are_never_flagged(self):
        # The report is REQUIRED to print these, and they come from the
        # system's own tables rather than from the model.
        content = self._content_with(
            "In Somalia the drought outlook is the reason. Ethiopia is "
            "the comparison."
        )
        result = validate_mod.check_proper_nouns(
            content, evidence_text=self.EVIDENCE
        )
        assert result.detail["violations"] == []

    def test_a_possessive_country_name_is_the_same_name(self):
        # "Afghanistan's drought" flagged on the guard's first live run and
        # cost a correction pass to remove a country name the report is
        # required to print. That is the false positive that gets a checker
        # switched off.
        content = self._content_with(
            "Afghanistan's outlook is the reason, and Somalia's is worse."
        )
        result = validate_mod.check_proper_nouns(
            content, evidence_text=self.EVIDENCE
        )
        assert result.detail["violations"] == []

    def test_a_possessive_invented_name_is_still_flagged(self):
        content = self._content_with("Typhoon Maysak's track shifted west.")
        result = validate_mod.check_proper_nouns(
            content, evidence_text=self.EVIDENCE
        )
        assert any("Maysak" in v for v in result.detail["violations"])

    def test_a_sentence_initial_capital_is_grammar_not_a_name(self):
        content = self._content_with("Rainfall drove the change.")
        result = validate_mod.check_proper_nouns(
            content, evidence_text=self.EVIDENCE
        )
        assert result.detail["violations"] == []

    def test_no_evidence_text_skips_rather_than_flagging_everything(self):
        content = self._content_with("Typhoon Maysak is approaching.")
        result = validate_mod.check_proper_nouns(content, evidence_text=None)
        assert result.skipped is True
        assert result.detail.get("violations") is None


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
        assert set(report.checks) == {
            "schema", "referential", "numeric", "prose", "style", "categories",
            "proper_nouns",
        }

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


class TestStyle:
    """The house-style rules that can be checked mechanically.

    Deliberately narrow: rhythm and variety are judged by a reader, not by a
    lint. What is here is unambiguous, and each rule was asked for because a
    reader tripped over it.
    """

    def test_em_dash_fails(self):
        content = _content()
        content["headline"] = "One country stands out — and it is not the usual one."
        result = validate_mod.check_style(content)
        assert not result.passed
        assert any("full stop" in e for e in result.errors)

    def test_en_dash_fails_too(self):
        content = _content(headline="A rise – of a kind we have not seen.")
        assert not validate_mod.check_style(content).passed

    def test_banned_words(self):
        for word in ("delve", "tapestry", "leverage", "pivotal", "landscape"):
            content = _content(headline=f"We {word} into the record.")
            result = validate_mod.check_style(content)
            assert not result.passed, word
            assert any(word in e for e in result.errors), word

    def test_both_sidesing_and_not_x_but_y(self):
        assert not validate_mod.check_style(
            _content(headline="It is bad. On the other hand, it could be worse.")
        ).passed
        assert not validate_mod.check_style(
            _content(headline="This is not a surprise, but a warning.")
        ).passed

    def test_codes_in_prose_fail(self):
        assert not validate_mod.check_style(
            _content(headline="ETH is the country to watch.")
        ).passed
        assert not validate_mod.check_style(
            _content(headline="The DR/PA forecast is high.")
        ).passed
        assert not validate_mod.check_style(
            _content(headline="EVENT_OCCURRENCE is elevated.")
        ).passed

    def test_known_acronyms_and_plain_prose_pass(self):
        content = _content(
            headline="Ethiopia stands out; the IPC analysis and OCHA reporting agree."
        )
        result = validate_mod.check_style(content)
        assert result.passed, result.errors

    def test_a_placeholder_is_not_a_code(self):
        content = _content(headline="The forecast sits at {{fig:log_ev_ratio}}.")
        assert validate_mod.check_style(content).passed


class TestCategories:
    """The pack decides the sections. The model copies that decision."""

    PACK = {QID: ("worsening", "conflict")}

    def test_matching_content_passes(self):
        result = validate_mod.check_categories(_content(), pack_categories=self.PACK)
        assert result.passed, result.errors

    def test_promotion_fails(self):
        content = _content()
        content["attention"][0]["category"] = "stable_major"
        result = validate_mod.check_categories(content, pack_categories=self.PACK)
        assert not result.passed
        assert any("category" in e for e in result.errors)

    def test_wrong_family_fails(self):
        content = _content()
        content["attention"][0]["hazard_family"] = "climate"
        assert not validate_mod.check_categories(
            content, pack_categories=self.PACK
        ).passed

    def test_invented_entry_fails(self):
        content = _content()
        content["attention"].append(dict(content["attention"][0]))
        content["attention"][1]["question_ids"] = ["MADE_UP_QID"]
        result = validate_mod.check_categories(content, pack_categories=self.PACK)
        assert not result.passed
        assert any("no question the pack put in a category" in e for e in result.errors)

    def test_dropped_entry_fails(self):
        pack = dict(self.PACK)
        pack["SOM_FL_PA_2026-08"] = ("worsening", "climate")
        result = validate_mod.check_categories(_content(), pack_categories=pack)
        assert not result.passed
        assert any("never covers" in e for e in result.errors)

    def test_no_categorised_rows_skips_rather_than_fails(self):
        result = validate_mod.check_categories(_content(), pack_categories={})
        assert result.passed and result.skipped


class TestFigureKeyNormalisation:
    """The model writes figure_refs as "fig:x" because the prose syntax is
    {{fig:x}}. The prefix carries no meaning; rejecting it failed a whole
    report over punctuation."""

    def test_prefixed_refs_resolve(self):
        content = _content()
        content["attention"][0]["figure_refs"] = [
            "js_vs_baserate", "fig:js_vs_baserate", "{{fig:js_vs_baserate}}",
        ]
        result = validate_mod.check_referential(
            content,
            valid_question_ids={QID},
            per_question={QID: {"js_vs_baserate": 0.3}},
            global_figures={"skill_brier_spd": 0.4},
        )
        assert result.passed, result.errors

    def test_a_genuinely_missing_key_still_fails(self):
        content = _content()
        content["attention"][0]["figure_refs"] = ["fig:not_a_real_key"]
        result = validate_mod.check_referential(
            content,
            valid_question_ids={QID},
            per_question={QID: {"js_vs_baserate": 0.3}},
            global_figures={},
        )
        assert not result.passed


class TestCodeCheckPrecision:
    """FAO and IOM are prose. SOM and NIC are codes a reader cannot decode."""

    def test_agency_acronyms_are_not_codes(self):
        for text in ("The FAO reported shortages.",
                     "IOM tracked the movement.",
                     "WHO and ICRC both responded."):
            assert validate_mod.find_codes_in_prose(text) == [], text

    def test_real_iso3_codes_are_still_caught(self):
        assert "SOM" in validate_mod.find_codes_in_prose("SOM is worsening.")
        assert "NIC" in validate_mod.find_codes_in_prose("Watch NIC this month.")

    def test_metric_enums_and_slash_forms_still_caught(self):
        assert validate_mod.find_codes_in_prose("EVENT_OCCURRENCE is up.")
        assert validate_mod.find_codes_in_prose("The DR/PA forecast.")


class TestEvMultipleIsVerifiable:
    """The report leads with ev_multiple, so the guard must be able to check it.

    It could not: the first passing live report had the numeric check report
    "0 checked, 9 unverifiable" — honest, and no protection at all, because
    every figure the model cited was one the guard did not know how to derive.
    ev_multiple is exp(log_ev_ratio) from the same row.
    """

    def test_ev_multiple_is_rederived_and_matched(self, tmp_path: Path):
        con = duckdb.connect(str(tmp_path / "ev.duckdb"))
        con.execute(
            "CREATE TABLE forecast_deviation (run_id TEXT, question_id TEXT, "
            "model_name TEXT, js_vs_baserate DOUBLE, log_ev_ratio DOUBLE, "
            "eiv_nominal DOUBLE, eiv_per_100k DOUBLE)"
        )
        con.execute(
            "INSERT INTO forecast_deviation VALUES "
            f"('fc_1', '{QID}', 'ensemble_mean_v2', 0.3466, 1.2, 200000.0, 180.0)"
        )
        content = _content()
        content["attention"][0]["figure_refs"] = ["ev_multiple"]
        content["attention"][0]["why_it_stands_out"] = (
            "About {{fig:ev_multiple}} the usual level."
        )
        result = check_numeric(
            content, con=con, run_id="fc_1",
            per_question=PER_QUESTION, global_figures=GLOBAL,
        )
        assert result.passed, result.errors
        assert result.detail["n_checked"] >= 1, result.detail
        con.close()

    def test_a_wrong_ev_multiple_is_caught(self, tmp_path: Path):
        con = duckdb.connect(str(tmp_path / "ev2.duckdb"))
        con.execute(
            "CREATE TABLE forecast_deviation (run_id TEXT, question_id TEXT, "
            "model_name TEXT, js_vs_baserate DOUBLE, log_ev_ratio DOUBLE, "
            "eiv_nominal DOUBLE, eiv_per_100k DOUBLE)"
        )
        con.execute(
            "INSERT INTO forecast_deviation VALUES "
            f"('fc_1', '{QID}', 'ensemble_mean_v2', 0.3466, 1.2, 200000.0, 180.0)"
        )
        figs = {QID: dict(PER_QUESTION[QID], ev_multiple=99.0)}
        content = _content()
        content["attention"][0]["figure_refs"] = ["ev_multiple"]
        result = check_numeric(
            content, con=con, run_id="fc_1",
            per_question=figs, global_figures=GLOBAL,
        )
        assert not result.passed
        con.close()
