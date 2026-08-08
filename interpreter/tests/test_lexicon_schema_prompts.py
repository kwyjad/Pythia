# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Lexicon bands, output-schema shape validation, template assembly."""

from __future__ import annotations

import pytest

from interpreter import lexicon, prompts, schema


class TestLexicon:
    def test_band_boundaries(self):
        assert lexicon.word_for(0.995) == "virtually certain"
        assert lexicon.word_for(0.99) == "virtually certain"
        assert lexicon.word_for(0.95) == "very likely"
        assert lexicon.word_for(0.90) == "very likely"
        assert lexicon.word_for(0.70) == "likely"
        assert lexicon.word_for(0.66) == "likely"
        assert lexicon.word_for(0.50) == "about as likely as not"
        assert lexicon.word_for(0.33) == "about as likely as not"
        assert lexicon.word_for(0.20) == "unlikely"
        assert lexicon.word_for(0.10) == "unlikely"
        assert lexicon.word_for(0.05) == "very unlikely"
        assert lexicon.word_for(0.01) == "very unlikely"
        assert lexicon.word_for(0.005) == "exceptionally unlikely"
        assert lexicon.word_for(0.0) == "exceptionally unlikely"
        assert lexicon.word_for(1.0) == "virtually certain"

    def test_word_matches_and_unknown_word(self):
        assert lexicon.word_matches("likely", 0.7)
        assert not lexicon.word_matches("likely", 0.5)
        assert not lexicon.word_matches("plausible", 0.5)  # not in the lexicon

    def test_markdown_table_lists_every_word(self):
        table = lexicon.markdown_table()
        for word in lexicon.WORDS:
            assert word in table


def _valid_content(kind: str = "combined") -> dict:
    content = {
        "schema_version": "1",
        "template_version": "v1",
        "kind": kind,
        "headline": "One country stands far above its base rate this month.",
        "attention": [
            {
                "rank": 1,
                "reason_code": "base_rate_deviation",
                "iso3": "ETH",
                "hazard_code": "ACE",
                "metric": "FATALITIES",
                "category": "worsening",
                "hazard_family": "conflict",
                "question_ids": ["ETH_ACE_FATALITIES_2026-08"],
                "figure_refs": ["js_vs_baserate", "eiv_nominal"],
                "why_it_stands_out": "The ensemble sits {{fig:js_vs_baserate}} from its base rate.",
                "impacts": ["Displacement pressure on neighbouring regions."],
                "lead_time_months": 3,
            }
        ],
        "performance": {"plain_summary": "No scored run exists yet."},
        "changes_since_last_run": ["A new risk entered the list."],
        "blind_spots": ["Coverage of impact reporting is thin."],
        "confidence_notes": ["The record behind the base rate is short."],
    }
    if kind == "scored":
        content.pop("attention")
    if kind == "current":
        content.pop("performance")
    return content


class TestSchema:
    def test_valid_content_passes(self):
        for kind in ("current", "scored", "combined"):
            errors = schema.validate_output(_valid_content(kind), kind=kind)
            assert errors == [], f"{kind}: {errors}"

    def test_missing_headline_fails(self):
        content = _valid_content()
        del content["headline"]
        assert any("headline" in e for e in schema.validate_output(content, kind="combined"))

    def test_bad_reason_code_fails(self):
        content = _valid_content()
        content["attention"][0]["reason_code"] = "vibes"
        assert schema.validate_output(content, kind="combined")

    def test_kind_conditional_requirements(self):
        content = _valid_content("combined")
        content["attention"] = []
        errors = schema.validate_output(content, kind="combined")
        assert any("attention" in e for e in errors)
        content2 = _valid_content("scored")
        del content2["performance"]
        errors2 = schema.validate_output(content2, kind="scored")
        assert any("performance" in e for e in errors2)

    def test_kind_mismatch_reported(self):
        content = _valid_content("current")
        errors = schema.validate_output(content, kind="scored")
        assert any("kind mismatch" in e for e in errors)

    def test_additional_properties_rejected(self):
        content = _valid_content()
        content["freeform_extra"] = "nope"
        assert schema.validate_output(content, kind="combined")


class TestPrompts:
    def test_system_prompt_carries_rules_schema_and_lexicon(self):
        text = prompts.build_system_prompt("v1")
        assert "NO NUMERALS IN PROSE" in text
        assert "{{fig:" in text
        assert "never blend" in text.lower() or "NEVER BLEND" in text
        assert '"schema_version"' in text  # embedded schema JSON
        assert "virtually certain" in text  # lexicon table
        # The fixed Track-1/2 caveat the model cannot drop.
        assert "disjoint" in text

    def test_user_prompt_kinds(self):
        combined = prompts.build_user_prompt(
            kind="combined", pack_text="PACKBODY", version="v1", scored_section=None
        )
        assert "PACKBODY" in combined
        assert "NO scored run exists yet" in combined
        combined2 = prompts.build_user_prompt(
            kind="combined", pack_text="PACKBODY", version="v1",
            scored_section='{"performance": {}}',
        )
        assert "already\nexists" in combined2 or "already exists" in combined2.replace("\n", " ")
        scored = prompts.build_user_prompt(kind="scored", pack_text="SP", version="v1")
        assert "SP" in scored and "performance" in scored
        current = prompts.build_user_prompt(kind="current", pack_text="CP", version="v1")
        assert "omit the" in current.lower()

    def test_prompt_hash_stable_and_sensitive(self):
        a = prompts.prompt_hash("s", "u")
        assert a == prompts.prompt_hash("s", "u")
        assert a != prompts.prompt_hash("s", "u2")

    def test_unknown_template_version_raises(self):
        with pytest.raises(FileNotFoundError):
            prompts.build_system_prompt("v999")
