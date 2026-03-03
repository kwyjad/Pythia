# Pythia / Copyright (c) 2025 Kevin Wyjad

"""Tests for conflict forecast prompt integration.

Validates :func:`format_conflict_forecasts_for_prompt`,
:func:`format_conflict_forecasts_for_research`, and the injection of
conflict forecast sections into the ACE RC and triage prompt builders.
"""

from __future__ import annotations

import pytest

from horizon_scanner.conflict_forecasts import (
    format_conflict_forecasts_for_prompt,
    format_conflict_forecasts_for_research,
)
from horizon_scanner.rc_prompts import build_rc_prompt_ace
from horizon_scanner.hs_triage_prompts import build_triage_prompt_ace


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

def _full_forecast_data() -> dict:
    """Return a complete conflict_forecasts dict (VIEWS + cf.org)."""
    return {
        "views_fatalities": [
            {"lead_months": 1, "value": 12.3},
            {"lead_months": 2, "value": 14.7},
            {"lead_months": 3, "value": 18.0},
        ],
        "views_p_gte25": [
            {"lead_months": 1, "value": 0.35},
            {"lead_months": 2, "value": 0.40},
        ],
        "views_issue_date": "2025-06-01",
        "views_model": "fatalities003",
        "views_stale": False,
        "cf_risk_3m": 0.42,
        "cf_risk_12m": 0.31,
        "cf_intensity_3m": 0.55,
        "cf_issue_date": "2025-06-15",
        "cf_stale": False,
    }


def _views_only_forecast_data() -> dict:
    """Return a forecast dict with only VIEWS data (no cf.org)."""
    return {
        "views_fatalities": [
            {"lead_months": 1, "value": 5.0},
        ],
        "views_p_gte25": [],
        "views_issue_date": "2025-07-01",
        "views_model": "fatalities003",
        "views_stale": False,
    }


def _stale_forecast_data() -> dict:
    """Return a forecast dict where both sources are stale."""
    return {
        "views_fatalities": [
            {"lead_months": 1, "value": 9.0},
        ],
        "views_p_gte25": [],
        "views_issue_date": "2025-01-01",
        "views_model": "fatalities003",
        "views_stale": True,
        "cf_risk_3m": 0.20,
        "cf_risk_12m": None,
        "cf_intensity_3m": None,
        "cf_issue_date": "2025-01-10",
        "cf_stale": True,
    }


def _minimal_resolver_features() -> dict:
    """Return a minimal resolver features dict for prompt builders."""
    return {"pa_trailing_12m": 50000, "fatalities_trailing_12m": 200}


# ---------------------------------------------------------------------------
# format_conflict_forecasts_for_prompt
# ---------------------------------------------------------------------------

class TestFormatConflictForecastsForPrompt:
    """Tests for format_conflict_forecasts_for_prompt."""

    def test_full_data_contains_views_section(self):
        result = format_conflict_forecasts_for_prompt(_full_forecast_data())
        assert "VIEWS" in result
        assert "fatalities003" in result
        assert "Predicted fatalities" in result

    def test_full_data_contains_cf_section(self):
        result = format_conflict_forecasts_for_prompt(_full_forecast_data())
        assert "conflictforecast.org" in result
        assert "Mueller/Rauh" in result
        assert "Armed conflict risk (3-month horizon)" in result

    def test_none_returns_empty_string(self):
        result = format_conflict_forecasts_for_prompt(None)
        assert result == ""

    def test_empty_dict_returns_empty_string(self):
        result = format_conflict_forecasts_for_prompt({})
        assert result == ""

    def test_views_only_no_cf_section(self):
        result = format_conflict_forecasts_for_prompt(_views_only_forecast_data())
        assert "VIEWS" in result
        # The cf.org *section header* (Mueller/Rauh) should be absent when
        # no cf.org data is provided.  The trailing interpretation guidance
        # may still reference conflictforecast.org by name.
        assert "Mueller/Rauh" not in result
        assert "Armed conflict risk (3-month horizon)" not in result

    def test_stale_data_contains_warning(self):
        result = format_conflict_forecasts_for_prompt(_stale_forecast_data())
        assert "WARNING" in result
        assert ">45 DAYS OLD" in result

    def test_full_data_contains_interpretation_guidance(self):
        result = format_conflict_forecasts_for_prompt(_full_forecast_data())
        assert "independent sources" in result


# ---------------------------------------------------------------------------
# format_conflict_forecasts_for_research
# ---------------------------------------------------------------------------

class TestFormatConflictForecastsForResearch:
    """Tests for format_conflict_forecasts_for_research (tabular format)."""

    def test_full_data_has_table_header(self):
        result = format_conflict_forecasts_for_research(_full_forecast_data())
        assert "QUANTITATIVE CONFLICT FORECASTS" in result
        assert "| Lead month |" in result
        assert "|--------" in result

    def test_full_data_has_table_rows(self):
        result = format_conflict_forecasts_for_research(_full_forecast_data())
        assert "Month 1" in result
        assert "Month 2" in result

    def test_full_data_includes_cf_section(self):
        result = format_conflict_forecasts_for_research(_full_forecast_data())
        assert "conflictforecast.org" in result
        assert "Armed conflict risk (3m)" in result

    def test_none_returns_empty_string(self):
        result = format_conflict_forecasts_for_research(None)
        assert result == ""

    def test_empty_dict_returns_empty_string(self):
        result = format_conflict_forecasts_for_research({})
        assert result == ""


# ---------------------------------------------------------------------------
# build_rc_prompt_ace — conflict forecast injection
# ---------------------------------------------------------------------------

class TestBuildRcPromptAceConflictForecasts:
    """Tests for conflict forecast injection in build_rc_prompt_ace."""

    def test_includes_conflict_forecast_section(self):
        prompt = build_rc_prompt_ace(
            country_name="Somalia",
            iso3="SOM",
            resolver_features=_minimal_resolver_features(),
            conflict_forecasts=_full_forecast_data(),
        )
        assert "VIEWS" in prompt
        assert "conflictforecast.org" in prompt
        assert "Compare these forward-looking estimates" in prompt

    def test_includes_icg_section(self):
        icg_text = "Somalia is at risk of renewed clan conflict in the south."
        prompt = build_rc_prompt_ace(
            country_name="Somalia",
            iso3="SOM",
            resolver_features=_minimal_resolver_features(),
            icg_on_the_horizon=icg_text,
        )
        assert "ON THE HORIZON" in prompt
        assert icg_text in prompt

    def test_none_conflict_forecasts_no_crash(self):
        prompt = build_rc_prompt_ace(
            country_name="Somalia",
            iso3="SOM",
            resolver_features=_minimal_resolver_features(),
            conflict_forecasts=None,
        )
        # Should still produce a valid prompt without the forecast section.
        assert "ACE-SPECIFIC GUIDANCE" in prompt
        # The forecast comparison block should be absent.
        assert "Compare these forward-looking estimates" not in prompt

    def test_none_icg_no_section(self):
        prompt = build_rc_prompt_ace(
            country_name="Somalia",
            iso3="SOM",
            resolver_features=_minimal_resolver_features(),
            icg_on_the_horizon=None,
        )
        assert "ON THE HORIZON" not in prompt


# ---------------------------------------------------------------------------
# build_triage_prompt_ace — conflict forecast injection
# ---------------------------------------------------------------------------

class TestBuildTriagePromptAceConflictForecasts:
    """Tests for conflict forecast injection in build_triage_prompt_ace."""

    def test_includes_conflict_forecast_section(self):
        prompt = build_triage_prompt_ace(
            country_name="Ethiopia",
            iso3="ETH",
            resolver_features=_minimal_resolver_features(),
            conflict_forecasts=_full_forecast_data(),
        )
        assert "VIEWS" in prompt
        assert "conflictforecast.org" in prompt

    def test_none_conflict_forecasts_no_crash(self):
        prompt = build_triage_prompt_ace(
            country_name="Ethiopia",
            iso3="ETH",
            resolver_features=_minimal_resolver_features(),
            conflict_forecasts=None,
        )
        assert "ACE-SPECIFIC TRIAGE GUIDANCE" in prompt
        # No EXTERNAL CONFLICT FORECASTS header when data is None.
        assert "EXTERNAL CONFLICT FORECASTS" not in prompt

    def test_empty_dict_conflict_forecasts_no_section(self):
        prompt = build_triage_prompt_ace(
            country_name="Ethiopia",
            iso3="ETH",
            resolver_features=_minimal_resolver_features(),
            conflict_forecasts={},
        )
        assert "EXTERNAL CONFLICT FORECASTS" not in prompt
