# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for the HDX Signals connector."""

from __future__ import annotations

import csv
import io
import textwrap
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import pytest

from horizon_scanner.hdx_signals import (
    HAZARD_TO_INDICATORS,
    INDICATOR_DISPLAY_NAMES,
    INDICATOR_TO_HAZARDS,
    clear_cache,
    format_hdx_signals_for_prompt,
    get_signals_for_country,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_CSV = textwrap.dedent("""\
    iso3,location,region,hrp_location,indicator_id,date,alert_level,value,plot,map,plot2,other_images,summary_long,summary_short,summary_source,hdx_url,source_url,other_urls,further_information,campaign_url,campaign_date,signals_version
    SDN,Sudan,Eastern Africa,True,acled_conflict,{recent_date},High concern,1250.0,,,,,Conflict in Sudan has intensified significantly.,Significant spike in conflict events.,,https://data.humdata.org/dataset/x,https://acleddata.com,,,,{recent_date},0.2.0
    SDN,Sudan,Eastern Africa,True,ipc_food_insecurity,{recent_date},High concern,5.0,,,,,"IPC Phase 5 declared in parts of Darfur, affecting 500,000 people.",IPC Phase 5 in Darfur.,,https://data.humdata.org/dataset/y,https://ipcinfo.org,,,,{recent_date},0.2.0
    SDN,Sudan,Eastern Africa,True,acaps_inform_severity,{recent_date},High concern,4.8,,,,,INFORM Severity Index for Sudan has reached its highest level.,INFORM Severity at record high.,,https://data.humdata.org/dataset/z,https://acaps.org,,,,{recent_date},0.2.0
    SOM,Somalia,Eastern Africa,True,wfp_market_monitor,{recent_date},Medium concern,22.0,,,,,,22% increase in the cost of the food basket.,,https://data.humdata.org/dataset/w,https://wfp.org,,,,{recent_date},0.2.0
    SOM,Somalia,Eastern Africa,True,idmc_displacement_disaster,{recent_date},High concern,85000.0,,,,,Major flooding has displaced 85000 people across southern Somalia.,85000 displaced by flooding.,,https://data.humdata.org/dataset/v,https://idmc.org,,,,{recent_date},0.2.0
    NZL,New Zealand,Oceania,False,wfp_market_monitor,{old_date},Medium concern,5.0,,,,,,5% increase in the cost of the food basket.,,https://data.humdata.org/dataset/u,https://wfp.org,,,,{old_date},0.2.0
""")


def _make_csv(recent_days_ago: int = 30, old_days_ago: int = 200) -> str:
    recent = (datetime.now() - timedelta(days=recent_days_ago)).strftime("%Y-%m-%d")
    old = (datetime.now() - timedelta(days=old_days_ago)).strftime("%Y-%m-%d")
    return _SAMPLE_CSV.format(recent_date=recent, old_date=old)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear the in-memory signals cache before and after each test."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture()
def mock_cache(tmp_path: Path):
    """Write sample CSV to a temporary cache file and patch CACHE_FILE."""
    csv_path = tmp_path / "hdx_signals.csv"
    csv_path.write_text(_make_csv(), encoding="utf-8")
    with mock.patch("horizon_scanner.hdx_signals.CACHE_FILE", csv_path):
        yield csv_path


# ---------------------------------------------------------------------------
# Mapping consistency
# ---------------------------------------------------------------------------


def test_indicator_to_hazards_and_reverse_are_consistent():
    """Every indicator in INDICATOR_TO_HAZARDS must appear in HAZARD_TO_INDICATORS."""
    for indicator, hazards in INDICATOR_TO_HAZARDS.items():
        for h in hazards:
            assert indicator in HAZARD_TO_INDICATORS[h], (
                f"{indicator} maps to {h} but is missing from HAZARD_TO_INDICATORS[{h}]"
            )


def test_hazard_to_indicators_and_reverse_are_consistent():
    """Every indicator in HAZARD_TO_INDICATORS must map back to that hazard."""
    for hazard, indicators in HAZARD_TO_INDICATORS.items():
        for ind in indicators:
            assert hazard in INDICATOR_TO_HAZARDS[ind], (
                f"HAZARD_TO_INDICATORS[{hazard}] contains {ind} but "
                f"INDICATOR_TO_HAZARDS[{ind}] = {INDICATOR_TO_HAZARDS[ind]}"
            )


def test_all_indicators_have_display_names():
    """Every indicator_id in the mapping should have a display name."""
    for indicator in INDICATOR_TO_HAZARDS:
        assert indicator in INDICATOR_DISPLAY_NAMES, (
            f"Missing display name for indicator: {indicator}"
        )


# ---------------------------------------------------------------------------
# get_signals_for_country
# ---------------------------------------------------------------------------


def test_get_signals_for_country_filters_by_iso3(mock_cache):
    """Should return only signals for the requested country."""
    results = get_signals_for_country("SDN", "ACE")
    assert all(r["iso3"] == "SDN" for r in results)
    assert len(results) >= 1


def test_get_signals_for_country_filters_by_hazard(mock_cache):
    """Should return only signals for indicators relevant to the hazard."""
    results = get_signals_for_country("SDN", "ACE")
    ace_indicators = set(HAZARD_TO_INDICATORS["ACE"])
    assert all(r["indicator_id"] in ace_indicators for r in results)


def test_get_signals_for_country_filters_by_age(mock_cache):
    """Signals older than max_age_days should be excluded."""
    # NZL has a signal at ~200 days ago, should be excluded with 180-day window.
    results = get_signals_for_country("NZL", "DR", max_age_days=180)
    assert len(results) == 0

    # But with 365-day window, it should appear.
    results = get_signals_for_country("NZL", "DR", max_age_days=365)
    assert len(results) == 1


def test_get_signals_for_country_empty_for_no_match(mock_cache):
    """Should return empty list for countries with no signals."""
    results = get_signals_for_country("USA", "ACE")
    assert results == []


def test_get_signals_for_country_empty_for_unknown_hazard(mock_cache):
    """Should return empty list for unmapped hazard codes."""
    results = get_signals_for_country("SDN", "UNKNOWN")
    assert results == []


def test_get_signals_sorted_by_date_descending(mock_cache):
    """Most recent signals should come first."""
    results = get_signals_for_country("SDN", "ACE")
    if len(results) > 1:
        dates = [(r.get("campaign_date") or "")[:10] for r in results]
        assert dates == sorted(dates, reverse=True)


# ---------------------------------------------------------------------------
# format_hdx_signals_for_prompt
# ---------------------------------------------------------------------------


def test_format_returns_empty_for_no_signals(mock_cache):
    """No signals → empty string."""
    result = format_hdx_signals_for_prompt("USA", "ACE")
    assert result == ""


def test_format_returns_header_and_note(mock_cache):
    """Output should include the header and the trailing NOTE."""
    result = format_hdx_signals_for_prompt("SDN", "ACE")
    assert "## HDX Signals" in result
    assert "OCHA" in result
    assert "NOTE:" in result


def test_format_includes_signal_details(mock_cache):
    """Output should include indicator name, alert level, and source."""
    result = format_hdx_signals_for_prompt("SDN", "ACE")
    assert "Conflict Events (ACLED)" in result
    assert "High concern" in result


def test_format_disaster_displacement_disclaimer(mock_cache):
    """Disaster displacement signals should include a disclaimer."""
    result = format_hdx_signals_for_prompt("SOM", "FL")
    assert "all disaster types" in result


def test_format_respects_max_signals(mock_cache):
    """Should limit the number of signals shown."""
    result = format_hdx_signals_for_prompt("SDN", "ACE", max_signals=1)
    # Should have at most 1 signal header (### ...)
    headers = [line for line in result.split("\n") if line.startswith("### ")]
    assert len(headers) <= 1


# ---------------------------------------------------------------------------
# Integration test (requires network)
# ---------------------------------------------------------------------------


@pytest.mark.allow_network
def test_fetch_and_cache_downloads_csv(tmp_path: Path):
    """Verify fetch_and_cache() can download from HDX (requires network)."""
    from horizon_scanner.hdx_signals import fetch_and_cache

    with mock.patch("horizon_scanner.hdx_signals.CACHE_DIR", tmp_path), \
         mock.patch("horizon_scanner.hdx_signals.CACHE_FILE", tmp_path / "hdx_signals.csv"):
        result = fetch_and_cache()
        assert result is not None
        assert result.exists()
        assert result.stat().st_size > 1000  # should be at least a few KB
