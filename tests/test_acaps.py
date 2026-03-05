# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for the ACAPS unified data connector."""

from __future__ import annotations

import os

import pytest

from pythia.acaps import (
    _access_category,
    _current_month_label,
    _month_labels_back,
    _safe_float,
    _safe_int,
    _truncate,
    format_daily_monitoring_for_prompt,
    format_daily_monitoring_for_spd,
    format_humanitarian_access_for_prompt,
    format_inform_severity_for_prompt,
    format_inform_severity_for_spd,
    format_risk_radar_for_prompt,
    format_risk_radar_for_spd,
)


# ---------------------------------------------------------------------------
# Fixtures — sample data
# ---------------------------------------------------------------------------

def _sample_severity() -> dict:
    return {
        "iso3": "SDN",
        "crisis_id": "SDN001",
        "crisis_name": "Sudan",
        "severity_score": 4.2,
        "severity_category": "Very High",
        "impact_score": 4.5,
        "conditions_score": 4.0,
        "complexity_score": 3.8,
        "snapshot_date": "Feb2026",
        "trend_6m": [
            {"date": "Sep2025", "score": 3.8},
            {"date": "Oct2025", "score": 3.9},
            {"date": "Nov2025", "score": 4.0},
            {"date": "Dec2025", "score": 4.0},
            {"date": "Jan2026", "score": 4.1},
        ],
        "delta_1m": 0.1,
        "delta_3m": 0.2,
        "top_indicators": [
            {"indicator": "People displaced", "figure": 5.0, "dimension": "impact"},
            {"indicator": "Food security", "figure": 4.8, "dimension": "conditions"},
        ],
        "fetched_at": "2026-03-05T10:00:00+00:00",
    }


def _sample_risk_radar() -> dict:
    return {
        "iso3": "SDN",
        "risks": [
            {
                "risk_id": "R123",
                "title": "Escalation of conflict in Darfur",
                "risk_type": "Marked deterioration in an existing crisis",
                "risk_level": "High",
                "probability": "High",
                "impact": 5,
                "risk_trend": "Worsening",
                "expected_exposure": ">250,000",
                "triggers_summary": "Collapse of ceasefire; RSF offensive",
                "rationale": "Ongoing conflict between SAF and RSF...",
                "triggers": [
                    {
                        "trigger_id": "T456",
                        "title": "RSF advance on El Fasher",
                        "completion_rate": "75%",
                        "trend": "Increasing",
                        "description": "RSF forces moving toward...",
                    },
                    {
                        "trigger_id": "T789",
                        "title": "Ceasefire collapse",
                        "completion_rate": "60%",
                        "trend": "Stable",
                        "description": "Negotiations stalled...",
                    },
                ],
            },
        ],
        "total_active_risks": 1,
        "highest_risk_level": "High",
        "fetched_at": "2026-03-05T10:00:00+00:00",
    }


def _sample_monitoring_entries() -> list[dict]:
    return [
        {
            "entry_id": "M001",
            "date": "2026-03-04",
            "latest_developments": "Heavy fighting reported in El Fasher.",
            "source": "OCHA",
            "weekly_pick": True,
        },
        {
            "entry_id": "M002",
            "date": "2026-03-03",
            "latest_developments": "Aid convoy blocked at checkpoint.",
            "source": "MSF",
            "weekly_pick": False,
        },
        {
            "entry_id": "M003",
            "date": "2026-03-01",
            "latest_developments": "Displacement from Wad Madani continues.",
            "source": "UNHCR",
            "weekly_pick": True,
        },
    ]


def _sample_access() -> dict:
    return {
        "iso3": "SDN",
        "crisis_id": "SDN001",
        "access_score": 4.2,
        "access_category": "Very High",
        "snapshot_date": "Jul2025",
        "stale": True,
        "fetched_at": "2026-03-05T10:00:00+00:00",
    }


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for shared helper functions."""

    def test_safe_float_valid(self):
        assert _safe_float(3.14) == 3.14

    def test_safe_float_string(self):
        assert _safe_float("2.5") == 2.5

    def test_safe_float_none(self):
        assert _safe_float(None) is None

    def test_safe_float_invalid(self):
        assert _safe_float("abc") is None

    def test_safe_int_valid(self):
        assert _safe_int(5) == 5

    def test_safe_int_string(self):
        assert _safe_int("42") == 42

    def test_safe_int_none(self):
        assert _safe_int(None) == 0

    def test_safe_int_invalid(self):
        assert _safe_int("abc") == 0

    def test_truncate_short(self):
        assert _truncate("hello", 10) == "hello"

    def test_truncate_long(self):
        result = _truncate("hello world foo bar", 12)
        assert result.endswith("...")
        assert len(result) <= 15  # word-boundary truncation

    def test_access_category_very_low(self):
        assert _access_category(0.5) == "Very Low"

    def test_access_category_low(self):
        assert _access_category(1.5) == "Low"

    def test_access_category_medium(self):
        assert _access_category(2.5) == "Medium"

    def test_access_category_high(self):
        assert _access_category(3.5) == "High"

    def test_access_category_very_high(self):
        assert _access_category(4.5) == "Very High"


class TestMonthLabels:
    """Tests for month label generation."""

    def test_current_month_label_format(self):
        label = _current_month_label()
        # Should be 7-8 chars like Mar2026
        assert len(label) >= 7
        assert label[-4:].isdigit()

    def test_month_labels_back_count(self):
        labels = _month_labels_back(6)
        assert len(labels) == 6

    def test_month_labels_back_first_is_current(self):
        labels = _month_labels_back(3)
        assert labels[0] == _current_month_label()


# ---------------------------------------------------------------------------
# INFORM Severity formatters
# ---------------------------------------------------------------------------


class TestFormatInformSeverity:
    """Tests for format_inform_severity_for_prompt and _for_spd."""

    def test_prompt_returns_empty_for_none(self):
        assert format_inform_severity_for_prompt(None) == ""

    def test_prompt_returns_empty_for_no_score(self):
        assert format_inform_severity_for_prompt({"severity_score": None}) == ""

    def test_prompt_contains_header(self):
        result = format_inform_severity_for_prompt(_sample_severity())
        assert "INFORM SEVERITY INDEX (SDN):" in result

    def test_prompt_contains_score(self):
        result = format_inform_severity_for_prompt(_sample_severity())
        assert "4.2/5.0" in result
        assert "Very High" in result

    def test_prompt_contains_dimensions(self):
        result = format_inform_severity_for_prompt(_sample_severity())
        assert "Impact 4.5/5" in result
        assert "Conditions 4.0/5" in result
        assert "Complexity 3.8/5" in result

    def test_prompt_contains_trend(self):
        result = format_inform_severity_for_prompt(_sample_severity())
        assert "+0.1 vs last month" in result
        assert "+0.2 vs 3 months ago" in result

    def test_prompt_contains_indicators(self):
        result = format_inform_severity_for_prompt(_sample_severity())
        assert "People displaced" in result

    def test_prompt_contains_guidance(self):
        result = format_inform_severity_for_prompt(_sample_severity())
        assert "severity benchmark" in result

    def test_spd_returns_empty_for_none(self):
        assert format_inform_severity_for_spd(None) == ""

    def test_spd_compact_format(self):
        result = format_inform_severity_for_spd(_sample_severity())
        assert "INFORM SEVERITY (SDN):" in result
        assert "4.2/5.0" in result
        assert "D1m:" in result

    def test_spd_no_deltas(self):
        data = _sample_severity()
        data["delta_1m"] = None
        data["delta_3m"] = None
        result = format_inform_severity_for_spd(data)
        assert "n/a" in result


# ---------------------------------------------------------------------------
# Risk Radar formatters
# ---------------------------------------------------------------------------


class TestFormatRiskRadar:
    """Tests for format_risk_radar_for_prompt and _for_spd."""

    def test_prompt_returns_empty_for_none(self):
        assert format_risk_radar_for_prompt(None) == ""

    def test_prompt_returns_empty_for_zero_risks(self):
        data = {"iso3": "NOR", "risks": [], "total_active_risks": 0,
                "highest_risk_level": "None"}
        assert format_risk_radar_for_prompt(data) == ""

    def test_prompt_contains_header(self):
        result = format_risk_radar_for_prompt(_sample_risk_radar())
        assert "ACAPS RISK RADAR (SDN):" in result

    def test_prompt_contains_risk_details(self):
        result = format_risk_radar_for_prompt(_sample_risk_radar())
        assert "Escalation of conflict in Darfur" in result
        assert "High" in result
        assert "Impact: 5/5" in result

    def test_prompt_contains_triggers(self):
        result = format_risk_radar_for_prompt(_sample_risk_radar())
        assert "RSF advance on El Fasher" in result
        assert "75% complete" in result

    def test_prompt_contains_guidance(self):
        result = format_risk_radar_for_prompt(_sample_risk_radar())
        assert "expert forward-looking assessments" in result

    def test_spd_returns_empty_for_none(self):
        assert format_risk_radar_for_spd(None) == ""

    def test_spd_returns_empty_for_no_risks(self):
        data = {"iso3": "NOR", "risks": [], "total_active_risks": 0}
        assert format_risk_radar_for_spd(data) == ""

    def test_spd_compact_format(self):
        result = format_risk_radar_for_spd(_sample_risk_radar())
        assert "ACAPS RISKS (SDN):" in result
        assert "[High]" in result
        assert "P:High" in result


# ---------------------------------------------------------------------------
# Daily Monitoring formatters
# ---------------------------------------------------------------------------


class TestFormatDailyMonitoring:
    """Tests for format_daily_monitoring_for_prompt and _for_spd."""

    def test_prompt_returns_empty_for_none(self):
        assert format_daily_monitoring_for_prompt(None) == ""

    def test_prompt_returns_empty_for_empty_list(self):
        assert format_daily_monitoring_for_prompt([]) == ""

    def test_prompt_contains_header(self):
        result = format_daily_monitoring_for_prompt(_sample_monitoring_entries())
        assert "ACAPS DAILY MONITORING" in result

    def test_prompt_marks_weekly_picks(self):
        result = format_daily_monitoring_for_prompt(_sample_monitoring_entries())
        assert "* [2026-03-04]" in result  # weekly pick
        assert "  [2026-03-03]" in result  # not a weekly pick

    def test_prompt_contains_entries(self):
        result = format_daily_monitoring_for_prompt(_sample_monitoring_entries())
        assert "Heavy fighting reported" in result
        assert "Aid convoy blocked" in result

    def test_prompt_contains_guidance(self):
        result = format_daily_monitoring_for_prompt(_sample_monitoring_entries())
        assert "analyst-curated" in result

    def test_spd_returns_empty_for_none(self):
        assert format_daily_monitoring_for_spd(None) == ""

    def test_spd_returns_empty_for_empty_list(self):
        assert format_daily_monitoring_for_spd([]) == ""

    def test_spd_prefers_weekly_picks(self):
        result = format_daily_monitoring_for_spd(_sample_monitoring_entries())
        assert "ACAPS MONITORING:" in result
        # Should include entries (weekly picks preferred)
        assert "2026-03-04" in result

    def test_spd_max_5_entries(self):
        entries = [
            {
                "entry_id": f"M{i}",
                "date": f"2026-03-{i + 1:02d}",
                "latest_developments": f"Event {i}.",
                "source": "Test",
                "weekly_pick": True,
            }
            for i in range(10)
        ]
        result = format_daily_monitoring_for_spd(entries)
        lines = [l for l in result.strip().split("\n") if l.startswith("- ")]
        assert len(lines) == 5


# ---------------------------------------------------------------------------
# Humanitarian Access formatter
# ---------------------------------------------------------------------------


class TestFormatHumanitarianAccess:
    """Tests for format_humanitarian_access_for_prompt."""

    def test_returns_empty_for_none(self):
        assert format_humanitarian_access_for_prompt(None) == ""

    def test_returns_empty_for_no_score(self):
        assert format_humanitarian_access_for_prompt(
            {"access_score": None}
        ) == ""

    def test_contains_header(self):
        result = format_humanitarian_access_for_prompt(_sample_access())
        assert "HUMANITARIAN ACCESS (SDN):" in result

    def test_contains_score(self):
        result = format_humanitarian_access_for_prompt(_sample_access())
        assert "4.2/5.0" in result
        assert "Very High" in result

    def test_contains_date(self):
        result = format_humanitarian_access_for_prompt(_sample_access())
        assert "Jul2025" in result

    def test_stale_warning(self):
        result = format_humanitarian_access_for_prompt(_sample_access())
        assert "STALE" in result

    def test_no_stale_warning_when_fresh(self):
        data = _sample_access()
        data["stale"] = False
        result = format_humanitarian_access_for_prompt(data)
        assert "STALE" not in result

    def test_contains_guidance(self):
        result = format_humanitarian_access_for_prompt(_sample_access())
        assert "access constraints" in result


# ---------------------------------------------------------------------------
# Live integration (conditional on ACAPS credentials)
# ---------------------------------------------------------------------------

_has_acaps_creds = bool(
    os.environ.get("ACAPS_USERNAME") and os.environ.get("ACAPS_PASSWORD")
)


@pytest.mark.allow_network
@pytest.mark.skipif(not _has_acaps_creds, reason="ACAPS credentials not set")
def test_live_auth():
    """Smoke test: verify ACAPS token retrieval."""
    from pythia.acaps import _get_acaps_token

    token = _get_acaps_token(force_refresh=True)
    assert token is not None
    assert len(token) > 10


@pytest.mark.allow_network
@pytest.mark.skipif(not _has_acaps_creds, reason="ACAPS credentials not set")
def test_live_inform_severity_sdn():
    """Smoke test: fetch INFORM Severity for Sudan."""
    from pythia.acaps import fetch_inform_severity

    data = fetch_inform_severity("SDN")
    assert data is not None
    assert data["iso3"] == "SDN"
    assert isinstance(data["severity_score"], float)
    assert 1.0 <= data["severity_score"] <= 5.0
    assert data["snapshot_date"]


@pytest.mark.allow_network
@pytest.mark.skipif(not _has_acaps_creds, reason="ACAPS credentials not set")
def test_live_risk_radar_sdn():
    """Smoke test: fetch Risk Radar for Sudan."""
    from pythia.acaps import fetch_risk_radar

    data = fetch_risk_radar("SDN")
    assert data is not None
    assert data["iso3"] == "SDN"
    assert isinstance(data["risks"], list)
    assert isinstance(data["total_active_risks"], int)


@pytest.mark.allow_network
@pytest.mark.skipif(not _has_acaps_creds, reason="ACAPS credentials not set")
def test_live_risk_radar_empty_nor():
    """Smoke test: fetch Risk Radar for Norway (expect zero risks)."""
    from pythia.acaps import fetch_risk_radar

    data = fetch_risk_radar("NOR")
    assert data is not None
    assert data["total_active_risks"] == 0
    assert data["risks"] == []


@pytest.mark.allow_network
@pytest.mark.skipif(not _has_acaps_creds, reason="ACAPS credentials not set")
def test_live_daily_monitoring_sdn():
    """Smoke test: fetch Daily Monitoring for Sudan."""
    from pythia.acaps import fetch_daily_monitoring

    entries = fetch_daily_monitoring("SDN")
    # Sudan should have entries; but the API might return None if
    # monitoring doesn't cover it in the requested window.
    if entries is not None:
        assert isinstance(entries, list)
        assert len(entries) > 0
        entry = entries[0]
        assert "date" in entry
        assert "latest_developments" in entry


@pytest.mark.allow_network
@pytest.mark.skipif(not _has_acaps_creds, reason="ACAPS credentials not set")
def test_live_humanitarian_access_sdn():
    """Smoke test: fetch Humanitarian Access for Sudan."""
    from pythia.acaps import fetch_humanitarian_access

    data = fetch_humanitarian_access("SDN")
    # Sudan should have access data
    if data is not None:
        assert data["iso3"] == "SDN"
        assert isinstance(data["access_score"], float)
        assert 0.0 <= data["access_score"] <= 5.0
