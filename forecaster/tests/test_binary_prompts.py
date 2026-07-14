# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for binary event prompt builder and base rate computation."""

from __future__ import annotations

import json
from datetime import date

import duckdb
import pytest

from forecaster.binary_prompts import (
    build_binary_base_rate,
    build_binary_event_prompt,
    get_binary_hazard_reasoning_block,
    parse_binary_response,
)


# ---- Prompt generation tests ----

def test_prompt_contains_all_sections():
    """Binary prompt should contain role, base rate, situation, reasoning, output sections."""
    question = {
        "iso3": "ETH",
        "hazard_code": "DR",
        "metric": "EVENT_OCCURRENCE",
        "country_name": "Ethiopia",
        "window_start_date": "2026-04",
    }
    base_rate = {
        "total_months": 120,
        "event_months": 18,
        "base_rate_pct": 15.0,
        "seasonal_pattern": {str(m): 10.0 + m for m in range(1, 13)},
        "recent_12m_events": 3,
        "recent_12m_rate": 25.0,
        "trend": "increasing",
    }
    prompt = build_binary_event_prompt(
        question=question,
        base_rate=base_rate,
        current_alerts=[],
        structured_data={},
        today="2026-03-21",
    )
    assert "ROLE AND TASK" in prompt
    assert "HISTORICAL BASE RATE" in prompt
    assert "CURRENT SITUATION" in prompt
    assert "HAZARD-SPECIFIC REASONING" in prompt
    assert "OUTPUT INSTRUCTIONS" in prompt
    assert "Brier score" in prompt
    assert "Ethiopia" in prompt
    assert "drought" in prompt


def test_prompt_with_current_alerts():
    """Prompt should include recent Orange/Red GDACS event months.

    facts_resolved rows carry no event name, so the section renders
    [level] ym only (the old event-name slot always printed '?')."""
    question = {
        "iso3": "BGD",
        "hazard_code": "FL",
        "metric": "EVENT_OCCURRENCE",
        "country_name": "Bangladesh",
        "window_start_date": "2026-04",
    }
    alerts = [
        {"alertlevel": "Orange", "ym": "2026-03"},
    ]
    prompt = build_binary_event_prompt(
        question=question,
        base_rate={},
        current_alerts=alerts,
        structured_data={},
        today="2026-03-21",
    )
    assert "[Orange] 2026-03" in prompt
    assert "Orange/Red alerts" in prompt
    alert_lines = [l for l in prompt.splitlines() if l.strip().startswith("- [")]
    assert alert_lines and all("?" not in l for l in alert_lines)


def test_prompt_with_structured_data():
    """Prompt should include structured data."""
    question = {
        "iso3": "ETH",
        "hazard_code": "DR",
        "metric": "EVENT_OCCURRENCE",
        "country_name": "Ethiopia",
        "window_start_date": date(2026, 4, 1),
    }
    structured = {
        "nmme_seasonal_outlook": "Below-normal rainfall expected",
        "enso": "La Ni\u00f1a conditions present",
    }
    prompt = build_binary_event_prompt(
        question=question,
        base_rate={},
        current_alerts=[],
        structured_data=structured,
        today="2026-03-21",
    )
    assert "Below-normal rainfall expected" in prompt
    assert "La Ni\u00f1a" in prompt


# ---- Hazard reasoning block tests ----

def test_binary_hazard_reasoning_dr():
    block = get_binary_hazard_reasoning_block("DR")
    assert "DROUGHT" in block
    assert "NMME" in block


def test_binary_hazard_reasoning_fl():
    block = get_binary_hazard_reasoning_block("FL")
    assert "FLOOD" in block
    assert "seasonal" in block.lower()


def test_binary_hazard_reasoning_tc():
    block = get_binary_hazard_reasoning_block("TC")
    assert "TROPICAL CYCLONE" in block
    assert "season" in block.lower()


def test_binary_hazard_reasoning_unknown():
    block = get_binary_hazard_reasoning_block("UNKNOWN")
    assert "BINARY EVENT" in block


# ---- Response parsing tests ----

def test_parse_valid_response():
    """Parse a well-formed binary response."""
    raw = json.dumps({
        "months": {
            "2026-04": {"prior": 0.12, "posterior": 0.08, "reasoning": "test"},
            "2026-05": {"prior": 0.15, "posterior": 0.22, "reasoning": "test"},
            "2026-06": {"prior": 0.10, "posterior": 0.10, "reasoning": "test"},
        }
    })
    result = parse_binary_response(raw)
    assert len(result) == 3
    assert result["2026-04"] == 0.08
    assert result["2026-05"] == 0.22
    assert result["2026-06"] == 0.10


def test_parse_response_with_code_fences():
    """Parse JSON wrapped in markdown code fences."""
    raw = '```json\n{"months": {"2026-04": {"posterior": 0.15}}}\n```'
    result = parse_binary_response(raw)
    assert result["2026-04"] == 0.15


def test_parse_response_clamps_values():
    """Out-of-range probabilities should be clamped to [0.001, 0.999]."""
    raw = json.dumps({
        "months": {
            "2026-04": {"posterior": 0.0},
            "2026-05": {"posterior": 1.0},
            "2026-06": {"posterior": -0.5},
        }
    })
    result = parse_binary_response(raw)
    assert result["2026-04"] == 0.001
    assert result["2026-05"] == 0.999
    assert result["2026-06"] == 0.001


def test_parse_malformed_json():
    """Malformed JSON should return empty dict."""
    result = parse_binary_response("this is not json at all")
    assert result == {}


def test_parse_flat_probability_values():
    """Accept month -> float format (without nested dict)."""
    raw = json.dumps({
        "months": {
            "2026-04": 0.12,
            "2026-05": 0.25,
        }
    })
    result = parse_binary_response(raw)
    assert result["2026-04"] == 0.12
    assert result["2026-05"] == 0.25


def test_parse_complete_expected_months_accepted():
    """A response covering every expected month passes through."""
    months = ["2026-04", "2026-05", "2026-06"]
    raw = json.dumps({"months": {m: {"posterior": 0.1} for m in months}})
    result = parse_binary_response(raw, expected_months=months)
    assert sorted(result) == months


def test_parse_incomplete_expected_months_rejected():
    """A response missing any expected month is a model-level failure —
    partial forecasts must not flow into aggregation."""
    raw = json.dumps({
        "months": {
            "2026-04": {"posterior": 0.1},
            "2026-05": {"posterior": 0.2},
        }
    })
    result = parse_binary_response(
        raw, expected_months=["2026-04", "2026-05", "2026-06"]
    )
    assert result == {}


def test_parse_partial_allowed_without_expected_months():
    """Without expected_months, partial results still pass (backward compat)."""
    raw = json.dumps({"months": {"2026-04": {"posterior": 0.1}}})
    result = parse_binary_response(raw)
    assert result == {"2026-04": 0.1}


# ---- Base rate tests ----

def _setup_facts_resolved(con, rows: list[tuple]):
    """Create facts_resolved table and insert test rows."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS facts_resolved (
            ym VARCHAR,
            iso3 VARCHAR,
            hazard_code VARCHAR,
            metric VARCHAR,
            value DOUBLE,
            alertlevel VARCHAR,
            publisher VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    for row in rows:
        con.execute(
            "INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value) VALUES (?, ?, ?, ?, ?)",
            list(row),
        )


def test_base_rate_with_data():
    """Base rate computation with event occurrence data."""
    con = duckdb.connect(":memory:")
    rows = []
    # 24 months of data, 6 events
    for y in (2024, 2025):
        for m in range(1, 13):
            ym = f"{y:04d}-{m:02d}"
            # Events in Jun, Jul, Aug each year (flood season)
            value = 1.0 if m in (6, 7, 8) else 0.0
            rows.append((ym, "BGD", "FL", "event_occurrence", value))

    _setup_facts_resolved(con, rows)
    result = build_binary_base_rate("BGD", "FL", conn=con)
    con.close()

    assert result["total_months"] == 24
    assert result["event_months"] == 6
    assert abs(result["base_rate_pct"] - 25.0) < 0.1
    # June should have 100% event rate
    assert result["seasonal_pattern"]["6"] == 100.0
    # January should have 0%
    assert result["seasonal_pattern"]["1"] == 0.0


def test_base_rate_no_data():
    """Base rate returns zeros when no data exists."""
    con = duckdb.connect(":memory:")
    _setup_facts_resolved(con, [])
    result = build_binary_base_rate("XYZ", "FL", conn=con)
    con.close()

    assert result["total_months"] == 0
    assert result["event_months"] == 0
    assert result["base_rate_pct"] == 0.0
    assert result["trend"] == "unknown"


def test_base_rate_no_table():
    """Base rate returns empty dict when facts_resolved doesn't exist."""
    con = duckdb.connect(":memory:")
    result = build_binary_base_rate("ETH", "DR", conn=con)
    con.close()
    assert result == {}


def test_base_rate_trend_detection():
    """Trend detection: increasing when recent rate > overall * 1.3."""
    con = duckdb.connect(":memory:")
    rows = []
    # 36 months: first 24 months = 2 events, last 12 months = 6 events
    for i in range(36):
        y = 2023 + i // 12
        m = (i % 12) + 1
        ym = f"{y:04d}-{m:02d}"
        if i >= 24:  # last 12 months
            value = 1.0 if m in (1, 3, 5, 7, 9, 11) else 0.0
        else:  # first 24 months
            value = 1.0 if m == 6 else 0.0
        rows.append((ym, "ETH", "DR", "event_occurrence", value))

    _setup_facts_resolved(con, rows)
    result = build_binary_base_rate("ETH", "DR", conn=con)
    con.close()

    assert result["trend"] == "increasing"

def test_base_rate_coverage_window_captured():
    """The base rate carries the ACTUAL coverage window (the header used to
    hardcode '2015-present' regardless of the data)."""
    con = duckdb.connect(":memory:")
    rows = [
        (f"{y:04d}-{m:02d}", "IRN", "DR", "event_occurrence", 0.0)
        for y in (2021, 2022)
        for m in range(10, 13)
    ]
    _setup_facts_resolved(con, rows)
    result = build_binary_base_rate("IRN", "DR", conn=con)
    con.close()

    assert result["coverage_start"] == "2021-10"
    assert result["coverage_end"] == "2022-12"


def test_base_rate_section_renders_dynamic_coverage():
    from forecaster.binary_prompts import _section_base_rate

    base_rate = {
        "total_months": 54,
        "event_months": 0,
        "base_rate_pct": 0.0,
        "seasonal_pattern": {str(m): 0.0 for m in range(1, 13)},
        "recent_12m_events": 0,
        "recent_12m_rate": 0.0,
        "trend": "stable",
        "coverage_start": "2021-10",
        "coverage_end": "2026-05",
    }
    text = _section_base_rate("Iran", "drought", base_rate)
    assert "GDACS, 2021-10 to 2026-05" in text
    assert "2015" not in text


def test_reliefweb_list_renders_with_header():
    """ReliefWeb titles must carry their own section header — headerless
    bullets rendered visually nested under ENSO STATE (July 2026 run)."""
    question = {
        "iso3": "IRN",
        "hazard_code": "DR",
        "metric": "EVENT_OCCURRENCE",
        "country_name": "Iran",
        "window_start_date": "2026-08",
    }
    prompt = build_binary_event_prompt(
        question=question,
        base_rate={},
        current_alerts=[],
        structured_data={
            "enso_context": "## ENSO State and Forecast (IRI/CPC)\nCurrent state: Neutral.",
            "reliefweb_reports": [
                {"title": "Situation Report 1"},
                {"title": "Situation Report 2"},
            ],
        },
        today="2026-07-14",
    )
    assert "RECENT RELIEFWEB REPORTS:" in prompt
    # Header must precede the first title.
    assert prompt.index("RECENT RELIEFWEB REPORTS:") < prompt.index("Situation Report 1")
