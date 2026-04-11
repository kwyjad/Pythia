# Pythia / Copyright (c) 2025 Kevin Wyjad
"""Unit tests for the GDELT conflict indicators connector."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date
from pathlib import Path

import pytest

from pythia.gdelt import (
    TIER_1_CODES,
    TIER_2_CODES,
    TIER_3_CODES,
    _load_fips_to_iso3,
    compute_country_indicators,
    format_gdelt_for_prompt,
    load_gdelt_conflict_indicators,
)


# ---------------------------------------------------------------------------
# Static data tests
# ---------------------------------------------------------------------------


def test_cameo_tier_sets_are_disjoint():
    assert TIER_1_CODES & TIER_2_CODES == set()
    assert TIER_1_CODES & TIER_3_CODES == set()
    assert TIER_2_CODES & TIER_3_CODES == set()


def test_cameo_tier_codes_content():
    # Spot-check known root codes
    assert "18" in TIER_1_CODES  # Unconventional violence
    assert "19" in TIER_1_CODES  # Fight
    assert "20" in TIER_1_CODES  # Unconventional mass violence
    assert "15" in TIER_2_CODES  # Exhibit force posture
    assert "17" in TIER_2_CODES  # Coerce
    assert "14" in TIER_3_CODES  # Protest
    assert "16" in TIER_3_CODES  # Reduce relations


def test_fips_to_iso3_covers_key_countries():
    mapping = _load_fips_to_iso3()
    # Spot-check countries commonly forecast by Pythia
    assert mapping["IZ"] == "IRQ"
    assert mapping["SY"] == "SYR"
    assert mapping["SU"] == "SDN"
    assert mapping["OD"] == "SSD"
    assert mapping["SO"] == "SOM"
    assert mapping["UP"] == "UKR"
    assert mapping["ET"] == "ETH"
    assert mapping["MY"] == "MYS"


# ---------------------------------------------------------------------------
# Indicator computation
# ---------------------------------------------------------------------------


def _make_event(
    country="IZ",
    root_code="19",
    event_code="190",
    quad=4,
    goldstein=-7.0,
    tone=-5.0,
):
    return {
        "SQLDATE": "20260401",
        "EventRootCode": root_code,
        "EventCode": event_code,
        "QuadClass": str(quad),
        "GoldsteinScale": str(goldstein),
        "AvgTone": str(tone),
        "ActionGeo_CountryCode": country,
    }


def test_compute_indicators_basic_tiers():
    events = [
        _make_event(root_code="19", event_code="190"),  # T1
        _make_event(root_code="18", event_code="182"),  # T1
        _make_event(root_code="17", event_code="173"),  # T2
        _make_event(root_code="14", event_code="141"),  # T3
        _make_event(root_code="01", event_code="010", quad=1),  # neutral
    ]
    out = compute_country_indicators(events)
    assert "IRQ" in out
    ind = out["IRQ"]
    assert ind["total_events"] == 5
    assert ind["tier1_events"] == 2
    assert ind["tier2_events"] == 1
    assert ind["tier3_events"] == 1
    # 4 of 5 events have QuadClass == 4
    assert ind["material_conflict_events"] == 4
    # Goldstein average from all events (all -7)
    assert ind["avg_goldstein"] == pytest.approx(-7.0)


def test_compute_indicators_drops_unknown_country():
    events = [
        _make_event(country="ZZ"),  # not in FIPS map
        _make_event(country="IZ"),
    ]
    out = compute_country_indicators(events)
    assert list(out.keys()) == ["IRQ"]
    assert out["IRQ"]["total_events"] == 1


def test_compute_indicators_top_codes_json():
    events = [
        _make_event(root_code="19", event_code="190"),
        _make_event(root_code="19", event_code="190"),
        _make_event(root_code="18", event_code="182"),
    ]
    out = compute_country_indicators(events)
    top = json.loads(out["IRQ"]["top_codes_json"])
    # Top should be [("190", 2), ("182", 1)]
    assert top[0] == ["190", 2]
    assert top[1] == ["182", 1]


def test_compute_indicators_empty_input():
    assert compute_country_indicators([]) == {}


# ---------------------------------------------------------------------------
# DB round-trip + prompt formatting
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db(monkeypatch):
    """Create a temp DuckDB file and point PYTHIA_DB_URL at it."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "gdelt_test.duckdb"
        monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db_path}")
        yield db_path


def test_load_empty_db_returns_none(temp_db):
    result = load_gdelt_conflict_indicators("IRQ")
    assert result is None


def test_format_empty_db_returns_empty_string(temp_db):
    assert format_gdelt_for_prompt("IRQ") == ""


def test_db_roundtrip_and_format(temp_db):
    from pythia.gdelt import _get_connection, _store_day

    con = _get_connection()
    try:
        # Insert 10 days of synthetic data for IRQ
        for i in range(10):
            d = date(2026, 3, 10 + i)
            indicators = {
                "IRQ": {
                    "total_events": 100,
                    "material_conflict_events": 30 + i,
                    "verbal_conflict_events": 20,
                    "tier1_events": 10 + i,
                    "tier2_events": 5,
                    "tier3_events": 8,
                    "avg_goldstein": -4.5,
                    "avg_tone_conflict": -6.0,
                    "top_codes_json": json.dumps([["190", 5], ["182", 2]]),
                }
            }
            _store_day(con, d, indicators)
    finally:
        con.close()

    data = load_gdelt_conflict_indicators("IRQ", lookback_days=365)
    assert data is not None
    assert data["iso3"] == "IRQ"
    assert data["days_covered"] == 10
    assert data["total_events"] == 1000
    # Material conflict share > 0
    assert data["material_conflict_share_30d"] is not None
    assert 0.0 <= data["material_conflict_share_30d"] <= 1.0
    # Goldstein average should be around -4.5
    assert data["avg_goldstein_30d"] == pytest.approx(-4.5)

    text = format_gdelt_for_prompt("IRQ", country_name="Iraq")
    assert "GDELT Media Conflict Indicators" in text
    assert "Iraq" in text
    assert "GDELT indicators are media-derived" in text
    assert "Material Conflict share" in text
