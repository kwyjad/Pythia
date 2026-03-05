# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for the IPC Food Security Phase Classification connector."""

from __future__ import annotations

import os

import pytest

from pythia.ipc_phases import (
    _compute_trend,
    _is_stale,
    _iso3_to_iso2,
    _safe_int,
    format_ipc_for_prompt,
    format_ipc_for_spd,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ipc_data(**overrides) -> dict:
    """Build a valid sample IPC data dict for testing."""
    data = {
        "iso3": "SDN",
        "analysis_id": "12345",
        "analysis_date": "2026-01-15",
        "analysis_period": "Oct 2025 - Feb 2026",
        "projection_period": "Mar 2026 - Jul 2026",
        "total_population": 45000000,
        "current": {
            "phase1": 15000000,
            "phase2": 12000000,
            "phase3": 8000000,
            "phase4": 5000000,
            "phase5": 500000,
            "phase3plus": 13500000,
            "phase3plus_pct": 30.0,
        },
        "projected": {
            "phase3": 9000000,
            "phase4": 7000000,
            "phase5": 1000000,
            "phase3plus": 17000000,
            "phase3plus_pct": 37.8,
            "projection_period": "Mar 2026 - Jul 2026",
        },
        "trend": "worsening",
        "areas_in_phase5": ["North Darfur", "West Darfur"],
        "stale": False,
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# Trend computation
# ---------------------------------------------------------------------------


class TestComputeTrend:
    def test_worsening(self):
        assert _compute_trend(10000, 15000) == "worsening"

    def test_improving(self):
        assert _compute_trend(15000, 10000) == "improving"

    def test_stable_equal(self):
        assert _compute_trend(10000, 10000) == "stable"

    def test_stable_no_projection(self):
        assert _compute_trend(10000, None) == "stable"

    def test_stable_no_current(self):
        assert _compute_trend(None, 10000) == "stable"

    def test_stable_both_none(self):
        assert _compute_trend(None, None) == "stable"


# ---------------------------------------------------------------------------
# Staleness detection
# ---------------------------------------------------------------------------


class TestStaleness:
    def test_recent_not_stale(self):
        assert _is_stale("2026-01-15") is False

    def test_old_is_stale(self):
        assert _is_stale("2020-01-15") is True

    def test_empty_string_not_stale(self):
        assert _is_stale("") is False

    def test_none_not_stale(self):
        assert _is_stale(None) is False

    def test_invalid_date_not_stale(self):
        assert _is_stale("not-a-date") is False


# ---------------------------------------------------------------------------
# ISO3 → ISO2 mapping
# ---------------------------------------------------------------------------


class TestIso3ToIso2:
    def test_known_mapping(self):
        assert _iso3_to_iso2("SDN") == "SD"
        assert _iso3_to_iso2("SOM") == "SO"
        assert _iso3_to_iso2("YEM") == "YE"
        assert _iso3_to_iso2("SSD") == "SS"

    def test_case_insensitive(self):
        assert _iso3_to_iso2("sdn") == "SD"

    def test_unknown_returns_none(self):
        assert _iso3_to_iso2("XYZ") is None
        assert _iso3_to_iso2("USA") is None


# ---------------------------------------------------------------------------
# Prompt formatter — full version
# ---------------------------------------------------------------------------


class TestFormatIpcForPrompt:
    def test_none_returns_empty(self):
        assert format_ipc_for_prompt(None) == ""

    def test_empty_dict_returns_empty(self):
        assert format_ipc_for_prompt({}) == ""

    def test_full_data_includes_all_sections(self):
        data = _make_ipc_data()
        result = format_ipc_for_prompt(data)
        assert "IPC FOOD SECURITY CLASSIFICATION (SDN):" in result
        assert "Phase 3 (Crisis):" in result
        assert "Phase 4 (Emergency):" in result
        assert "Phase 5 (Famine):" in result
        assert "Total Phase 3+:" in result
        assert "13,500,000" in result
        assert "30.0%" in result
        assert "Projected situation" in result
        assert "17,000,000" in result
        assert "worsening" in result
        assert "FAMINE (Phase 5) detected in:" in result
        assert "North Darfur" in result
        assert "West Darfur" in result
        assert "calibration anchors" in result

    def test_no_projection_omits_projected_section(self):
        data = _make_ipc_data(projected=None, trend="stable")
        result = format_ipc_for_prompt(data)
        assert "Projected situation" not in result
        assert "Current situation:" in result

    def test_phase5_areas_highlighted(self):
        data = _make_ipc_data(areas_in_phase5=["Zamzam Camp"])
        result = format_ipc_for_prompt(data)
        assert "FAMINE (Phase 5) detected in: Zamzam Camp" in result

    def test_no_phase5_areas_omits_famine_line(self):
        data = _make_ipc_data(areas_in_phase5=[])
        result = format_ipc_for_prompt(data)
        assert "FAMINE (Phase 5) detected in:" not in result

    def test_stale_warning_included(self):
        data = _make_ipc_data(stale=True)
        result = format_ipc_for_prompt(data)
        assert "[WARNING: IPC ANALYSIS >6 MONTHS OLD]" in result

    def test_no_stale_warning_when_fresh(self):
        data = _make_ipc_data(stale=False)
        result = format_ipc_for_prompt(data)
        assert "WARNING" not in result

    def test_delta_calculation(self):
        data = _make_ipc_data()
        result = format_ipc_for_prompt(data)
        # 17,000,000 - 13,500,000 = +3,500,000
        assert "+3,500,000" in result


# ---------------------------------------------------------------------------
# SPD formatter — compact version
# ---------------------------------------------------------------------------


class TestFormatIpcForSpd:
    def test_none_returns_empty(self):
        assert format_ipc_for_spd(None) == ""

    def test_empty_dict_returns_empty(self):
        assert format_ipc_for_spd({}) == ""

    def test_compact_format(self):
        data = _make_ipc_data()
        result = format_ipc_for_spd(data)
        assert "IPC PHASES (SDN):" in result
        assert "Current Phase 3+: 13,500,000" in result
        assert "Projected: 17,000,000" in result
        assert "[worsening]" in result

    def test_calibration_check_present(self):
        data = _make_ipc_data()
        result = format_ipc_for_spd(data)
        assert "CALIBRATION CHECK:" in result
        assert "17,000,000" in result
        assert "reconcile the discrepancy" in result

    def test_no_calibration_check_without_projection(self):
        data = _make_ipc_data(projected=None, trend="stable")
        result = format_ipc_for_spd(data)
        assert "CALIBRATION CHECK:" not in result

    def test_stale_warning(self):
        data = _make_ipc_data(stale=True)
        result = format_ipc_for_spd(data)
        assert "[WARNING: IPC ANALYSIS >6 MONTHS OLD]" in result

    def test_phase5_areas_in_spd(self):
        data = _make_ipc_data(areas_in_phase5=["Zamzam Camp", "El Fasher"])
        result = format_ipc_for_spd(data)
        assert "FAMINE (Phase 5) areas:" in result
        assert "Zamzam Camp" in result

    def test_no_phase5_areas_omitted(self):
        data = _make_ipc_data(areas_in_phase5=[])
        result = format_ipc_for_spd(data)
        assert "FAMINE (Phase 5) areas:" not in result


# ---------------------------------------------------------------------------
# Safe int helper
# ---------------------------------------------------------------------------


class TestSafeInt:
    def test_none(self):
        assert _safe_int(None) == 0

    def test_string_number(self):
        assert _safe_int("42") == 42

    def test_invalid(self):
        assert _safe_int("abc") == 0

    def test_normal_int(self):
        assert _safe_int(100) == 100

    def test_float(self):
        assert _safe_int(3.7) == 3


# ---------------------------------------------------------------------------
# Live integration (conditional)
# ---------------------------------------------------------------------------

_has_ipc_key = bool(os.environ.get("IPC_API_KEY"))


@pytest.mark.allow_network
@pytest.mark.skipif(not _has_ipc_key, reason="IPC_API_KEY not set")
def test_live_fetch_sdn():
    """Smoke test: fetch IPC phases for Sudan."""
    from pythia.ipc_phases import fetch_ipc_phases

    result = fetch_ipc_phases("SDN")
    # Sudan almost always has IPC data
    assert result is None or isinstance(result, dict)
    if result:
        assert result["iso3"] == "SDN"
        assert "current" in result
        assert "phase3plus" in result["current"]
        assert result["current"]["phase3plus"] >= 0
        assert result["trend"] in ("worsening", "improving", "stable")
