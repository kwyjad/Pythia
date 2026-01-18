import pytest

pytest.importorskip("duckdb")

from horizon_scanner.horizon_scanner import _select_tail_pack_hazards


def test_hs_tail_pack_selection_limits_and_ordering():
    triage = {
        "country": "USA",
        "hazards": {
            "ACE": {
                "triage_score": 0.7,
                "regime_change": {
                    "likelihood": 0.8,
                    "magnitude": 0.7,
                    "direction": "up",
                    "window": "month_1-2",
                },
            },
            "DI": {
                "triage_score": 0.4,
                "regime_change": {
                    "likelihood": 0.65,
                    "magnitude": 0.55,
                    "direction": "down",
                    "window": "month_3-4",
                },
            },
            "DR": {
                "triage_score": 0.9,
                "regime_change": {
                    "likelihood": 0.6,
                    "magnitude": 0.5,
                    "direction": "down",
                    "window": "month_5-6",
                },
            },
            "FL": {
                "triage_score": 0.2,
                "regime_change": {
                    "likelihood": 0.2,
                    "magnitude": 0.1,
                    "direction": "unclear",
                    "window": "month_1-2",
                },
            },
        },
    }

    selected = _select_tail_pack_hazards(triage, expected_hazards=["ACE", "DI", "DR", "FL", "HW", "TC"])

    assert len(selected) <= 2
    assert [item["hazard_code"] for item in selected] == ["ACE", "DI"]
    assert all(item["rc_level"] >= 2 for item in selected)
