# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Regression tests for forecast-month anchoring and window-anchored writes.

Guards against the +5-month forecast shift (runs 2026-03-21 → 2026-07-01):
prompts, month-offset expansion, and every writer must anchor month 1 at
the question's window_start_date — never at the questions-table
target_month, which is the 6th (last) window month.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest

duckdb = pytest.importorskip("duckdb")

import forecaster.cli as cli
import forecaster.prompts as prompts
from forecaster.aggregate import aggregate_spd_v2_mean
from forecaster.providers import ModelSpec
from pythia.db import schema as db_schema


# ---------------------------------------------------------------------------
# _anchor_month_for_question
# ---------------------------------------------------------------------------

def test_anchor_prefers_window_start_date() -> None:
    rec = {"window_start_date": date(2026, 7, 1), "target_month": "2026-12"}
    assert cli._anchor_month_for_question(rec) == "2026-07"


def test_anchor_accepts_iso_string_window_start() -> None:
    rec = {"window_start_date": "2026-07-01"}
    assert cli._anchor_month_for_question(rec) == "2026-07"


def test_anchor_falls_back_to_target_month_minus_five() -> None:
    # target_month is the 6th window month → anchor = target - 5.
    rec = {"window_start_date": None, "target_month": "2026-12"}
    assert cli._anchor_month_for_question(rec) == "2026-07"


def test_anchor_none_when_no_dates() -> None:
    assert cli._anchor_month_for_question({}) is None


# ---------------------------------------------------------------------------
# _month_index_for_label
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "label,anchor,expected",
    [
        ("2026-07", "2026-07", 1),
        ("2026-12", "2026-07", 6),
        ("2027-01", "2026-07", None),  # past the window
        ("2026-06", "2026-07", None),  # before the window
        ("month_1", "2026-07", 1),
        ("month_6", None, 6),          # month_N needs no anchor
        ("month_7", "2026-07", None),
        ("2026-08-15", "2026-07", 2),  # YYYY-MM-DD accepted
        ("garbage", "2026-07", None),
        ("2026-08", None, None),       # calendar label without anchor
    ],
)
def test_month_index_for_label(label: str, anchor: str | None, expected: int | None) -> None:
    assert cli._month_index_for_label(label, anchor) == expected


# ---------------------------------------------------------------------------
# Prompt month keys (the original +5-month shift entry point)
# ---------------------------------------------------------------------------

def test_forecast_month_keys_anchor_at_window_start() -> None:
    keys = prompts._forecast_month_keys_from_question(
        {"window_start_date": date(2026, 7, 1), "target_month": "2026-12"}
    )
    assert keys == ["2026-07", "2026-08", "2026-09", "2026-10", "2026-11", "2026-12"]


def test_forecast_month_keys_target_fallback_shifts_back() -> None:
    keys = prompts._forecast_month_keys_from_question({"target_month": "2026-12"})
    assert keys == ["2026-07", "2026-08", "2026-09", "2026-10", "2026-11", "2026-12"]


def test_forecast_month_keys_year_rollover() -> None:
    keys = prompts._forecast_month_keys_from_question(
        {"window_start_date": "2026-10-01"}
    )
    assert keys == ["2026-10", "2026-11", "2026-12", "2027-01", "2027-02", "2027-03"]


# ---------------------------------------------------------------------------
# Window-anchored binary writes: off-window labels are dropped, in-window
# labels keep their calendar-derived index even when some are missing.
# ---------------------------------------------------------------------------

def _binary_question_row() -> dict:
    return {
        "question_id": "ETH_FL_EVENT_OCCURRENCE_2026-07",
        "iso3": "ETH",
        "hazard_code": "FL",
        "metric": "EVENT_OCCURRENCE",
        "wording": "test",
        "target_month": "2026-12",
        "window_start_date": date(2026, 7, 1),
        "hs_run_id": "test",
    }


def test_write_binary_outputs_drops_off_window_and_keeps_indices(tmp_path) -> None:
    db_path = str(tmp_path / "binary_anchor.duckdb")
    con = duckdb.connect(db_path)
    db_schema.ensure_schema(con)
    con.close()

    # 2026-08 missing, 2027-01 off-window: indices must come from the
    # labels (1, 3) — not positional enumeration (which would give 1, 2, 3).
    month_probs = {"2026-07": 0.10, "2026-09": 0.30, "2027-01": 0.99}

    with patch("forecaster.cli.connect") as mock_connect:
        mock_connect.return_value = duckdb.connect(db_path)
        cli._write_binary_outputs(
            "test-run",
            _binary_question_row(),
            month_probs,
            resolution_source="GDACS",
            usage={},
            model_name="ensemble_mean_v2",
        )

    con = duckdb.connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT month_index, probability FROM forecasts_raw
            WHERE question_id = 'ETH_FL_EVENT_OCCURRENCE_2026-07' AND bucket_index = 1
            ORDER BY month_index
            """
        ).fetchall()
    finally:
        con.close()

    assert [(int(m), round(p, 2)) for m, p in rows] == [(1, 0.10), (3, 0.30)]


def test_write_spd_outputs_maps_calendar_labels_to_window_indices(tmp_path) -> None:
    db_path = str(tmp_path / "spd_anchor.duckdb")
    con = duckdb.connect(db_path)
    db_schema.ensure_schema(con)
    con.close()

    question_row = {
        "question_id": "ETH_ACE_PA_2026-07",
        "hs_run_id": "test",
        "scenario_ids_json": "[]",
        "iso3": "ETH",
        "hazard_code": "ACE",
        "metric": "PA",
        "target_month": "2026-12",
        "window_start_date": "2026-07-01",
        "window_end_date": None,
        "wording": "test",
        "status": "active",
        "pythia_metadata_json": None,
    }
    probs = [0.2, 0.2, 0.2, 0.2, 0.2]
    spd_obj = {
        "spds": {
            "2026-07": {"probs": probs},
            "2026-09": {"probs": probs},   # 2026-08 missing → index 3, not 2
            "2027-02": {"probs": probs},   # off-window → dropped
        }
    }

    with patch("forecaster.cli.connect") as mock_connect:
        mock_connect.return_value = duckdb.connect(db_path)
        cli._write_spd_outputs(
            "test-run",
            question_row,
            spd_obj,
            resolution_source="ACLED",
            usage={},
            model_name="ensemble_mean_v2",
        )

    con = duckdb.connect(db_path)
    try:
        idx = sorted(
            int(r[0])
            for r in con.execute(
                """
                SELECT DISTINCT month_index FROM forecasts_ensemble
                WHERE question_id = 'ETH_ACE_PA_2026-07'
                """
            ).fetchall()
        )
    finally:
        con.close()

    assert idx == [1, 3]


# ---------------------------------------------------------------------------
# Calibration-weighted aggregation
# ---------------------------------------------------------------------------

def test_weighted_mean_shifts_toward_heavier_member() -> None:
    a = {"2026-07": [1.0, 0.0, 0.0, 0.0, 0.0]}
    b = {"2026-07": [0.0, 1.0, 0.0, 0.0, 0.0]}
    out = aggregate_spd_v2_mean([a, b], member_weights=[3.0, 1.0])
    assert abs(out["2026-07"][0] - 0.75) < 1e-9
    # None keeps the legacy unweighted mean
    out = aggregate_spd_v2_mean([a, b], member_weights=None)
    assert abs(out["2026-07"][0] - 0.5) < 1e-9


def test_resolve_member_weights_matches_plain_and_disambiguated_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = [
        ModelSpec(name="Claude", provider="anthropic", model_id="claude-sonnet-4-6", active=True),
        ModelSpec(name="Gemini", provider="google", model_id="gemini-3-flash-preview", active=True),
        ModelSpec(name="Gemini", provider="google", model_id="gemini-3.1-pro-preview", active=True),
    ]
    monkeypatch.setattr(
        cli,
        "_load_calibration_weights_db",
        lambda hz, mt: {
            "Claude": 0.5,
            "Gemini (gemini-3-flash-preview)": 0.3,
            "Gemini (gemini-3.1-pro-preview)": 0.2,
        },
    )
    cli._CALIB_WEIGHTS_CACHE.clear()
    weights_by_key, keys, weight_list = cli._resolve_member_weights(specs, "ACE", "FATALITIES")
    assert keys == [
        "Claude (claude-sonnet-4-6)",
        "Gemini (gemini-3-flash-preview)",
        "Gemini (gemini-3.1-pro-preview)",
    ]
    # Rescaled to mean 1.0: raw (0.5, 0.3, 0.2) → (1.5, 0.9, 0.6)
    assert weight_list is not None
    assert [round(w, 6) for w in weight_list] == [1.5, 0.9, 0.6]
    assert weights_by_key["Claude (claude-sonnet-4-6)"] == pytest.approx(1.5)
    cli._CALIB_WEIGHTS_CACHE.clear()


def test_resolve_member_weights_disabled_or_unmatched(monkeypatch: pytest.MonkeyPatch) -> None:
    specs = [ModelSpec(name="Claude", provider="anthropic", model_id="c1", active=True)]

    monkeypatch.setattr(cli, "_load_calibration_weights_db", lambda hz, mt: {"Other": 0.9})
    cli._CALIB_WEIGHTS_CACHE.clear()
    weights_by_key, _keys, weight_list = cli._resolve_member_weights(specs, "FL", "PA")
    assert weights_by_key is None and weight_list is None

    monkeypatch.setenv("PYTHIA_USE_CALIBRATION_WEIGHTS", "0")
    cli._CALIB_WEIGHTS_CACHE.clear()
    weights_by_key, _keys, weight_list = cli._resolve_member_weights(specs, "FL", "PA")
    assert weights_by_key is None and weight_list is None
    cli._CALIB_WEIGHTS_CACHE.clear()
