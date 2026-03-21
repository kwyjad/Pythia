# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for binary event forecast pipeline support (Prompt 2.3)."""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import duckdb
import pytest

from forecaster.binary_prompts import parse_binary_response


# ---- Binary question detection ----

def test_binary_question_detection():
    """EVENT_OCCURRENCE metric should be detected as binary."""
    metric = "EVENT_OCCURRENCE"
    is_binary = metric.upper() == "EVENT_OCCURRENCE"
    assert is_binary

    for non_binary in ("PA", "FATALITIES", "PHASE3PLUS_IN_NEED"):
        assert non_binary.upper() != "EVENT_OCCURRENCE"


# ---- Binary response parsing ----

def test_parse_binary_response_valid():
    """Parse well-formed binary response JSON."""
    raw = json.dumps({
        "months": {
            "2026-04": {"prior": 0.12, "posterior": 0.08, "reasoning": "low season"},
            "2026-05": {"prior": 0.15, "posterior": 0.22, "reasoning": "onset"},
            "2026-06": {"prior": 0.20, "posterior": 0.35, "reasoning": "peak"},
            "2026-07": {"prior": 0.20, "posterior": 0.30, "reasoning": "ongoing"},
            "2026-08": {"prior": 0.15, "posterior": 0.18, "reasoning": "waning"},
            "2026-09": {"prior": 0.10, "posterior": 0.10, "reasoning": "end season"},
        }
    })
    result = parse_binary_response(raw)
    assert len(result) == 6
    assert result["2026-04"] == 0.08
    assert result["2026-06"] == 0.35


def test_parse_binary_response_malformed():
    """Malformed JSON should return empty dict."""
    assert parse_binary_response("not json") == {}
    assert parse_binary_response("") == {}


def test_parse_binary_response_clamping():
    """Out-of-range probabilities should be clamped."""
    raw = json.dumps({
        "months": {
            "2026-04": {"posterior": 0.0},
            "2026-05": {"posterior": 1.0},
        }
    })
    result = parse_binary_response(raw)
    assert result["2026-04"] == 0.01
    assert result["2026-05"] == 0.99


# ---- Binary ensemble aggregation ----

def test_binary_ensemble_aggregation():
    """Weighted average of binary probabilities."""
    model_probs = [
        {"2026-04": 0.10, "2026-05": 0.20},
        {"2026-04": 0.30, "2026-05": 0.40},
    ]
    all_months = set()
    for mp in model_probs:
        all_months.update(mp.keys())
    aggregated = {}
    for month in sorted(all_months):
        probs = [mp[month] for mp in model_probs if month in mp]
        aggregated[month] = sum(probs) / len(probs)

    assert abs(aggregated["2026-04"] - 0.20) < 1e-6
    assert abs(aggregated["2026-05"] - 0.30) < 1e-6


# ---- Binary storage convention ----

def test_binary_storage_convention():
    """bucket_1 = P(yes), bucket_2 = P(no), rest = 0."""
    p_yes = 0.35
    probs = [p_yes, 1.0 - p_yes, 0.0, 0.0, 0.0]
    assert len(probs) == 5
    assert abs(probs[0] - 0.35) < 1e-6
    assert abs(probs[1] - 0.65) < 1e-6
    assert probs[2] == 0.0
    assert probs[3] == 0.0
    assert probs[4] == 0.0
    assert abs(sum(probs) - 1.0) < 1e-6


# ---- Resolution source inference ----

def test_infer_resolution_source_binary():
    """EVENT_OCCURRENCE should infer GDACS as resolution source."""
    from forecaster.cli import _infer_resolution_source
    assert _infer_resolution_source("FL", "EVENT_OCCURRENCE") == "GDACS"
    assert _infer_resolution_source("DR", "EVENT_OCCURRENCE") == "GDACS"
    assert _infer_resolution_source("TC", "EVENT_OCCURRENCE") == "GDACS"


def test_infer_resolution_source_spd_unchanged():
    """SPD resolution sources should be unchanged."""
    from forecaster.cli import _infer_resolution_source
    assert _infer_resolution_source("ACE", "FATALITIES") == "ACLED"
    assert _infer_resolution_source("ACE", "PA") == "IDMC"
    assert _infer_resolution_source("FL", "PA") == "IFRC"


# ---- Integration: write + read binary outputs ----

def test_write_binary_outputs_roundtrip(tmp_path):
    """Write binary outputs to DB and verify bucket convention."""
    db_path = str(tmp_path / "test.duckdb")
    con = duckdb.connect(db_path)

    # Create minimal tables
    from pythia.db.schema import ensure_schema
    ensure_schema(con)
    con.close()

    question_row = {
        "question_id": "ETH_DR_EVENT_OCCURRENCE_2026-04",
        "iso3": "ETH",
        "hazard_code": "DR",
        "metric": "EVENT_OCCURRENCE",
        "wording": "test",
        "target_month": "2026-09",
        "window_start_date": date(2026, 4, 1),
        "window_end_date": date(2026, 9, 30),
        "hs_run_id": "test",
        "track": 1,
    }

    month_probs = {
        "2026-04": 0.15,
        "2026-05": 0.25,
        "2026-06": 0.35,
        "2026-07": 0.30,
        "2026-08": 0.20,
        "2026-09": 0.10,
    }

    # Patch connect to use our test DB
    with patch("forecaster.cli.connect") as mock_connect:
        mock_connect.return_value = duckdb.connect(db_path)
        from forecaster.cli import _write_binary_outputs
        _write_binary_outputs(
            "test-run",
            question_row,
            month_probs,
            resolution_source="GDACS",
            usage={},
            model_name="ensemble_mean_v2",
        )

    # Verify storage
    con = duckdb.connect(db_path)
    try:
        # Check forecasts_raw
        raw_rows = con.execute(
            """SELECT month_index, bucket_index, probability
               FROM forecasts_raw
               WHERE question_id = 'ETH_DR_EVENT_OCCURRENCE_2026-04'
               ORDER BY month_index, bucket_index"""
        ).fetchall()
        assert len(raw_rows) == 30  # 6 months × 5 buckets

        # Check bucket_1 for month_1 = 0.15 (April)
        month1_bucket1 = [r for r in raw_rows if r[0] == 1 and r[1] == 1]
        assert len(month1_bucket1) == 1
        assert abs(month1_bucket1[0][2] - 0.15) < 1e-6

        # Check bucket_2 for month_1 = 0.85 (1 - 0.15)
        month1_bucket2 = [r for r in raw_rows if r[0] == 1 and r[1] == 2]
        assert len(month1_bucket2) == 1
        assert abs(month1_bucket2[0][2] - 0.85) < 1e-6

        # Check bucket_3 for month_1 = 0.0
        month1_bucket3 = [r for r in raw_rows if r[0] == 1 and r[1] == 3]
        assert len(month1_bucket3) == 1
        assert month1_bucket3[0][2] == 0.0

        # Check forecasts_ensemble
        ens_rows = con.execute(
            """SELECT month_index, bucket_index, probability
               FROM forecasts_ensemble
               WHERE question_id = 'ETH_DR_EVENT_OCCURRENCE_2026-04'
               ORDER BY month_index, bucket_index"""
        ).fetchall()
        assert len(ens_rows) == 30  # 6 months × 5 buckets
    finally:
        con.close()
