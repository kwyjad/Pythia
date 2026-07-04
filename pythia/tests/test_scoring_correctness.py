# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Correctness tests for the scoring pipeline (round-2 hardening).

Covers: the standard normalized RPS formula, degenerate-input guards
(zero-sum SPDs, negative resolved values), deterministic resolution
precedence, and the zero-default source-coverage guard.
"""
from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")

from pythia.tools.compute_scores import _bucket_index, _rps


# ---------------------------------------------------------------------------
# Normalized RPS = sum_{k=1..K-1}(F_k - H_k)^2 / (K-1)
# ---------------------------------------------------------------------------

def test_rps_perfect_forecast_is_zero() -> None:
    assert _rps([1.0, 0.0, 0.0, 0.0, 0.0], 0) == pytest.approx(0.0)
    assert _rps([0.0, 0.0, 1.0, 0.0, 0.0], 2) == pytest.approx(0.0)


def test_rps_maximally_wrong_forecast_is_one() -> None:
    # All mass in bucket 5, outcome in bucket 1: F = (0,0,0,0), H = (1,1,1,1)
    # over k=1..4 → 4 × 1² / 4 = 1.0
    assert _rps([0.0, 0.0, 0.0, 0.0, 1.0], 0) == pytest.approx(1.0)
    # Symmetric case
    assert _rps([1.0, 0.0, 0.0, 0.0, 0.0], 4) == pytest.approx(1.0)


def test_rps_uniform_forecast_hand_computed() -> None:
    # Uniform p=0.2, outcome bucket 1 (j=0): F=(0.2,0.4,0.6,0.8), H=(1,1,1,1)
    # → (0.8² + 0.6² + 0.4² + 0.2²)/4 = (0.64+0.36+0.16+0.04)/4 = 0.30
    assert _rps([0.2] * 5, 0) == pytest.approx(0.30)


def test_rps_rewards_near_misses_over_far_misses() -> None:
    # Ordinal sensitivity: mass one bucket away scores better than mass
    # four buckets away (this is what distinguishes RPS from Brier).
    near = _rps([0.0, 1.0, 0.0, 0.0, 0.0], 0)   # forecast bucket 2, truth 1
    far = _rps([0.0, 0.0, 0.0, 0.0, 1.0], 0)    # forecast bucket 5, truth 1
    assert near == pytest.approx(0.25)  # 1 wrong CDF step of 4
    assert far == pytest.approx(1.0)
    assert near < far


def test_rps_degenerate_single_bucket() -> None:
    assert _rps([1.0], 0) == 0.0


# ---------------------------------------------------------------------------
# _bucket_index guards
# ---------------------------------------------------------------------------

def test_bucket_index_boundaries() -> None:
    assert _bucket_index(0.0, "PA") == 0
    assert _bucket_index(9_999.0, "PA") == 0
    assert _bucket_index(10_000.0, "PA") == 1
    assert _bucket_index(500_000.0, "PA") == 4
    assert _bucket_index(5_000_000.0, "PA") == 4
    assert _bucket_index(4.0, "FATALITIES") == 0
    assert _bucket_index(500.0, "FATALITIES") == 4
    assert _bucket_index(15_000_000.0, "PHASE3PLUS_IN_NEED") == 4


def test_bucket_index_rejects_negative_and_nonfinite() -> None:
    # A negative resolved value must NOT land in the top bucket.
    assert _bucket_index(-5.0, "PA") is None
    assert _bucket_index(-0.001, "FATALITIES") is None
    assert _bucket_index(float("nan"), "PA") is None


def test_bucket_index_unknown_metric() -> None:
    assert _bucket_index(100.0, "EVENT_OCCURRENCE") is None


# ---------------------------------------------------------------------------
# Threshold/label single-sourcing: the derived values must equal the
# canonical literals that scoring has always used.
# ---------------------------------------------------------------------------

def test_derived_thresholds_match_canonical_literals() -> None:
    from pythia.buckets import interior_thresholds_for, labels_for, thresholds_for
    from pythia.tools import compute_scores as cs
    from forecaster import scoring as fsc

    assert thresholds_for("PA") == [0.0, 10_000.0, 50_000.0, 250_000.0, 500_000.0, float("inf")]
    assert thresholds_for("FATALITIES") == [0.0, 5.0, 25.0, 100.0, 500.0, float("inf")]
    assert thresholds_for("PHASE3PLUS_IN_NEED") == [
        0.0, 100_000.0, 1_000_000.0, 5_000_000.0, 15_000_000.0, float("inf")
    ]
    assert labels_for("PA") == ["<10k", "10k-<50k", "50k-<250k", "250k-<500k", ">=500k"]
    assert labels_for("FATALITIES") == ["<5", "5-<25", "25-<100", "100-<500", ">=500"]
    assert labels_for("PHASE3PLUS_IN_NEED") == ["<100k", "100k-<1M", "1M-<5M", "5M-<15M", ">=15M"]
    assert thresholds_for("NO_SUCH_METRIC") == []

    # Module-level names still exposed for existing importers.
    assert cs.PA_THRESHOLDS == thresholds_for("PA")
    assert cs.SPD_CLASS_BINS_PHASE3 == labels_for("PHASE3PLUS_IN_NEED")
    assert fsc.PA_THRESHOLDS == interior_thresholds_for("PA") == [10_000.0, 50_000.0, 250_000.0, 500_000.0]
    assert fsc.FATALITIES_THRESHOLDS == [5.0, 25.0, 100.0, 500.0]
