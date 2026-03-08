# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for calibration system improvements (Prompt 9).

Covers:
  - Adaptive softmax temperature (Part B)
  - JS divergence helper (Part F)
"""

from __future__ import annotations

import math


def test_adaptive_softmax_temp():
    """Verify adaptive softmax temperature at key N values."""
    from pythia.tools.compute_calibration_pythia import _adaptive_softmax_temp

    t20 = _adaptive_softmax_temp(20)
    t50 = _adaptive_softmax_temp(50)
    t200 = _adaptive_softmax_temp(200)

    assert 0.35 < t20 < 0.45, f"Expected ~0.39 at n=20, got {t20}"
    assert 0.25 < t50 < 0.35, f"Expected ~0.30 at n=50, got {t50}"
    assert 0.10 < t200 < 0.15, f"Expected ~0.12 at n=200, got {t200}"

    # Temperature should be monotonically decreasing
    assert t20 > t50 > t200


def test_adaptive_softmax_weights_more_uniform_at_low_n():
    """At low N, weights should be closer to uniform for same Brier spread."""
    from pythia.tools.compute_calibration_pythia import _adaptive_softmax_temp

    # Simulate two models with fixed Brier scores
    brier_a = 0.20
    brier_b = 0.30

    def _softmax_weights(temp: float) -> tuple[float, float]:
        raw_a = -brier_a
        raw_b = -brier_b
        max_raw = max(raw_a, raw_b)
        x_a = math.exp((raw_a - max_raw) / temp)
        x_b = math.exp((raw_b - max_raw) / temp)
        denom = x_a + x_b
        return x_a / denom, x_b / denom

    t_low = _adaptive_softmax_temp(20)
    t_high = _adaptive_softmax_temp(200)

    w_a_low, w_b_low = _softmax_weights(t_low)
    w_a_high, w_b_high = _softmax_weights(t_high)

    # At low N, weights should be closer to 0.5/0.5 (more uniform)
    spread_low = abs(w_a_low - w_b_low)
    spread_high = abs(w_a_high - w_b_high)
    assert spread_low < spread_high, (
        f"Weights at low N (spread={spread_low:.3f}) should be more "
        f"uniform than at high N (spread={spread_high:.3f})"
    )


def test_js_divergence_identical():
    """JSD of identical distributions should be 0."""
    import numpy as np
    from pythia.tools.generate_calibration_advice import _js_divergence

    p = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    jsd = _js_divergence(p, p)
    assert jsd < 1e-10, f"JSD of identical distributions should be ~0, got {jsd}"


def test_js_divergence_different():
    """JSD of different distributions should be positive."""
    import numpy as np
    from pythia.tools.generate_calibration_advice import _js_divergence

    p = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
    q = np.array([0.05, 0.05, 0.1, 0.3, 0.5])
    jsd = _js_divergence(p, q)
    assert jsd > 0.1, f"JSD of very different distributions should be > 0.1, got {jsd}"
    # JSD is bounded by ln(2) ~ 0.693
    assert jsd <= math.log(2) + 1e-10, f"JSD should be <= ln(2), got {jsd}"
