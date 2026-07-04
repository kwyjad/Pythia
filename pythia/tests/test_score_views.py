# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for pythia.tools.score_views — point-to-SPD conversion and scoring."""

from __future__ import annotations

import math

import pytest

from pythia.tools.score_views import (
    LOGNORMAL_SIGMA,
    MIN_POINT_FOR_LOGNORMAL,
    point_to_spd_fatalities,
)


class TestPointToSpdFatalities:
    """Tests for the log-normal point-to-SPD conversion."""

    def test_sums_to_one_typical(self):
        """SPD sums to 1.0 for typical forecasts."""
        for pf in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]:
            spd = point_to_spd_fatalities(pf)
            assert abs(sum(spd) - 1.0) < 1e-9, f"SPD for pf={pf} sums to {sum(spd)}"

    def test_sums_to_one_various_sigmas(self):
        """SPD sums to 1.0 for different sigma values."""
        for sigma in [0.3, 0.5, 1.0, 1.5, 2.0, 2.5]:
            spd = point_to_spd_fatalities(50.0, sigma=sigma)
            assert abs(sum(spd) - 1.0) < 1e-9, f"SPD for sigma={sigma} sums to {sum(spd)}"

    def test_bucket_count_matches_canonical(self):
        """SPD always has exactly K elements (K from pythia.buckets)."""
        from pythia.buckets import n_buckets_for

        k = n_buckets_for("FATALITIES")
        for pf in [0.0, 0.1, 10.0, 500.0, 10000.0]:
            spd = point_to_spd_fatalities(pf)
            assert len(spd) == k, f"Expected {k} buckets, got {len(spd)} for pf={pf}"

    def test_all_probabilities_positive(self):
        """All bucket probabilities are strictly positive."""
        for pf in [0.0, 0.1, 10.0, 500.0, 10000.0]:
            spd = point_to_spd_fatalities(pf)
            for i, p in enumerate(spd):
                assert p > 0, f"Bucket {i} has zero probability for pf={pf}"

    def test_near_zero_zero_bucket_dominates(self):
        """Near-zero forecasts should have the "0" bucket dominating."""
        spd = point_to_spd_fatalities(0.1)
        assert spd[0] > 0.80, f"Expected '0' bucket > 80%, got {spd[0]*100:.1f}%"
        for i in range(1, len(spd)):
            assert spd[i] <= spd[i - 1] + 1e-12, (
                "Near-zero should be (non-strictly) decreasing across buckets"
            )

    def test_zero_forecast(self):
        """Zero forecast should behave like near-zero."""
        spd = point_to_spd_fatalities(0.0)
        assert spd[0] > 0.80
        assert abs(sum(spd) - 1.0) < 1e-9

    def test_negative_forecast_clamped(self):
        """Negative forecast should be clamped to zero."""
        spd = point_to_spd_fatalities(-5.0)
        assert spd[0] > 0.80
        assert abs(sum(spd) - 1.0) < 1e-9

    def test_low_forecast_bottom_buckets_favored(self):
        """A forecast of 2.0 should favor the 1-<5 bucket over the top bucket."""
        spd = point_to_spd_fatalities(2.0)
        assert spd[1] > spd[-1], (
            "1-<5 bucket should be higher than >=1000 for low forecast"
        )

    def test_mid_forecast_spread(self):
        """A forecast of 50 should spread across middle buckets."""
        spd = point_to_spd_fatalities(50.0)
        # With sigma=1.0, forecast of 50 should have meaningful mass in the
        # 5-<25, 25-<100, and 100-<500 buckets (indices 2-4).
        assert spd[2] > 0.05, "5-<25 bucket should have some mass"
        assert spd[3] > 0.05, "25-<100 bucket should have some mass"
        assert spd[4] > 0.05, "100-<500 bucket should have some mass"

    def test_high_forecast_top_bucket_dominates(self):
        """A very high forecast (1000+) should favor the >=1000 bucket."""
        spd = point_to_spd_fatalities(1000.0)
        assert spd[-1] > spd[0], "Top bucket should dominate the '0' bucket"

    def test_very_high_forecast(self):
        """Extreme forecast (10000) should heavily weight the >=1000 bucket."""
        spd = point_to_spd_fatalities(10000.0)
        assert spd[-1] > 0.3, f"Expected >=1000 bucket > 30%, got {spd[-1]*100:.1f}%"
        assert abs(sum(spd) - 1.0) < 1e-9

    def test_monotonicity_with_increasing_forecast(self):
        """As forecast increases, top-bucket probability should generally increase."""
        prev_top = 0.0
        for pf in [1.0, 10.0, 50.0, 200.0, 1000.0]:
            spd = point_to_spd_fatalities(pf)
            # Top bucket should generally increase with forecast
            # (not strictly monotonic at every step due to log-normal shape,
            # but should increase across this range)
            if pf >= 50.0:
                assert spd[-1] >= prev_top - 0.01, (
                    f"Top bucket should not decrease significantly: "
                    f"pf={pf}, top={spd[-1]:.4f}, prev={prev_top:.4f}"
                )
            prev_top = spd[-1]

    def test_narrow_sigma_more_concentrated(self):
        """Narrower sigma should produce more concentrated SPDs."""
        narrow = point_to_spd_fatalities(50.0, sigma=0.3)
        wide = point_to_spd_fatalities(50.0, sigma=2.0)
        # Max probability should be higher with narrow sigma
        assert max(narrow) > max(wide), (
            "Narrow sigma should produce more concentrated distribution"
        )

    def test_threshold_boundary_below_min(self):
        """Forecast just below MIN_POINT_FOR_LOGNORMAL uses spike distribution."""
        spd = point_to_spd_fatalities(MIN_POINT_FOR_LOGNORMAL - 0.01)
        assert spd[0] == 0.90, "Below threshold should use spike distribution"

    def test_threshold_boundary_at_min(self):
        """Forecast at MIN_POINT_FOR_LOGNORMAL uses log-normal."""
        spd = point_to_spd_fatalities(MIN_POINT_FOR_LOGNORMAL)
        # Should use log-normal, not spike
        assert spd[0] != 0.90, "At threshold should use log-normal, not spike"
        assert abs(sum(spd) - 1.0) < 1e-9
