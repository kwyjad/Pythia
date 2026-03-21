# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for bucket definitions including DR Phase 3+ (PHASE3PLUS_IN_NEED)."""

from __future__ import annotations

from pythia.buckets import (
    BUCKET_SPECS,
    DR_PHASE3_BUCKETS,
    FATALITIES_BUCKETS,
    PA_BUCKETS,
    BucketSpec,
    get_bucket_specs,
)


class TestGetBucketSpecs:
    def test_pa_returns_5_buckets(self):
        specs = get_bucket_specs("PA")
        assert len(specs) == 5

    def test_fatalities_returns_5_buckets(self):
        specs = get_bucket_specs("FATALITIES")
        assert len(specs) == 5

    def test_phase3plus_in_need_returns_5_buckets(self):
        specs = get_bucket_specs("PHASE3PLUS_IN_NEED")
        assert len(specs) == 5

    def test_unknown_metric_returns_empty(self):
        specs = get_bucket_specs("NONEXISTENT")
        assert len(specs) == 0

    def test_case_insensitive(self):
        specs = get_bucket_specs("phase3plus_in_need")
        assert len(specs) == 5


class TestDrPhase3Buckets:
    def test_bucket_count(self):
        assert len(DR_PHASE3_BUCKETS) == 5

    def test_bucket_indices_sequential(self):
        indices = [b.idx for b in DR_PHASE3_BUCKETS]
        assert indices == [1, 2, 3, 4, 5]

    def test_bucket_boundaries(self):
        specs = DR_PHASE3_BUCKETS
        assert specs[0].lower == 0.0
        assert specs[0].upper == 100_000.0
        assert specs[1].lower == 100_000.0
        assert specs[1].upper == 1_000_000.0
        assert specs[2].lower == 1_000_000.0
        assert specs[2].upper == 5_000_000.0
        assert specs[3].lower == 5_000_000.0
        assert specs[3].upper == 15_000_000.0
        assert specs[4].lower == 15_000_000.0
        assert specs[4].upper is None  # open-ended

    def test_centroids_within_boundaries(self):
        for spec in DR_PHASE3_BUCKETS:
            assert spec.centroid >= spec.lower
            if spec.upper is not None:
                assert spec.centroid < spec.upper

    def test_boundaries_contiguous(self):
        """Each bucket's lower bound equals the previous bucket's upper bound."""
        for i in range(1, len(DR_PHASE3_BUCKETS)):
            prev = DR_PHASE3_BUCKETS[i - 1]
            curr = DR_PHASE3_BUCKETS[i]
            assert prev.upper == curr.lower, (
                f"Gap between bucket {prev.idx} upper={prev.upper} "
                f"and bucket {curr.idx} lower={curr.lower}"
            )

    def test_labels(self):
        labels = [b.label for b in DR_PHASE3_BUCKETS]
        assert labels == ["<100k", "100k-<1M", "1M-<5M", "5M-<15M", ">=15M"]


class TestBucketSpecsRegistry:
    def test_phase3plus_in_registry(self):
        assert "PHASE3PLUS_IN_NEED" in BUCKET_SPECS

    def test_pa_in_registry(self):
        assert "PA" in BUCKET_SPECS

    def test_fatalities_in_registry(self):
        assert "FATALITIES" in BUCKET_SPECS

    def test_registry_has_three_metrics(self):
        assert len(BUCKET_SPECS) == 3
