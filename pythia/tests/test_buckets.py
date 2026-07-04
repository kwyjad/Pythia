# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for bucket definitions including DR Phase 3+ (PHASE3PLUS_IN_NEED)."""

from __future__ import annotations

import pytest

from pythia.buckets import (
    BUCKET_SPECS,
    DR_PHASE3_BUCKETS,
    FATALITIES_BUCKETS,
    PA_BUCKETS,
    BucketSpec,
    centroids_for,
    get_bucket_specs,
    label_index_map,
    labels_for,
    max_bucket_count,
    n_buckets_for,
    thresholds_for,
)


class TestGetBucketSpecs:
    def test_pa_returns_6_buckets(self):
        specs = get_bucket_specs("PA")
        assert len(specs) == 6

    def test_fatalities_returns_7_buckets(self):
        specs = get_bucket_specs("FATALITIES")
        assert len(specs) == 7

    def test_phase3plus_in_need_returns_6_buckets(self):
        specs = get_bucket_specs("PHASE3PLUS_IN_NEED")
        assert len(specs) == 6

    def test_unknown_metric_returns_empty(self):
        specs = get_bucket_specs("NONEXISTENT")
        assert len(specs) == 0

    def test_case_insensitive(self):
        specs = get_bucket_specs("phase3plus_in_need")
        assert len(specs) == 6


class TestZeroBuckets:
    """Every SPD metric leads with a dedicated '0' bucket (centroid 0)."""

    @pytest.mark.parametrize("metric", ["PA", "FATALITIES", "PHASE3PLUS_IN_NEED"])
    def test_first_bucket_is_zero(self, metric):
        first = get_bucket_specs(metric)[0]
        assert first.label == "0"
        assert first.centroid == 0.0
        assert first.lower == 0.0
        assert first.upper == 1.0

    @pytest.mark.parametrize("metric", ["PA", "FATALITIES", "PHASE3PLUS_IN_NEED"])
    def test_second_bucket_starts_at_one(self, metric):
        second = get_bucket_specs(metric)[1]
        assert second.lower == 1.0

    def test_fatalities_top_bucket_is_1000(self):
        top = get_bucket_specs("FATALITIES")[-1]
        assert top.label == ">=1000"
        assert top.lower == 1_000.0
        assert top.upper is None


class TestDrPhase3Buckets:
    def test_bucket_count(self):
        assert len(DR_PHASE3_BUCKETS) == 6

    def test_bucket_indices_sequential(self):
        indices = [b.idx for b in DR_PHASE3_BUCKETS]
        assert indices == [1, 2, 3, 4, 5, 6]

    def test_bucket_boundaries(self):
        specs = DR_PHASE3_BUCKETS
        assert specs[0].lower == 0.0
        assert specs[0].upper == 1.0
        assert specs[1].lower == 1.0
        assert specs[1].upper == 100_000.0
        assert specs[2].lower == 100_000.0
        assert specs[2].upper == 1_000_000.0
        assert specs[3].lower == 1_000_000.0
        assert specs[3].upper == 5_000_000.0
        assert specs[4].lower == 5_000_000.0
        assert specs[4].upper == 15_000_000.0
        assert specs[5].lower == 15_000_000.0
        assert specs[5].upper is None  # open-ended

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
        assert labels == ["0", "1-<100k", "100k-<1M", "1M-<5M", "5M-<15M", ">=15M"]


class TestAllMetricsStructuralInvariants:
    @pytest.mark.parametrize("specs", [PA_BUCKETS, FATALITIES_BUCKETS, DR_PHASE3_BUCKETS])
    def test_indices_sequential(self, specs):
        assert [b.idx for b in specs] == list(range(1, len(specs) + 1))

    @pytest.mark.parametrize("specs", [PA_BUCKETS, FATALITIES_BUCKETS, DR_PHASE3_BUCKETS])
    def test_boundaries_contiguous(self, specs):
        for i in range(1, len(specs)):
            assert specs[i - 1].upper == specs[i].lower

    @pytest.mark.parametrize("specs", [PA_BUCKETS, FATALITIES_BUCKETS, DR_PHASE3_BUCKETS])
    def test_centroids_within_boundaries(self, specs):
        for spec in specs:
            assert spec.centroid >= spec.lower
            if spec.upper is not None:
                assert spec.centroid < spec.upper


class TestHelpers:
    def test_n_buckets_for(self):
        assert n_buckets_for("PA") == 6
        assert n_buckets_for("FATALITIES") == 7
        assert n_buckets_for("PHASE3PLUS_IN_NEED") == 6
        assert n_buckets_for("NONEXISTENT") == 0

    def test_max_bucket_count(self):
        assert max_bucket_count() == 7

    def test_centroids_for_pa(self):
        assert centroids_for("PA") == [
            0.0, 5_000.0, 30_000.0, 150_000.0, 375_000.0, 700_000.0
        ]

    def test_centroids_for_fatalities(self):
        assert centroids_for("FATALITIES") == [
            0.0, 3.0, 15.0, 62.0, 300.0, 750.0, 1_500.0
        ]

    def test_thresholds_for_fatalities(self):
        thr = thresholds_for("FATALITIES")
        assert thr[:-1] == [0.0, 1.0, 5.0, 25.0, 100.0, 500.0, 1_000.0]
        assert thr[-1] == float("inf")

    def test_label_index_map(self):
        m = label_index_map("PA")
        assert m["0"] == 1
        assert m["1-<10k"] == 2
        assert m[">=500k"] == 6

    def test_label_index_map_lowercase(self):
        m = label_index_map("PHASE3PLUS_IN_NEED", lowercase=True)
        assert m["100k-<1m"] == 3
        assert m[">=15m"] == 6

    @pytest.mark.parametrize("metric", ["PA", "FATALITIES", "PHASE3PLUS_IN_NEED"])
    def test_labels_unique_after_lowercasing(self, metric):
        """downloads.py / eiv_sql.py lowercase labels before lookup — the
        lowercased labels must stay unique per metric."""
        labels = [x.lower() for x in labels_for(metric)]
        assert len(labels) == len(set(labels))


class TestBucketSpecsRegistry:
    def test_phase3plus_in_registry(self):
        assert "PHASE3PLUS_IN_NEED" in BUCKET_SPECS

    def test_pa_in_registry(self):
        assert "PA" in BUCKET_SPECS

    def test_fatalities_in_registry(self):
        assert "FATALITIES" in BUCKET_SPECS

    def test_registry_has_three_metrics(self):
        assert len(BUCKET_SPECS) == 3
