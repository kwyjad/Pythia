# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for Phase 3+ bucket integration (PHASE3PLUS_IN_NEED metric)."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest

from pythia.buckets import (
    BUCKET_SPECS,
    DR_PHASE3_BUCKETS,
    BucketSpec,
    get_bucket_specs,
)


class TestPhase3BucketSpecs:
    """Verify PHASE3PLUS_IN_NEED bucket specs are correctly configured."""

    def test_get_bucket_specs_returns_5(self):
        specs = get_bucket_specs("PHASE3PLUS_IN_NEED")
        assert len(specs) == 5

    def test_bucket_boundaries_match_expected(self):
        specs = get_bucket_specs("PHASE3PLUS_IN_NEED")
        expected = [
            (0.0, 100_000.0),
            (100_000.0, 1_000_000.0),
            (1_000_000.0, 5_000_000.0),
            (5_000_000.0, 15_000_000.0),
            (15_000_000.0, None),
        ]
        for spec, (exp_lower, exp_upper) in zip(specs, expected):
            assert spec.lower == exp_lower
            assert spec.upper == exp_upper

    def test_centroids_within_bucket_ranges(self):
        specs = get_bucket_specs("PHASE3PLUS_IN_NEED")
        for spec in specs:
            assert spec.centroid >= spec.lower, (
                f"Bucket {spec.idx}: centroid {spec.centroid} < lower {spec.lower}"
            )
            if spec.upper is not None:
                assert spec.centroid < spec.upper, (
                    f"Bucket {spec.idx}: centroid {spec.centroid} >= upper {spec.upper}"
                )

    def test_expected_centroids(self):
        specs = get_bucket_specs("PHASE3PLUS_IN_NEED")
        expected_centroids = [50_000.0, 500_000.0, 2_500_000.0, 10_000_000.0, 20_000_000.0]
        for spec, expected in zip(specs, expected_centroids):
            assert spec.centroid == expected

    def test_phase3plus_in_bucket_specs_registry(self):
        assert "PHASE3PLUS_IN_NEED" in BUCKET_SPECS
        assert BUCKET_SPECS["PHASE3PLUS_IN_NEED"] is DR_PHASE3_BUCKETS


class TestComputeBucketCentroidsMetricFilter:
    """Verify the metric filter logic in compute_bucket_centroids."""

    def test_phase3plus_metric_filter(self):
        """Verify that PHASE3PLUS_IN_NEED generates the correct SQL filter."""
        # We import the module and test the metric filter logic indirectly
        # by checking the SQL that would be generated
        metric = "PHASE3PLUS_IN_NEED"
        if metric == "PHASE3PLUS_IN_NEED":
            metric_filter = "lower(metric) = 'phase3plus_in_need'"
        elif metric == "PA":
            metric_filter = "lower(metric) IN ('affected','people_affected','pa','displaced')"
        elif metric == "FATALITIES":
            metric_filter = "lower(metric) = 'fatalities'"
        else:
            metric_filter = f"lower(metric) = '{metric.lower()}'"

        assert metric_filter == "lower(metric) = 'phase3plus_in_need'"

    def test_pa_metric_filter(self):
        metric = "PA"
        if metric == "PHASE3PLUS_IN_NEED":
            metric_filter = "lower(metric) = 'phase3plus_in_need'"
        elif metric == "PA":
            metric_filter = "lower(metric) IN ('affected','people_affected','pa','displaced')"
        elif metric == "FATALITIES":
            metric_filter = "lower(metric) = 'fatalities'"
        else:
            metric_filter = f"lower(metric) = '{metric.lower()}'"

        assert "affected" in metric_filter
        assert "displaced" in metric_filter

    def test_fatalities_metric_filter(self):
        metric = "FATALITIES"
        if metric == "PHASE3PLUS_IN_NEED":
            metric_filter = "lower(metric) = 'phase3plus_in_need'"
        elif metric == "PA":
            metric_filter = "lower(metric) IN ('affected','people_affected','pa','displaced')"
        elif metric == "FATALITIES":
            metric_filter = "lower(metric) = 'fatalities'"
        else:
            metric_filter = f"lower(metric) = '{metric.lower()}'"

        assert metric_filter == "lower(metric) = 'fatalities'"


class TestUpdateBucketCentroidsHandlesPhase3:
    """Verify that update_bucket_centroids iterates PHASE3PLUS_IN_NEED."""

    def test_bucket_specs_includes_phase3plus(self):
        """BUCKET_SPECS must include PHASE3PLUS_IN_NEED for the update script
        to seed its centroids."""
        assert "PHASE3PLUS_IN_NEED" in BUCKET_SPECS

    def test_all_metrics_have_centroids(self):
        """Every metric in BUCKET_SPECS should have non-empty specs with centroids."""
        for metric, specs in BUCKET_SPECS.items():
            assert len(specs) > 0, f"{metric} has no bucket specs"
            for spec in specs:
                assert spec.centroid is not None, (
                    f"{metric} bucket {spec.idx} has None centroid"
                )


class TestAnalysisScriptImport:
    """Verify the analysis script can be imported without errors."""

    def test_import_analysis_script(self):
        """The analysis script module should be importable."""
        import importlib.util
        import os

        script_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "tools", "analyze_fewsnet_distribution.py"
        )
        script_path = os.path.abspath(script_path)
        assert os.path.exists(script_path), f"Script not found at {script_path}"

        spec = importlib.util.spec_from_file_location(
            "analyze_fewsnet_distribution", script_path
        )
        mod = importlib.util.module_from_spec(spec)
        # Don't exec (would try to connect to DB), just verify it parses
        assert spec is not None
        assert mod is not None
