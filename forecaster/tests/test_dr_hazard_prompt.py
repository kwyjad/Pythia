# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for the DR / PHASE3PLUS_IN_NEED hazard prompt and FEWS NET projection loader."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from forecaster.hazard_prompts import get_hazard_reasoning_block


# ---------------------------------------------------------------------------
# Dispatch tests
# ---------------------------------------------------------------------------


class TestDRPhase3Dispatch:
    """get_hazard_reasoning_block routes DR + PHASE3PLUS_IN_NEED to _DR_PHASE3."""

    def test_dr_phase3_returns_fewsnet_text(self):
        block = get_hazard_reasoning_block("DR", "PHASE3PLUS_IN_NEED")
        assert "FEWS NET" in block
        assert "Phase 3+" in block

    def test_dr_phase3_does_not_mention_ifrc_montandon(self):
        block = get_hazard_reasoning_block("DR", "PHASE3PLUS_IN_NEED")
        assert "IFRC Montandon" not in block
        assert "IFRC" not in block
        assert "EM-DAT" not in block

    def test_dr_pa_returns_old_block(self):
        """Backward compat: DR + PA still gets the generic DR block."""
        block = get_hazard_reasoning_block("DR", "PA")
        assert "IFRC Montandon" in block
        assert "FEWS NET" not in block

    def test_dr_fatalities_returns_old_block(self):
        block = get_hazard_reasoning_block("DR", "FATALITIES")
        assert "IFRC Montandon" in block

    def test_dr_phase3_case_insensitive(self):
        block = get_hazard_reasoning_block("dr", "phase3plus_in_need")
        assert "FEWS NET" in block


# ---------------------------------------------------------------------------
# Bucket boundary tests
# ---------------------------------------------------------------------------


class TestDRPhase3Buckets:
    """_DR_PHASE3 block references correct bucket boundaries from DR_PHASE3_BUCKETS."""

    def test_bucket_boundary_100k(self):
        block = get_hazard_reasoning_block("DR", "PHASE3PLUS_IN_NEED")
        assert "100k" in block.lower() or "<100k" in block

    def test_bucket_boundary_1m(self):
        block = get_hazard_reasoning_block("DR", "PHASE3PLUS_IN_NEED")
        # 1M boundary
        assert "1M" in block or "1m" in block.lower()

    def test_bucket_boundary_5m(self):
        block = get_hazard_reasoning_block("DR", "PHASE3PLUS_IN_NEED")
        assert "5M" in block or "5m" in block.lower()

    def test_bucket_boundary_15m(self):
        block = get_hazard_reasoning_block("DR", "PHASE3PLUS_IN_NEED")
        assert "15M" in block or "15m" in block.lower() or "15 million" in block.lower()

    def test_all_five_buckets_referenced(self):
        block = get_hazard_reasoning_block("DR", "PHASE3PLUS_IN_NEED")
        for label in ["Bucket 1", "Bucket 2", "Bucket 3", "Bucket 4", "Bucket 5"]:
            assert label in block, f"{label} not found in _DR_PHASE3 block"


# ---------------------------------------------------------------------------
# Content tests
# ---------------------------------------------------------------------------


class TestDRPhase3Content:
    """Key reasoning principles are present in the _DR_PHASE3 block."""

    def test_mentions_persistent_slow_changing(self):
        block = get_hazard_reasoning_block("DR", "PHASE3PLUS_IN_NEED")
        assert "PERSISTENT" in block or "SLOW-CHANGING" in block

    def test_mentions_analysis_cycles(self):
        block = get_hazard_reasoning_block("DR", "PHASE3PLUS_IN_NEED")
        assert "ANALYSIS CYCLES" in block or "analysis cycle" in block.lower()

    def test_mentions_seasonal_drivers(self):
        block = get_hazard_reasoning_block("DR", "PHASE3PLUS_IN_NEED")
        assert "SEASONAL" in block or "lean season" in block.lower()

    def test_mentions_conflict_driver(self):
        block = get_hazard_reasoning_block("DR", "PHASE3PLUS_IN_NEED")
        assert "conflict" in block.lower()

    def test_mentions_resolution_source_distinction(self):
        block = get_hazard_reasoning_block("DR", "PHASE3PLUS_IN_NEED")
        assert "Current Situation" in block
        assert "Most Likely" in block

    def test_mentions_nmme(self):
        block = get_hazard_reasoning_block("DR", "PHASE3PLUS_IN_NEED")
        assert "NMME" in block

    def test_mentions_cumulative_effects(self):
        block = get_hazard_reasoning_block("DR", "PHASE3PLUS_IN_NEED")
        assert "CUMULATIVE" in block or "cumulative" in block


# ---------------------------------------------------------------------------
# _load_fewsnet_projection tests
# ---------------------------------------------------------------------------


class TestLoadFewsnetProjection:
    """Test _load_fewsnet_projection with mocked DuckDB."""

    def test_returns_table_with_mock_data(self):
        from forecaster.prompts import _load_fewsnet_projection

        mock_con = MagicMock()
        mock_con.execute.return_value.fetchall.return_value = [
            ("2026-04", 1_500_000, "2026-03-15"),
            ("2026-05", 1_800_000, "2026-03-15"),
        ]

        with patch("forecaster.prompts._pythia_db_url_from_config", return_value=None), \
             patch.dict("os.environ", {"RESOLVER_DB_URL": ""}), \
             patch("resolver.db.duckdb_io.get_db", return_value=mock_con), \
             patch("resolver.db.duckdb_io.close_db"):
            result = _load_fewsnet_projection("ETH", ["2026-04", "2026-05", "2026-06"])

        assert "FEWS NET MOST LIKELY PROJECTION" in result
        assert "1,500,000" in result
        assert "1,800,000" in result
        assert "2026-04" in result
        assert "2026-03-15" in result

    def test_returns_empty_when_no_rows(self):
        from forecaster.prompts import _load_fewsnet_projection

        mock_con = MagicMock()
        mock_con.execute.return_value.fetchall.return_value = []

        with patch("forecaster.prompts._pythia_db_url_from_config", return_value=None), \
             patch.dict("os.environ", {"RESOLVER_DB_URL": ""}), \
             patch("resolver.db.duckdb_io.get_db", return_value=mock_con), \
             patch("resolver.db.duckdb_io.close_db"):
            result = _load_fewsnet_projection("XYZ", ["2026-04", "2026-05"])

        assert result == ""

    def test_returns_empty_when_no_iso3(self):
        from forecaster.prompts import _load_fewsnet_projection

        result = _load_fewsnet_projection("", ["2026-04"])
        assert result == ""

    def test_returns_empty_when_no_forecast_keys(self):
        from forecaster.prompts import _load_fewsnet_projection

        result = _load_fewsnet_projection("ETH", [])
        assert result == ""

    def test_returns_empty_on_db_exception(self):
        from forecaster.prompts import _load_fewsnet_projection

        with patch("forecaster.prompts._pythia_db_url_from_config", return_value=None), \
             patch.dict("os.environ", {"RESOLVER_DB_URL": ""}), \
             patch("resolver.db.duckdb_io.get_db", side_effect=Exception("db error")):
            result = _load_fewsnet_projection("ETH", ["2026-04"])

        assert result == ""
