# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Unit tests for IFRC GO connector dtype mapping and multi-metric extraction."""

from __future__ import annotations

import pytest

from resolver.ingestion.ifrc_go_client import (
    DTYPE_TO_HAZARD,
    _dtype_id_from_record,
    detect_hazard,
    extract_all_metrics,
    load_cfg,
)


# ---------------------------------------------------------------------------
# _dtype_id_from_record
# ---------------------------------------------------------------------------


class TestDtypeIdFromRecord:
    def test_dict_with_id(self):
        rec = {"dtype": {"id": 12, "name": "Flood"}}
        assert _dtype_id_from_record(rec) == 12

    def test_nested_dtype_details(self):
        rec = {"dtype_details": {"id": 4, "name": "Cyclone"}}
        assert _dtype_id_from_record(rec) == 4

    def test_disaster_type_details_dict(self):
        rec = {"disaster_type_details": {"id": 20, "name": "Drought"}}
        assert _dtype_id_from_record(rec) == 20

    def test_disaster_type_list(self):
        rec = {"disaster_type": [{"id": 19, "name": "Heat Wave"}]}
        assert _dtype_id_from_record(rec) == 19

    def test_integer_dtype(self):
        rec = {"dtype": 12}
        assert _dtype_id_from_record(rec) == 12

    def test_no_dtype_returns_none(self):
        rec = {"title": "Some report"}
        assert _dtype_id_from_record(rec) is None

    def test_empty_dict_returns_none(self):
        rec = {"dtype": {}}
        assert _dtype_id_from_record(rec) is None


# ---------------------------------------------------------------------------
# detect_hazard with dtype priority
# ---------------------------------------------------------------------------


class TestDetectHazardDtype:
    @pytest.fixture
    def cfg(self):
        return load_cfg()

    def test_flood_dtype(self, cfg):
        rec = {"dtype": {"id": 12, "name": "Flood"}}
        assert detect_hazard("", cfg, record=rec) == "FL"

    def test_flash_flood_dtype(self, cfg):
        rec = {"dtype": {"id": 27, "name": "Pluvial/Flash Flood"}}
        assert detect_hazard("", cfg, record=rec) == "FL"

    def test_cyclone_dtype(self, cfg):
        rec = {"dtype": {"id": 4, "name": "Cyclone"}}
        assert detect_hazard("", cfg, record=rec) == "TC"

    def test_storm_surge_maps_to_tc(self, cfg):
        rec = {"dtype": {"id": 23, "name": "Storm Surge"}}
        assert detect_hazard("", cfg, record=rec) == "TC"

    def test_drought_dtype(self, cfg):
        rec = {"dtype": {"id": 20, "name": "Drought"}}
        assert detect_hazard("", cfg, record=rec) == "DR"

    def test_heat_wave_dtype(self, cfg):
        rec = {"dtype": {"id": 19, "name": "Heat Wave"}}
        assert detect_hazard("", cfg, record=rec) == "HW"

    def test_epidemic_dtype(self, cfg):
        rec = {"dtype": {"id": 1, "name": "Epidemic"}}
        assert detect_hazard("", cfg, record=rec) == "PHE"

    def test_biological_emergency_maps_to_phe(self, cfg):
        rec = {"dtype": {"id": 66, "name": "Biological Emergency"}}
        assert detect_hazard("", cfg, record=rec) == "PHE"

    def test_complex_emergency_maps_to_ace(self, cfg):
        rec = {"dtype": {"id": 6, "name": "Complex Emergency"}}
        assert detect_hazard("", cfg, record=rec) == "ACE"

    def test_earthquake_dtype(self, cfg):
        rec = {"dtype": {"id": 2, "name": "Earthquake"}}
        assert detect_hazard("", cfg, record=rec) == "EQ"

    def test_volcanic_eruption_dtype(self, cfg):
        rec = {"dtype": {"id": 8, "name": "Volcanic Eruption"}}
        assert detect_hazard("", cfg, record=rec) == "VO"

    def test_fire_dtype(self, cfg):
        rec = {"dtype": {"id": 15, "name": "Fire"}}
        assert detect_hazard("", cfg, record=rec) == "FIRE"

    def test_food_insecurity_dtype(self, cfg):
        rec = {"dtype": {"id": 21, "name": "Food Insecurity"}}
        assert detect_hazard("", cfg, record=rec) == "FI"

    def test_unmapped_dtype_falls_back_to_keywords(self, cfg):
        """Transport Accident (54) is not mapped; keyword fallback should work."""
        rec = {"dtype": {"id": 54, "name": "Transport Accident"}}
        # Text contains "flood" keyword → should detect FL via keyword fallback
        assert detect_hazard("severe flood damage", cfg, record=rec) == "FL"

    def test_unmapped_dtype_no_keywords_returns_none(self, cfg):
        rec = {"dtype": {"id": 54, "name": "Transport Accident"}}
        assert detect_hazard("transport accident report", cfg, record=rec) is None

    def test_dtype_takes_priority_over_keywords(self, cfg):
        """Even if text says 'flood', dtype 20 (Drought) should win."""
        rec = {"dtype": {"id": 20, "name": "Drought"}}
        assert detect_hazard("flood and drought conditions", cfg, record=rec) == "DR"

    def test_keyword_fallback_when_no_record(self, cfg):
        """Without a record, detect_hazard uses keyword matching."""
        assert detect_hazard("severe flood damage", cfg) == "FL"
        assert detect_hazard("tropical cyclone warning", cfg) == "TC"
        assert detect_hazard("drought conditions", cfg) == "DR"

    def test_keyword_no_match(self, cfg):
        assert detect_hazard("no recognizable hazard here", cfg) is None


# ---------------------------------------------------------------------------
# DTYPE_TO_HAZARD coverage
# ---------------------------------------------------------------------------


def test_dtype_mapping_covers_all_natural_hazards():
    """Verify all 18 mapped disaster types are present."""
    expected_dtypes = {1, 2, 4, 5, 6, 7, 8, 11, 12, 14, 15, 19, 20, 21, 23, 24, 27, 66}
    assert set(DTYPE_TO_HAZARD.keys()) == expected_dtypes


def test_subtype_merges():
    """Flood subtypes, cyclone subtypes, and epidemic subtypes merge correctly."""
    # Flood subtypes
    assert DTYPE_TO_HAZARD[12] == "FL"
    assert DTYPE_TO_HAZARD[27] == "FL"
    # Cyclone subtypes
    assert DTYPE_TO_HAZARD[4] == "TC"
    assert DTYPE_TO_HAZARD[23] == "TC"
    # Epidemic subtypes
    assert DTYPE_TO_HAZARD[1] == "PHE"
    assert DTYPE_TO_HAZARD[66] == "PHE"


# ---------------------------------------------------------------------------
# extract_all_metrics
# ---------------------------------------------------------------------------


class TestExtractAllMetrics:
    def test_single_affected(self):
        rec = {"num_affected": 5000}
        result = extract_all_metrics(rec)
        assert len(result) == 1
        assert result[0] == ("affected", 5000, "persons", "num_affected")

    def test_multiple_metrics(self):
        rec = {"num_affected": 5000, "num_dead": 12, "num_displaced": 3000}
        result = extract_all_metrics(rec)
        metrics = {r[0] for r in result}
        assert metrics == {"affected", "fatalities", "displaced"}
        assert len(result) == 3

    def test_all_five_metrics(self):
        rec = {
            "num_affected": 10000,
            "num_dead": 50,
            "num_injured": 200,
            "num_displaced": 5000,
            "num_missing": 10,
        }
        result = extract_all_metrics(rec)
        assert len(result) == 5

    def test_gov_variant_used(self):
        rec = {"gov_num_dead": 15}
        result = extract_all_metrics(rec)
        assert len(result) == 1
        assert result[0] == ("fatalities", 15, "persons", "gov_num_dead")

    def test_max_of_variants(self):
        """Should take the maximum of primary/gov/other variants."""
        rec = {"num_dead": 10, "gov_num_dead": 5, "other_num_dead": 25}
        result = extract_all_metrics(rec)
        assert len(result) == 1
        assert result[0][0] == "fatalities"
        assert result[0][1] == 25  # max value
        assert result[0][3] == "other_num_dead"

    def test_zero_values_skipped(self):
        rec = {"num_affected": 0, "num_dead": 0}
        result = extract_all_metrics(rec)
        assert len(result) == 0

    def test_none_values_skipped(self):
        rec = {"num_affected": None, "num_dead": None}
        result = extract_all_metrics(rec)
        assert len(result) == 0

    def test_empty_string_skipped(self):
        rec = {"num_affected": ""}
        result = extract_all_metrics(rec)
        assert len(result) == 0

    def test_empty_record(self):
        result = extract_all_metrics({})
        assert len(result) == 0

    def test_string_number_parsed(self):
        rec = {"num_affected": "12,500"}
        result = extract_all_metrics(rec)
        assert len(result) == 1
        assert result[0][1] == 12500
