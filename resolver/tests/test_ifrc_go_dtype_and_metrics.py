# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Unit tests for IFRC GO connector dtype mapping and multi-metric extraction."""

from __future__ import annotations

import pytest

import pandas as pd

from resolver.ingestion.ifrc_go_client import (
    DTYPE_TO_HAZARD,
    _APPEAL_SOURCE_PREFIX,
    _DERIVED_SOURCE_PREFIX,
    _dtype_id_from_record,
    _maybe_derive_affected,
    _maybe_use_beneficiaries_as_affected,
    detect_hazard,
    extract_all_metrics,
    iso3_pairs_from_go,
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


# ---------------------------------------------------------------------------
# Derived `affected` imputation
# ---------------------------------------------------------------------------


class TestDerivedAffected:
    """When IFRC reports collateral impact metrics (fatalities / injured /
    displaced / missing) but no `affected`, ``extract_all_metrics`` should
    synthesise an `affected` row as their sum so PA-eligible resolution can
    fire. The synthesised row is marked via the source-field prefix."""

    def test_derives_affected_from_fatalities_plus_injured(self):
        rec = {"num_dead": 10, "num_injured": 50}
        result = extract_all_metrics(rec)
        metrics_by_name = {r[0]: r for r in result}
        assert "affected" in metrics_by_name, (
            "Two collateral metrics present, no num_affected — must synthesise."
        )
        affected = metrics_by_name["affected"]
        assert affected[1] == 60  # 10 + 50
        assert affected[2] == "persons"
        assert affected[3].startswith(_DERIVED_SOURCE_PREFIX)
        assert "fatalities" in affected[3]
        assert "injured" in affected[3]

    def test_derives_from_all_four_collateral(self):
        rec = {
            "num_dead": 5,
            "num_injured": 20,
            "num_displaced": 1000,
            "num_missing": 3,
        }
        result = extract_all_metrics(rec)
        affected = [r for r in result if r[0] == "affected"]
        assert len(affected) == 1
        assert affected[0][1] == 1028  # 5 + 20 + 1000 + 3
        # Components show up in alphabet-of-insertion order matching
        # COLLATERAL_FOR_AFFECTED tuple
        assert "fatalities" in affected[0][3]
        assert "injured" in affected[0][3]
        assert "displaced" in affected[0][3]
        assert "missing" in affected[0][3]

    def test_no_derivation_when_only_one_collateral(self):
        """Single isolated metric is not enough to extrapolate from."""
        rec = {"num_dead": 10}
        result = extract_all_metrics(rec)
        metrics = {r[0] for r in result}
        assert metrics == {"fatalities"}
        assert "affected" not in metrics

    def test_no_derivation_when_real_affected_present(self):
        """If num_affected is reported, do NOT replace or augment it."""
        rec = {"num_affected": 5000, "num_dead": 10, "num_injured": 50}
        result = extract_all_metrics(rec)
        affected_rows = [r for r in result if r[0] == "affected"]
        assert len(affected_rows) == 1
        assert affected_rows[0][1] == 5000  # the real one, not the sum (60)
        assert affected_rows[0][3] == "num_affected"
        assert not affected_rows[0][3].startswith(_DERIVED_SOURCE_PREFIX)

    def test_no_derivation_when_only_displaced_present(self):
        """displaced alone is < threshold (need 2 collateral)."""
        rec = {"num_displaced": 3000}
        result = extract_all_metrics(rec)
        metrics = {r[0] for r in result}
        assert metrics == {"displaced"}
        assert "affected" not in metrics

    def test_displaced_plus_fatalities_derives(self):
        rec = {"num_displaced": 3000, "num_dead": 12}
        result = extract_all_metrics(rec)
        affected = [r for r in result if r[0] == "affected"]
        assert len(affected) == 1
        assert affected[0][1] == 3012
        assert affected[0][3].startswith(_DERIVED_SOURCE_PREFIX)

    def test_gov_variants_count_toward_derivation(self):
        """Derivation works on the consolidated metric, not on the raw
        field name — gov_num_dead still produces a 'fatalities' result."""
        rec = {"gov_num_dead": 7, "other_num_injured": 30}
        result = extract_all_metrics(rec)
        affected = [r for r in result if r[0] == "affected"]
        assert len(affected) == 1
        assert affected[0][1] == 37  # 7 + 30

    def test_zero_collateral_does_not_derive(self):
        """If all collateral metrics are zero, no derived row even if 2+
        nominally present (extract_all_metrics filters zeros out before
        we see them)."""
        rec = {"num_dead": 0, "num_injured": 0}
        result = extract_all_metrics(rec)
        assert result == []


class TestAppealBeneficiaries:
    """When a record (typically a DREF / Emergency Appeal) has no
    ``num_affected`` but does carry ``num_beneficiaries`` (the appeal's
    "people targeted" figure), ``extract_all_metrics`` should promote it to
    an ``affected`` row marked with the appeal source prefix."""

    def test_beneficiaries_promoted_to_affected(self):
        rec = {"num_beneficiaries": 50000}
        result = extract_all_metrics(rec)
        affected = [r for r in result if r[0] == "affected"]
        assert len(affected) == 1
        assert affected[0][1] == 50000
        assert affected[0][2] == "persons"
        assert affected[0][3].startswith(_APPEAL_SOURCE_PREFIX)
        assert affected[0][3] == f"{_APPEAL_SOURCE_PREFIX}num_beneficiaries"

    def test_real_num_affected_wins_over_beneficiaries(self):
        """If both are present, num_affected (the actual impact field) wins;
        num_beneficiaries is the planning estimate."""
        rec = {"num_affected": 80000, "num_beneficiaries": 50000}
        result = extract_all_metrics(rec)
        affected = [r for r in result if r[0] == "affected"]
        assert len(affected) == 1
        assert affected[0][1] == 80000
        assert affected[0][3] == "num_affected"
        assert not affected[0][3].startswith(_APPEAL_SOURCE_PREFIX)

    def test_beneficiaries_with_collateral_picks_beneficiaries(self):
        """Beneficiaries fallback fires before the imputation fallback —
        an official IFRC planning figure beats a sum of collateral metrics."""
        rec = {"num_beneficiaries": 100000, "num_dead": 10, "num_injured": 50}
        result = extract_all_metrics(rec)
        affected = [r for r in result if r[0] == "affected"]
        assert len(affected) == 1
        assert affected[0][1] == 100000  # not 60 (collateral sum)
        assert affected[0][3].startswith(_APPEAL_SOURCE_PREFIX)

    def test_zero_beneficiaries_skipped(self):
        rec = {"num_beneficiaries": 0}
        result = extract_all_metrics(rec)
        assert result == []  # no metrics at all

    def test_amount_beneficiaries_variant(self):
        """Fallback to ``amount_beneficiaries`` if ``num_beneficiaries`` is
        absent (schema variant guard)."""
        rec = {"amount_beneficiaries": 25000}
        result = extract_all_metrics(rec)
        affected = [r for r in result if r[0] == "affected"]
        assert len(affected) == 1
        assert affected[0][1] == 25000
        assert affected[0][3] == f"{_APPEAL_SOURCE_PREFIX}amount_beneficiaries"

    def test_beneficiaries_coexists_with_collateral_rows(self):
        """Promoting beneficiaries to 'affected' must not suppress the other
        fields' rows — fatalities and injured should still appear."""
        rec = {"num_beneficiaries": 80000, "num_dead": 5, "num_injured": 20}
        result = extract_all_metrics(rec)
        metrics = {r[0] for r in result}
        assert metrics == {"affected", "fatalities", "injured"}


class TestMaybeUseBeneficiariesAsAffectedUnit:
    """Direct unit tests for the appeals helper."""

    def test_no_results_with_no_beneficiaries(self):
        assert _maybe_use_beneficiaries_as_affected({}, []) is None

    def test_existing_affected_short_circuits(self):
        rec = {"num_beneficiaries": 50000}
        results = [("affected", 10000, "persons", "num_affected")]
        assert _maybe_use_beneficiaries_as_affected(rec, results) is None

    def test_string_beneficiaries_parsed(self):
        rec = {"num_beneficiaries": "12,345"}
        out = _maybe_use_beneficiaries_as_affected(rec, [])
        assert out is not None
        assert out[1] == 12345


class TestIso3PairsFromGo:
    """Country-detail extraction must handle both Field Report / Event shape
    (``countries_details`` — list) and Appeals shape (``country`` — single
    dict). Before this PR, only the plural list shape was recognised, so
    every appeal silently failed at the no-country gate."""

    @pytest.fixture
    def countries_df(self):
        return pd.DataFrame(
            [
                {"iso3": "ECU", "country_name": "Ecuador"},
                {"iso3": "PAK", "country_name": "Pakistan"},
                {"iso3": "BGD", "country_name": "Bangladesh"},
            ]
        )

    def test_field_report_countries_details_plural(self, countries_df):
        """Existing behaviour — list of country dicts."""
        rec = {"countries_details": [{"iso3": "PAK", "name": "Pakistan"}]}
        result = iso3_pairs_from_go(countries_df, rec)
        assert result == [("Pakistan", "PAK")]

    def test_appeals_country_singular_dict(self, countries_df):
        """Appeal records use the singular ``country`` field with a dict —
        this is the shape we needed to support."""
        rec = {"country": {"iso3": "ECU", "name": "Ecuador", "iso": "EC"}}
        result = iso3_pairs_from_go(countries_df, rec)
        assert result == [("Ecuador", "ECU")]

    def test_country_details_singular_also_handled(self, countries_df):
        """If the API ever returns ``country_details`` (singular) we should
        accept that shape too — belt-and-suspenders."""
        rec = {"country_details": {"iso3": "BGD", "name": "Bangladesh"}}
        result = iso3_pairs_from_go(countries_df, rec)
        assert result == [("Bangladesh", "BGD")]

    def test_appeal_with_multi_country_field_report_payload(self, countries_df):
        """If both plural and singular shapes are present (defensive), accept
        both — duplicates are deduped naturally by the caller's
        seen_ids set."""
        rec = {
            "countries_details": [{"iso3": "PAK", "name": "Pakistan"}],
            "country": {"iso3": "ECU", "name": "Ecuador"},
        }
        result = iso3_pairs_from_go(countries_df, rec)
        # Order: plural first (existing precedence), then singular
        assert ("Pakistan", "PAK") in result
        assert ("Ecuador", "ECU") in result

    def test_no_country_returns_empty(self, countries_df):
        rec = {"title": "No country info"}
        result = iso3_pairs_from_go(countries_df, rec)
        assert result == []

    def test_unknown_iso3_returns_empty(self, countries_df):
        """If the iso3 isn't in our registry, drop the candidate."""
        rec = {"country": {"iso3": "XYZ", "name": "Nowhere"}}
        result = iso3_pairs_from_go(countries_df, rec)
        assert result == []


class TestMaybeDeriveAffectedUnit:
    """Direct unit tests for the helper, independent of extract_all_metrics."""

    def test_no_results_returns_none(self):
        assert _maybe_derive_affected([]) is None

    def test_real_affected_present_returns_none(self):
        results = [("affected", 1000, "persons", "num_affected"),
                   ("fatalities", 10, "persons", "num_dead"),
                   ("injured", 20, "persons", "num_injured")]
        assert _maybe_derive_affected(results) is None

    def test_one_collateral_returns_none(self):
        results = [("fatalities", 10, "persons", "num_dead")]
        assert _maybe_derive_affected(results) is None

    def test_two_collateral_derives(self):
        results = [("fatalities", 10, "persons", "num_dead"),
                   ("injured", 50, "persons", "num_injured")]
        derived = _maybe_derive_affected(results)
        assert derived is not None
        assert derived[0] == "affected"
        assert derived[1] == 60
        assert derived[2] == "persons"
        assert derived[3].startswith(_DERIVED_SOURCE_PREFIX)
