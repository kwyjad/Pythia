# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Unit tests for the ACLED normalize adapter."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from resolver.transform.adapters.acled import ACLEDAdapter
from resolver.transform.adapters.base import CANONICAL_COLUMNS


@pytest.fixture()
def acled_staging_csv(tmp_path: Path) -> Path:
    """Write a minimal ACLED staging CSV matching the 21-column format."""

    csv_text = textwrap.dedent("""\
        event_id,country_name,iso3,hazard_code,hazard_label,hazard_class,metric,series_semantics,value,unit,as_of_date,publication_date,publisher,source_type,source_url,doc_title,definition_text,method,confidence,revision,ingested_at
        SDN-ACLED-ACE-fatalities_battle_month-2024-01-abc,Sudan,SDN,ACE,Armed Conflict Escalation,conflict,fatalities_battle_month,new,150,persons,2024-01,2024-02-15,ACLED,other,https://acleddata.com,ACLED monthly aggregation,Battle fatalities,api,high,1,2024-02-20
        UKR-ACLED-ACE-fatalities_battle_month-2024-01-def,Ukraine,UKR,ACE,Armed Conflict Escalation,conflict,fatalities_battle_month,new,3200,persons,2024-01,2024-02-15,ACLED,other,https://acleddata.com,ACLED monthly aggregation,Battle fatalities,api,high,1,2024-02-20
        MMR-ACLED-CU-events-2024-02-ghi,Myanmar,MMR,CU,Civil Unrest,unrest,events,new,47,events,2024-02,2024-03-10,ACLED,other,https://acleddata.com,ACLED monthly aggregation,Civil unrest events,api,med,1,2024-03-15
    """)
    path = tmp_path / "acled.csv"
    path.write_text(csv_text, encoding="utf-8")
    return path


@pytest.fixture()
def adapter() -> ACLEDAdapter:
    return ACLEDAdapter("acled")


class TestACLEDAdapterLoad:
    def test_load_returns_dataframe(
        self, adapter: ACLEDAdapter, acled_staging_csv: Path
    ) -> None:
        df = adapter.load(acled_staging_csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_load_has_21_columns(
        self, adapter: ACLEDAdapter, acled_staging_csv: Path
    ) -> None:
        df = adapter.load(acled_staging_csv)
        assert len(df.columns) == 21


class TestACLEDAdapterMap:
    def test_produces_canonical_columns(
        self, adapter: ACLEDAdapter, acled_staging_csv: Path
    ) -> None:
        raw = adapter.load(acled_staging_csv)
        canonical = adapter.map(raw)
        assert list(canonical.columns) == CANONICAL_COLUMNS

    def test_as_of_date_converted_to_month_end(
        self, adapter: ACLEDAdapter, acled_staging_csv: Path
    ) -> None:
        raw = adapter.load(acled_staging_csv)
        canonical = adapter.map(raw)
        # 2024-01 → last day of January
        sdn_row = canonical[canonical["iso3"] == "SDN"].iloc[0]
        assert sdn_row["as_of_date"] == "2024-01-31"
        # 2024-02 → last day of February (leap year)
        mmr_row = canonical[canonical["iso3"] == "MMR"].iloc[0]
        assert mmr_row["as_of_date"] == "2024-02-29"

    def test_metric_remapped_to_canonical(
        self, adapter: ACLEDAdapter, acled_staging_csv: Path
    ) -> None:
        raw = adapter.load(acled_staging_csv)
        canonical = adapter.map(raw)
        metrics = set(canonical["metric"].unique())
        # fatalities_battle_month should be mapped to fatalities
        assert "fatalities" in metrics
        assert "fatalities_battle_month" not in metrics
        # events stays as events
        assert "events" in metrics

    def test_source_is_acled(
        self, adapter: ACLEDAdapter, acled_staging_csv: Path
    ) -> None:
        raw = adapter.load(acled_staging_csv)
        canonical = adapter.map(raw)
        assert (canonical["source"] == "acled").all()

    def test_series_semantics_is_new(
        self, adapter: ACLEDAdapter, acled_staging_csv: Path
    ) -> None:
        raw = adapter.load(acled_staging_csv)
        canonical = adapter.map(raw)
        assert (canonical["series_semantics"] == "new").all()

    def test_value_is_numeric(
        self, adapter: ACLEDAdapter, acled_staging_csv: Path
    ) -> None:
        raw = adapter.load(acled_staging_csv)
        canonical = adapter.map(raw)
        assert canonical["value"].dtype == float
        assert canonical.loc[canonical["iso3"] == "SDN", "value"].iloc[0] == 150.0

    def test_iso3_uppercased(
        self, adapter: ACLEDAdapter, acled_staging_csv: Path
    ) -> None:
        raw = adapter.load(acled_staging_csv)
        canonical = adapter.map(raw)
        for iso3 in canonical["iso3"]:
            assert iso3 == iso3.upper()

    def test_empty_input_returns_empty_canonical(
        self, adapter: ACLEDAdapter
    ) -> None:
        empty = pd.DataFrame(columns=[
            "event_id", "country_name", "iso3", "hazard_code",
            "hazard_label", "hazard_class", "metric", "series_semantics",
            "value", "unit", "as_of_date", "publication_date", "publisher",
            "source_type", "source_url", "doc_title", "definition_text",
            "method", "confidence", "revision", "ingested_at",
        ])
        result = adapter.map(empty)
        assert result.empty
        assert list(result.columns) == CANONICAL_COLUMNS


class TestACLEDAdapterNormalize:
    def test_normalize_end_to_end(
        self, adapter: ACLEDAdapter, acled_staging_csv: Path
    ) -> None:
        canonical = adapter.normalize(acled_staging_csv.parent)
        assert len(canonical) == 3
        assert list(canonical.columns) == CANONICAL_COLUMNS
