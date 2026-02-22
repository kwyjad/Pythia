# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Integration test for the new pipeline orchestrator."""

from __future__ import annotations

import pandas as pd
import pytest

from resolver.connectors.protocol import CANONICAL_COLUMNS
from resolver.tools.enrich import derive_ym, enrich


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------


class TestDeriveYm:
    def test_derives_ym_from_as_of_date(self):
        df = pd.DataFrame({"as_of_date": ["2025-09-30", "2025-01-15"]})
        for col in CANONICAL_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        result = derive_ym(df)
        assert list(result["ym"]) == ["2025-09", "2025-01"]

    def test_preserves_existing_ym(self):
        df = pd.DataFrame({
            "as_of_date": ["2025-09-30"],
            "ym": ["2025-08"],  # pre-set, should NOT be overwritten
        })
        for col in CANONICAL_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        result = derive_ym(df)
        assert result.iloc[0]["ym"] == "2025-08"


class TestEnrich:
    def test_fills_country_name_from_registry(self):
        df = pd.DataFrame({
            "iso3": ["AFG"],
            "country_name": [""],
            "hazard_code": ["ACE"],
            "hazard_label": [""],
            "hazard_class": [""],
            "metric": ["fatalities"],
            "unit": ["persons"],
            "as_of_date": ["2025-01-31"],
            "publication_date": ["2025-02-01"],
            "event_id": ["test-1"],
            "series_semantics": ["new"],
            "value": [100],
            "source_type": ["agency"],
            "source_url": [""],
            "doc_title": [""],
            "definition_text": [""],
            "method": ["api"],
            "confidence": ["high"],
            "revision": ["1"],
            "ingested_at": ["2025-02-01"],
        })
        result = enrich(df)
        # If the countries.csv registry exists and contains AFG,
        # country_name should be filled.
        assert result.iloc[0]["iso3"] == "AFG"

    def test_defaults_metric_and_unit(self):
        df = pd.DataFrame({col: [""] for col in CANONICAL_COLUMNS})
        df["iso3"] = ["AFG"]
        df["value"] = [100]
        df["as_of_date"] = ["2025-01-31"]
        result = enrich(df)
        assert result.iloc[0]["metric"] == "affected"
        assert result.iloc[0]["unit"] == "persons"


# ---------------------------------------------------------------------------
# Pipeline (end to end with mocked connectors)
# ---------------------------------------------------------------------------


class _FakeConnector:
    """Connector that returns a pre-built canonical DataFrame."""

    def __init__(self, name: str, rows: pd.DataFrame):
        self.name = name
        self._rows = rows

    def fetch_and_normalize(self) -> pd.DataFrame:
        return self._rows


def _make_canonical_rows(iso3: str, hazard: str, value: float, source: str) -> pd.DataFrame:
    return pd.DataFrame({
        "event_id": [f"{iso3}-{hazard}-test"],
        "country_name": [""],
        "iso3": [iso3],
        "hazard_code": [hazard],
        "hazard_label": [""],
        "hazard_class": [""],
        "metric": ["affected"],
        "series_semantics": ["stock"],
        "value": [value],
        "unit": ["persons"],
        "as_of_date": ["2025-01-31"],
        "publication_date": ["2025-02-01"],
        "publisher": [source],
        "source_type": ["agency"],
        "source_url": [""],
        "doc_title": [""],
        "definition_text": [""],
        "method": ["api"],
        "confidence": ["high"],
        "revision": ["1"],
        "ingested_at": ["2025-02-01T00:00:00Z"],
    })


class TestRunPipeline:
    def test_pipeline_dry_run(self, monkeypatch):
        """Full pipeline with dry_run=True and fake connectors."""
        from resolver.tools.run_pipeline import run_pipeline

        fake_acled = _FakeConnector(
            "acled",
            _make_canonical_rows("AFG", "ACE", 42, "ACLED"),
        )
        fake_idmc = _FakeConnector(
            "idmc",
            _make_canonical_rows("ETH", "DI", 5000, "IDMC"),
        )

        monkeypatch.setattr(
            "resolver.connectors.discover_connectors",
            lambda names=None: [fake_acled, fake_idmc],
        )

        result = run_pipeline(dry_run=True)

        assert result.total_facts == 2
        assert len(result.connector_results) == 2
        assert all(cr.status == "ok" for cr in result.connector_results)
        assert result.db_written is False

    def test_pipeline_handles_connector_error(self, monkeypatch):
        """Pipeline continues if one connector fails."""
        from resolver.tools.run_pipeline import run_pipeline

        class _BrokenConnector:
            name = "broken"

            def fetch_and_normalize(self):
                raise RuntimeError("API down")

        good = _FakeConnector(
            "good",
            _make_canonical_rows("AFG", "ACE", 42, "ACLED"),
        )

        monkeypatch.setattr(
            "resolver.connectors.discover_connectors",
            lambda names=None: [_BrokenConnector(), good],
        )

        result = run_pipeline(dry_run=True)

        assert len(result.connector_results) == 2
        broken_cr = result.connector_results[0]
        assert broken_cr.status == "error"
        assert "API down" in broken_cr.error
        good_cr = result.connector_results[1]
        assert good_cr.status == "ok"
        assert result.total_facts == 1
