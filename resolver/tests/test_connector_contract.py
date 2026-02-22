# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Contract tests: every connector must return exactly CANONICAL_COLUMNS."""

from __future__ import annotations

import pandas as pd
import pytest

from resolver.connectors.protocol import CANONICAL_COLUMNS
from resolver.connectors.validate import validate_canonical, empty_canonical


# ---------------------------------------------------------------------------
# Validate helper
# ---------------------------------------------------------------------------


class TestValidateCanonical:
    def test_accepts_valid_frame(self):
        df = pd.DataFrame(
            {col: ["val"] for col in CANONICAL_COLUMNS}
        )
        df["value"] = [100.0]
        df["iso3"] = ["AFG"]
        result = validate_canonical(df, source="test")
        assert len(result) == 1

    def test_rejects_missing_columns(self):
        df = pd.DataFrame({"iso3": ["AFG"]})
        with pytest.raises(ValueError, match="missing columns"):
            validate_canonical(df, source="test")

    def test_rejects_extra_columns(self):
        df = pd.DataFrame(
            {col: ["val"] for col in CANONICAL_COLUMNS}
        )
        df["value"] = [100.0]
        df["iso3"] = ["AFG"]
        df["extra_col"] = ["oops"]
        with pytest.raises(ValueError, match="unexpected columns"):
            validate_canonical(df, source="test")

    def test_rejects_non_numeric_value(self):
        df = pd.DataFrame(
            {col: ["val"] for col in CANONICAL_COLUMNS}
        )
        df["value"] = ["not_a_number"]
        df["iso3"] = ["AFG"]
        with pytest.raises(ValueError, match="non-numeric"):
            validate_canonical(df, source="test")

    def test_rejects_bad_iso3(self):
        df = pd.DataFrame(
            {col: ["val"] for col in CANONICAL_COLUMNS}
        )
        df["value"] = [100.0]
        df["iso3"] = ["TOOLONG"]
        with pytest.raises(ValueError, match="iso3"):
            validate_canonical(df, source="test")

    def test_empty_frame_passes(self):
        df = empty_canonical()
        result = validate_canonical(df, source="test")
        assert len(result) == 0
        assert list(result.columns) == CANONICAL_COLUMNS


# ---------------------------------------------------------------------------
# ACLED connector (offline — uses fixture data if no credentials)
# ---------------------------------------------------------------------------


class TestAcledConnectorContract:
    """Verify ACLED wrapper returns the canonical schema."""

    def _make_fixture_rows(self):
        """Build minimal ACLED-style rows without calling the API."""
        from resolver.ingestion.acled_client import CANONICAL_HEADERS

        row = {h: "" for h in CANONICAL_HEADERS}
        row.update(
            event_id="AFG-ACE-fatalities_battle_month-2025-01-abc123",
            country_name="Afghanistan",
            iso3="AFG",
            hazard_code="ACE",
            hazard_label="Armed conflict escalation",
            hazard_class="conflict",
            metric="fatalities_battle_month",
            series_semantics="new",
            value="42",
            unit="persons",
            as_of_date="2025-01-31",
            publication_date="2025-02-01",
            publisher="ACLED",
            source_type="other",
            source_url="https://acleddata.com",
            doc_title="ACLED monthly aggregation",
            definition_text="Battle fatalities",
            method="api",
            confidence="high",
            revision="1",
            ingested_at="2025-02-01T00:00:00Z",
        )
        return [row]

    def test_acled_connector_maps_to_canonical(self, monkeypatch):
        """Monkey-patch collect_rows to avoid hitting the API."""
        fixture_rows = self._make_fixture_rows()
        monkeypatch.setattr(
            "resolver.ingestion.acled_client.collect_rows",
            lambda: fixture_rows,
        )

        from resolver.connectors.acled import AcledConnector

        connector = AcledConnector()
        df = connector.fetch_and_normalize()

        assert list(df.columns) == CANONICAL_COLUMNS
        assert len(df) == 1
        assert df.iloc[0]["iso3"] == "AFG"
        assert df.iloc[0]["hazard_code"] == "ACE"
        validate_canonical(df, source="acled")


# ---------------------------------------------------------------------------
# IDMC connector (offline — mock the client)
# ---------------------------------------------------------------------------


class TestIdmcConnectorContract:
    """Verify IDMC wrapper returns the canonical schema."""

    def test_idmc_connector_maps_to_canonical(self, monkeypatch):
        """Mock the IDMC client chain to avoid network calls."""
        # Minimal IDMC-style normalized data with the 6 fact columns.
        fake_facts = pd.DataFrame(
            {
                "iso3": ["AFG", "ETH"],
                "as_of_date": ["2025-01-31", "2025-01-31"],
                "metric": ["new_displacements", "new_displacements"],
                "value": [5000, 12000],
                "series_semantics": ["new", "new"],
                "source": ["IDMC", "IDMC"],
            }
        )

        # Patch at the source modules (where the deferred imports resolve).
        monkeypatch.setattr(
            "resolver.ingestion.idmc.export.build_resolution_ready_facts",
            lambda normalized: fake_facts,
        )
        monkeypatch.setattr(
            "resolver.ingestion.idmc.client.IdmcClient",
            lambda config: type("FakeClient", (), {"fetch": lambda self: pd.DataFrame({"x": [1]})})(),
        )
        monkeypatch.setattr(
            "resolver.ingestion.idmc.config.load",
            lambda: {},
        )
        monkeypatch.setattr(
            "resolver.ingestion.idmc.normalize.normalize_all",
            lambda raw, config: raw,
        )

        from resolver.connectors.idmc import IdmcConnector

        connector = IdmcConnector()
        df = connector.fetch_and_normalize()

        assert list(df.columns) == CANONICAL_COLUMNS
        assert len(df) == 2
        assert set(df["iso3"]) == {"AFG", "ETH"}
        validate_canonical(df, source="idmc")


# ---------------------------------------------------------------------------
# IFRC Montandon connector (offline — mock collect_rows)
# ---------------------------------------------------------------------------


class TestIfrcMontandonConnectorContract:
    """Verify IFRC Montandon wrapper returns the canonical schema."""

    def _make_fixture_rows(self):
        """Build minimal IFRC GO-style rows (List[List[str]])."""
        return [
            [
                "SDN-FL-ifrcgo-12345",   # event_id
                "Sudan",                  # country_name
                "SDN",                    # iso3
                "FL",                     # hazard_code
                "Flood",                  # hazard_label
                "natural",                # hazard_class
                "affected",               # metric
                "stock",                  # series_semantics
                "75000",                  # value
                "persons",                # unit
                "2025-01-15",             # as_of_date
                "2025-01-16",             # publication_date
                "IFRC",                   # publisher
                "sitrep",                 # source_type
                "https://go.ifrc.org",    # source_url
                "Sudan Floods DREF",      # doc_title
                "Extracted affected via num_affected from IFRC GO field report.",
                "api",                    # method
                "med",                    # confidence
                "1",                      # revision
                "2025-01-16T12:00:00Z",   # ingested_at
            ],
            [
                "BGD-TC-ifrcgo-67890",
                "Bangladesh",
                "BGD",
                "TC",
                "Tropical Cyclone",
                "natural",
                "in_need",
                "stock",
                "120000",
                "persons",
                "2025-02-01",
                "2025-02-02",
                "IFRC",
                "sitrep",
                "https://go.ifrc.org",
                "Cyclone Appeal",
                "Extracted in_need via people in need from IFRC GO appeal.",
                "api",
                "med",
                "1",
                "2025-02-02T08:00:00Z",
            ],
        ]

    def test_ifrc_montandon_connector_maps_to_canonical(self, monkeypatch):
        """Monkey-patch collect_rows to avoid hitting the API."""
        fixture_rows = self._make_fixture_rows()
        monkeypatch.setattr(
            "resolver.ingestion.ifrc_go_client.collect_rows",
            lambda: fixture_rows,
        )
        monkeypatch.delenv("RESOLVER_INGESTION_MODE", raising=False)

        from resolver.connectors.ifrc_montandon import IfrcMontandonConnector

        connector = IfrcMontandonConnector()
        df = connector.fetch_and_normalize()

        assert list(df.columns) == CANONICAL_COLUMNS
        assert len(df) == 2
        assert set(df["iso3"]) == {"SDN", "BGD"}
        assert set(df["hazard_code"]) == {"FL", "TC"}
        assert df.iloc[0]["publisher"] == "IFRC"
        assert df.iloc[0]["series_semantics"] == "stock"
        validate_canonical(df, source="ifrc_montandon")

    def test_ifrc_montandon_stubs_mode_returns_empty(self, monkeypatch):
        """In stubs mode the connector should return empty without calling the API."""
        monkeypatch.setenv("RESOLVER_INGESTION_MODE", "stubs")

        from resolver.connectors.ifrc_montandon import IfrcMontandonConnector

        connector = IfrcMontandonConnector()
        df = connector.fetch_and_normalize()

        assert list(df.columns) == CANONICAL_COLUMNS
        assert len(df) == 0
