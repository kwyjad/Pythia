# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for the FEWS NET IPC connector."""

from __future__ import annotations

import json
import textwrap
from datetime import date, datetime, timezone
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

try:
    import pycountry  # noqa: F401
except ImportError:
    pytest.skip("pycountry not installed", allow_module_level=True)

from resolver.connectors.fewsnet_ipc import (
    FewsnetIpcConnector,
    _iso2_to_iso3,
    _write_country_list,
    _FEWSNET_COUNTRIES_JSON,
)
from resolver.connectors.protocol import CANONICAL_COLUMNS
from resolver.connectors.validate import validate_canonical

# ---------------------------------------------------------------------------
# Sample CSV fixture (UTF-8 BOM header, 2 countries, 2 scenarios)
# ---------------------------------------------------------------------------

SAMPLE_CSV = textwrap.dedent("""\
    country_code,phase,scenario_name,projection_start,projection_end,value,low_value,high_value,phase_name,population_range,fnid,reporting_date
    ET,3+,Current Situation,2025-06-01,2025-06-30,17000000.0000000000000000,15000000.0,19000000.0,Phase 3 and above,15-19 million,ET,2025-05-15
    ET,3+,Most Likely,2025-06-01,2025-06-30,18000000.0000000000000000,16000000.0,20000000.0,Phase 3 and above,16-20 million,ET,2025-05-15
    SO,3+,Current Situation,2025-06-01,2025-06-30,4500000.0000000000000000,4000000.0,5000000.0,Phase 3 and above,4-5 million,SO,2025-05-15
    SO,3+,Most Likely,2025-06-01,2025-06-30,5000000.0000000000000000,4500000.0,5500000.0,Phase 3 and above,4.5-5.5 million,SO,2025-05-15
    ET,3+,Current Situation,2025-07-01,2025-07-31,17500000.0000000000000000,15500000.0,19500000.0,Phase 3 and above,15.5-19.5 million,ET,2025-06-10
    ET,3+,Peak Needs,2025-07-01,2025-07-31,20000000.0000000000000000,18000000.0,22000000.0,Phase 3 and above,18-22 million,ET,2025-06-10
    XX,3+,Current Situation,2025-06-01,2025-06-30,100000.0,90000.0,110000.0,Phase 3 and above,90k-110k,XX,2025-05-15
""")

# Same CSV with BOM prefix
SAMPLE_CSV_BOM = "\ufeff" + SAMPLE_CSV

# Duplicate reporting dates — ET has two Current Situation rows for 2025-06
# with different reporting dates.
SAMPLE_CSV_DEDUP = textwrap.dedent("""\
    country_code,phase,scenario_name,projection_start,projection_end,value,low_value,high_value,phase_name,population_range,fnid,reporting_date
    ET,3+,Current Situation,2025-06-01,2025-06-30,17000000.0,15000000.0,19000000.0,Phase 3 and above,15-19 million,ET,2025-05-01
    ET,3+,Current Situation,2025-06-01,2025-06-30,17500000.0,15500000.0,19500000.0,Phase 3 and above,15.5-19.5 million,ET,2025-05-15
""")


# ---------------------------------------------------------------------------
# ISO2 → ISO3 conversion tests
# ---------------------------------------------------------------------------


class TestIso2ToIso3:
    def test_known_codes(self):
        assert _iso2_to_iso3("ET") == "ETH"
        assert _iso2_to_iso3("SO") == "SOM"
        assert _iso2_to_iso3("US") == "USA"
        assert _iso2_to_iso3("KE") == "KEN"

    def test_lowercase(self):
        assert _iso2_to_iso3("et") == "ETH"

    def test_whitespace(self):
        assert _iso2_to_iso3("  ET  ") == "ETH"

    def test_invalid_code(self):
        assert _iso2_to_iso3("XX") is None
        assert _iso2_to_iso3("") is None
        assert _iso2_to_iso3("ABC") is None


# ---------------------------------------------------------------------------
# Scenario filtering tests
# ---------------------------------------------------------------------------


class TestScenarioFiltering:
    def _parse_sample(self) -> pd.DataFrame:
        return pd.read_csv(StringIO(SAMPLE_CSV))

    def test_keeps_current_situation_and_most_likely(self):
        df = self._parse_sample()
        wanted = {"Current Situation", "Most Likely"}
        filtered = df[df["scenario_name"].isin(wanted)]
        # Should drop the "Peak Needs" row and the XX row (kept by filter,
        # but XX is invalid ISO2 — tested separately)
        assert "Peak Needs" not in filtered["scenario_name"].values
        assert "Current Situation" in filtered["scenario_name"].values
        assert "Most Likely" in filtered["scenario_name"].values

    def test_peak_needs_excluded(self):
        df = self._parse_sample()
        wanted = {"Current Situation", "Most Likely"}
        filtered = df[df["scenario_name"].isin(wanted)]
        assert not any(filtered["scenario_name"] == "Peak Needs")


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_latest_reporting_date_wins(self):
        df = pd.read_csv(StringIO(SAMPLE_CSV_DEDUP))
        df["projection_start"] = pd.to_datetime(df["projection_start"])
        df["reporting_date"] = pd.to_datetime(df["reporting_date"])
        df["ym"] = df["projection_start"].dt.strftime("%Y-%m")

        df = df.sort_values("reporting_date", ascending=False)
        df = df.drop_duplicates(
            subset=["country_code", "scenario_name", "ym"], keep="first"
        )

        assert len(df) == 1
        assert df.iloc[0]["value"] == 17500000.0
        assert df.iloc[0]["reporting_date"].strftime("%Y-%m-%d") == "2025-05-15"


# ---------------------------------------------------------------------------
# Country list generation tests
# ---------------------------------------------------------------------------


class TestCountryList:
    def test_writes_json(self, tmp_path):
        json_path = tmp_path / "fewsnet_countries.json"
        with patch(
            "resolver.connectors.fewsnet_ipc._FEWSNET_COUNTRIES_JSON", json_path
        ):
            _write_country_list(["ETH", "SOM", "ETH", "KEN"])

        data = json.loads(json_path.read_text())
        assert data == ["ETH", "KEN", "SOM"]  # sorted, deduplicated


# ---------------------------------------------------------------------------
# Full connector integration test (mocked HTTP)
# ---------------------------------------------------------------------------


class TestFewsnetIpcConnector:
    def _mock_response(self, csv_text: str) -> MagicMock:
        resp = MagicMock()
        resp.status_code = 200
        resp.content = csv_text.encode("utf-8-sig")
        resp.raise_for_status = MagicMock()
        return resp

    @patch("resolver.connectors.fewsnet_ipc._write_country_list")
    @patch("resolver.connectors.fewsnet_ipc._build_session")
    def test_canonical_output(self, mock_session_fn, mock_write_countries):
        session = MagicMock()
        session.get.return_value = self._mock_response(SAMPLE_CSV_BOM)
        mock_session_fn.return_value = session

        connector = FewsnetIpcConnector()
        with patch.dict("os.environ", {"FEWSNET_MONTHS": "12", "FEWSNET_REQUEST_DELAY": "0"}):
            df = connector.fetch_and_normalize()

        # Should have correct columns
        assert list(df.columns) == CANONICAL_COLUMNS

        # Should only have Current Situation and Most Likely rows
        assert set(df["metric"].unique()) == {
            "phase3plus_in_need",
            "phase3plus_projection",
        }

        # Should not have XX (invalid ISO2) or Peak Needs
        assert "XX" not in df["iso3"].values
        assert "Peak Needs" not in df["doc_title"].values

        # All rows should have hazard_code DR
        assert (df["hazard_code"] == "DR").all()

        # All rows should have unit "persons"
        assert (df["unit"] == "persons").all()

        # All rows should have publisher "FEWS NET"
        assert (df["publisher"] == "FEWS NET").all()

        # Validate canonical schema
        validate_canonical(df, source="test_fewsnet_ipc")

        # Country list should have been written
        mock_write_countries.assert_called_once()
        iso3_list = mock_write_countries.call_args[0][0]
        assert "ETH" in iso3_list
        assert "SOM" in iso3_list

    @patch("resolver.connectors.fewsnet_ipc._write_country_list")
    @patch("resolver.connectors.fewsnet_ipc._build_session")
    def test_deduplication_in_connector(self, mock_session_fn, mock_write_countries):
        """If duplicate (iso3, scenario, ym) rows exist, keep latest reporting_date."""
        session = MagicMock()
        csv_bom = "\ufeff" + SAMPLE_CSV_DEDUP
        session.get.return_value = self._mock_response(csv_bom)
        mock_session_fn.return_value = session

        connector = FewsnetIpcConnector()
        with patch.dict("os.environ", {"FEWSNET_MONTHS": "12", "FEWSNET_REQUEST_DELAY": "0"}):
            df = connector.fetch_and_normalize()

        # Should only have 1 row (deduplicated)
        eth_current = df[
            (df["iso3"] == "ETH") & (df["metric"] == "phase3plus_in_need")
        ]
        assert len(eth_current) == 1
        assert eth_current.iloc[0]["value"] == 17500000.0

    @patch("resolver.connectors.fewsnet_ipc._build_session")
    def test_empty_response(self, mock_session_fn):
        """Empty CSV should return empty canonical DataFrame."""
        resp = MagicMock()
        resp.status_code = 200
        resp.content = b""
        resp.raise_for_status = MagicMock()

        session = MagicMock()
        session.get.return_value = resp
        mock_session_fn.return_value = session

        connector = FewsnetIpcConnector()
        with patch.dict("os.environ", {"FEWSNET_MONTHS": "12", "FEWSNET_REQUEST_DELAY": "0"}):
            df = connector.fetch_and_normalize()

        assert df.empty
        assert list(df.columns) == CANONICAL_COLUMNS

    @patch("resolver.connectors.fewsnet_ipc._build_session")
    def test_http_error(self, mock_session_fn):
        """HTTP error should return empty canonical DataFrame."""
        session = MagicMock()
        import requests as req
        session.get.side_effect = req.RequestException("Connection failed")
        mock_session_fn.return_value = session

        connector = FewsnetIpcConnector()
        with patch.dict("os.environ", {"FEWSNET_MONTHS": "12", "FEWSNET_REQUEST_DELAY": "0"}):
            df = connector.fetch_and_normalize()

        assert df.empty
        assert list(df.columns) == CANONICAL_COLUMNS
