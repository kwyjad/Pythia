# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for the IPC API connector."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

try:
    import pycountry  # noqa: F401
except ImportError:
    pytest.skip("pycountry not installed", allow_module_level=True)

from resolver.connectors.ipc_api import (
    IpcApiConnector,
    _iso2_to_iso3,
    _load_fewsnet_country_list,
    _write_country_list,
)
from resolver.connectors.protocol import CANONICAL_COLUMNS
from resolver.connectors.validate import validate_canonical

# ---------------------------------------------------------------------------
# Sample JSON fixture (IPC API response format)
# ---------------------------------------------------------------------------

SAMPLE_RECORDS = [
    {
        "country": "AF",
        "ipc_period": "A",
        "p3plus": 15800000,
        "analysis_period_start": "2025-06-01",
        "analysis_period_end": "2025-06-30",
        "analysis_date": "2025-05-20",
    },
    {
        "country": "AF",
        "ipc_period": "P",
        "p3plus": 17200000,
        "analysis_period_start": "2025-06-01",
        "analysis_period_end": "2025-10-31",
        "analysis_date": "2025-05-20",
    },
    {
        "country": "PK",
        "ipc_period": "A",
        "p3plus": 9400000,
        "analysis_period_start": "2025-06-01",
        "analysis_period_end": "2025-06-30",
        "analysis_date": "2025-05-15",
    },
    # This record has country in FEWS NET list — should be excluded
    {
        "country": "ET",
        "ipc_period": "A",
        "p3plus": 17000000,
        "analysis_period_start": "2025-06-01",
        "analysis_period_end": "2025-06-30",
        "analysis_date": "2025-05-15",
    },
    # Invalid country code — should be skipped
    {
        "country": "XX",
        "ipc_period": "A",
        "p3plus": 100000,
        "analysis_period_start": "2025-06-01",
        "analysis_period_end": "2025-06-30",
        "analysis_date": "2025-05-15",
    },
]

# Duplicate records for dedup testing
SAMPLE_RECORDS_DEDUP = [
    {
        "country": "AF",
        "ipc_period": "A",
        "p3plus": 15000000,
        "analysis_period_start": "2025-06-01",
        "analysis_period_end": "2025-06-30",
        "analysis_date": "2025-04-01",
    },
    {
        "country": "AF",
        "ipc_period": "A",
        "p3plus": 15800000,
        "analysis_period_start": "2025-06-01",
        "analysis_period_end": "2025-06-30",
        "analysis_date": "2025-05-20",
    },
]

# FEWS NET countries to exclude
FEWSNET_COUNTRIES = ["ETH", "SOM", "KEN", "SDN", "SSD"]


# ---------------------------------------------------------------------------
# ISO2 → ISO3 conversion tests
# ---------------------------------------------------------------------------


class TestIso2ToIso3:
    def test_known_codes(self):
        assert _iso2_to_iso3("AF") == "AFG"
        assert _iso2_to_iso3("PK") == "PAK"
        assert _iso2_to_iso3("LB") == "LBN"
        assert _iso2_to_iso3("GT") == "GTM"

    def test_lowercase(self):
        assert _iso2_to_iso3("af") == "AFG"

    def test_whitespace(self):
        assert _iso2_to_iso3("  AF  ") == "AFG"

    def test_invalid_code(self):
        assert _iso2_to_iso3("XX") is None
        assert _iso2_to_iso3("") is None
        assert _iso2_to_iso3("ABC") is None


# ---------------------------------------------------------------------------
# FEWS NET exclusion tests
# ---------------------------------------------------------------------------


class TestFewsnetExclusion:
    def test_loads_fewsnet_list(self, tmp_path):
        json_path = tmp_path / "fewsnet_countries.json"
        json_path.write_text(json.dumps(FEWSNET_COUNTRIES))

        with patch(
            "resolver.connectors.ipc_api._FEWSNET_COUNTRIES_JSON", json_path
        ):
            result = _load_fewsnet_country_list()

        assert "ETH" in result
        assert "SOM" in result
        assert len(result) == len(FEWSNET_COUNTRIES)

    def test_missing_file_returns_empty(self, tmp_path):
        json_path = tmp_path / "nonexistent.json"
        with patch(
            "resolver.connectors.ipc_api._FEWSNET_COUNTRIES_JSON", json_path
        ):
            result = _load_fewsnet_country_list()
        assert result == set()


# ---------------------------------------------------------------------------
# Country list generation tests
# ---------------------------------------------------------------------------


class TestCountryList:
    def test_writes_json(self, tmp_path):
        json_path = tmp_path / "ipc_countries.json"
        with patch(
            "resolver.connectors.ipc_api._IPC_COUNTRIES_JSON", json_path
        ):
            _write_country_list(["AFG", "PAK", "AFG", "GTM"])

        data = json.loads(json_path.read_text())
        assert data == ["AFG", "GTM", "PAK"]  # sorted, deduplicated


# ---------------------------------------------------------------------------
# Full connector integration test (mocked HTTP)
# ---------------------------------------------------------------------------


class TestIpcApiConnector:
    def _mock_response(self, payload) -> MagicMock:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = payload
        resp.raise_for_status = MagicMock()
        return resp

    @patch("resolver.connectors.ipc_api._write_country_list")
    @patch("resolver.connectors.ipc_api._load_fewsnet_country_list")
    @patch("resolver.connectors.ipc_api._build_session")
    def test_canonical_output(
        self, mock_session_fn, mock_fewsnet, mock_write_countries
    ):
        session = MagicMock()
        session.get.return_value = self._mock_response(SAMPLE_RECORDS)
        mock_session_fn.return_value = session
        mock_fewsnet.return_value = {"ETH", "SOM", "KEN", "SDN", "SSD"}

        connector = IpcApiConnector()
        with patch.dict(
            "os.environ",
            {"IPC_API_KEY": "test-key", "IPC_API_MONTHS": "24", "IPC_API_REQUEST_DELAY": "0"},
        ):
            df = connector.fetch_and_normalize()

        # Should have correct columns
        assert list(df.columns) == CANONICAL_COLUMNS

        # Should only have Current Situation and First Projection metrics
        assert set(df["metric"].unique()) == {
            "phase3plus_in_need",
            "phase3plus_projection",
        }

        # Should not have ETH (FEWS NET country) or XX (invalid ISO2)
        assert "ETH" not in df["iso3"].values
        assert "XX" not in df["iso3"].values  # XX is invalid, _iso2_to_iso3 returns None

        # Should have AFG and PAK
        assert "AFG" in df["iso3"].values
        assert "PAK" in df["iso3"].values

        # All rows should have hazard_code DR
        assert (df["hazard_code"] == "DR").all()

        # All rows should have unit "persons"
        assert (df["unit"] == "persons").all()

        # All rows should have publisher "IPC"
        assert (df["publisher"] == "IPC").all()

        # Validate canonical schema
        validate_canonical(df, source="test_ipc_api")

        # Country list should have been written
        mock_write_countries.assert_called_once()
        iso3_list = mock_write_countries.call_args[0][0]
        assert "AFG" in iso3_list
        assert "PAK" in iso3_list
        assert "ETH" not in iso3_list

    @patch("resolver.connectors.ipc_api._write_country_list")
    @patch("resolver.connectors.ipc_api._load_fewsnet_country_list")
    @patch("resolver.connectors.ipc_api._build_session")
    def test_deduplication(
        self, mock_session_fn, mock_fewsnet, mock_write_countries
    ):
        """If duplicate (iso3, period, ym) rows exist, keep latest analysis_date."""
        session = MagicMock()
        session.get.return_value = self._mock_response(SAMPLE_RECORDS_DEDUP)
        mock_session_fn.return_value = session
        mock_fewsnet.return_value = set()

        connector = IpcApiConnector()
        with patch.dict(
            "os.environ",
            {"IPC_API_KEY": "test-key", "IPC_API_MONTHS": "24", "IPC_API_REQUEST_DELAY": "0"},
        ):
            df = connector.fetch_and_normalize()

        # Should only have 1 row (deduplicated)
        afg_current = df[
            (df["iso3"] == "AFG") & (df["metric"] == "phase3plus_in_need")
        ]
        assert len(afg_current) == 1
        assert afg_current.iloc[0]["value"] == 15800000.0

    @patch("resolver.connectors.ipc_api._build_session")
    def test_no_api_key(self, mock_session_fn):
        """Missing API key should return empty canonical DataFrame."""
        connector = IpcApiConnector()
        with patch.dict("os.environ", {"IPC_API_KEY": ""}, clear=False):
            df = connector.fetch_and_normalize()

        assert df.empty
        assert list(df.columns) == CANONICAL_COLUMNS
        mock_session_fn.assert_not_called()

    @patch("resolver.connectors.ipc_api._build_session")
    def test_http_error(self, mock_session_fn):
        """HTTP error should return empty canonical DataFrame."""
        import requests as req

        session = MagicMock()
        session.get.side_effect = req.RequestException("Connection failed")
        mock_session_fn.return_value = session

        connector = IpcApiConnector()
        with patch.dict(
            "os.environ",
            {"IPC_API_KEY": "test-key", "IPC_API_MONTHS": "24", "IPC_API_REQUEST_DELAY": "0"},
        ):
            df = connector.fetch_and_normalize()

        assert df.empty
        assert list(df.columns) == CANONICAL_COLUMNS

    @patch("resolver.connectors.ipc_api._build_session")
    def test_empty_response(self, mock_session_fn):
        """Empty response should return empty canonical DataFrame."""
        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = []
        resp.raise_for_status = MagicMock()
        session.get.return_value = resp
        mock_session_fn.return_value = session

        connector = IpcApiConnector()
        with patch.dict(
            "os.environ",
            {"IPC_API_KEY": "test-key", "IPC_API_MONTHS": "24", "IPC_API_REQUEST_DELAY": "0"},
        ):
            df = connector.fetch_and_normalize()

        assert df.empty
        assert list(df.columns) == CANONICAL_COLUMNS

    @patch("resolver.connectors.ipc_api._write_country_list")
    @patch("resolver.connectors.ipc_api._load_fewsnet_country_list")
    @patch("resolver.connectors.ipc_api._build_session")
    def test_phase_sum_fallback(
        self, mock_session_fn, mock_fewsnet, mock_write_countries
    ):
        """If p3plus is not present, sum phase3+phase4+phase5."""
        records = [
            {
                "country": "AF",
                "ipc_period": "A",
                "phase3_population": 10000000,
                "phase4_population": 4000000,
                "phase5_population": 1000000,
                "analysis_period_start": "2025-06-01",
                "analysis_period_end": "2025-06-30",
                "analysis_date": "2025-05-20",
            },
        ]
        session = MagicMock()
        session.get.return_value = self._mock_response(records)
        mock_session_fn.return_value = session
        mock_fewsnet.return_value = set()

        connector = IpcApiConnector()
        with patch.dict(
            "os.environ",
            {"IPC_API_KEY": "test-key", "IPC_API_MONTHS": "24", "IPC_API_REQUEST_DELAY": "0"},
        ):
            df = connector.fetch_and_normalize()

        assert len(df) == 1
        assert df.iloc[0]["value"] == 15000000.0
