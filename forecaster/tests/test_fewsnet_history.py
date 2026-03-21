# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

duckdb = pytest.importorskip("duckdb")

import forecaster.cli as cli  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_connection(rows):
    """Return a mock duckdb connection that returns *rows* from execute().fetchall()."""
    mock_con = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchall.return_value = rows
    mock_con.execute.return_value = mock_result
    return mock_con


def _sample_rows(today: date | None = None):
    """Build sample DB rows for a few months of FEWS NET data.

    Returns rows in DESC ym order (as the SQL query would).
    """
    today = today or date.today()
    y, m = today.year, today.month

    def _ym_back(offset):
        nm = m - offset
        ny = y
        while nm < 1:
            nm += 12
            ny -= 1
        return date(ny, nm, 1)

    return [
        # (ym, value, created_at) — most recent first
        (_ym_back(0), 1_200_000, date(y, m, 15)),
        (_ym_back(1), 1_100_000, date(y, m - 1 if m > 1 else 12, 15)),
        (_ym_back(2), None, None),  # gap month — null
        (_ym_back(3), 900_000, date(y, m, 10)),
        (_ym_back(5), 800_000, date(y, m, 10)),
        (_ym_back(7), 700_000, date(y, m, 10)),
        (_ym_back(9), 600_000, date(y, m, 10)),
        (_ym_back(11), 500_000, date(y, m, 10)),
    ]


# ---------------------------------------------------------------------------
# Tests for _load_fewsnet_phase3_history
# ---------------------------------------------------------------------------

class TestLoadFewsnetPhase3History:
    """Tests for _load_fewsnet_phase3_history."""

    @patch("forecaster.cli.duckdb")
    @patch("forecaster.cli._pythia_db_path_from_config", return_value=":memory:")
    def test_returns_correct_structure(self, mock_path, mock_duckdb):
        """Result dict has the expected top-level keys and type."""
        rows = _sample_rows()
        mock_con = _make_mock_connection(rows)
        mock_duckdb.connect.return_value = mock_con

        result = cli._load_fewsnet_phase3_history("ETH", months=36)

        assert result["type"] == "fewsnet_phase3"
        assert result["source"] == "FEWSNET_IPC"
        assert "history_length_months" in result
        assert "observed_months" in result
        assert "coverage_pct" in result
        assert "recent_mean" in result
        assert "recent_max" in result
        assert "trend" in result
        assert "data_quality" in result
        assert "last_6m_values" in result
        assert "notes" in result

    @patch("forecaster.cli.duckdb")
    @patch("forecaster.cli._pythia_db_path_from_config", return_value=":memory:")
    def test_null_months_preserved(self, mock_path, mock_duckdb):
        """Months without data should appear as None, not zero."""
        rows = _sample_rows()
        mock_con = _make_mock_connection(rows)
        mock_duckdb.connect.return_value = mock_con

        result = cli._load_fewsnet_phase3_history("ETH", months=36)

        last_6m = result["last_6m_values"]
        assert isinstance(last_6m, list)
        assert len(last_6m) == 6

        # There should be at least one None value in last_6m (the gap month
        # or months outside the data_by_ym set).
        values = [e["value"] for e in last_6m]
        assert None in values, "Null months should be preserved as None, not converted to zero"

        # No value should be 0 when the original was None
        for entry in last_6m:
            assert entry["value"] is None or entry["value"] > 0, (
                f"Value for {entry['ym']} should be None or positive, got {entry['value']}"
            )

    @patch("forecaster.cli.duckdb")
    @patch("forecaster.cli._pythia_db_path_from_config", return_value=":memory:")
    def test_statistics_over_observed_only(self, mock_path, mock_duckdb):
        """Statistics (mean, max) should be computed over observed months only."""
        rows = _sample_rows()
        mock_con = _make_mock_connection(rows)
        mock_duckdb.connect.return_value = mock_con

        result = cli._load_fewsnet_phase3_history("ETH", months=36)

        # observed_months should count only non-null, positive values
        assert result["observed_months"] > 0
        # The None row should NOT be counted
        # We have 7 non-null rows out of 8 total
        assert result["observed_months"] == 7

    @patch("forecaster.cli.duckdb")
    @patch("forecaster.cli._pythia_db_path_from_config", return_value=":memory:")
    def test_coverage_percentage(self, mock_path, mock_duckdb):
        """Coverage pct = observed / total months * 100."""
        rows = _sample_rows()
        mock_con = _make_mock_connection(rows)
        mock_duckdb.connect.return_value = mock_con

        result = cli._load_fewsnet_phase3_history("ETH", months=36)

        expected_coverage = round(7 / 36 * 100, 1)
        assert result["coverage_pct"] == expected_coverage

    @patch("forecaster.cli.duckdb")
    @patch("forecaster.cli._pythia_db_path_from_config", return_value=":memory:")
    def test_empty_rows_returns_zero_coverage(self, mock_path, mock_duckdb):
        """When no data exists, return a note dict with 0 coverage."""
        mock_con = _make_mock_connection([])
        mock_duckdb.connect.return_value = mock_con

        result = cli._load_fewsnet_phase3_history("ZZZ", months=36)

        assert result["type"] == "fewsnet_phase3"
        assert result["observed_months"] == 0
        assert result["coverage_pct"] == 0.0
        assert "note" in result

    def test_db_connection_failure_returns_error(self):
        """When the DB cannot be connected, return an error dict."""
        with patch("forecaster.cli._pythia_db_path_from_config", side_effect=Exception("boom")):
            with patch("forecaster.cli.duckdb") as mock_duckdb:
                mock_duckdb.connect.side_effect = Exception("boom")
                result = cli._load_fewsnet_phase3_history("ETH")

        assert result["type"] == "fewsnet_phase3"
        assert result.get("error") == "missing_db"


# ---------------------------------------------------------------------------
# Tests for _format_base_rate_for_prompt (fewsnet_phase3 type)
# ---------------------------------------------------------------------------

class TestFormatFewsnetPhase3:
    """Tests for fewsnet_phase3 formatting in _format_base_rate_for_prompt."""

    def test_null_months_shown_as_null_in_output(self):
        """Formatted output should show 'null' for months with no data."""
        summary = {
            "type": "fewsnet_phase3",
            "source": "FEWSNET_IPC",
            "history_length_months": 36,
            "observed_months": 4,
            "coverage_pct": 11.1,
            "recent_mean": 1_000_000,
            "recent_max": 1_200_000,
            "trend": "stable",
            "trend_pct": 2.0,
            "data_quality": "low",
            "last_6m_values": [
                {"ym": "2025-10", "value": None},
                {"ym": "2025-11", "value": 900_000},
                {"ym": "2025-12", "value": None},
                {"ym": "2026-01", "value": 1_100_000},
                {"ym": "2026-02", "value": None},
                {"ym": "2026-03", "value": 1_200_000},
            ],
            "notes": "Test notes.",
        }

        output = cli._format_base_rate_for_prompt(
            summary,
            forecast_keys=["2026-04", "2026-05"],
            iso3="ETH",
            hazard_code="DR",
        )

        assert "FEWS NET IPC Phase 3+" in output
        assert "null" in output
        # Verify observed values are formatted with commas
        assert "1,200,000" in output or "1200000" in output
        assert "Note:" in output


# ---------------------------------------------------------------------------
# Tests for _infer_resolution_source
# ---------------------------------------------------------------------------

class TestInferResolutionSource:
    """Test _infer_resolution_source returns FEWSNET_IPC for DR/PHASE3PLUS_IN_NEED."""

    def test_dr_phase3plus_returns_fewsnet(self):
        assert cli._infer_resolution_source("DR", "PHASE3PLUS_IN_NEED") == "FEWSNET_IPC"

    def test_dr_phase3plus_case_insensitive(self):
        assert cli._infer_resolution_source("dr", "phase3plus_in_need") == "FEWSNET_IPC"

    def test_other_metrics_unchanged(self):
        """Existing dispatch should not be broken."""
        assert cli._infer_resolution_source("ACE", "FATALITIES") == "ACLED"
        assert cli._infer_resolution_source("ACE", "PA") == "IDMC"
        assert cli._infer_resolution_source("DR", "PA") == "IFRC"
        assert cli._infer_resolution_source("FL", "EVENT_OCCURRENCE") == "GDACS"
