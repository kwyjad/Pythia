# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for the conflictforecast.org connector.

Covers:
- File listing parsing from the Backendless API
- _find_file_url regex matching against file name patterns
- CSV download and parse via _download_csv
- Metric name and lead_months mapping from _FILE_PATTERNS
- _derive_issue_date from file name and created-timestamp metadata
- _transform_csv column detection, ISO3 validation, value extraction
- Error handling when the API is unreachable
- Missing ISO3 column detection
"""

from __future__ import annotations

import math
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from resolver.connectors.conflictforecast import (
    ConflictForecastOrgConnector,
    _FILE_PATTERNS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_listing(
    *names: str,
    created: int | float | None = None,
) -> list[dict]:
    """Build a fake file listing with the given file names."""
    entries = []
    for name in names:
        entry: dict = {
            "name": name,
            "publicUrl": f"https://files.example.com/{name}",
        }
        if created is not None:
            entry["created"] = created
        entries.append(entry)
    return entries


def _csv_text(rows: list[list[str]], header: list[str] | None = None) -> str:
    """Build a simple CSV string from rows."""
    if header is None:
        header = rows[0]
        rows = rows[1:]
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(c) for c in row))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File listing parsing
# ---------------------------------------------------------------------------


class TestGetFileListing:
    """Test _get_file_listing HTTP interaction."""

    @patch("resolver.connectors.conflictforecast.requests.get")
    def test_returns_list_from_api(self, mock_get: MagicMock) -> None:
        listing = _make_listing("Armed_Conflict_3m_01-2025.csv")
        mock_resp = MagicMock()
        mock_resp.json.return_value = listing
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        connector = ConflictForecastOrgConnector()
        result = connector._get_file_listing()

        assert result == listing
        mock_get.assert_called_once()
        mock_resp.raise_for_status.assert_called_once()

    @patch("resolver.connectors.conflictforecast.requests.get")
    def test_returns_empty_list_when_non_list_response(
        self, mock_get: MagicMock
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": "unexpected"}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        connector = ConflictForecastOrgConnector()
        result = connector._get_file_listing()

        assert result == []

    @patch("resolver.connectors.conflictforecast.requests.get")
    def test_raises_on_http_error(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("503 Service Unavailable")
        mock_get.return_value = mock_resp

        connector = ConflictForecastOrgConnector()
        with pytest.raises(Exception, match="503"):
            connector._get_file_listing()


# ---------------------------------------------------------------------------
# _find_file_url
# ---------------------------------------------------------------------------


class TestFindFileUrl:
    """Test regex-based file URL matching."""

    def test_matches_armed_conflict_3m(self) -> None:
        listing = _make_listing(
            "Armed_Conflict_3m_01-2025.csv",
            "Violence_Intensity_3m_01-2025.csv",
        )
        url = ConflictForecastOrgConnector._find_file_url(
            listing, r"armed.?conflict.*3"
        )
        assert url == "https://files.example.com/Armed_Conflict_3m_01-2025.csv"

    def test_matches_armed_conflict_12m(self) -> None:
        listing = _make_listing(
            "Armed_Conflict_12m_01-2025.csv",
            "Armed_Conflict_3m_01-2025.csv",
        )
        url = ConflictForecastOrgConnector._find_file_url(
            listing, r"armed.?conflict.*12"
        )
        assert url == "https://files.example.com/Armed_Conflict_12m_01-2025.csv"

    def test_matches_violence_intensity_3m(self) -> None:
        listing = _make_listing(
            "ArmedConflict_3m.csv",
            "Violence_Intensity_3m_01-2025.csv",
        )
        url = ConflictForecastOrgConnector._find_file_url(
            listing, r"violence.?intensity.*3"
        )
        assert url == "https://files.example.com/Violence_Intensity_3m_01-2025.csv"

    def test_case_insensitive_matching(self) -> None:
        listing = _make_listing("ARMED_CONFLICT_3M_FORECAST.csv")
        url = ConflictForecastOrgConnector._find_file_url(
            listing, r"armed.?conflict.*3"
        )
        assert url is not None

    def test_returns_none_when_no_match(self) -> None:
        listing = _make_listing("unrelated_file.csv", "README.txt")
        url = ConflictForecastOrgConnector._find_file_url(
            listing, r"armed.?conflict.*3"
        )
        assert url is None

    def test_returns_none_for_empty_listing(self) -> None:
        url = ConflictForecastOrgConnector._find_file_url([], r"armed.?conflict.*3")
        assert url is None

    def test_handles_entry_with_missing_name(self) -> None:
        listing = [{"publicUrl": "https://example.com/file.csv"}]
        url = ConflictForecastOrgConnector._find_file_url(
            listing, r"armed.?conflict.*3"
        )
        assert url is None

    def test_returns_empty_string_when_publicUrl_missing(self) -> None:
        listing = [{"name": "Armed_Conflict_3m.csv"}]
        url = ConflictForecastOrgConnector._find_file_url(
            listing, r"armed.?conflict.*3"
        )
        assert url == ""


# ---------------------------------------------------------------------------
# CSV download and parse
# ---------------------------------------------------------------------------


class TestDownloadCsv:
    """Test _download_csv fetches and parses CSV content."""

    @patch("resolver.connectors.conflictforecast.requests.get")
    def test_parses_csv_content(self, mock_get: MagicMock) -> None:
        csv_content = _csv_text([
            ["country", "iso3", "probability"],
            ["Afghanistan", "AFG", "0.85"],
            ["Kenya", "KEN", "0.32"],
        ])
        mock_resp = MagicMock()
        mock_resp.text = csv_content
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        df = ConflictForecastOrgConnector._download_csv("https://example.com/f.csv")

        assert len(df) == 2
        assert list(df.columns) == ["country", "iso3", "probability"]
        assert df.iloc[0]["iso3"] == "AFG"
        assert df.iloc[1]["iso3"] == "KEN"

    @patch("resolver.connectors.conflictforecast.requests.get")
    def test_raises_on_http_error(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("404 Not Found")
        mock_get.return_value = mock_resp

        with pytest.raises(Exception, match="404"):
            ConflictForecastOrgConnector._download_csv("https://example.com/gone.csv")


# ---------------------------------------------------------------------------
# Metric / lead_months mapping
# ---------------------------------------------------------------------------


class TestFilePatterns:
    """Verify _FILE_PATTERNS map to the correct metric names and lead months."""

    def test_armed_conflict_3m_pattern(self) -> None:
        pattern, metric, lead = _FILE_PATTERNS[0]
        assert metric == "cf_armed_conflict_risk_3m"
        assert lead == 3

    def test_armed_conflict_12m_pattern(self) -> None:
        pattern, metric, lead = _FILE_PATTERNS[1]
        assert metric == "cf_armed_conflict_risk_12m"
        assert lead == 12

    def test_violence_intensity_3m_pattern(self) -> None:
        pattern, metric, lead = _FILE_PATTERNS[2]
        assert metric == "cf_violence_intensity_3m"
        assert lead == 3

    def test_three_patterns_defined(self) -> None:
        assert len(_FILE_PATTERNS) == 3


# ---------------------------------------------------------------------------
# _derive_issue_date
# ---------------------------------------------------------------------------


class TestDeriveIssueDate:
    """Test issue date extraction from file listing metadata."""

    def test_from_filename_mm_yyyy(self) -> None:
        listing = _make_listing("Armed_Conflict_3m_03-2025.csv")
        result = ConflictForecastOrgConnector._derive_issue_date(listing)
        assert result == date(2025, 3, 1)

    def test_from_filename_takes_first_match(self) -> None:
        listing = _make_listing(
            "Armed_Conflict_3m_06-2025.csv",
            "Violence_Intensity_3m_07-2025.csv",
        )
        result = ConflictForecastOrgConnector._derive_issue_date(listing)
        assert result == date(2025, 6, 1)

    def test_from_created_timestamp_millis(self) -> None:
        # 2025-01-15 12:00:00 UTC in milliseconds
        ts_ms = 1736942400000
        listing = [{"name": "some_file.csv", "created": ts_ms}]
        result = ConflictForecastOrgConnector._derive_issue_date(listing)
        # Should be first of the month of the timestamp
        assert result.day == 1

    def test_filename_preferred_over_created(self) -> None:
        listing = [
            {
                "name": "Armed_Conflict_3m_09-2025.csv",
                "created": 1736942400000,
            }
        ]
        result = ConflictForecastOrgConnector._derive_issue_date(listing)
        assert result == date(2025, 9, 1)

    def test_falls_back_to_today_when_no_metadata(self) -> None:
        listing = [{"name": "readme.txt"}]
        result = ConflictForecastOrgConnector._derive_issue_date(listing)
        today_first = date.today().replace(day=1)
        assert result == today_first

    def test_rejects_invalid_month(self) -> None:
        listing = _make_listing("file_13-2025.csv")
        result = ConflictForecastOrgConnector._derive_issue_date(listing)
        # Month 13 is out of range, should fall back to today
        assert result == date.today().replace(day=1)

    def test_rejects_year_out_of_range(self) -> None:
        listing = _make_listing("file_03-2019.csv")
        result = ConflictForecastOrgConnector._derive_issue_date(listing)
        # Year 2019 is before the 2020 threshold, should fall back
        assert result == date.today().replace(day=1)

    def test_empty_listing_returns_today(self) -> None:
        result = ConflictForecastOrgConnector._derive_issue_date([])
        assert result == date.today().replace(day=1)


# ---------------------------------------------------------------------------
# _transform_csv
# ---------------------------------------------------------------------------


class TestTransformCsv:
    """Test CSV transformation to conflict_forecasts row format."""

    def test_basic_transform(self) -> None:
        df = pd.DataFrame({
            "country": ["Afghanistan", "Kenya", "Ethiopia"],
            "iso3": ["AFG", "KEN", "ETH"],
            "probability": [0.85, 0.32, 0.67],
        })
        issue = date(2025, 3, 1)
        rows = ConflictForecastOrgConnector._transform_csv(
            df, "cf_armed_conflict_risk_3m", 3, issue
        )
        assert len(rows) == 3

        afg = rows[0]
        assert afg["source"] == "conflictforecast_org"
        assert afg["iso3"] == "AFG"
        assert afg["hazard_code"] == "AC"
        assert afg["metric"] == "cf_armed_conflict_risk_3m"
        assert afg["lead_months"] == 3
        assert afg["value"] == pytest.approx(0.85)
        assert afg["forecast_issue_date"] == date(2025, 3, 1)
        assert afg["target_month"] == date(2025, 6, 1)
        assert afg["model_version"] == "conflictforecast_org"

    def test_detects_iso_column_variants(self) -> None:
        """ISO3 column can be named iso3, iso, isocode, iso_code, or isoab."""
        for col_name in ("iso3", "iso", "isocode", "iso_code", "isoab"):
            df = pd.DataFrame({
                "country": ["Somalia"],
                col_name: ["SOM"],
                "value_col": [0.45],
            })
            rows = ConflictForecastOrgConnector._transform_csv(
                df, "cf_armed_conflict_risk_3m", 3, date(2025, 1, 1)
            )
            assert len(rows) == 1, f"Failed for ISO3 column named '{col_name}'"
            assert rows[0]["iso3"] == "SOM"

    def test_uses_last_numeric_column_as_value(self) -> None:
        df = pd.DataFrame({
            "country": ["Mali"],
            "iso3": ["MLI"],
            "first_metric": [0.10],
            "second_metric": [0.99],
        })
        rows = ConflictForecastOrgConnector._transform_csv(
            df, "cf_violence_intensity_3m", 3, date(2025, 1, 1)
        )
        assert len(rows) == 1
        assert rows[0]["value"] == pytest.approx(0.99)

    def test_skips_invalid_iso3(self) -> None:
        df = pd.DataFrame({
            "country": ["Good", "Bad", "Short"],
            "iso3": ["AFG", "", "XY"],
            "probability": [0.5, 0.5, 0.5],
        })
        rows = ConflictForecastOrgConnector._transform_csv(
            df, "cf_armed_conflict_risk_3m", 3, date(2025, 1, 1)
        )
        assert len(rows) == 1
        assert rows[0]["iso3"] == "AFG"

    def test_skips_nan_values(self) -> None:
        df = pd.DataFrame({
            "country": ["Afghanistan", "Kenya"],
            "iso3": ["AFG", "KEN"],
            "probability": [0.85, float("nan")],
        })
        rows = ConflictForecastOrgConnector._transform_csv(
            df, "cf_armed_conflict_risk_3m", 3, date(2025, 1, 1)
        )
        assert len(rows) == 1
        assert rows[0]["iso3"] == "AFG"

    def test_empty_dataframe_returns_empty(self) -> None:
        df = pd.DataFrame()
        rows = ConflictForecastOrgConnector._transform_csv(
            df, "cf_armed_conflict_risk_3m", 3, date(2025, 1, 1)
        )
        assert rows == []

    def test_target_month_wraps_year(self) -> None:
        """Issue date 2025-11 + 3 lead months = 2026-02."""
        df = pd.DataFrame({
            "country": ["Chad"],
            "iso3": ["TCD"],
            "probability": [0.40],
        })
        rows = ConflictForecastOrgConnector._transform_csv(
            df, "cf_armed_conflict_risk_3m", 3, date(2025, 11, 1)
        )
        assert len(rows) == 1
        assert rows[0]["target_month"] == date(2026, 2, 1)

    def test_target_month_12m_lead(self) -> None:
        """Issue date 2025-03 + 12 lead months = 2026-03."""
        df = pd.DataFrame({
            "country": ["Nigeria"],
            "iso3": ["NGA"],
            "probability": [0.55],
        })
        rows = ConflictForecastOrgConnector._transform_csv(
            df, "cf_armed_conflict_risk_12m", 12, date(2025, 3, 1)
        )
        assert len(rows) == 1
        assert rows[0]["target_month"] == date(2026, 3, 1)

    def test_coerces_string_numeric_values(self) -> None:
        """Value column containing string numbers should be coerced."""
        df = pd.DataFrame({
            "country": ["Niger"],
            "iso3": ["NER"],
            "probability": ["0.72"],
        })
        rows = ConflictForecastOrgConnector._transform_csv(
            df, "cf_armed_conflict_risk_3m", 3, date(2025, 1, 1)
        )
        assert len(rows) == 1
        assert rows[0]["value"] == pytest.approx(0.72)

    def test_iso3_uppercased(self) -> None:
        df = pd.DataFrame({
            "country": ["Sudan"],
            "iso3": ["sdn"],
            "probability": [0.90],
        })
        rows = ConflictForecastOrgConnector._transform_csv(
            df, "cf_armed_conflict_risk_3m", 3, date(2025, 1, 1)
        )
        assert len(rows) == 1
        assert rows[0]["iso3"] == "SDN"

    def test_skip_columns_not_used_as_value(self) -> None:
        """Columns named 'country', 'name', 'region' are skipped for value detection."""
        df = pd.DataFrame({
            "region": ["East Africa"],
            "country": ["Eritrea"],
            "iso3": ["ERI"],
            "forecast_prob": [0.61],
        })
        # Force region/country to non-numeric; forecast_prob is the real value
        rows = ConflictForecastOrgConnector._transform_csv(
            df, "cf_armed_conflict_risk_3m", 3, date(2025, 1, 1)
        )
        assert len(rows) == 1
        assert rows[0]["value"] == pytest.approx(0.61)


# ---------------------------------------------------------------------------
# Missing ISO3 column detection
# ---------------------------------------------------------------------------


class TestMissingIso3Column:
    """Test behaviour when the CSV lacks an ISO3 column."""

    def test_returns_empty_when_no_iso3_column(self) -> None:
        df = pd.DataFrame({
            "country": ["Afghanistan", "Kenya"],
            "probability": [0.85, 0.32],
        })
        rows = ConflictForecastOrgConnector._transform_csv(
            df, "cf_armed_conflict_risk_3m", 3, date(2025, 1, 1)
        )
        assert rows == []

    def test_returns_empty_when_no_numeric_column(self) -> None:
        df = pd.DataFrame({
            "country": ["Afghanistan"],
            "iso3": ["AFG"],
            "notes": ["some text"],
        })
        rows = ConflictForecastOrgConnector._transform_csv(
            df, "cf_armed_conflict_risk_3m", 3, date(2025, 1, 1)
        )
        assert rows == []


# ---------------------------------------------------------------------------
# Error handling: API down returns empty DataFrame
# ---------------------------------------------------------------------------


class TestFetchForecastsErrorHandling:
    """Test top-level fetch_forecasts resilience."""

    @patch.object(
        ConflictForecastOrgConnector,
        "_get_file_listing",
        side_effect=ConnectionError("API unreachable"),
    )
    def test_api_down_returns_empty_dataframe(self, mock_listing: MagicMock) -> None:
        connector = ConflictForecastOrgConnector()
        result = connector.fetch_forecasts()
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @patch.object(
        ConflictForecastOrgConnector,
        "_get_file_listing",
        return_value=[],
    )
    def test_empty_listing_returns_empty_dataframe(
        self, mock_listing: MagicMock
    ) -> None:
        connector = ConflictForecastOrgConnector()
        result = connector.fetch_forecasts()
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @patch.object(ConflictForecastOrgConnector, "_download_csv")
    @patch.object(ConflictForecastOrgConnector, "_get_file_listing")
    def test_csv_download_failure_skips_metric(
        self, mock_listing: MagicMock, mock_download: MagicMock
    ) -> None:
        mock_listing.return_value = _make_listing(
            "Armed_Conflict_3m_01-2025.csv",
            "Armed_Conflict_12m_01-2025.csv",
            "Violence_Intensity_3m_01-2025.csv",
        )
        # First download fails, rest succeed
        good_df = pd.DataFrame({
            "country": ["Nigeria"],
            "iso3": ["NGA"],
            "probability": [0.60],
        })
        mock_download.side_effect = [
            Exception("download failed"),
            good_df,
            good_df,
        ]

        connector = ConflictForecastOrgConnector()
        result = connector.fetch_forecasts()
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        # Should have rows from the two successful downloads
        assert len(result) == 2

    @patch.object(ConflictForecastOrgConnector, "_download_csv")
    @patch.object(ConflictForecastOrgConnector, "_get_file_listing")
    def test_all_downloads_fail_returns_empty(
        self, mock_listing: MagicMock, mock_download: MagicMock
    ) -> None:
        mock_listing.return_value = _make_listing(
            "Armed_Conflict_3m_01-2025.csv",
            "Armed_Conflict_12m_01-2025.csv",
            "Violence_Intensity_3m_01-2025.csv",
        )
        mock_download.side_effect = Exception("always fails")

        connector = ConflictForecastOrgConnector()
        result = connector.fetch_forecasts()
        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ---------------------------------------------------------------------------
# Integration-style: full fetch_forecasts with mocked HTTP
# ---------------------------------------------------------------------------


class TestFetchForecastsIntegration:
    """End-to-end test of fetch_forecasts with all HTTP mocked."""

    @patch("resolver.connectors.conflictforecast.requests.get")
    def test_full_pipeline_produces_expected_rows(
        self, mock_get: MagicMock
    ) -> None:
        listing = _make_listing(
            "Armed_Conflict_3m_03-2025.csv",
            "Armed_Conflict_12m_03-2025.csv",
            "Violence_Intensity_3m_03-2025.csv",
        )

        csv_ac3 = _csv_text([
            ["country", "iso3", "prob"],
            ["Afghanistan", "AFG", "0.85"],
            ["Kenya", "KEN", "0.32"],
        ])
        csv_ac12 = _csv_text([
            ["country", "iso3", "prob"],
            ["Afghanistan", "AFG", "0.60"],
        ])
        csv_vi3 = _csv_text([
            ["country", "isocode", "intensity"],
            ["Nigeria", "NGA", "0.77"],
        ])

        def side_effect(url: str, timeout: int = 60) -> MagicMock:
            resp = MagicMock()
            resp.raise_for_status = MagicMock()

            if "get-latest-file-listing" in url:
                resp.json.return_value = listing
            elif "Armed_Conflict_3m" in url:
                resp.text = csv_ac3
            elif "Armed_Conflict_12m" in url:
                resp.text = csv_ac12
            elif "Violence_Intensity_3m" in url:
                resp.text = csv_vi3
            else:
                resp.raise_for_status.side_effect = Exception("unexpected URL")
            return resp

        mock_get.side_effect = side_effect

        connector = ConflictForecastOrgConnector()
        result = connector.fetch_forecasts()

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        # 2 rows from ac3 + 1 from ac12 + 1 from vi3 = 4
        assert len(result) == 4

        # Check that all three metrics appear
        metrics = set(result["metric"].unique())
        assert metrics == {
            "cf_armed_conflict_risk_3m",
            "cf_armed_conflict_risk_12m",
            "cf_violence_intensity_3m",
        }

        # Verify issue date extracted from file name "03-2025"
        assert all(result["forecast_issue_date"] == date(2025, 3, 1))

        # Verify target months
        ac3_rows = result[result["metric"] == "cf_armed_conflict_risk_3m"]
        assert all(ac3_rows["target_month"] == date(2025, 6, 1))

        ac12_rows = result[result["metric"] == "cf_armed_conflict_risk_12m"]
        assert all(ac12_rows["target_month"] == date(2026, 3, 1))

        vi3_rows = result[result["metric"] == "cf_violence_intensity_3m"]
        assert all(vi3_rows["target_month"] == date(2025, 6, 1))

        # All rows should have source and hazard code
        assert all(result["source"] == "conflictforecast_org")
        assert all(result["hazard_code"] == "AC")

    @patch("resolver.connectors.conflictforecast.requests.get")
    def test_no_matching_files_returns_empty(
        self, mock_get: MagicMock
    ) -> None:
        listing = _make_listing("unrelated_data.csv", "readme.txt")

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = listing
        mock_get.return_value = resp

        connector = ConflictForecastOrgConnector()
        result = connector.fetch_forecasts()

        assert isinstance(result, pd.DataFrame)
        assert result.empty
