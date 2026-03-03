# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for the VIEWS (Uppsala/PRIO) conflict forecast connector.

Covers:
- Issue date derivation from record year/month fields
- Transformation of API records into conflict_forecasts rows
- Lead month filtering (only months 1-6 kept)
- ISO3 validation (invalid/missing isoab skipped)
- fetch_forecasts with mocked successful API response
- fetch_forecasts when API returns error (empty DataFrame, no exception)
- Pagination handling (follows next_page links)
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from resolver.connectors.views import ViewsConnector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    isoab: str = "GUY",
    year: int = 2026,
    month: int = 1,
    main_mean: float = 0.0043,
    main_dich: float = 0.0,
    **extra,
) -> dict:
    """Build a single VIEWS-style API record."""
    rec = {
        "country_id": 1,
        "month_id": 553,
        "name": "Guyana",
        "gwcode": 110,
        "isoab": isoab,
        "year": year,
        "month": month,
        "main_mean_ln": main_mean,
        "main_mean": main_mean,
        "main_dich": main_dich,
    }
    rec.update(extra)
    return rec


def _make_api_page(data: list, *, next_page: str = "", page_cur: int = 1, page_count: int = 1) -> dict:
    """Build a VIEWS-style paginated API response body."""
    return {
        "next_page": next_page,
        "page_count": page_count,
        "page_cur": page_cur,
        "row_count": len(data),
        "data": data,
    }


# ---------------------------------------------------------------------------
# _derive_issue_date
# ---------------------------------------------------------------------------


class TestDeriveIssueDate:
    """ViewsConnector._derive_issue_date logic."""

    def test_returns_month_before_earliest_target(self):
        records = [
            _make_record(year=2026, month=3),
            _make_record(year=2026, month=5),
            _make_record(year=2026, month=1),
        ]
        result = ViewsConnector._derive_issue_date(records)
        # Earliest target is 2026-01, so issue date is 2025-12-01
        assert result == date(2025, 12, 1)

    def test_january_wraps_to_previous_december(self):
        records = [_make_record(year=2026, month=1)]
        result = ViewsConnector._derive_issue_date(records)
        assert result == date(2025, 12, 1)

    def test_non_january_subtracts_one_month(self):
        records = [_make_record(year=2026, month=6)]
        result = ViewsConnector._derive_issue_date(records)
        assert result == date(2026, 5, 1)

    def test_missing_year_month_falls_back_to_today(self):
        records = [{"isoab": "GUY", "main_mean": 1.0}]
        result = ViewsConnector._derive_issue_date(records)
        expected = date.today().replace(day=1)
        assert result == expected

    def test_empty_records_falls_back_to_today(self):
        result = ViewsConnector._derive_issue_date([])
        expected = date.today().replace(day=1)
        assert result == expected


# ---------------------------------------------------------------------------
# _transform
# ---------------------------------------------------------------------------


class TestTransform:
    """ViewsConnector._transform produces correct rows."""

    def setup_method(self):
        self.connector = ViewsConnector()
        # Issue date 2025-12-01 means lead month 1 = 2026-01
        self.issue_date = date(2025, 12, 1)
        self.model_version = "fatalities003"

    def test_two_rows_per_country_per_lead_month(self):
        records = [_make_record(isoab="GUY", year=2026, month=1, main_mean=5.0, main_dich=0.3)]
        rows = self.connector._transform(records, self.issue_date, self.model_version)
        assert len(rows) == 2
        metrics = {r["metric"] for r in rows}
        assert metrics == {"views_predicted_fatalities", "views_p_gte25_brd"}

    def test_row_fields_are_correct(self):
        records = [_make_record(isoab="GUY", year=2026, month=1, main_mean=5.0, main_dich=0.3)]
        rows = self.connector._transform(records, self.issue_date, self.model_version)

        fatalities_row = [r for r in rows if r["metric"] == "views_predicted_fatalities"][0]
        assert fatalities_row["source"] == "VIEWS"
        assert fatalities_row["iso3"] == "GUY"
        assert fatalities_row["hazard_code"] == "AC"
        assert fatalities_row["lead_months"] == 1
        assert fatalities_row["value"] == 5.0
        assert fatalities_row["forecast_issue_date"] == self.issue_date
        assert fatalities_row["target_month"] == date(2026, 1, 1)
        assert fatalities_row["model_version"] == "fatalities003"

        dich_row = [r for r in rows if r["metric"] == "views_p_gte25_brd"][0]
        assert dich_row["value"] == 0.3

    def test_multiple_countries(self):
        records = [
            _make_record(isoab="GUY", year=2026, month=1, main_mean=5.0, main_dich=0.3),
            _make_record(isoab="KEN", year=2026, month=1, main_mean=12.0, main_dich=0.8),
        ]
        rows = self.connector._transform(records, self.issue_date, self.model_version)
        # 2 metrics x 2 countries = 4 rows
        assert len(rows) == 4
        iso3s = {r["iso3"] for r in rows}
        assert iso3s == {"GUY", "KEN"}

    def test_lead_months_computed_correctly(self):
        records = [
            _make_record(year=2026, month=1),  # lead 1
            _make_record(year=2026, month=3),  # lead 3
            _make_record(year=2026, month=6),  # lead 6
        ]
        rows = self.connector._transform(records, self.issue_date, self.model_version)
        leads = sorted({r["lead_months"] for r in rows})
        assert leads == [1, 3, 6]


# ---------------------------------------------------------------------------
# Lead month filtering
# ---------------------------------------------------------------------------


class TestLeadMonthFiltering:
    """Only lead months 1-6 are kept; anything outside is dropped."""

    def setup_method(self):
        self.connector = ViewsConnector()
        # Issue date 2025-12-01: lead 1 = 2026-01, lead 6 = 2026-06
        self.issue_date = date(2025, 12, 1)
        self.model_version = "fatalities003"

    def test_lead_month_7_is_dropped(self):
        records = [_make_record(year=2026, month=7, main_mean=1.0, main_dich=0.5)]
        rows = self.connector._transform(records, self.issue_date, self.model_version)
        assert len(rows) == 0

    def test_lead_month_0_is_dropped(self):
        # month=12 of 2025 with issue_date 2025-12 => lead = 0
        records = [_make_record(year=2025, month=12, main_mean=1.0, main_dich=0.5)]
        rows = self.connector._transform(records, self.issue_date, self.model_version)
        assert len(rows) == 0

    def test_negative_lead_month_is_dropped(self):
        records = [_make_record(year=2025, month=11, main_mean=1.0, main_dich=0.5)]
        rows = self.connector._transform(records, self.issue_date, self.model_version)
        assert len(rows) == 0

    def test_lead_months_1_through_6_are_kept(self):
        records = [_make_record(year=2026, month=m) for m in range(1, 7)]
        rows = self.connector._transform(records, self.issue_date, self.model_version)
        leads = sorted({r["lead_months"] for r in rows})
        assert leads == [1, 2, 3, 4, 5, 6]

    def test_mixed_valid_and_invalid_lead_months(self):
        records = [
            _make_record(year=2026, month=1, main_mean=1.0, main_dich=0.1),   # lead 1, kept
            _make_record(year=2026, month=6, main_mean=2.0, main_dich=0.2),   # lead 6, kept
            _make_record(year=2026, month=7, main_mean=3.0, main_dich=0.3),   # lead 7, dropped
            _make_record(year=2026, month=12, main_mean=4.0, main_dich=0.4),  # lead 12, dropped
        ]
        rows = self.connector._transform(records, self.issue_date, self.model_version)
        # 2 valid lead months x 2 metrics each = 4 rows
        assert len(rows) == 4
        leads = sorted({r["lead_months"] for r in rows})
        assert leads == [1, 6]


# ---------------------------------------------------------------------------
# ISO3 validation
# ---------------------------------------------------------------------------


class TestIso3Validation:
    """Records with invalid or missing isoab are skipped."""

    def setup_method(self):
        self.connector = ViewsConnector()
        self.issue_date = date(2025, 12, 1)
        self.model_version = "fatalities003"

    def test_missing_isoab_is_skipped(self):
        records = [_make_record(isoab="", year=2026, month=1)]
        rows = self.connector._transform(records, self.issue_date, self.model_version)
        assert len(rows) == 0

    def test_none_isoab_is_skipped(self):
        rec = _make_record(year=2026, month=1)
        rec["isoab"] = None
        rows = self.connector._transform([rec], self.issue_date, self.model_version)
        assert len(rows) == 0

    def test_two_letter_code_is_skipped(self):
        records = [_make_record(isoab="GU", year=2026, month=1)]
        rows = self.connector._transform(records, self.issue_date, self.model_version)
        assert len(rows) == 0

    def test_four_letter_code_is_skipped(self):
        records = [_make_record(isoab="GUYA", year=2026, month=1)]
        rows = self.connector._transform(records, self.issue_date, self.model_version)
        assert len(rows) == 0

    def test_whitespace_only_isoab_is_skipped(self):
        records = [_make_record(isoab="   ", year=2026, month=1)]
        rows = self.connector._transform(records, self.issue_date, self.model_version)
        assert len(rows) == 0

    def test_valid_iso3_is_uppercased(self):
        records = [_make_record(isoab="guy", year=2026, month=1)]
        rows = self.connector._transform(records, self.issue_date, self.model_version)
        assert len(rows) == 2
        assert all(r["iso3"] == "GUY" for r in rows)

    def test_valid_records_mixed_with_invalid(self):
        records = [
            _make_record(isoab="GUY", year=2026, month=1),   # valid
            _make_record(isoab="", year=2026, month=1),       # invalid: empty
            _make_record(isoab="KE", year=2026, month=1),     # invalid: 2 chars
            _make_record(isoab="KEN", year=2026, month=1),    # valid
        ]
        rows = self.connector._transform(records, self.issue_date, self.model_version)
        iso3s = {r["iso3"] for r in rows}
        assert iso3s == {"GUY", "KEN"}


# ---------------------------------------------------------------------------
# fetch_forecasts — successful response
# ---------------------------------------------------------------------------


class TestFetchForecastsSuccess:
    """fetch_forecasts with mocked API returning valid data."""

    @patch("resolver.connectors.views.requests.get")
    def test_returns_correct_dataframe_schema(self, mock_get):
        data = [
            _make_record(isoab="GUY", year=2026, month=1, main_mean=0.5, main_dich=0.1),
            _make_record(isoab="KEN", year=2026, month=2, main_mean=12.0, main_dich=0.8),
        ]

        # First call: _fetch_all_pages (page 1)
        page_response = MagicMock()
        page_response.status_code = 200
        page_response.json.return_value = _make_api_page(data, next_page="", page_cur=1, page_count=1)
        page_response.raise_for_status = MagicMock()

        # Second call: _detect_run_id (API root)
        root_response = MagicMock()
        root_response.status_code = 200
        root_response.json.return_value = ["fatalities003_2026_01_t01"]
        root_response.raise_for_status = MagicMock()

        mock_get.side_effect = [page_response, root_response]

        connector = ViewsConnector()
        df = connector.fetch_forecasts()

        assert isinstance(df, pd.DataFrame)
        assert not df.empty

        expected_columns = {
            "source", "iso3", "hazard_code", "metric", "lead_months",
            "value", "forecast_issue_date", "target_month", "model_version",
        }
        assert set(df.columns) == expected_columns

    @patch("resolver.connectors.views.requests.get")
    def test_all_rows_have_source_views(self, mock_get):
        data = [
            _make_record(isoab="GUY", year=2026, month=1, main_mean=5.0, main_dich=0.3),
        ]
        page_resp = MagicMock()
        page_resp.json.return_value = _make_api_page(data)
        page_resp.raise_for_status = MagicMock()

        root_resp = MagicMock()
        root_resp.json.return_value = ["fatalities003_2026_01_t01"]
        root_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [page_resp, root_resp]

        df = ViewsConnector().fetch_forecasts()
        assert (df["source"] == "VIEWS").all()
        assert (df["hazard_code"] == "AC").all()

    @patch("resolver.connectors.views.requests.get")
    def test_correct_row_count(self, mock_get):
        data = [
            _make_record(isoab="GUY", year=2026, month=1, main_mean=5.0, main_dich=0.3),
            _make_record(isoab="KEN", year=2026, month=2, main_mean=12.0, main_dich=0.8),
            _make_record(isoab="ETH", year=2026, month=3, main_mean=8.0, main_dich=0.5),
        ]
        page_resp = MagicMock()
        page_resp.json.return_value = _make_api_page(data)
        page_resp.raise_for_status = MagicMock()

        root_resp = MagicMock()
        root_resp.json.return_value = ["fatalities003_2026_01_t01"]
        root_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [page_resp, root_resp]

        df = ViewsConnector().fetch_forecasts()
        # 3 countries x 2 metrics each = 6 rows
        assert len(df) == 6


# ---------------------------------------------------------------------------
# fetch_forecasts — error handling
# ---------------------------------------------------------------------------


class TestFetchForecastsError:
    """fetch_forecasts returns empty DataFrame on API errors, no exception raised."""

    @patch("resolver.connectors.views.requests.get")
    def test_http_error_returns_empty_dataframe(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("HTTP 500 Server Error")
        mock_get.return_value = mock_resp

        connector = ViewsConnector()
        df = connector.fetch_forecasts()

        assert isinstance(df, pd.DataFrame)
        assert df.empty

    @patch("resolver.connectors.views.requests.get")
    def test_connection_error_returns_empty_dataframe(self, mock_get):
        mock_get.side_effect = ConnectionError("Connection refused")

        connector = ViewsConnector()
        df = connector.fetch_forecasts()

        assert isinstance(df, pd.DataFrame)
        assert df.empty

    @patch("resolver.connectors.views.requests.get")
    def test_timeout_returns_empty_dataframe(self, mock_get):
        mock_get.side_effect = TimeoutError("Request timed out")

        connector = ViewsConnector()
        df = connector.fetch_forecasts()

        assert isinstance(df, pd.DataFrame)
        assert df.empty

    @patch("resolver.connectors.views.requests.get")
    def test_empty_data_returns_empty_dataframe(self, mock_get):
        page_resp = MagicMock()
        page_resp.json.return_value = _make_api_page([])
        page_resp.raise_for_status = MagicMock()

        mock_get.return_value = page_resp

        connector = ViewsConnector()
        df = connector.fetch_forecasts()

        assert isinstance(df, pd.DataFrame)
        assert df.empty


# ---------------------------------------------------------------------------
# Pagination handling
# ---------------------------------------------------------------------------


class TestPaginationHandling:
    """Verify connector follows next_page links across multiple pages."""

    @patch("resolver.connectors.views.requests.get")
    def test_follows_next_page_links(self, mock_get):
        page1_data = [
            _make_record(isoab="GUY", year=2026, month=1, main_mean=1.0, main_dich=0.1),
        ]
        page2_data = [
            _make_record(isoab="KEN", year=2026, month=2, main_mean=2.0, main_dich=0.2),
        ]
        page3_data = [
            _make_record(isoab="ETH", year=2026, month=3, main_mean=3.0, main_dich=0.3),
        ]

        page1_resp = MagicMock()
        page1_resp.json.return_value = _make_api_page(
            page1_data, next_page="https://api.viewsforecasting.org/current/cm/sb?page=2", page_cur=1, page_count=3,
        )
        page1_resp.raise_for_status = MagicMock()

        page2_resp = MagicMock()
        page2_resp.json.return_value = _make_api_page(
            page2_data, next_page="https://api.viewsforecasting.org/current/cm/sb?page=3", page_cur=2, page_count=3,
        )
        page2_resp.raise_for_status = MagicMock()

        page3_resp = MagicMock()
        page3_resp.json.return_value = _make_api_page(
            page3_data, next_page="", page_cur=3, page_count=3,
        )
        page3_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [page1_resp, page2_resp, page3_resp]

        connector = ViewsConnector()
        records = connector._fetch_all_pages()

        assert len(records) == 3
        # Verify all three pages were fetched
        assert mock_get.call_count == 3

    @patch("resolver.connectors.views.requests.get")
    def test_stops_when_next_page_is_empty(self, mock_get):
        data = [_make_record(isoab="GUY", year=2026, month=1)]

        page_resp = MagicMock()
        page_resp.json.return_value = _make_api_page(data, next_page="", page_cur=1, page_count=1)
        page_resp.raise_for_status = MagicMock()

        mock_get.return_value = page_resp

        connector = ViewsConnector()
        records = connector._fetch_all_pages()

        assert len(records) == 1
        assert mock_get.call_count == 1

    @patch("resolver.connectors.views.requests.get")
    def test_stops_when_next_page_is_whitespace(self, mock_get):
        data = [_make_record(isoab="GUY", year=2026, month=1)]

        page_resp = MagicMock()
        page_resp.json.return_value = _make_api_page(data, next_page="   ", page_cur=1, page_count=1)
        page_resp.raise_for_status = MagicMock()

        mock_get.return_value = page_resp

        connector = ViewsConnector()
        records = connector._fetch_all_pages()

        assert len(records) == 1
        assert mock_get.call_count == 1

    @patch("resolver.connectors.views.requests.get")
    def test_first_page_sends_params_subsequent_do_not(self, mock_get):
        page1_data = [_make_record(isoab="GUY", year=2026, month=1)]
        page2_data = [_make_record(isoab="KEN", year=2026, month=2)]

        page1_resp = MagicMock()
        page1_resp.json.return_value = _make_api_page(
            page1_data, next_page="https://api.viewsforecasting.org/current/cm/sb?page=2", page_cur=1, page_count=2,
        )
        page1_resp.raise_for_status = MagicMock()

        page2_resp = MagicMock()
        page2_resp.json.return_value = _make_api_page(
            page2_data, next_page="", page_cur=2, page_count=2,
        )
        page2_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [page1_resp, page2_resp]

        connector = ViewsConnector()
        connector._fetch_all_pages()

        # First call: includes params (page=1, pagesize=1000)
        first_call = mock_get.call_args_list[0]
        assert first_call.kwargs.get("params") is not None or first_call[1].get("params") is not None

        # Second call: params should be None (URL already contains pagination)
        second_call = mock_get.call_args_list[1]
        second_params = second_call.kwargs.get("params") if second_call.kwargs else second_call[1].get("params")
        assert second_params is None

    @patch("resolver.connectors.views.requests.get")
    def test_pagination_aggregates_all_records(self, mock_get):
        """End-to-end: paginated fetch feeds into fetch_forecasts correctly."""
        page1_data = [
            _make_record(isoab="GUY", year=2026, month=1, main_mean=1.0, main_dich=0.1),
            _make_record(isoab="GUY", year=2026, month=2, main_mean=2.0, main_dich=0.2),
        ]
        page2_data = [
            _make_record(isoab="KEN", year=2026, month=1, main_mean=3.0, main_dich=0.3),
        ]

        page1_resp = MagicMock()
        page1_resp.json.return_value = _make_api_page(
            page1_data, next_page="https://api.viewsforecasting.org/current/cm/sb?page=2", page_cur=1, page_count=2,
        )
        page1_resp.raise_for_status = MagicMock()

        page2_resp = MagicMock()
        page2_resp.json.return_value = _make_api_page(
            page2_data, next_page="", page_cur=2, page_count=2,
        )
        page2_resp.raise_for_status = MagicMock()

        # Third call: _detect_run_id
        root_resp = MagicMock()
        root_resp.json.return_value = {"fatalities003_2026_01_t01": {}}
        root_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [page1_resp, page2_resp, root_resp]

        df = ViewsConnector().fetch_forecasts()

        # 3 records x 2 metrics = 6 rows
        assert len(df) == 6
        assert set(df["iso3"].unique()) == {"GUY", "KEN"}


# ---------------------------------------------------------------------------
# _extract_model_version
# ---------------------------------------------------------------------------


class TestExtractModelVersion:
    """ViewsConnector._extract_model_version parses run_id strings."""

    def test_standard_run_id(self):
        assert ViewsConnector._extract_model_version("fatalities003_2025_12_t01") == "fatalities003"

    def test_simple_model_name(self):
        assert ViewsConnector._extract_model_version("fatalities003") == "fatalities003"

    def test_empty_string(self):
        assert ViewsConnector._extract_model_version("") == ""
