# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for the ACLED CAST conflict forecast connector.

Covers:
- Month name to number mapping
- Admin1 to country aggregation (summing)
- Issue date derivation from timestamp field
- Transformation of aggregated records into conflict_forecasts rows
- Lead month filtering (only months 1-6 kept)
- Country name to ISO3 resolution
- fetch_forecasts with mocked API response
- fetch_forecasts when API returns error (empty DataFrame, no exception)
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from resolver.connectors.acled_cast import (
    AcledCastConnector,
    _MONTH_MAP,
    _CAST_ALIASES,
    _METRIC_MAP,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cast_record(
    country: str = "Nigeria",
    admin1: str = "Lagos",
    month: str = "March",
    year: int = 2026,
    total_forecast: int = 30,
    battles_forecast: int = 10,
    erv_forecast: int = 8,
    vac_forecast: int = 12,
    timestamp: str = "2026-02-06T00:00:00",
    **extra,
) -> dict:
    """Build a single CAST-style API record."""
    rec = {
        "country": country,
        "admin1": admin1,
        "month": month,
        "year": year,
        "total_forecast": total_forecast,
        "battles_forecast": battles_forecast,
        "erv_forecast": erv_forecast,
        "vac_forecast": vac_forecast,
        "timestamp": timestamp,
    }
    rec.update(extra)
    return rec


def _make_api_response(data: list, **extra) -> dict:
    """Build a CAST-style API response body."""
    body = {
        "status": 200,
        "success": True,
        "count": len(data),
        "data": data,
    }
    body.update(extra)
    return body


# ---------------------------------------------------------------------------
# Month mapping
# ---------------------------------------------------------------------------


class TestMonthMapping:
    """Verify month name to number mapping."""

    def test_all_twelve_months(self):
        expected = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
        }
        assert _MONTH_MAP == expected

    def test_month_map_has_twelve_entries(self):
        assert len(_MONTH_MAP) == 12


# ---------------------------------------------------------------------------
# Admin1 to country aggregation
# ---------------------------------------------------------------------------


class TestAggregateToCountry:
    """Verify admin1-level data is correctly summed to country level."""

    def setup_method(self):
        self.connector = AcledCastConnector()

    @patch("resolver.ingestion.utils.iso_normalize.to_iso3", return_value="NGA")
    def test_sums_admin1_regions(self, mock_iso3):
        records = [
            _make_cast_record(country="Nigeria", admin1="Lagos", total_forecast=20, battles_forecast=5),
            _make_cast_record(country="Nigeria", admin1="Abuja", total_forecast=15, battles_forecast=8),
            _make_cast_record(country="Nigeria", admin1="Kano", total_forecast=10, battles_forecast=3),
        ]
        result = self.connector._aggregate_to_country(records)
        assert len(result) == 1  # 1 country
        row = result.iloc[0]
        assert row["total_forecast"] == 45  # 20 + 15 + 10
        assert row["battles_forecast"] == 16  # 5 + 8 + 3

    @patch("resolver.ingestion.utils.iso_normalize.to_iso3", side_effect=lambda name, aliases: {
        "Nigeria": "NGA", "Kenya": "KEN"
    }.get(name))
    def test_multiple_countries_kept_separate(self, mock_iso3):
        records = [
            _make_cast_record(country="Nigeria", admin1="Lagos", total_forecast=20),
            _make_cast_record(country="Kenya", admin1="Nairobi", total_forecast=15),
        ]
        result = self.connector._aggregate_to_country(records)
        assert len(result) == 2

    @patch("resolver.ingestion.utils.iso_normalize.to_iso3", return_value="NGA")
    def test_different_months_kept_separate(self, mock_iso3):
        records = [
            _make_cast_record(month="March", total_forecast=20),
            _make_cast_record(month="April", total_forecast=15),
        ]
        result = self.connector._aggregate_to_country(records)
        assert len(result) == 2

    @patch("resolver.ingestion.utils.iso_normalize.to_iso3", return_value=None)
    def test_unmapped_country_is_dropped(self, mock_iso3):
        records = [_make_cast_record(country="UnknownLand")]
        result = self.connector._aggregate_to_country(records)
        # ISO3 is None → should be dropped
        assert result.empty or result[result["iso3"].notna()].empty

    def test_empty_records_returns_empty_df(self):
        result = self.connector._aggregate_to_country([])
        assert result.empty

    def test_invalid_month_name_is_skipped(self):
        records = [_make_cast_record(month="InvalidMonth")]
        result = self.connector._aggregate_to_country(records)
        assert result.empty


# ---------------------------------------------------------------------------
# Issue date derivation
# ---------------------------------------------------------------------------


class TestDeriveIssueDate:
    """AcledCastConnector._derive_issue_date logic."""

    def test_iso_string_timestamp(self):
        records = [
            _make_cast_record(timestamp="2026-02-06T12:00:00"),
            _make_cast_record(timestamp="2026-02-10T08:00:00"),
        ]
        result = AcledCastConnector._derive_issue_date(records)
        assert result == date(2026, 2, 1)

    def test_epoch_int_timestamp(self):
        import time
        from datetime import datetime
        # 2026-03-15 00:00:00 UTC
        epoch = int(datetime(2026, 3, 15).timestamp())
        records = [_make_cast_record(timestamp=epoch)]
        result = AcledCastConnector._derive_issue_date(records)
        assert result == date(2026, 3, 1)

    def test_missing_timestamps_falls_back_to_today(self):
        records = [{"country": "Nigeria", "month": "March", "year": 2026}]
        result = AcledCastConnector._derive_issue_date(records)
        expected = date.today().replace(day=1)
        assert result == expected

    def test_empty_records_falls_back_to_today(self):
        result = AcledCastConnector._derive_issue_date([])
        expected = date.today().replace(day=1)
        assert result == expected


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


class TestTransform:
    """AcledCastConnector._transform produces correct rows."""

    def setup_method(self):
        self.connector = AcledCastConnector()
        self.issue_date = date(2026, 2, 1)

    def test_four_rows_per_country_per_lead_month(self):
        agg = pd.DataFrame([{
            "country": "Nigeria",
            "month_num": 3,
            "year": 2026,
            "total_forecast": 45.0,
            "battles_forecast": 18.0,
            "erv_forecast": 12.0,
            "vac_forecast": 15.0,
            "iso3": "NGA",
        }])
        rows = self.connector._transform(agg, self.issue_date)
        assert len(rows) == 4
        metrics = {r["metric"] for r in rows}
        assert metrics == {
            "cast_total_events",
            "cast_battles_events",
            "cast_erv_events",
            "cast_vac_events",
        }

    def test_row_fields_are_correct(self):
        agg = pd.DataFrame([{
            "country": "Nigeria",
            "month_num": 3,
            "year": 2026,
            "total_forecast": 45.0,
            "battles_forecast": 18.0,
            "erv_forecast": 12.0,
            "vac_forecast": 15.0,
            "iso3": "NGA",
        }])
        rows = self.connector._transform(agg, self.issue_date)

        total_row = [r for r in rows if r["metric"] == "cast_total_events"][0]
        assert total_row["source"] == "ACLED_CAST"
        assert total_row["iso3"] == "NGA"
        assert total_row["hazard_code"] == "AC"
        assert total_row["lead_months"] == 1  # March - Feb = 1
        assert total_row["value"] == 45.0
        assert total_row["forecast_issue_date"] == self.issue_date
        assert total_row["target_month"] == date(2026, 3, 1)
        assert total_row["model_version"] == "cast"

    def test_multiple_countries(self):
        agg = pd.DataFrame([
            {"country": "Nigeria", "month_num": 3, "year": 2026,
             "total_forecast": 45, "battles_forecast": 18,
             "erv_forecast": 12, "vac_forecast": 15, "iso3": "NGA"},
            {"country": "Kenya", "month_num": 3, "year": 2026,
             "total_forecast": 20, "battles_forecast": 8,
             "erv_forecast": 5, "vac_forecast": 7, "iso3": "KEN"},
        ])
        rows = self.connector._transform(agg, self.issue_date)
        # 2 countries x 4 metrics = 8 rows
        assert len(rows) == 8
        iso3s = {r["iso3"] for r in rows}
        assert iso3s == {"NGA", "KEN"}


# ---------------------------------------------------------------------------
# Lead month filtering
# ---------------------------------------------------------------------------


class TestLeadMonthFiltering:
    """Only lead months 1-6 are kept; anything outside is dropped."""

    def setup_method(self):
        self.connector = AcledCastConnector()
        self.issue_date = date(2026, 2, 1)

    def _make_agg_row(self, month_num: int, year: int = 2026) -> dict:
        return {
            "country": "Nigeria", "month_num": month_num, "year": year,
            "total_forecast": 10.0, "battles_forecast": 5.0,
            "erv_forecast": 3.0, "vac_forecast": 2.0, "iso3": "NGA",
        }

    def test_lead_month_7_is_dropped(self):
        # Sep 2026 - Feb 2026 = 7 months
        agg = pd.DataFrame([self._make_agg_row(month_num=9)])
        rows = self.connector._transform(agg, self.issue_date)
        assert len(rows) == 0

    def test_lead_month_0_is_dropped(self):
        # Feb 2026 - Feb 2026 = 0 months
        agg = pd.DataFrame([self._make_agg_row(month_num=2)])
        rows = self.connector._transform(agg, self.issue_date)
        assert len(rows) == 0

    def test_negative_lead_month_is_dropped(self):
        # Jan 2026 - Feb 2026 = -1
        agg = pd.DataFrame([self._make_agg_row(month_num=1)])
        rows = self.connector._transform(agg, self.issue_date)
        assert len(rows) == 0

    def test_lead_months_1_through_6_are_kept(self):
        agg = pd.DataFrame([
            self._make_agg_row(month_num=m) for m in range(3, 9)
        ])  # March(1) through August(6)
        rows = self.connector._transform(agg, self.issue_date)
        leads = sorted({r["lead_months"] for r in rows})
        assert leads == [1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# Country name to ISO3
# ---------------------------------------------------------------------------


class TestCastAliases:
    """CAST-specific country name aliases resolve correctly."""

    def test_common_aliases_present(self):
        assert _CAST_ALIASES["Republic of Congo"] == "COG"
        assert _CAST_ALIASES["Palestine"] == "PSE"
        assert _CAST_ALIASES["Burma/Myanmar"] == "MMR"
        assert _CAST_ALIASES["Ivory Coast"] == "CIV"
        assert _CAST_ALIASES["eSwatini"] == "SWZ"
        assert _CAST_ALIASES["Türkiye"] == "TUR"


# ---------------------------------------------------------------------------
# fetch_forecasts — successful response
# ---------------------------------------------------------------------------


class TestFetchForecastsSuccess:
    """fetch_forecasts with mocked API returning valid data."""

    @patch("resolver.ingestion.utils.iso_normalize.to_iso3", return_value="NGA")
    @patch("resolver.connectors.acled_cast.requests.get")
    @patch("resolver.ingestion.acled_auth.get_auth_header", return_value={"Authorization": "Bearer test"})
    def test_returns_correct_dataframe_schema(self, mock_auth, mock_get, mock_iso3):
        data = [
            _make_cast_record(country="Nigeria", admin1="Lagos", month="March", year=2026),
            _make_cast_record(country="Nigeria", admin1="Abuja", month="March", year=2026),
        ]

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = _make_api_response(data)
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        connector = AcledCastConnector()
        df = connector.fetch_forecasts()

        assert isinstance(df, pd.DataFrame)
        assert not df.empty

        expected_columns = {
            "source", "iso3", "hazard_code", "metric", "lead_months",
            "value", "forecast_issue_date", "target_month", "model_version",
        }
        assert set(df.columns) == expected_columns

    @patch("resolver.ingestion.utils.iso_normalize.to_iso3", return_value="NGA")
    @patch("resolver.connectors.acled_cast.requests.get")
    @patch("resolver.ingestion.acled_auth.get_auth_header", return_value={"Authorization": "Bearer test"})
    def test_all_rows_have_source_acled_cast(self, mock_auth, mock_get, mock_iso3):
        data = [
            _make_cast_record(country="Nigeria", admin1="Lagos", month="March", year=2026),
        ]
        resp = MagicMock()
        resp.json.return_value = _make_api_response(data)
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        df = AcledCastConnector().fetch_forecasts()
        assert (df["source"] == "ACLED_CAST").all()
        assert (df["hazard_code"] == "AC").all()

    @patch("resolver.ingestion.utils.iso_normalize.to_iso3", return_value="NGA")
    @patch("resolver.connectors.acled_cast.requests.get")
    @patch("resolver.ingestion.acled_auth.get_auth_header", return_value={"Authorization": "Bearer test"})
    def test_admin1_aggregation_in_fetch(self, mock_auth, mock_get, mock_iso3):
        """Two admin1 regions for same country/month should aggregate."""
        data = [
            _make_cast_record(country="Nigeria", admin1="Lagos", month="March",
                              total_forecast=20, battles_forecast=8),
            _make_cast_record(country="Nigeria", admin1="Abuja", month="March",
                              total_forecast=10, battles_forecast=4),
        ]
        resp = MagicMock()
        resp.json.return_value = _make_api_response(data)
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        df = AcledCastConnector().fetch_forecasts()
        total_rows = df[df["metric"] == "cast_total_events"]
        assert len(total_rows) == 1
        assert total_rows.iloc[0]["value"] == 30.0  # 20 + 10


# ---------------------------------------------------------------------------
# fetch_forecasts — error handling
# ---------------------------------------------------------------------------


class TestFetchForecastsError:
    """fetch_forecasts returns empty DataFrame on errors, no exception raised."""

    @patch("resolver.connectors.acled_cast.requests.get")
    @patch("resolver.ingestion.acled_auth.get_auth_header", return_value={"Authorization": "Bearer test"})
    def test_http_error_returns_empty_dataframe(self, mock_auth, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("HTTP 500 Server Error")
        mock_get.return_value = mock_resp

        df = AcledCastConnector().fetch_forecasts()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    @patch("resolver.connectors.acled_cast.requests.get")
    @patch("resolver.ingestion.acled_auth.get_auth_header", return_value={"Authorization": "Bearer test"})
    def test_connection_error_returns_empty_dataframe(self, mock_auth, mock_get):
        mock_get.side_effect = ConnectionError("Connection refused")

        df = AcledCastConnector().fetch_forecasts()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    @patch("resolver.ingestion.acled_auth.get_auth_header")
    def test_auth_error_returns_empty_dataframe(self, mock_auth):
        mock_auth.side_effect = RuntimeError("ACLED authentication failed")

        df = AcledCastConnector().fetch_forecasts()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    @patch("resolver.connectors.acled_cast.requests.get")
    @patch("resolver.ingestion.acled_auth.get_auth_header", return_value={"Authorization": "Bearer test"})
    def test_empty_data_returns_empty_dataframe(self, mock_auth, mock_get):
        resp = MagicMock()
        resp.json.return_value = _make_api_response([])
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        df = AcledCastConnector().fetch_forecasts()
        assert isinstance(df, pd.DataFrame)
        assert df.empty


# ---------------------------------------------------------------------------
# Metric mapping
# ---------------------------------------------------------------------------


class TestMetricMapping:
    """Verify the metric field mapping is correct."""

    def test_four_metrics_defined(self):
        assert len(_METRIC_MAP) == 4

    def test_metric_names(self):
        assert _METRIC_MAP["total_forecast"] == "cast_total_events"
        assert _METRIC_MAP["battles_forecast"] == "cast_battles_events"
        assert _METRIC_MAP["erv_forecast"] == "cast_erv_events"
        assert _METRIC_MAP["vac_forecast"] == "cast_vac_events"
