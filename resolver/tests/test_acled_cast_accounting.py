# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""What the CAST connector must account for (run 33841370196, Group B).

Nothing is inferred: the vintage's target months are logged as found, a
duplicate record is dropped before the SUM and counted, a country-month
that leaves the pipeline is named with its reason, and an epoch timestamp
is read in UTC rather than in whatever zone the runner happens to be in.
"""

from __future__ import annotations

import logging
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from resolver.connectors.acled_cast import AcledCastConnector
from resolver.tests.test_acled_cast_connector import _make_api_response, _make_cast_record


def _iso3(name, aliases=None):
    return {"Nigeria": "NGA", "Kenya": "KEN"}.get(name)


class TestDeduplication:
    @patch("resolver.ingestion.utils.iso_normalize.to_iso3", side_effect=_iso3)
    def test_a_record_repeated_across_a_page_boundary_is_counted_once(self, _iso, caplog):
        rec = _make_cast_record(country="Nigeria", admin1="Lagos", month="March", total_forecast=30)
        connector = AcledCastConnector()
        with caplog.at_level(logging.INFO):
            grouped = connector._aggregate_to_country([rec, dict(rec), _make_cast_record(
                country="Nigeria", admin1="Kano", month="March", total_forecast=5)])
        assert float(grouped["total_forecast"].iloc[0]) == 35.0      # 30 + 5, not 30 + 30 + 5
        assert connector.summary["distinct_admin1_keys"] == 2
        assert connector.summary["duplicate_records_dropped"] == 1
        assert any("3 records fetched, 2 distinct" in r.message for r in caplog.records)

    @patch("resolver.ingestion.utils.iso_normalize.to_iso3", side_effect=_iso3)
    def test_no_duplicates_costs_nothing_and_says_zero(self, _iso):
        connector = AcledCastConnector()
        connector._aggregate_to_country([
            _make_cast_record(admin1="Lagos"), _make_cast_record(admin1="Kano"),
        ])
        assert connector.summary["duplicate_records_dropped"] == 0


class TestCountryMonthAccounting:
    @patch("resolver.ingestion.utils.iso_normalize.to_iso3", side_effect=_iso3)
    def test_a_country_month_dropped_for_want_of_an_iso3_is_named(self, _iso):
        connector = AcledCastConnector()
        connector._aggregate_to_country([
            _make_cast_record(country="Nigeria", month="March"),
            _make_cast_record(country="Atlantis", month="March"),
        ])
        assert connector.summary["country_months_aggregated"] == 2
        assert connector.summary["country_months_dropped_no_iso3"] == ["Atlantis 2026-03"]

    def test_a_country_month_outside_the_lead_window_is_named(self):
        import pandas as pd

        connector = AcledCastConnector()
        aggregated = pd.DataFrame([
            {"country": "Nigeria", "iso3": "NGA", "year": 2026, "month_num": 3,
             "total_forecast": 1.0, "battles_forecast": 1.0, "erv_forecast": 1.0, "vac_forecast": 1.0},
            {"country": "Nigeria", "iso3": "NGA", "year": 2026, "month_num": 9,
             "total_forecast": 1.0, "battles_forecast": 1.0, "erv_forecast": 1.0, "vac_forecast": 1.0},
        ])
        rows = connector._transform(aggregated, date(2026, 2, 1))
        assert {r["target_month"] for r in rows} == {date(2026, 3, 1)}
        assert connector.summary["country_months_dropped_by_lead_filter"] == ["NGA 2026-09 (lead 7)"]


class TestTargetMonthsAreReportedNotInferred:
    @patch("resolver.ingestion.utils.iso_normalize.to_iso3", side_effect=_iso3)
    @patch("resolver.connectors.acled_cast.requests.get")
    @patch("resolver.ingestion.acled_auth.get_auth_header",
           return_value={"Authorization": "Bearer test"})
    def test_a_five_month_vintage_is_logged_as_five_months_with_gaps_named(self, _auth, mock_get, _iso, caplog):
        """The December 2025 vintage carries January to May: June is absent
        from the API, and one country carrying four months is named."""
        months = ["January", "February", "March", "April", "May"]
        data = [
            _make_cast_record(country="Nigeria", admin1="Lagos", month=m, year=2026,
                              timestamp="2025-12-10T00:00:00")
            for m in months
        ] + [
            _make_cast_record(country="Kenya", admin1="Nairobi", month=m, year=2026,
                              timestamp="2025-12-10T00:00:00")
            for m in months[:4]
        ]
        resp = MagicMock()
        resp.json.return_value = _make_api_response(data)
        empty = MagicMock()
        empty.json.return_value = _make_api_response([])
        mock_get.side_effect = [resp, empty]

        connector = AcledCastConnector()
        with caplog.at_level(logging.INFO):
            df = connector.fetch_forecasts()

        assert not df.empty
        assert connector.summary["target_months"] == [
            "2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01",
        ]
        assert connector.summary["countries"] == 2
        assert connector.summary["country_months_written"] == 9
        assert connector.summary["countries_with_month_gaps"] == {"KEN": 1}
        assert connector.summary["issue_date"] == "2025-12-01"
        assert any("carries 5 target month(s)" in r.message for r in caplog.records)
        assert any("countries with gaps: {'KEN': 1}" in r.message for r in caplog.records)


class TestIssueDateIsUtc:
    def test_an_epoch_timestamp_is_read_in_utc_not_runner_local(self, monkeypatch):
        monkeypatch.setenv("TZ", "Pacific/Kiritimati")   # UTC+14: 2025-12-10 22:00 UTC is the 11th here
        import time

        time.tzset()
        try:
            # 1765375966 = 2025-12-10T14:12:46Z, the timestamp on every live CAST record.
            assert AcledCastConnector._derive_issue_date([{"timestamp": 1765375966}]) == date(2025, 12, 1)
            # An evening epoch near a month boundary must not roll into the next month.
            late = 1767225599  # 2025-12-31T23:59:59Z
            assert AcledCastConnector._derive_issue_date([{"timestamp": late}]) == date(2025, 12, 1)
        finally:
            monkeypatch.delenv("TZ")
            time.tzset()
