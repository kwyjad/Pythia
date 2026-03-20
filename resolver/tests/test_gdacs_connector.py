# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for the GDACS RSS archive connector.

Covers:
- Connector satisfies the Connector protocol
- XML parsing with sample RSS items (FL, DR, TC, EQ filtered out)
- Month expansion logic (event spanning 2 months produces 2 rows)
- Multi-country population-weighted split
- TC zero-fill logic
- Deduplication (same eventid appears twice, keep latest)
- Output passes validate_canonical()
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from resolver.connectors.gdacs import (
    GdacsConnector,
    _HAZARD_MAP,
    _POPULATION,
    _build_session,
    _last_day,
    _month_range,
    _overlapping_months,
    _parse_date,
)
from resolver.connectors.protocol import CANONICAL_COLUMNS, Connector
from resolver.connectors.validate import validate_canonical


# ---------------------------------------------------------------------------
# Sample RSS XML fixture
# ---------------------------------------------------------------------------

_SAMPLE_RSS = b"""\
<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"
     xmlns:gdacs="http://www.gdacs.org"
     xmlns:geo="http://www.w3.org/2003/01/geo/wgs84_pos#">
  <channel>
    <title>GDACS</title>
    <item>
      <title>Flood in Bangladesh</title>
      <gdacs:eventtype>FL</gdacs:eventtype>
      <gdacs:eventid>1001</gdacs:eventid>
      <gdacs:alertlevel>Orange</gdacs:alertlevel>
      <gdacs:alertscore>2.5</gdacs:alertscore>
      <gdacs:population value="500000" unit="people"/>
      <gdacs:iso3>BGD</gdacs:iso3>
      <gdacs:country>Bangladesh</gdacs:country>
      <gdacs:fromdate>2024-07-10</gdacs:fromdate>
      <gdacs:todate>2024-07-25</gdacs:todate>
      <pubDate>Fri, 25 Jul 2024 12:00:00 GMT</pubDate>
      <geo:Point><geo:lat>23.5</geo:lat><geo:long>90.3</geo:long></geo:Point>
    </item>
    <item>
      <title>Drought in Kenya</title>
      <gdacs:eventtype>DR</gdacs:eventtype>
      <gdacs:eventid>2001</gdacs:eventid>
      <gdacs:alertlevel>Red</gdacs:alertlevel>
      <gdacs:alertscore>3.0</gdacs:alertscore>
      <gdacs:population value="1200000" unit="people"/>
      <gdacs:iso3>KEN</gdacs:iso3>
      <gdacs:country>Kenya</gdacs:country>
      <gdacs:fromdate>2024-06-01</gdacs:fromdate>
      <gdacs:todate>2024-07-31</gdacs:todate>
      <pubDate>Wed, 31 Jul 2024 08:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Tropical Cyclone in Philippines</title>
      <gdacs:eventtype>TC</gdacs:eventtype>
      <gdacs:eventid>3001</gdacs:eventid>
      <gdacs:alertlevel>Red</gdacs:alertlevel>
      <gdacs:alertscore>4.0</gdacs:alertscore>
      <gdacs:population value="2000000" unit="people"/>
      <gdacs:iso3>PHL</gdacs:iso3>
      <gdacs:country>Philippines</gdacs:country>
      <gdacs:fromdate>2024-09-15</gdacs:fromdate>
      <gdacs:todate>2024-09-20</gdacs:todate>
      <pubDate>Fri, 20 Sep 2024 06:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Earthquake in Turkey</title>
      <gdacs:eventtype>EQ</gdacs:eventtype>
      <gdacs:eventid>4001</gdacs:eventid>
      <gdacs:alertlevel>Orange</gdacs:alertlevel>
      <gdacs:alertscore>2.0</gdacs:alertscore>
      <gdacs:population value="300000" unit="people"/>
      <gdacs:iso3>TUR</gdacs:iso3>
      <gdacs:country>Turkey</gdacs:country>
      <gdacs:fromdate>2024-08-01</gdacs:fromdate>
      <gdacs:todate>2024-08-01</gdacs:todate>
      <pubDate>Thu, 01 Aug 2024 10:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    """GdacsConnector satisfies the Connector protocol."""

    def test_is_connector_instance(self):
        connector = GdacsConnector()
        assert isinstance(connector, Connector)

    def test_has_name(self):
        assert GdacsConnector.name == "gdacs"

    def test_has_fetch_and_normalize(self):
        assert callable(getattr(GdacsConnector, "fetch_and_normalize", None))


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------


class TestXmlParsing:
    """Parse sample RSS and verify event extraction."""

    def setup_method(self):
        self.connector = GdacsConnector()
        self.name_to_iso3 = {
            "bangladesh": "BGD",
            "kenya": "KEN",
            "philippines": "PHL",
            "turkey": "TUR",
        }

    def test_parses_three_wanted_events(self):
        events = self.connector._parse_rss(_SAMPLE_RSS, self.name_to_iso3)
        # FL, DR, TC kept; EQ filtered out
        assert len(events) == 3

    def test_filters_out_earthquake(self):
        events = self.connector._parse_rss(_SAMPLE_RSS, self.name_to_iso3)
        types = {e["eventtype"] for e in events}
        assert "EQ" not in types
        assert types == {"FL", "DR", "TC"}

    def test_extracts_population(self):
        events = self.connector._parse_rss(_SAMPLE_RSS, self.name_to_iso3)
        fl_event = [e for e in events if e["eventtype"] == "FL"][0]
        assert fl_event["population"] == 500000.0

    def test_extracts_iso3(self):
        events = self.connector._parse_rss(_SAMPLE_RSS, self.name_to_iso3)
        iso3s = {e["iso3"] for e in events}
        assert iso3s == {"BGD", "KEN", "PHL"}

    def test_extracts_dates(self):
        events = self.connector._parse_rss(_SAMPLE_RSS, self.name_to_iso3)
        dr_event = [e for e in events if e["eventtype"] == "DR"][0]
        assert dr_event["fromdate"] == date(2024, 6, 1)
        assert dr_event["todate"] == date(2024, 7, 31)

    def test_extracts_alert_level(self):
        events = self.connector._parse_rss(_SAMPLE_RSS, self.name_to_iso3)
        fl_event = [e for e in events if e["eventtype"] == "FL"][0]
        assert fl_event["alertlevel"] == "Orange"


# ---------------------------------------------------------------------------
# Month expansion
# ---------------------------------------------------------------------------


class TestMonthExpansion:
    """Event spanning multiple months produces one row per month."""

    def test_two_month_event_produces_two_rows(self):
        connector = GdacsConnector()
        event = {
            "eventtype": "DR",
            "eventid": "2001",
            "population": 1200000.0,
            "iso3": "KEN",
            "country": "Kenya",
            "fromdate": date(2024, 6, 1),
            "todate": date(2024, 7, 31),
            "alertlevel": "Red",
            "pub_date": date(2024, 7, 31),
        }
        name_to_iso3 = {"kenya": "KEN"}
        rows = connector._expand_to_country_months([event], name_to_iso3)
        # Should produce rows for June 2024 and July 2024
        assert len(rows) == 2
        months = {(r["year"], r["month"]) for r in rows}
        assert months == {(2024, 6), (2024, 7)}

    def test_single_month_event_produces_one_row(self):
        connector = GdacsConnector()
        event = {
            "eventtype": "FL",
            "eventid": "1001",
            "population": 500000.0,
            "iso3": "BGD",
            "country": "Bangladesh",
            "fromdate": date(2024, 7, 10),
            "todate": date(2024, 7, 25),
            "alertlevel": "Orange",
            "pub_date": date(2024, 7, 25),
        }
        name_to_iso3 = {"bangladesh": "BGD"}
        rows = connector._expand_to_country_months([event], name_to_iso3)
        assert len(rows) == 1
        assert rows[0]["year"] == 2024
        assert rows[0]["month"] == 7

    def test_three_month_event(self):
        connector = GdacsConnector()
        event = {
            "eventtype": "DR",
            "eventid": "5001",
            "population": 900000.0,
            "iso3": "ETH",
            "country": "Ethiopia",
            "fromdate": date(2024, 1, 15),
            "todate": date(2024, 3, 10),
            "alertlevel": "Orange",
            "pub_date": date(2024, 3, 10),
        }
        name_to_iso3 = {"ethiopia": "ETH"}
        rows = connector._expand_to_country_months([event], name_to_iso3)
        assert len(rows) == 3
        months = sorted((r["year"], r["month"]) for r in rows)
        assert months == [(2024, 1), (2024, 2), (2024, 3)]


# ---------------------------------------------------------------------------
# Multi-country population split
# ---------------------------------------------------------------------------


class TestMultiCountryPopulationSplit:
    """Events affecting multiple countries split population by weight."""

    def test_two_country_split(self):
        connector = GdacsConnector()
        # Event with comma-separated ISO3
        event = {
            "eventtype": "FL",
            "eventid": "6001",
            "population": 1000000.0,
            "iso3": "BGD,IND",
            "country": "Bangladesh,India",
            "fromdate": date(2024, 8, 1),
            "todate": date(2024, 8, 31),
            "alertlevel": "Orange",
            "pub_date": date(2024, 8, 31),
        }
        name_to_iso3 = {"bangladesh": "BGD", "india": "IND"}
        rows = connector._expand_to_country_months([event], name_to_iso3)
        assert len(rows) == 2
        iso3s = {r["iso3"] for r in rows}
        assert iso3s == {"BGD", "IND"}

        # Values should sum to total
        total = sum(r["value"] for r in rows)
        assert abs(total - 1000000.0) < 0.01

        # India has much larger population, so should get larger share
        ind_row = [r for r in rows if r["iso3"] == "IND"][0]
        bgd_row = [r for r in rows if r["iso3"] == "BGD"][0]
        assert ind_row["value"] > bgd_row["value"]

    def test_single_country_gets_full_population(self):
        connector = GdacsConnector()
        shares = connector._population_split(["KEN"], 500000.0)
        assert shares == {"KEN": 500000.0}

    def test_population_weighted_proportional(self):
        connector = GdacsConnector()
        shares = connector._population_split(["BGD", "IND"], 1000000.0)
        bgd_pop = _POPULATION["BGD"]
        ind_pop = _POPULATION["IND"]
        total_pop = bgd_pop + ind_pop
        expected_bgd = 1000000.0 * bgd_pop / total_pop
        expected_ind = 1000000.0 * ind_pop / total_pop
        assert abs(shares["BGD"] - expected_bgd) < 0.01
        assert abs(shares["IND"] - expected_ind) < 0.01


# ---------------------------------------------------------------------------
# TC zero-fill
# ---------------------------------------------------------------------------


class TestTcZeroFill:
    """TC events get zero-filled for months with no cyclone activity."""

    def test_zero_fill_for_missing_tc_months(self):
        connector = GdacsConnector()
        # Start with a TC event in September 2024 only
        agg_df = pd.DataFrame([{
            "iso3": "PHL",
            "hazard_code": "TC",
            "year": 2024,
            "month": 9,
            "value": 2000000.0,
            "alertlevel": "Red",
            "todate": date(2024, 9, 20),
        }])
        name_to_iso3 = {"philippines": "PHL"}
        start = date(2024, 8, 1)
        end = date(2024, 10, 31)

        result = connector._apply_no_event_logic(agg_df, start, end, name_to_iso3)
        # Should have 3 months: Aug (zero), Sep (event), Oct (zero)
        phl_tc = result[(result["iso3"] == "PHL") & (result["hazard_code"] == "TC")]
        assert len(phl_tc) == 3

        aug = phl_tc[(phl_tc["year"] == 2024) & (phl_tc["month"] == 8)]
        assert len(aug) == 1
        assert aug.iloc[0]["value"] == 0.0

        sep = phl_tc[(phl_tc["year"] == 2024) & (phl_tc["month"] == 9)]
        assert len(sep) == 1
        assert sep.iloc[0]["value"] == 2000000.0

        oct_rows = phl_tc[(phl_tc["year"] == 2024) & (phl_tc["month"] == 10)]
        assert len(oct_rows) == 1
        assert oct_rows.iloc[0]["value"] == 0.0

    def test_no_zero_fill_for_fl(self):
        connector = GdacsConnector()
        agg_df = pd.DataFrame([{
            "iso3": "BGD",
            "hazard_code": "FL",
            "year": 2024,
            "month": 7,
            "value": 500000.0,
            "alertlevel": "Orange",
            "todate": date(2024, 7, 25),
        }])
        start = date(2024, 6, 1)
        end = date(2024, 8, 31)

        result = connector._apply_no_event_logic(agg_df, start, end, {})
        fl_rows = result[result["hazard_code"] == "FL"]
        # Should NOT zero-fill — only the 1 original row
        assert len(fl_rows) == 1

    def test_no_zero_fill_for_dr(self):
        connector = GdacsConnector()
        agg_df = pd.DataFrame([{
            "iso3": "KEN",
            "hazard_code": "DR",
            "year": 2024,
            "month": 6,
            "value": 1200000.0,
            "alertlevel": "Red",
            "todate": date(2024, 6, 30),
        }])
        start = date(2024, 5, 1)
        end = date(2024, 7, 31)

        result = connector._apply_no_event_logic(agg_df, start, end, {})
        dr_rows = result[result["hazard_code"] == "DR"]
        assert len(dr_rows) == 1


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Same (eventtype, eventid) appearing twice — keep latest."""

    def test_keeps_latest_episode(self):
        connector = GdacsConnector()
        events = [
            {
                "eventtype": "DR",
                "eventid": "2001",
                "population": 800000.0,
                "iso3": "KEN",
                "country": "Kenya",
                "fromdate": date(2024, 6, 1),
                "todate": date(2024, 6, 30),
                "alertlevel": "Orange",
                "pub_date": date(2024, 6, 30),
            },
            {
                "eventtype": "DR",
                "eventid": "2001",
                "population": 1200000.0,
                "iso3": "KEN",
                "country": "Kenya",
                "fromdate": date(2024, 6, 1),
                "todate": date(2024, 7, 31),
                "alertlevel": "Red",
                "pub_date": date(2024, 7, 31),
            },
        ]
        result = connector._deduplicate(events)
        assert len(result) == 1
        assert result[0]["population"] == 1200000.0
        assert result[0]["todate"] == date(2024, 7, 31)

    def test_different_event_ids_kept(self):
        connector = GdacsConnector()
        events = [
            {
                "eventtype": "FL",
                "eventid": "1001",
                "population": 500000.0,
                "iso3": "BGD",
                "country": "Bangladesh",
                "fromdate": date(2024, 7, 10),
                "todate": date(2024, 7, 25),
                "alertlevel": "Orange",
                "pub_date": date(2024, 7, 25),
            },
            {
                "eventtype": "FL",
                "eventid": "1002",
                "population": 300000.0,
                "iso3": "BGD",
                "country": "Bangladesh",
                "fromdate": date(2024, 8, 1),
                "todate": date(2024, 8, 15),
                "alertlevel": "Green",
                "pub_date": date(2024, 8, 15),
            },
        ]
        result = connector._deduplicate(events)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Validate canonical output
# ---------------------------------------------------------------------------


class TestCanonicalOutput:
    """Full pipeline with mocked HTTP produces valid canonical output."""

    def _make_mock_response(self, xml_bytes: bytes) -> MagicMock:
        resp = MagicMock()
        resp.content = xml_bytes
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        return resp

    @patch("resolver.connectors.gdacs.time.sleep")
    @patch("resolver.connectors.gdacs._build_session")
    def test_output_passes_validate_canonical(self, mock_session_fn, mock_sleep, monkeypatch):
        """Full fetch_and_normalize with mocked HTTP returns valid canonical df."""
        monkeypatch.setenv("GDACS_MONTHS", "4")  # Oct 2024 - 4 = Jul 2024

        mock_session = MagicMock()
        mock_session.get.return_value = self._make_mock_response(_SAMPLE_RSS)
        mock_session_fn.return_value = mock_session

        connector = GdacsConnector()
        with patch.object(connector, "_today", return_value=date(2024, 10, 31)):
            df = connector.fetch_and_normalize()

        assert not df.empty
        assert list(df.columns) == CANONICAL_COLUMNS

        # Should have FL, DR, TC rows (no EQ)
        hazards = set(df["hazard_code"].unique())
        assert hazards <= {"FL", "DR", "TC"}
        assert "EQ" not in hazards

        # All iso3 codes should be 3 chars
        assert (df["iso3"].str.len() == 3).all()

        # Values should be numeric
        assert pd.to_numeric(df["value"], errors="coerce").notna().all()

        # Publisher should be GDACS / JRC
        assert (df["publisher"] == "GDACS / JRC").all()

        # validate_canonical should not raise
        validate_canonical(df, source="gdacs")

    @patch("resolver.connectors.gdacs.time.sleep")
    @patch("resolver.connectors.gdacs._build_session")
    def test_tc_zero_fill_in_full_pipeline(self, mock_session_fn, mock_sleep, monkeypatch):
        """TC zero-fill rows appear in final output."""
        monkeypatch.setenv("GDACS_MONTHS", "2")  # Oct 2024 - 2 = Sep 2024

        # RSS with only a TC event in September
        tc_only_rss = b"""\
<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"
     xmlns:gdacs="http://www.gdacs.org"
     xmlns:geo="http://www.w3.org/2003/01/geo/wgs84_pos#">
  <channel>
    <item>
      <gdacs:eventtype>TC</gdacs:eventtype>
      <gdacs:eventid>3001</gdacs:eventid>
      <gdacs:alertlevel>Red</gdacs:alertlevel>
      <gdacs:population value="2000000" unit="people"/>
      <gdacs:iso3>PHL</gdacs:iso3>
      <gdacs:country>Philippines</gdacs:country>
      <gdacs:fromdate>2024-09-15</gdacs:fromdate>
      <gdacs:todate>2024-09-20</gdacs:todate>
    </item>
  </channel>
</rss>
"""
        mock_session = MagicMock()
        mock_session.get.return_value = self._make_mock_response(tc_only_rss)
        mock_session_fn.return_value = mock_session

        connector = GdacsConnector()
        with patch.object(connector, "_today", return_value=date(2024, 10, 31)):
            df = connector.fetch_and_normalize()

        tc_rows = df[df["hazard_code"] == "TC"]
        # GDACS_MONTHS=2 from Oct 2024 → start Aug 2024
        # Aug (zero-fill) + Sep (event) + Oct (zero-fill) = 3 months
        assert len(tc_rows) == 3
        zero_rows = tc_rows[tc_rows["value"].astype(float) == 0.0]
        assert len(zero_rows) == 2

    @patch("resolver.connectors.gdacs.time.sleep")
    @patch("resolver.connectors.gdacs._build_session")
    def test_empty_response_returns_empty_canonical(self, mock_session_fn, mock_sleep, monkeypatch):
        """HTTP returning no items produces empty canonical DataFrame."""
        monkeypatch.setenv("GDACS_MONTHS", "2")  # Oct 2024 - 2 = Sep 2024

        empty_rss = b"""\
<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"><channel></channel></rss>
"""
        mock_session = MagicMock()
        mock_session.get.return_value = self._make_mock_response(empty_rss)
        mock_session_fn.return_value = mock_session

        connector = GdacsConnector()
        with patch.object(connector, "_today", return_value=date(2024, 10, 31)):
            df = connector.fetch_and_normalize()

        assert list(df.columns) == CANONICAL_COLUMNS
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    """Unit tests for module-level helper functions."""

    def test_last_day(self):
        assert _last_day(2024, 2) == date(2024, 2, 29)  # leap year
        assert _last_day(2023, 2) == date(2023, 2, 28)
        assert _last_day(2024, 12) == date(2024, 12, 31)

    def test_month_range(self):
        months = list(_month_range(date(2024, 11, 1), date(2025, 2, 28)))
        assert months == [(2024, 11), (2024, 12), (2025, 1), (2025, 2)]

    def test_overlapping_months(self):
        result = _overlapping_months(date(2024, 6, 15), date(2024, 8, 10))
        assert result == [(2024, 6), (2024, 7), (2024, 8)]

    def test_parse_date_iso(self):
        assert _parse_date("2024-07-15") == date(2024, 7, 15)

    def test_parse_date_iso_with_time(self):
        assert _parse_date("2024-07-15T12:00:00") == date(2024, 7, 15)

    def test_parse_date_rfc822(self):
        assert _parse_date("Fri, 25 Jul 2024 12:00:00 GMT") == date(2024, 7, 25)

    def test_parse_date_none(self):
        assert _parse_date(None) is None

    def test_parse_date_invalid(self):
        assert _parse_date("not-a-date") is None

    def test_hazard_map(self):
        assert _HAZARD_MAP == {"DR": "DR", "FL": "FL", "TC": "TC"}


# ---------------------------------------------------------------------------
# Aggregation (multiple events in same country-month are summed)
# ---------------------------------------------------------------------------


class TestAggregation:
    """Multiple events in same country-month should SUM population values."""

    def test_sums_two_floods_same_month(self):
        connector = GdacsConnector()
        rows = [
            {"iso3": "BGD", "hazard_code": "FL", "year": 2024, "month": 7,
             "value": 500000.0, "alertlevel": "Orange", "todate": date(2024, 7, 25)},
            {"iso3": "BGD", "hazard_code": "FL", "year": 2024, "month": 7,
             "value": 300000.0, "alertlevel": "Green", "todate": date(2024, 7, 28)},
        ]
        result = connector._aggregate(rows)
        assert len(result) == 1
        assert result.iloc[0]["value"] == 800000.0
        # Max alert level should be Orange
        assert result.iloc[0]["alertlevel"] == "Orange"
