# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Group B of the run-33946954189 repairs: GDACS.

Three faults, all of which cost the impact ladder its upper bound:

* **532 of 1,050** per-event enrichment requests came back HTTP 403 — rate
  limiting, from six workers with no delay and a retry list that does not
  carry 403 — so half the events lost their exposure figure and the cells
  that needed one reconciled with no upper bound at all;
* seventeen tropical cyclones were dropped because `_parse_item` never read
  a position out of the RSS, leaving the geometry resolver nothing to place
  them with; and
* every one of the 1,324 figures-ledger rows carried an empty ceiling beside
  a constant `ceiling_field`, so a blank ceiling could not be told from a
  ceiling that had been evaluated.

Network-free: the transport is injected.
"""

from __future__ import annotations

import datetime as dt
import xml.etree.ElementTree as ET

import duckdb
import pytest

from resolver.connectors import gdacs as connector_mod
from resolver.connectors.gdacs import GdacsConnector

_ITEM_WITH_GEO = """<?xml version="1.0" encoding="UTF-8"?>
<rss xmlns:gdacs="http://www.gdacs.org"
     xmlns:geo="http://www.w3.org/2003/01/geo/wgs84_pos#"
     xmlns:georss="http://www.georss.org/georss">
  <channel>
    <item>
      <gdacs:eventtype>TC</gdacs:eventtype>
      <gdacs:eventid>1001314</gdacs:eventid>
      <gdacs:fromdate>2026-08-01</gdacs:fromdate>
      <gdacs:todate>2026-08-04</gdacs:todate>
      <gdacs:alertlevel>Orange</gdacs:alertlevel>
      <geo:lat>18.5</geo:lat>
      <geo:long>-72.3</geo:long>
    </item>
  </channel>
</rss>"""

_ITEM_WITH_GEORSS_ONLY = _ITEM_WITH_GEO.replace(
    "<geo:lat>18.5</geo:lat>\n      <geo:long>-72.3</geo:long>",
    "<georss:point>18.5 -72.3</georss:point>",
)

_ITEM_WITHOUT_POSITION = _ITEM_WITH_GEO.replace(
    "<geo:lat>18.5</geo:lat>\n      <geo:long>-72.3</geo:long>", ""
)


class TestRssPosition:
    """B2 — a country-less event now carries something to place it with."""

    def setup_method(self):
        self.connector = GdacsConnector()

    def test_geo_lat_long_are_read(self):
        events = self.connector._parse_rss(_ITEM_WITH_GEO.encode(), {})

        assert len(events) == 1
        assert events[0]["lat"] == pytest.approx(18.5)
        assert events[0]["lon"] == pytest.approx(-72.3)

    def test_georss_point_is_read(self):
        events = self.connector._parse_rss(_ITEM_WITH_GEORSS_ONLY.encode(), {})

        assert events[0]["lat"] == pytest.approx(18.5)
        assert events[0]["lon"] == pytest.approx(-72.3)

    def test_an_item_with_no_position_is_still_parsed(self):
        events = self.connector._parse_rss(_ITEM_WITHOUT_POSITION.encode(), {})

        assert len(events) == 1
        assert events[0]["lat"] is None
        assert events[0]["lon"] is None

    def test_the_georss_namespace_is_registered(self):
        # Without it, ElementTree finds nothing under georss: and the point
        # element reads as absent rather than as unparseable.
        assert "georss" in connector_mod._NS


class _Response:
    def __init__(self, status_code: int, content: bytes = b""):
        self.status_code = status_code
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Session:
    """A session that refuses n times and then answers."""

    def __init__(self, refusals: int, status: int = 403, body: bytes = b""):
        self.refusals = refusals
        self.status = status
        self.body = body
        self.calls = 0

    def get(self, url, timeout=None):  # noqa: ARG002
        self.calls += 1
        if self.calls <= self.refusals:
            return _Response(self.status)
        return _Response(200, self.body)


_ENRICH_RSS = _ITEM_WITH_GEO.replace(
    "<gdacs:alertlevel>Orange</gdacs:alertlevel>",
    '<gdacs:alertlevel>Orange</gdacs:alertlevel>\n'
    '      <gdacs:population value="40000" unit="people">40000 people</gdacs:population>',
).encode()


class TestEnrichmentRetry:
    """B1 — a refusal is rate limiting, and rate limiting is retried.

    **That premise was refuted by measurement and this class records how.**
    It was a reasonable reading in September 2026: GDACS answers a throttled
    caller 403, urllib3's retry list does not carry it, and six workers with
    no delay had lost 532 of 1,050 exposure figures. So the ladder was built
    to ride the throttle out.

    Two runs then measured it. Cutting volume by 73% moved the refusal rate
    from 79.6% to 76%, and a rate limit eases when the rate falls. And of
    291 distinct events, 153 were served on the FIRST request while 138 were
    refused on all four attempts — a decision made per event, not a throttle
    riding out. 138 x 4 = 552, exactly the refusals.

    The tests below are kept and inverted rather than deleted, because the
    refuted belief is the most useful thing here: the next person to see a
    403 from a source will reach for a retry, and this is the evidence that
    it does not work. What SURVIVES intact is the intent underneath: a
    refusal must never silently lose a figure. It no longer loses one
    because the retry catches it. It no longer loses one because the cache
    already holds it.
    """

    def setup_method(self, method):  # noqa: ARG002
        self.connector = GdacsConnector()
        connector_mod.reset_exposure_memo()

    def teardown_method(self, method):  # noqa: ARG002
        connector_mod.reset_exposure_memo()

    def test_a_403_is_not_retried(self, monkeypatch):
        """Four requests to learn what the first one already said."""

        monkeypatch.setattr(connector_mod.time, "sleep", lambda _s: None)
        session = _Session(refusals=2, body=_ENRICH_RSS)
        event = {"eventtype": "TC", "eventid": "1001314", "population": 0.0}

        out = self.connector._enrich_one_event(session, event, {})

        assert session.calls == 1
        assert out["population_refused"] == 403

    def test_a_refused_figure_still_lands_from_the_cache(self, tmp_path, monkeypatch):
        """The intent of the retry, kept, by the mechanism that works.

        A refusal must never silently lose a figure. It used to be caught
        by asking again; it is now caught by never having needed to ask.
        """

        from resolver.hazard_resolution import gdacs as machine
        from resolver.hazard_resolution.schema import ensure_haz_schema
        from resolver.hazard_resolution.sources import RawRecord, store_raw_records

        con = duckdb.connect(str(tmp_path / "haz.duckdb"))
        ensure_haz_schema(con)
        store_raw_records(
            con, machine.SOURCE,
            [RawRecord(
                record_id="TC-1001314",
                payload={
                    "event_id": "1001314",
                    "exposed_population": 40000.0,
                    "end_date": "2024-03-04",
                },
                iso3="HTI", hazard="TC",
            )],
        )
        machine.seed_exposure_memo(
            con, refresh_days=21, today=dt.date(2026, 9, 7)
        )

        monkeypatch.setattr(connector_mod.time, "sleep", lambda _s: None)
        session = _Session(refusals=99)
        event = {"eventtype": "TC", "eventid": "1001314", "population": 0.0}

        out = self.connector._enrich_one_event(session, event, {})

        assert session.calls == 0, "the cache answered; nothing was asked"
        assert out["population"] == pytest.approx(40000.0)
        assert out["population_enriched"] is True
        assert out["population_source"] == "cache", (
            "a stale figure is worth more than none and only worth "
            "anything if it says it is stale"
        )

    def test_a_persistent_refusal_is_recorded_not_silently_dropped(self, monkeypatch):
        """Unchanged in intent: a refusal is a fact and must be on the row."""

        monkeypatch.setattr(connector_mod.time, "sleep", lambda _s: None)
        session = _Session(refusals=99)
        event = {"eventtype": "TC", "eventid": "1001314", "population": 0.0}

        out = self.connector._enrich_one_event(session, event, {})

        assert session.calls == 1
        assert out["population_refused"] == 403
        assert out["population_enriched"] is False

    def test_a_404_is_not_retried(self, monkeypatch):
        monkeypatch.setattr(connector_mod.time, "sleep", lambda _s: None)
        session = _Session(refusals=99, status=404)
        event = {"eventtype": "TC", "eventid": "1", "population": 0.0}

        self.connector._enrich_one_event(session, event, {})

        assert session.calls == 1, "an absent event is not a throttled one"

    def test_403_left_the_retryable_set_and_429_stayed(self):
        """The inverted assertion, with the reason on the row.

        429 is the code a source uses when it means "slow down" and is
        worth obeying. 403 is not that, whatever it looked like.
        """

        assert 403 not in connector_mod._RETRYABLE_ENRICH_STATUS
        assert 403 in connector_mod._REFUSAL_STATUS
        assert 429 in connector_mod._RETRYABLE_ENRICH_STATUS

    def test_the_pool_is_one_by_default(self, monkeypatch):
        monkeypatch.delenv("GDACS_ENRICH_WORKERS", raising=False)

        assert connector_mod._DEFAULT_ENRICH_WORKERS == 1, (
            "six workers with no delay of their own drew the refusals, and "
            "two with a quarter of a second did not stop them — the pace is "
            "held by a process-wide bucket now, not by the worker count"
        )

    def test_the_enrichment_delay_has_its_own_knob(self, monkeypatch):
        monkeypatch.delenv("GDACS_ENRICH_DELAY", raising=False)
        # Unset, the caller's value stands — a caller asking for none gets none.
        assert connector_mod._enrich_delay(0.0) == pytest.approx(0.0)
        assert connector_mod._enrich_delay(1.0) == pytest.approx(1.0)

        monkeypatch.setenv("GDACS_ENRICH_DELAY", "0.25")
        assert connector_mod._enrich_delay(1.0) == pytest.approx(0.25), (
            "an operator must be able to slow enrichment without touching "
            "the discovery delay the caller passes"
        )

    def test_backoff_is_jittered(self, monkeypatch):
        seen = {connector_mod._backoff_seconds(2) for _ in range(20)}

        assert len(seen) > 1, (
            "a pool that all sleeps the same interval wakes together and "
            "reproduces the burst that drew the refusal"
        )


class TestExposureCarryForward:
    """B1 — a refused run must not overwrite a stored exposure with nothing."""

    def _con(self, tmp_path):
        from resolver.hazard_resolution.schema import ensure_haz_schema

        con = duckdb.connect(str(tmp_path / "haz.duckdb"))
        ensure_haz_schema(con)
        return con

    def test_a_missing_exposure_is_filled_from_the_cache(self, tmp_path):
        from resolver.hazard_resolution import gdacs as machine
        from resolver.hazard_resolution.sources import RawRecord, store_raw_records

        con = self._con(tmp_path)
        store_raw_records(
            con,
            machine.SOURCE,
            [
                RawRecord(
                    record_id="TC-1001314",
                    payload={"event_id": "1001314", "exposed_population": 40000.0},
                    iso3="HTI",
                    hazard="TC",
                )
            ],
        )

        record = RawRecord(
            record_id="TC-1001314",
            payload={"event_id": "1001314", "exposed_population": 0.0},
            iso3="HTI",
            hazard="TC",
        )
        filled = machine._carry_forward_exposure(con, [record])

        assert filled == 1
        assert record.payload["exposed_population"] == pytest.approx(40000.0)
        assert record.payload["exposed_population_source"] == "cache"
        con.close()

    def test_a_fresh_figure_is_never_overwritten_by_the_cache(self, tmp_path):
        from resolver.hazard_resolution import gdacs as machine
        from resolver.hazard_resolution.sources import RawRecord, store_raw_records

        con = self._con(tmp_path)
        store_raw_records(
            con,
            machine.SOURCE,
            [
                RawRecord(
                    record_id="TC-1",
                    payload={"event_id": "1", "exposed_population": 40000.0},
                    hazard="TC",
                )
            ],
        )
        record = RawRecord(
            record_id="TC-1",
            payload={"event_id": "1", "exposed_population": 12345.0},
            hazard="TC",
        )

        assert machine._carry_forward_exposure(con, [record]) == 0
        assert record.payload["exposed_population"] == pytest.approx(12345.0)
        assert "exposed_population_source" not in record.payload
        con.close()

    def test_an_unreadable_cache_leaves_the_records_alone(self, tmp_path):
        from resolver.hazard_resolution import gdacs as machine
        from resolver.hazard_resolution.sources import RawRecord

        con = self._con(tmp_path)
        con.close()  # every query now raises
        record = RawRecord(record_id="TC-1", payload={"exposed_population": 0.0})

        assert machine._carry_forward_exposure(con, [record]) == 0


class TestCeilingIsRecorded:
    """B3 — a blank ceiling must say why it is blank."""

    def test_no_usable_exposure_names_no_field_and_states_its_basis(self, tmp_path):
        from resolver.hazard_resolution import candidates as cand
        from resolver.hazard_resolution.schema import ensure_haz_schema

        con = duckdb.connect(str(tmp_path / "haz.duckdb"))
        ensure_haz_schema(con)

        basis = cand.exposure_ceiling_basis(con, "HTI", "2026-08", "TC")

        assert basis["value"] is None
        assert basis["basis"] == "no_usable_gdacs_exposure"
        assert basis["field"] is None, (
            "a constant field beside a blank ceiling reads as a ceiling that "
            "was evaluated and came out empty"
        )
        assert basis["source"] is None
        con.close()

    def test_a_real_exposure_names_the_field_it_came_from(self, tmp_path, monkeypatch):
        from resolver.hazard_resolution import candidates as cand
        from resolver.hazard_resolution.schema import ensure_haz_schema

        con = duckdb.connect(str(tmp_path / "haz.duckdb"))
        ensure_haz_schema(con)

        monkeypatch.setattr(
            cand.gdacs_mod,
            "events_for_country_month",
            lambda *_a, **_k: [
                {
                    "event_id": "1",
                    "exposed_population": 40000.0,
                    "exposed_population_unit": "people",
                    "alert_level": "Orange",
                    "iso3_list": ["HTI"],
                }
            ],
        )
        basis = cand.exposure_ceiling_basis(con, "HTI", "2026-08", "TC")

        assert basis["basis"] == "gdacs_exposed"
        assert basis["field"] == cand.CEILING_FIELD
        assert basis["value"] == pytest.approx(40000.0)
        con.close()

    def test_the_ledger_carries_the_basis(self):
        import inspect

        from resolver.hazard_resolution import cell_ledger

        assert "ceiling_basis" in inspect.signature(
            cell_ledger.record_figure
        ).parameters
