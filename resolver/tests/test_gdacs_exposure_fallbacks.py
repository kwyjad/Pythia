# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Two cheaper answers before the route that refuses.

``fetch_gdacs_events`` discovers events through the JSON search API, which
carries no exposure figure, and then asks the per-event datareport RSS for
each one. That route is a published-REPORT artefact: it answers 403 for any
event GDACS never wrote a report for, which says nothing about whether the
event happened or how many people it exposed — and a cell with no exposure
reconciles with no upper bound at all.

Two routes already answer and were not being asked:

* the 3-month static feeds (``_STATIC_RSS``) carry ``gdacs:population`` for
  every event they list, in ONE request, and this module already fetches
  them successfully elsewhere;
* the API's own ``geteventdata`` route describes an event whatever the
  report tree says, and is asked once after a refusal.

Network-free: the transport is injected everywhere.
"""

from __future__ import annotations

import datetime as dt

import pytest

from resolver.connectors import gdacs as core
from resolver.hazard_resolution import gdacs as haz

TODAY = dt.date(2026, 9, 8)


def _reset() -> None:
    core.reset_exposure_memo()
    getattr(core, "reset_search_property_keys", lambda: None)()


@pytest.fixture(autouse=True)
def clean_memo():
    _reset()
    yield
    _reset()


def _rss(items: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<rss xmlns:gdacs="http://www.gdacs.org" '
        '     xmlns:geo="http://www.w3.org/2003/01/geo/wgs84_pos#">'
        f"<channel>{items}</channel></rss>"
    ).encode("utf-8")


def _item(eid: str, etype: str = "FL", *, population: str = "84000",
          unit: str = "Pop74", iso3: str = "PHL") -> str:
    return (
        "<item>"
        f"<gdacs:eventtype>{etype}</gdacs:eventtype>"
        f"<gdacs:eventid>{eid}</gdacs:eventid>"
        f'<gdacs:population value="{population}" unit="{unit}">'
        f"{population} people</gdacs:population>"
        f"<gdacs:iso3>{iso3}</gdacs:iso3>"
        "<gdacs:country>Philippines</gdacs:country>"
        "<gdacs:fromdate>2026-08-01</gdacs:fromdate>"
        "<gdacs:todate>2026-08-05</gdacs:todate>"
        "<gdacs:alertlevel>Orange</gdacs:alertlevel>"
        "</item>"
    )


class _Resp:
    def __init__(self, status=200, content=b"", payload=None):
        self.status_code = status
        self.content = content
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Session:
    """Answers by URL substring; records every URL it was handed."""

    def __init__(self, routes):
        self.routes = routes
        self.calls: list[str] = []

    def get(self, url, **kwargs):
        self.calls.append(url)
        for needle, resp in self.routes.items():
            if needle in url:
                return resp() if callable(resp) else resp
        return _Resp(404)


# ---------------------------------------------------------------------------
# The static feed answers first
# ---------------------------------------------------------------------------


def test_a_static_feed_hit_costs_no_per_event_request(monkeypatch):
    """The acceptance case: the feed states it, so nobody asks again."""

    session = _Session({"rss_fl_3m.xml": _Resp(200, _rss(_item("1104004")))})
    monkeypatch.setattr(core, "_load_countries", lambda: ({}, {}))

    counts = haz.seed_exposure_memo_from_static_feed(
        "FL", window_end=dt.date(2026, 8, 31), session=session, today=TODAY
    )
    assert counts["seeded"] == 1

    connector = core.GdacsConnector()
    event = connector._enrich_one_event(
        session, {"eventtype": "FL", "eventid": "1104004"}, {}
    )
    assert event["population"] == 84000.0
    assert event["population_source"] == "static_feed"
    assert not any("datareport" in url for url in session.calls), (
        "the feed already stated the figure; asking the per-event route "
        "again is the request this change exists to save"
    )


def test_an_event_the_feed_does_not_list_still_falls_through(monkeypatch):
    session = _Session({"rss_fl_3m.xml": _Resp(200, _rss(_item("1104004")))})
    monkeypatch.setattr(core, "_load_countries", lambda: ({}, {}))

    haz.seed_exposure_memo_from_static_feed(
        "FL", window_end=dt.date(2026, 8, 31), session=session, today=TODAY
    )
    assert ("FL", "9999999") not in core._EXPOSURE_MEMO


def test_a_listed_event_with_no_readable_figure_is_not_seeded(monkeypatch):
    """UNKNOWN, not zero: it must still be asked about individually."""

    session = _Session(
        {"rss_fl_3m.xml": _Resp(200, _rss(_item("1104004", population="0")))}
    )
    monkeypatch.setattr(core, "_load_countries", lambda: ({}, {}))

    counts = haz.seed_exposure_memo_from_static_feed(
        "FL", window_end=dt.date(2026, 8, 31), session=session, today=TODAY
    )
    assert (counts["seeded"], counts["no_usable_figure"]) == (0, 1)
    assert ("FL", "1104004") not in core._EXPOSURE_MEMO


def test_a_window_older_than_the_feed_costs_no_request(monkeypatch):
    session = _Session({})
    monkeypatch.setattr(core, "_load_countries", lambda: ({}, {}))

    counts = haz.seed_exposure_memo_from_static_feed(
        "FL", window_end=dt.date(2024, 3, 31), session=session, today=TODAY
    )
    assert counts["seeded"] == 0
    assert session.calls == [], (
        "the feed reaches back three months; a 2024 window cannot be in it "
        "and asking spends a request to be told nothing"
    )


def test_drought_has_no_static_feed_and_asks_for_none(monkeypatch):
    """GDACS's drought feed has 404'd since 2026-03."""

    session = _Session({})
    monkeypatch.setattr(core, "_load_countries", lambda: ({}, {}))

    haz.seed_exposure_memo_from_static_feed(
        "DR", window_end=dt.date(2026, 8, 31), session=session, today=TODAY
    )
    assert session.calls == []


def test_an_unreadable_feed_seeds_nothing_and_never_raises(monkeypatch):
    session = _Session({"rss_fl_3m.xml": _Resp(503)})
    monkeypatch.setattr(core, "_load_countries", lambda: ({}, {}))

    counts = haz.seed_exposure_memo_from_static_feed(
        "FL", window_end=dt.date(2026, 8, 31), session=session, today=TODAY
    )
    assert counts["seeded"] == 0


def test_the_feed_overwrites_a_cache_entry_for_the_same_event(monkeypatch):
    """The cache holds a figure read on some earlier day; the feed is today's."""

    core._EXPOSURE_MEMO[("FL", "1104004")] = (
        {"population": 12.0, "population_source": "cache"},
        None,
    )
    session = _Session({"rss_fl_3m.xml": _Resp(200, _rss(_item("1104004")))})
    monkeypatch.setattr(core, "_load_countries", lambda: ({}, {}))

    haz.seed_exposure_memo_from_static_feed(
        "FL", window_end=dt.date(2026, 8, 31), session=session, today=TODAY
    )
    episode, _refused = core._EXPOSURE_MEMO[("FL", "1104004")]
    assert (episode["population"], episode["population_source"]) == (
        84000.0, "static_feed",
    )


def test_the_machine_still_borrows_the_static_feed_map():
    """The reuse contract: one address, one implementation."""

    api = haz._connector_api()
    assert "FL" in api._STATIC_RSS
    assert hasattr(api.GdacsConnector, "fetch_static_rss_for_type")


# ---------------------------------------------------------------------------
# A 403 on the report tree, then the API's own route
# ---------------------------------------------------------------------------


def test_a_403_then_geteventdata_success_yields_a_figure():
    """A 403 on the report tree says a REPORT is missing, not the event."""

    session = _Session({
        "datareport": _Resp(403),
        "geteventdata": _Resp(
            200, payload={"impacts": [{"impacttype": "wind", "pop39": 250000}]}
        ),
    })
    connector = core.GdacsConnector()
    best, refused = connector._fetch_event_exposure(session, "TC", "1001273", {})

    assert refused is None, "the API answered; the event is not refused"
    assert best is not None and best["population"] == 250000.0
    assert best["population_source"] == "geteventdata"
    assert sum("datareport" in u for u in session.calls) == 1, (
        "the refusing route is asked once, not retried"
    )
    assert sum("geteventdata" in u for u in session.calls) == 1


def test_a_403_then_geteventdata_failure_is_still_a_refusal():
    """An event GDACS genuinely will not describe still counts as refused."""

    session = _Session({"datareport": _Resp(403), "geteventdata": _Resp(500)})
    connector = core.GdacsConnector()
    best, refused = connector._fetch_event_exposure(session, "TC", "1001273", {})

    assert (best, refused) == (None, 403)
    assert sum("geteventdata" in u for u in session.calls) == 1, (
        "the fallback is asked once too — it is a fallback, not a second ladder"
    )


def test_geteventdata_with_no_population_field_is_a_refusal_not_a_zero():
    session = _Session({
        "datareport": _Resp(403),
        "geteventdata": _Resp(200, payload={"impacts": [{"impacttype": "wind"}]}),
    })
    connector = core.GdacsConnector()
    assert connector._fetch_event_exposure(session, "TC", "1001273", {}) == (None, 403)


def test_a_200_from_the_report_tree_never_asks_geteventdata():
    session = _Session({"datareport": _Resp(200, _rss(_item("1104004")))})
    connector = core.GdacsConnector()
    best, refused = connector._fetch_event_exposure(session, "FL", "1104004", {})

    assert refused is None and best is not None
    assert not any("geteventdata" in u for u in session.calls)


def test_a_404_from_the_report_tree_never_asks_geteventdata():
    """404 is the report tree saying the event is not there at all."""

    session = _Session({"datareport": _Resp(404)})
    connector = core.GdacsConnector()
    assert connector._fetch_event_exposure(session, "FL", "1104004", {}) == (None, None)
    assert not any("geteventdata" in u for u in session.calls)


def test_the_refusal_fallback_is_remembered_like_any_other_answer():
    """One answer per event per process, whichever route supplied it."""

    session = _Session({
        "datareport": _Resp(403),
        "geteventdata": _Resp(200, payload={"impacts": [{"pop39": 250000}]}),
    })
    connector = core.GdacsConnector()
    for _ in range(3):
        connector._enrich_one_event(
            session, {"eventtype": "TC", "eventid": "1001273"}, {}
        )
    assert sum("geteventdata" in u for u in session.calls) == 1


# ---------------------------------------------------------------------------
# Reading the body
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["pop39", "pop74", "POPAFFECTED"])
def test_each_documented_population_field_is_read(field):
    people, detail = core.parse_geteventdata_population(
        {"impacts": [{field: 40000}]}
    )
    assert people == 40000.0
    assert detail["field"] == field.lower()


def test_the_largest_candidate_wins():
    """The figure bounds plausible impact; a bound set too low rejects
    correct figures, which is the fault the ceiling multiplier ended."""

    people, detail = core.parse_geteventdata_population(
        {"impacts": [{"pop74": 10000, "pop39": 250000, "popaffected": 90000}]}
    )
    assert people == 250000.0
    assert detail["field"] == "pop39"
    assert set(detail["candidates"]) == {"pop74", "pop39", "popaffected"}


def test_a_field_nested_under_properties_is_still_found():
    people, _ = core.parse_geteventdata_population(
        {"properties": {"impacts": [{"severity": {"pop39": 1234}}]}}
    )
    assert people == 1234.0


def test_a_body_with_no_population_field_names_the_keys_it_saw():
    people, detail = core.parse_geteventdata_population(
        {"eventid": 1, "alertlevel": "Orange", "impacts": []}
    )
    assert people is None
    assert detail["outcome"] == "no_population_field"
    assert "alertlevel" in detail["keys_seen"], (
        "a changed shape must be settled by evidence in the next bundle, "
        "not by re-reading the parser"
    )


def test_an_unreadable_value_is_unknown_never_the_bare_number():
    people, _ = core.parse_geteventdata_population({"impacts": [{"pop39": "n/a"}]})
    assert people is None


def test_a_non_positive_figure_is_unknown():
    people, _ = core.parse_geteventdata_population({"impacts": [{"pop39": 0}]})
    assert people is None
