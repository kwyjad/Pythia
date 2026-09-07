# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Group C of the run-34124705852 repairs: asking a refusing source less.

Two runs measured the same thing. Run 33946954189 was refused 532 of 1,050
per-event requests. Run 34124705852 cut volume by 73% with a process memo
and the refusal rate barely moved, 79.6% to 76%. A rate limit eases when
the rate falls; this did not. And of 291 distinct events, 153 were served
on the FIRST request and 138 were refused on all four attempts, which is a
decision made per event rather than a throttle riding out.

So the retry was spending four requests to learn what one already said, and
the cache was read too late to prevent a single request: ``_carry_forward_
exposure`` ran AFTER every fetch, repairing refusals it could have avoided.

The repairs, in the order they save a request: the search response is used
when it already states an exposure; the cache answers for any event that
ended long enough ago for its figure to be settled; a refusal is asked once;
and the pace is held by a process-wide bucket rather than by a per-worker
sleep that multiplies by the worker count.

Network-free — the transport is injected everywhere.
"""

from __future__ import annotations

import datetime as dt
import time

import duckdb
import pytest

from resolver.connectors import gdacs as core
from resolver.hazard_resolution import gdacs as haz
from resolver.hazard_resolution.schema import ensure_haz_schema
from resolver.hazard_resolution.sources import RawRecord, store_raw_records
from resolver.tests.hazard_resolution_utils import make_rulebook


@pytest.fixture()
def con(tmp_path):
    connection = duckdb.connect(str(tmp_path / "haz.duckdb"))
    ensure_haz_schema(connection)
    return connection


def _reset() -> None:
    """Clear both per-run stores.

    Tolerant of a missing reset by design: without it every test in this
    file errors in setup against the pre-fix code, and a wall of setup
    errors says nothing about WHICH assertion the old behaviour breaks.
    """

    core.reset_exposure_memo()
    getattr(core, "reset_search_property_keys", lambda: None)()


@pytest.fixture(autouse=True)
def clean_memo():
    _reset()
    yield
    _reset()


def _cached_event(
    event_id: str, *, population: float, end_date: str, event_type: str = "FL"
) -> RawRecord:
    return RawRecord(
        record_id=f"{event_type}-{event_id}",
        payload={
            "event_id": event_id,
            "event_type": event_type,
            "hazard": event_type,
            "exposed_population": population,
            "exposed_population_unit": "people",
            "start_date": end_date,
            "end_date": end_date,
        },
        iso3="PHL",
        ym=end_date[:7],
        hazard=event_type,
        source_url=f"https://example.test/{event_id}",
    )


# ---------------------------------------------------------------------------
# C4: a refusal is recorded once, not four times
# ---------------------------------------------------------------------------


def test_a_403_is_not_retried():
    """138 events were refused on all four attempts: 552 requests to learn
    what 138 already said."""

    assert 403 not in core._RETRYABLE_ENRICH_STATUS
    assert 403 in core._REFUSAL_STATUS


def test_a_429_is_still_retried():
    """That is the code a source uses when it means slow down."""

    assert 429 in core._RETRYABLE_ENRICH_STATUS


def test_a_refused_event_costs_exactly_one_request():
    calls: list[str] = []

    class _Resp:
        status_code = 403
        content = b""

    class _Session:
        def get(self, url, **kwargs):
            calls.append(url)
            return _Resp()

    connector = core.GdacsConnector()
    best, refused = connector._fetch_event_exposure(_Session(), "FL", "1104004", {})
    assert (best, refused) == (None, 403)
    assert len(calls) == 1, (
        f"a refusal must be asked once, not {len(calls)} times — the answer "
        "to a refusing source is fewer requests, not a harder retry"
    )


def test_a_server_error_is_still_retried():
    calls: list[str] = []

    class _Resp:
        status_code = 503
        content = b""

    class _Session:
        def get(self, url, **kwargs):
            calls.append(url)
            return _Resp()

    connector = core.GdacsConnector()
    _, refused = connector._fetch_event_exposure(_Session(), "FL", "1", {})
    assert refused == 503
    assert len(calls) > 1, "a 5xx is worth asking again about"


# ---------------------------------------------------------------------------
# C4: the pace is process-wide
# ---------------------------------------------------------------------------


def test_the_rate_limiter_holds_an_interval():
    limiter = core._RateLimiter(0.05)
    started = time.monotonic()
    for _ in range(4):
        limiter.acquire()
    # Three intervals for four acquisitions; the first is free.
    assert time.monotonic() - started >= 0.13


def test_a_zero_interval_never_sleeps():
    limiter = core._RateLimiter(0.0)
    started = time.monotonic()
    for _ in range(50):
        limiter.acquire()
    assert time.monotonic() - started < 0.5


def test_one_worker_is_the_default():
    assert core._DEFAULT_ENRICH_WORKERS == 1


# ---------------------------------------------------------------------------
# C3: the cache is read BEFORE the network, not after it
# ---------------------------------------------------------------------------


def test_a_settled_exposure_is_served_from_the_cache(con):
    """An event that ended long ago has a settled figure. Asking again buys
    a request and nothing else."""

    store_raw_records(
        con, haz.SOURCE,
        [_cached_event("1104004", population=900_000, end_date="2024-03-04")],
    )
    counts = haz.seed_exposure_memo(
        con, refresh_days=21, today=dt.date(2026, 9, 7)
    )
    assert counts["seeded"] == 1
    assert core._EXPOSURE_MEMO[("FL", "1104004")][0]["population"] == 900_000


def test_a_live_event_is_still_asked_about(con):
    """While an event is live its exposure is still being revised, and
    serving last week's figure as this week's is the serve-old-data
    failure in another costume."""

    store_raw_records(
        con, haz.SOURCE,
        [_cached_event("1104004", population=900_000, end_date="2026-09-01")],
    )
    counts = haz.seed_exposure_memo(
        con, refresh_days=21, today=dt.date(2026, 9, 7)
    )
    assert counts["seeded"] == 0
    assert counts["still_live"] == 1
    assert ("FL", "1104004") not in core._EXPOSURE_MEMO


def test_an_event_with_no_usable_figure_is_not_seeded(con):
    """A zero is GDACS declining to say, not GDACS saying nobody was
    exposed — seeding it would make the refusal permanent."""

    store_raw_records(
        con, haz.SOURCE,
        [_cached_event("1104004", population=0.0, end_date="2024-03-04")],
    )
    counts = haz.seed_exposure_memo(
        con, refresh_days=21, today=dt.date(2026, 9, 7)
    )
    assert counts == {"seeded": 0, "still_live": 0, "no_usable_figure": 1}


def test_a_seeded_event_makes_no_request(con):
    store_raw_records(
        con, haz.SOURCE,
        [_cached_event("1104004", population=900_000, end_date="2024-03-04")],
    )
    haz.seed_exposure_memo(con, refresh_days=21, today=dt.date(2026, 9, 7))

    calls: list[str] = []

    class _Session:
        def get(self, url, **kwargs):  # pragma: no cover - must never run
            calls.append(url)
            raise AssertionError("the cache already knows this event")

    connector = core.GdacsConnector()
    event = {"eventtype": "FL", "eventid": "1104004"}
    enriched = connector._enrich_one_event(_Session(), event, {})
    assert calls == []
    assert enriched["population"] == 900_000
    assert enriched["population_enriched"] is True


def test_a_cache_served_figure_says_it_came_from_the_cache(con):
    """A stale figure is worth more than none, and only worth anything if
    the row says it is stale."""

    store_raw_records(
        con, haz.SOURCE,
        [_cached_event("1104004", population=900_000, end_date="2024-03-04")],
    )
    haz.seed_exposure_memo(con, refresh_days=21, today=dt.date(2026, 9, 7))
    connector = core.GdacsConnector()
    enriched = connector._enrich_one_event(
        object(), {"eventtype": "FL", "eventid": "1104004"}, {}
    )
    assert enriched["population_source"] == "cache"
    assert enriched["population_cached_at"]


def test_a_live_fetch_is_labelled_live():
    class _Resp:
        status_code = 200
        content = b""

        def raise_for_status(self):
            return None

    class _Session:
        def get(self, url, **kwargs):
            return _Resp()

    connector = core.GdacsConnector()
    connector._parse_rss = lambda content, names: [  # type: ignore[method-assign]
        {"todate": dt.date(2024, 3, 4), "pub_date": None, "population": 5000.0}
    ]
    enriched = connector._enrich_one_event(
        _Session(), {"eventtype": "FL", "eventid": "9"}, {}
    )
    assert enriched["population"] == 5000.0
    assert enriched["population_source"] == "live"


def test_seeding_never_raises_on_an_unreadable_cache(tmp_path):
    empty = duckdb.connect(str(tmp_path / "no-schema.duckdb"))
    counts = haz.seed_exposure_memo(empty, refresh_days=21)
    assert counts["seeded"] == 0


# ---------------------------------------------------------------------------
# C5: what the bulk route actually carries
# ---------------------------------------------------------------------------


def test_a_population_stated_by_the_search_response_is_used():
    """If the bulk route already says it, the per-event route is a request
    spent for nothing — and the code assumed it does not without ever
    recording what the response carried."""

    assert core._population_from_properties({"population": "12,500"}) == (
        12_500.0, "search:population",
    )
    assert core._population_from_properties(
        {"severitydata": {"populationexposed": 4200}}
    ) == (4200.0, "search:severitydata.populationexposed")


def test_a_non_positive_or_unreadable_bulk_figure_is_absent():
    """UNKNOWN everywhere else in this connector, and here too — otherwise
    the bulk route is a way to smuggle a guess in."""

    for props in ({"population": 0}, {"population": "lots"}, {"alertlevel": "Red"}):
        assert core._population_from_properties(props) == (0.0, "")


def test_the_search_property_keys_are_recorded():
    core.reset_search_property_keys()
    core._note_search_properties({"eventid": 1, "alertlevel": "Red"})
    core._note_search_properties({"eventid": 2, "severitydata": {}})
    assert core.observed_search_property_keys() == [
        "alertlevel", "eventid", "severitydata",
    ]


# ---------------------------------------------------------------------------
# The rulebook owns the pacing it claims to own
# ---------------------------------------------------------------------------


def test_the_rulebook_pacing_keys_exist_and_are_conservative():
    rb = make_rulebook()
    assert rb.get("flood.gdacs.enrich_workers") == 1
    assert rb.get("flood.gdacs.enrich_min_interval_sec") >= 1.0
    assert rb.get("flood.gdacs.exposure_refresh_days") >= 1


def test_the_worker_count_is_taken_from_the_argument_not_the_env(monkeypatch):
    """It was validated by the rulebook and read by nothing, so lowering it
    did nothing at all."""

    monkeypatch.setenv("GDACS_ENRICH_WORKERS", "8")
    seen: dict[str, int] = {}
    connector = core.GdacsConnector()

    def _one(session, ev, names):
        seen["calls"] = seen.get("calls", 0) + 1
        return ev

    connector._enrich_one_event = _one  # type: ignore[method-assign]
    connector._enrich_with_population(
        object(), [{"eventtype": "FL", "eventid": "1"}], 0.0, {},
        workers=1, min_interval=0.0,
    )
    assert seen["calls"] == 1


def test_cyclone_borrows_the_gdacs_block_for_its_fingerprint():
    """The GDACS block lives under `flood` and configures the fetch for
    EVERY hazard, so a pacing change moves cyclone's answers too. Without
    this, cyclone's ledger claims months decided under rules that moved."""

    rb = make_rulebook()
    moved = make_rulebook({"flood": {"gdacs": {"enrich_min_interval_sec": 9.0}}})
    assert rb.hazard_fingerprint("cyclone") != moved.hazard_fingerprint("cyclone")
    assert rb.hazard_fingerprint("flood") != moved.hazard_fingerprint("flood")
    assert rb.hazard_fingerprint("drought") == moved.hazard_fingerprint("drought"), (
        "drought has no ladder and no ceiling, so nothing in the GDACS "
        "block reaches it — re-walking its 114 months would be waste"
    )


# ---------------------------------------------------------------------------
# A polite pace must not become a killed step
# ---------------------------------------------------------------------------


def test_the_pass_stops_asking_rather_than_outrunning_its_step():
    """3,272 events at one request every two seconds is 109 minutes against
    a 60-minute reset step. The answer is to stop asking, not to speed up."""

    asked: list[str] = []
    connector = core.GdacsConnector()

    def _one(session, ev, names):
        asked.append(str(ev["eventid"]))
        time.sleep(0.02)
        ev["population_enriched"] = True
        return ev

    connector._enrich_one_event = _one  # type: ignore[method-assign]
    events = [{"eventtype": "FL", "eventid": str(i)} for i in range(50)]
    out = connector._enrich_with_population(
        object(), events, 0.0, {}, workers=1, min_interval=0.0, max_seconds=0.1,
    )
    assert len(out) == 50, "every event is returned, asked or not"
    assert 0 < len(asked) < 50, (
        f"the pass must stop early, not ask all 50 ({len(asked)} asked)"
    )


def test_an_unasked_event_keeps_no_invented_figure():
    connector = core.GdacsConnector()
    connector._enrich_one_event = lambda s, ev, n: ev  # type: ignore[method-assign]
    events = [{"eventtype": "FL", "eventid": "1"}]
    out = connector._enrich_with_population(
        object(), events, 0.0, {}, workers=1, min_interval=0.0, max_seconds=-1,
    )
    assert out[0].get("population_enriched") in (None, False)
    assert "population" not in out[0]


def test_no_budget_means_no_ceiling():
    connector = core.GdacsConnector()
    asked: list[str] = []
    connector._enrich_one_event = lambda s, ev, n: (  # type: ignore[method-assign]
        asked.append(str(ev["eventid"])) or ev
    )
    events = [{"eventtype": "FL", "eventid": str(i)} for i in range(10)]
    connector._enrich_with_population(
        object(), events, 0.0, {}, workers=1, min_interval=0.0, max_seconds=0,
    )
    assert len(asked) == 10


def test_the_rulebook_budget_fits_inside_the_step():
    """The flood and cyclone machine steps are 90 minutes each."""

    rb = make_rulebook()
    assert 0 < rb.get("flood.gdacs.enrich_max_seconds") <= 85 * 60
