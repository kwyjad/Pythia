# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Phase-2 tests for the ladder's source connectors and the raw caches.

All network-free: every connector exposes an injectable transport seam
(``post``/``get``), matching the ``PostFn`` convention Phase 1 established
in ``reliefweb_sweep.py``.

The recurring theme is the distinction the whole machine rests on:
**"the source says nothing happened" and "we could not ask the source"
are different facts**, and a connector that blurs them can manufacture a
zero out of an outage.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from pathlib import Path

import duckdb
import pytest

from resolver.hazard_resolution import emdat as emdat_mod
from resolver.hazard_resolution import gdacs as gdacs_mod
from resolver.hazard_resolution import idmc_idu as idu_mod
from resolver.hazard_resolution import ifrc_go as go_mod
from resolver.hazard_resolution import reliefweb_docs as reliefweb_docs_mod
from resolver.hazard_resolution import sources as sources_mod
from resolver.hazard_resolution.schema import ensure_haz_schema
from resolver.hazard_resolution.sources import (
    VOLATILE_PAYLOAD_KEYS,
    RawRecord,
    content_hash,
    fetch_window,
    load_raw_records,
    parse_date,
    parse_number,
    shift_month,
    stable_payload,
    store_raw_records,
)
from resolver.tests.hazard_resolution_utils import (
    SYNTHETIC_COUNTRIES_GEOJSON,
    make_rulebook,
)


@pytest.fixture()
def rulebook():
    return make_rulebook()


@pytest.fixture()
def con():
    con = duckdb.connect(":memory:")
    ensure_haz_schema(con)
    return con


# ---------------------------------------------------------------------------
# Raw-cache plumbing
# ---------------------------------------------------------------------------


def test_identical_refetch_is_a_no_op_but_a_revision_appends(con):
    """The content-hash idiom Phase 1 established, on a Phase 2 cache."""
    record = RawRecord(
        record_id="emdat-1", payload={"total_affected": 100}, iso3="PHL",
        ym="2024-03", hazard="FL",
    )
    assert store_raw_records(con, "emdat", [record])["inserted"] == 1
    assert store_raw_records(con, "emdat", [record])["inserted"] == 0  # no-op

    revised = RawRecord(
        record_id="emdat-1", payload={"total_affected": 250}, iso3="PHL",
        ym="2024-03", hazard="FL",
    )
    assert store_raw_records(con, "emdat", [revised])["inserted"] == 1
    # Both revisions survive; the reader takes the newest.
    assert con.execute("SELECT COUNT(*) FROM haz_raw_emdat").fetchone()[0] == 2
    loaded = load_raw_records(con, "emdat")
    assert len(loaded) == 1
    assert loaded[0]["total_affected"] == 250


def test_a_wall_clock_in_the_payload_does_not_defeat_dedup(con):
    """The defect that grew the canonical DB from 3.0 GB to 17.7 GB.

    Seven connectors stamped ``"fetched_at": utcnow_iso()`` into the payload
    that ``content_hash`` covers, so the hash differed on every fetch,
    ``INSERT OR IGNORE`` never ignored, and each run appended a full
    duplicate of every record it touched -- ReliefWeb bodies included, at
    ~720 KB a cell. Content-only hashing is what makes the cache idempotent.
    """
    first = RawRecord(
        record_id="emdat-1",
        payload={"total_affected": 100, "fetched_at": "2026-08-05T20:30:00Z"},
        iso3="PHL", ym="2024-03", hazard="FL",
    )
    later = RawRecord(
        record_id="emdat-1",
        payload={"total_affected": 100, "fetched_at": "2026-08-06T20:30:00Z"},
        iso3="PHL", ym="2024-03", hazard="FL",
    )
    assert store_raw_records(con, "emdat", [first])["inserted"] == 1
    assert store_raw_records(con, "emdat", [later])["inserted"] == 0
    assert con.execute("SELECT COUNT(*) FROM haz_raw_emdat").fetchone()[0] == 1

    # A genuine content change still appends -- dedup must not become deafness.
    revised = RawRecord(
        record_id="emdat-1",
        payload={"total_affected": 250, "fetched_at": "2026-08-07T20:30:00Z"},
        iso3="PHL", ym="2024-03", hazard="FL",
    )
    assert store_raw_records(con, "emdat", [revised])["inserted"] == 1


def test_stored_payload_carries_no_volatile_keys(con):
    """Strip from the STORED payload too, not only from the hash input.

    Hashing a stripped copy while storing an unstripped one would make
    payload_json and content_hash disagree, and the surviving row's
    fetched_at would be a permanently frozen lie.
    """
    store_raw_records(con, "emdat", [RawRecord(
        record_id="emdat-9",
        payload={"total_affected": 7, "fetched_at": "2026-08-05T20:30:00Z"},
        iso3="PHL", ym="2024-03", hazard="FL",
    )])
    payload_json = con.execute(
        "SELECT payload_json FROM haz_raw_emdat WHERE record_id = 'emdat-9'"
    ).fetchone()[0]
    stored = json.loads(payload_json)
    assert not VOLATILE_PAYLOAD_KEYS & set(stored)
    assert stored["total_affected"] == 7

    # The retrieval time is not lost: it is the column, re-injected on read.
    loaded = load_raw_records(con, "emdat")
    assert loaded[0]["_retrieved_at"]


def test_no_connector_embeds_a_wall_clock_in_its_cached_payload():
    """Source-text guard against the copy-paste that produced all seven.

    A behavioural test cannot catch connector #8 naming its key
    ``retrieved_time``; this catches the literal that actually spread.
    """
    machine = Path(sources_mod.__file__).parent
    offenders = [
        path.name
        for path in sorted(machine.glob("*.py"))
        # sources.py is the storage layer, not a connector: its docstring
        # quotes the literal in order to explain why it is stripped.
        if path.name != "sources.py"
        and re.search(r"""["']fetched_at["']\s*:\s*_?utcnow_iso\(\)""", path.read_text())
    ]
    assert offenders == [], (
        "these connectors stamp a wall clock into the hashed payload, which "
        f"defeats the raw-cache dedup: {offenders}"
    )


@pytest.mark.parametrize("build", ["gdacs", "reliefweb_docs"])
def test_record_builders_hash_identically_across_two_clocks(build, rulebook, monkeypatch):
    """End-to-end on the real builders: same content, two clocks, one hash."""
    def _hash_with_clock(stamp: str) -> str:
        monkeypatch.setattr(sources_mod, "utcnow_iso", lambda: stamp)
        if build == "gdacs":
            record = gdacs_mod._event_record({
                "eventid": "1", "eventtype": "FL", "alertlevel": "Orange",
                "fromdate": "2024-03-01", "todate": "2024-03-05",
                "iso3": "PHL", "country": "Philippines",
            }, "flood")
            payload = record.payload
        else:
            payload = reliefweb_docs_mod._document(
                {"id": "42", "fields": {"title": "Floods", "body": "Body text",
                                        "url": "https://x", "date": {"created": "2024-03-02"}}},
                "PHL", "2024-03", "flood", rulebook,
            )
        return content_hash(
            json.dumps(stable_payload(payload), separators=(",", ":"), sort_keys=True)
        )

    assert _hash_with_clock("2026-08-05T20:30:00Z") == _hash_with_clock("2026-08-06T20:30:00Z")


def test_loaded_records_carry_cache_provenance(con):
    store_raw_records(
        con, "emdat",
        [RawRecord(record_id="e1", payload={"x": 1}, source_url="https://example.test/e1")],
    )
    loaded = load_raw_records(con, "emdat")[0]
    assert loaded["_record_id"] == "e1"
    assert loaded["_source_url"] == "https://example.test/e1"
    assert loaded["_retrieved_at"]


def test_fetch_window_frames_the_target_month(rulebook):
    start, end = fetch_window("2024-03", rulebook, "emdat")  # 3 back, 1 ahead
    assert (start.isoformat(), end.isoformat()) == ("2023-12-01", "2024-04-30")


def test_shift_month_crosses_year_boundaries():
    assert shift_month("2024-01", -1) == "2023-12"
    assert shift_month("2024-12", 1) == "2025-01"
    assert shift_month("2024-03", -15) == "2022-12"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1,234", 1234.0), (" 5 000 ", 5000.0), (0, 0.0), ("", None),
        (None, None), ("abc", None), (-5, None), (True, None),
    ],
)
def test_parse_number_is_strict_about_junk_and_negatives(raw, expected):
    """A negative people-affected count is malformed, not a small one."""
    assert parse_number(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2024-03-05", dt.date(2024, 3, 5)),
        ("2024-03-05T12:00:00Z", dt.date(2024, 3, 5)),
        ("", None), (None, None), ("not-a-date", None),
    ],
)
def test_parse_date_handles_the_shapes_sources_use(raw, expected):
    assert parse_date(raw) == expected


# ---------------------------------------------------------------------------
# EM-DAT
# ---------------------------------------------------------------------------


def _emdat_response(rows):
    return {"data": {"public_emdat": {"total_available": len(rows), "data": rows}}}


_EMDAT_ROW = {
    "disno": "2024-0123-PHL",
    "iso": "PHL",
    "classif_key": "nat-hyd-flo",
    "event_name": "Floods",
    "start_year": 2024, "start_month": 3, "start_day": 5,
    "end_year": 2024, "end_month": 3, "end_day": 9,
    "total_affected": 45000,
    "total_deaths": 12,
}


def test_emdat_stores_and_reads_back_a_record(con, rulebook, monkeypatch):
    monkeypatch.setenv("EMDAT_API_KEY", "test-key")
    outcome = emdat_mod.fetch_emdat(
        con, "2024-03", "FL", rulebook,
        post=lambda *a: _emdat_response([_EMDAT_ROW]),
    )
    assert outcome.ok is True and outcome.records == 1

    records = emdat_mod.records_for_country_month(con, "PHL", "2024-03", "FL")
    assert len(records) == 1
    assert records[0]["total_affected"] == 45000
    assert records[0]["months_overlapped"] == ["2024-03"]


def test_emdat_sends_the_key_from_the_environment_only(con, rulebook, monkeypatch):
    """Hard rule: credentials come from env, never from the rulebook."""
    monkeypatch.setenv("EMDAT_API_KEY", "secret-token")
    seen = {}

    def capture(url, payload, headers, timeout):
        seen.update(headers=headers, payload=payload)
        return _emdat_response([])

    emdat_mod.fetch_emdat(con, "2024-03", "FL", rulebook, post=capture)
    assert seen["headers"]["Authorization"] == "Bearer secret-token"
    assert "secret-token" not in str(rulebook.raw)


def test_emdat_without_a_key_reports_unavailable_not_empty(con, rulebook, monkeypatch):
    monkeypatch.delenv("EMDAT_API_KEY", raising=False)
    outcome = emdat_mod.fetch_emdat(con, "2024-03", "FL", rulebook)
    assert outcome.ok is False
    assert "EMDAT_API_KEY" in outcome.error


def test_emdat_api_failure_reports_unavailable_not_empty(con, rulebook, monkeypatch):
    monkeypatch.setenv("EMDAT_API_KEY", "test-key")

    def boom(*args):
        raise RuntimeError("503 Service Unavailable")

    outcome = emdat_mod.fetch_emdat(con, "2024-03", "FL", rulebook, post=boom)
    assert outcome.ok is False
    assert "503" in outcome.error
    assert outcome.records == 0


def test_emdat_serves_the_cache_when_the_live_call_fails(con, rulebook, monkeypatch):
    """A STALE rung and an UNREAD rung are different facts.

    api.emdat.be returned 500 on all six month-hazard passes of the 2026-08
    run, so every impact decision recorded EM-DAT as unavailable and lost its
    top rung — while haz_raw_emdat still held the previous pull, and
    records_for_country_month would happily have read it.
    """

    monkeypatch.setenv("EMDAT_API_KEY", "test-key")
    emdat_mod.fetch_emdat(
        con, "2024-03", "FL", rulebook, post=lambda *a: _emdat_response([_EMDAT_ROW])
    )

    def boom(*args):
        raise RuntimeError("500 Internal Server Error")

    outcome = emdat_mod.fetch_emdat(con, "2024-03", "FL", rulebook, post=boom)

    # The rung answered, so it is not in the run's unavailable_sources and
    # the ladder still has a top rung to walk.
    assert outcome.ok is True
    assert outcome.detail["served_from_cache"] is True
    assert "500" in outcome.detail["live_fetch_error"]
    assert outcome.records == 1
    # Nothing new was written: this is a read of what was already there.
    assert outcome.inserted == 0

    records = emdat_mod.records_for_country_month(con, "PHL", "2024-03", "FL")
    assert records and records[0]["total_affected"] == 45000


def test_emdat_serves_the_cache_when_the_key_is_missing(con, rulebook, monkeypatch):
    """An expired key does not delete what was already fetched."""

    monkeypatch.setenv("EMDAT_API_KEY", "test-key")
    emdat_mod.fetch_emdat(
        con, "2024-03", "FL", rulebook, post=lambda *a: _emdat_response([_EMDAT_ROW])
    )

    monkeypatch.delenv("EMDAT_API_KEY", raising=False)
    outcome = emdat_mod.fetch_emdat(con, "2024-03", "FL", rulebook)

    assert outcome.ok is True
    assert outcome.detail["served_from_cache"] is True
    assert "EMDAT_API_KEY" in outcome.detail["live_fetch_error"]


def test_emdat_with_an_empty_cache_is_still_unread(con, rulebook, monkeypatch):
    """The error in the other direction.

    With nothing to serve, claiming the rung answered would manufacture a
    missing rung out of an outage.
    """

    monkeypatch.setenv("EMDAT_API_KEY", "test-key")

    def boom(*args):
        raise RuntimeError("500 Internal Server Error")

    outcome = emdat_mod.fetch_emdat(con, "2024-03", "FL", rulebook, post=boom)
    assert outcome.ok is False
    assert "500" in outcome.error


def test_emdat_cache_fallback_only_counts_records_in_the_window(
    con, rulebook, monkeypatch
):
    """A cached record from another season does not make this month answered."""

    monkeypatch.setenv("EMDAT_API_KEY", "test-key")
    old_row = {
        **_EMDAT_ROW,
        "disno": "2019-0001-PHL",
        "start_year": 2019, "end_year": 2019,
    }
    emdat_mod.fetch_emdat(
        con, "2019-03", "FL", rulebook, post=lambda *a: _emdat_response([old_row])
    )

    def boom(*args):
        raise RuntimeError("500 Internal Server Error")

    outcome = emdat_mod.fetch_emdat(con, "2024-03", "FL", rulebook, post=boom)
    assert outcome.ok is False


def test_emdat_falls_back_to_summing_its_own_components():
    """EM-DAT defines total affected = injured + affected + homeless."""
    assert emdat_mod.affected_from_record({"total_affected": 500}) == 500
    assert emdat_mod.affected_from_record(
        {"no_injured": 10, "no_affected": 400, "no_homeless": 90}
    ) == 500
    # No stated figure is an ABSENT rung, never a zero.
    assert emdat_mod.affected_from_record({"total_deaths": 3}) is None


def test_emdat_record_without_a_month_is_dropped_not_parked_in_january(con, rulebook, monkeypatch):
    monkeypatch.setenv("EMDAT_API_KEY", "test-key")
    undated = {**_EMDAT_ROW, "start_month": None, "end_month": None}
    outcome = emdat_mod.fetch_emdat(
        con, "2024-03", "FL", rulebook, post=lambda *a: _emdat_response([undated])
    )
    assert outcome.ok is True
    assert outcome.records == 0


def test_emdat_records_outside_the_month_window_are_trimmed(con, rulebook, monkeypatch):
    """EM-DAT filters by YEAR, so the month trim has to happen here."""
    monkeypatch.setenv("EMDAT_API_KEY", "test-key")
    far = {**_EMDAT_ROW, "disno": "2024-9999-PHL",
           "start_month": 9, "end_month": 9}  # Sept, outside Dec..Apr
    emdat_mod.fetch_emdat(
        con, "2024-03", "FL", rulebook,
        post=lambda *a: _emdat_response([_EMDAT_ROW, far]),
    )
    stored = load_raw_records(con, "emdat")
    assert [r["disno"] for r in stored] == ["2024-0123-PHL"]


# ---------------------------------------------------------------------------
# IFRC GO
# ---------------------------------------------------------------------------


_GO_REPORT = {
    "id": 9001,
    "summary": "Philippines: Floods",
    "dtype": {"name": "Flood"},
    "report_date": "2024-03-11",
    "num_affected": 30000,
    "countries_details": [{"iso3": "PHL", "name": "Philippines"}],
}


def test_ifrc_go_stores_a_stated_figure(con, rulebook):
    outcome = go_mod.fetch_ifrc_go(
        con, "2024-03", "FL", rulebook,
        get=lambda *a: {"results": [_GO_REPORT], "next": None},
    )
    assert outcome.ok is True
    records = go_mod.records_for_country_month(con, "PHL", "2024-03", "FL")
    assert len(records) == 1
    assert records[0]["num_affected"] == 30000
    assert records[0]["affected_field"] == "num_affected"


def test_ifrc_go_field_priority_follows_the_rulebook(rulebook):
    """The record must say WHICH field stated the figure."""
    assert go_mod.affected_from_record(
        {"num_affected": 100, "gov_num_affected": 999}, rulebook
    ) == (100, "num_affected")
    assert go_mod.affected_from_record({"gov_num_affected": 999}, rulebook) == (
        999, "gov_num_affected",
    )
    assert go_mod.affected_from_record({"num_injured": 5}, rulebook) == (None, None)


def test_ifrc_go_skips_reports_about_another_hazard(con, rulebook):
    quake = {**_GO_REPORT, "id": 9002, "summary": "Earthquake", "dtype": {"name": "Earthquake"}}
    go_mod.fetch_ifrc_go(
        con, "2024-03", "FL", rulebook,
        get=lambda *a: {"results": [quake], "next": None},
    )
    assert go_mod.records_for_country_month(con, "PHL", "2024-03", "FL") == []


def test_ifrc_go_multi_country_report_is_not_split(con, rulebook):
    """GO states one figure for the operation; splitting would invent data."""
    multi = {
        **_GO_REPORT,
        "id": 9003,
        "countries_details": [
            {"iso3": "PHL", "name": "Philippines"},
            {"iso3": "VNM", "name": "Viet Nam"},
        ],
    }
    go_mod.fetch_ifrc_go(
        con, "2024-03", "FL", rulebook,
        get=lambda *a: {"results": [multi], "next": None},
    )
    for iso3 in ("PHL", "VNM"):
        records = go_mod.records_for_country_month(con, iso3, "2024-03", "FL")
        assert len(records) == 1
        assert records[0]["num_affected"] == 30000  # whole figure, not a share


def test_ifrc_go_api_failure_reports_unavailable(con, rulebook):
    def boom(*args):
        raise RuntimeError("connection reset")

    outcome = go_mod.fetch_ifrc_go(con, "2024-03", "FL", rulebook, get=boom)
    assert outcome.ok is False
    assert "connection reset" in outcome.error


# ---------------------------------------------------------------------------
# IDMC IDU
# ---------------------------------------------------------------------------


_IDU_ROW = {
    "id": 55001,
    "iso3": "PHL",
    "displacement_type": "Disaster",
    "type_name": "Flood",
    "event_name": "Floods March 2024",
    "displacement_start_date": "2024-03-06",
    "displacement_end_date": "2024-03-08",
    "figure": 8000,
}


def test_idu_stores_displacement_and_never_calls_it_affected(con, rulebook, monkeypatch):
    monkeypatch.setenv("IDMC_API_KEY", "test-key")
    outcome = idu_mod.fetch_idmc_idu(
        con, "2024-03", "FL", rulebook, get=lambda *a: [_IDU_ROW]
    )
    assert outcome.ok is True
    record = idu_mod.records_for_country_month(con, "PHL", "2024-03", "FL")[0]
    assert record["displaced"] == 8000
    # The payload must not carry an "affected" key at all.
    assert "affected" not in record and "num_affected" not in record


def test_idu_sends_the_key_as_client_id_from_the_environment(con, rulebook, monkeypatch):
    monkeypatch.setenv("IDMC_API_KEY", "idmc-secret")
    seen = {}

    def capture(url, params, timeout):
        seen.update(params=params)
        return []

    idu_mod.fetch_idmc_idu(con, "2024-03", "FL", rulebook, get=capture)
    assert seen["params"]["client_id"] == "idmc-secret"
    assert "idmc-secret" not in str(rulebook.raw)


def test_idu_without_a_key_reports_unavailable(con, rulebook, monkeypatch):
    monkeypatch.delenv("IDMC_API_KEY", raising=False)
    monkeypatch.delenv("IDMC_HELIX_CLIENT_ID", raising=False)
    outcome = idu_mod.fetch_idmc_idu(con, "2024-03", "FL", rulebook)
    assert outcome.ok is False
    # The error must name BOTH accepted vars — an operator reading it should
    # not have to grep the source to learn what would fix it.
    assert "IDMC_API_KEY" in outcome.error
    assert "IDMC_HELIX_CLIENT_ID" in outcome.error


def test_idu_falls_back_to_the_helix_client_id(con, rulebook, monkeypatch):
    """One IDMC client id serves both consumers.

    IDMC_HELIX_CLIENT_ID is the same credential the ingestion path already
    sends as client_id to the same external-api host — accepting it here
    means the operator keeps one secret instead of two that can drift.
    """

    monkeypatch.delenv("IDMC_API_KEY", raising=False)
    monkeypatch.setenv("IDMC_HELIX_CLIENT_ID", "helix-client-id")
    seen = {}

    def capture(url, params, timeout):
        seen.update(params=params)
        return []

    outcome = idu_mod.fetch_idmc_idu(con, "2024-03", "FL", rulebook, get=capture)
    assert outcome.ok is True
    assert seen["params"]["client_id"] == "helix-client-id"
    assert outcome.detail["credential_env"] == "IDMC_HELIX_CLIENT_ID"


def test_idu_prefers_its_own_key_over_the_fallback(con, rulebook, monkeypatch):
    """IDMC_API_KEY wins, so this rung can be pointed at a different id."""

    monkeypatch.setenv("IDMC_API_KEY", "rung-specific")
    monkeypatch.setenv("IDMC_HELIX_CLIENT_ID", "helix-client-id")
    seen = {}

    def capture(url, params, timeout):
        seen.update(params=params)
        return []

    outcome = idu_mod.fetch_idmc_idu(con, "2024-03", "FL", rulebook, get=capture)
    assert seen["params"]["client_id"] == "rung-specific"
    assert outcome.detail["credential_env"] == "IDMC_API_KEY"


def test_idu_never_reads_the_bearer_token_credential(con, rulebook, monkeypatch):
    """IDMC_API_TOKEN is a DIFFERENT credential and must never be used here.

    It is a bearer token for backend.idmcdb.org, and its mere presence is a
    feature flag for the ingestion path (run_connectors sets
    RESOLVER_SKIP_IDMC from it) — repurposing it would be two bugs at once.
    """

    monkeypatch.delenv("IDMC_API_KEY", raising=False)
    monkeypatch.delenv("IDMC_HELIX_CLIENT_ID", raising=False)
    monkeypatch.setenv("IDMC_API_TOKEN", "bearer-token-for-a-different-host")

    outcome = idu_mod.fetch_idmc_idu(con, "2024-03", "FL", rulebook)
    assert outcome.ok is False, "the bearer token must not satisfy this rung"
    assert "bearer-token-for-a-different-host" not in str(outcome.error)


def test_idu_skips_conflict_displacement(con, rulebook, monkeypatch):
    monkeypatch.setenv("IDMC_API_KEY", "test-key")
    conflict = {**_IDU_ROW, "id": 55002, "displacement_type": "Conflict"}
    outcome = idu_mod.fetch_idmc_idu(
        con, "2024-03", "FL", rulebook, get=lambda *a: [conflict]
    )
    assert outcome.records == 0
    assert outcome.detail["skipped_displacement_type"] == 1


def test_idu_hazard_mapping_prefers_the_longest_keyword(rulebook):
    """'flash flood' must not be claimed by a shorter entry elsewhere."""
    assert idu_mod.hazard_for_record({"type_name": "Flash flood"}, rulebook) == "FL"
    assert idu_mod.hazard_for_record({"type_name": "Tropical cyclone"}, rulebook) == "TC"
    assert idu_mod.hazard_for_record({"type_name": "Drought"}, rulebook) == "DR"
    assert idu_mod.hazard_for_record({"type_name": "Wildfire"}, rulebook) is None
    assert idu_mod.hazard_for_record({}, rulebook) is None


# ---------------------------------------------------------------------------
# GDACS adapter
# ---------------------------------------------------------------------------


def test_gdacs_adapter_reuses_the_core_connector():
    """The borrowed helpers must exist — a rename upstream fails loudly here.

    This module deliberately reuses resolver/connectors/gdacs.py instead of
    duplicating its client; that reuse needs a guard, or a refactor there
    breaks flood detection only at runtime, in production.
    """
    core = gdacs_mod._connector_api()
    for name in gdacs_mod._BORROWED:
        assert hasattr(core, name), name
    assert hasattr(core.GdacsConnector, "_search_events")
    assert hasattr(core.GdacsConnector, "_enrich_with_population")


def test_gdacs_adapter_raises_a_clear_error_if_the_reuse_breaks(monkeypatch):
    from resolver.connectors import gdacs as core

    monkeypatch.delattr(core, "_build_session")
    with pytest.raises(AttributeError, match="no longer provides"):
        gdacs_mod._connector_api()


def test_gdacs_event_record_labels_exposure_not_impact():
    """The payload key must make the exposure/impact distinction unmissable."""
    record = gdacs_mod._event_record(
        {
            "eventid": "1000001", "eventtype": "FL", "iso3": "PHL",
            "iso3_list": ["PHL"], "country": "Philippines",
            "alertlevel": "Orange", "alertscore": 1.5, "population": 500000,
            "fromdate": "2024-03-05", "todate": "2024-03-09",
            "pub_date": "2024-03-10",
        },
        "FL",
    )
    assert record.payload["exposed_population"] == 500000
    assert "affected" not in record.payload
    assert record.payload["months_overlapped"] == ["2024-03"]
    assert record.ym == "2024-03"  # anchored to the START month


def test_gdacs_fetch_failure_reports_unavailable(con, rulebook, monkeypatch):
    class BoomConnector:
        def _search_events(self, *a, **kw):
            raise RuntimeError("gdacs down")

    from resolver.connectors import gdacs as core

    monkeypatch.setattr(core, "GdacsConnector", BoomConnector)
    monkeypatch.setattr(core, "_build_session", lambda: object())
    monkeypatch.setattr(core, "_load_countries", lambda: ({}, {}))

    outcome = gdacs_mod.fetch_gdacs_events(con, "2024-03", "FL", rulebook)
    assert outcome.ok is False
    assert "gdacs down" in outcome.error


def test_gdacs_event_record_clamps_a_reversed_date_range(caplog):
    """The real 2021 shape: todate 2021-09-01 precedes fromdate 2021-09-28.

    One such event raised out of event_months and killed four backcast
    months (2021-08..2021-11) on eleven consecutive nightly runs. A
    reversed range is upstream data damage, not a reason to lose the
    month: clamp to the start day, keep the raw todate in provenance,
    and say so in the log.
    """
    with caplog.at_level(logging.WARNING):
        record = gdacs_mod._event_record(
            {
                "eventid": "1000496", "eventtype": "FL", "iso3": "PHL",
                "iso3_list": ["PHL"], "country": "Philippines",
                "alertlevel": "Green", "population": 1000,
                "fromdate": "2021-09-28", "todate": "2021-09-01",
            },
            "FL",
        )
    assert record is not None
    assert record.payload["start_date"] == "2021-09-28"
    assert record.payload["end_date"] == "2021-09-28"  # clamped
    assert record.payload["end_date_raw"] == "2021-09-01"  # provenance
    assert record.payload["months_overlapped"] == ["2021-09"]
    assert record.ym == "2021-09"
    assert any("1000496" in m for m in caplog.messages)


def test_gdacs_event_record_keeps_end_date_raw_none_when_not_clamped():
    record = gdacs_mod._event_record(
        {
            "eventid": "1000001", "eventtype": "FL", "iso3": "PHL",
            "iso3_list": ["PHL"], "fromdate": "2024-03-05",
            "todate": "2024-03-09",
        },
        "FL",
    )
    assert record.payload["end_date_raw"] is None


def test_gdacs_fetch_survives_one_malformed_event(con, rulebook, monkeypatch):
    """The fetch's no-raise contract applies per EVENT, not just per request.

    A single garbled upstream event must be skipped with a warning while
    the rest of the month's events store normally — GDACS answered.
    """
    good = {
        "eventid": "1", "eventtype": "FL", "iso3": "PHL",
        "iso3_list": ["PHL"], "fromdate": "2024-03-05",
        "todate": "2024-03-09",
    }
    bad = {**good, "eventid": "2", "iso3_list": 42}  # not iterable -> TypeError

    class TwoEventConnector:
        def _search_events(self, *a, **kw):
            return [good, bad]

        def _enrich_with_population(self, session, events, delay, name_to_iso3):
            return events

    from resolver.connectors import gdacs as core

    monkeypatch.setattr(core, "GdacsConnector", TwoEventConnector)
    monkeypatch.setattr(core, "_build_session", lambda: object())
    monkeypatch.setattr(core, "_load_countries", lambda: ({}, {}))

    outcome = gdacs_mod.fetch_gdacs_events(con, "2024-03", "FL", rulebook)
    assert outcome.ok is True
    assert outcome.records == 1
    assert outcome.detail["events_skipped_malformed"] == 1


# ---------------------------------------------------------------------------
# GDACS names no country for an event still over open ocean
# ---------------------------------------------------------------------------


class TestGdacsGeometryAttribution:
    """An event with no country is written and then read by no cell.

    GDACS leaves ``affectedcountries`` empty while a tropical cyclone is
    still offshore, and 16 TC events were dropped that way in the 2026-08 run
    (ids 1001273..1001315). The event's own position resolves it, against the
    same vendored boundaries cyclone detection already uses.
    """

    @staticmethod
    def _event(**overrides):
        event = {
            "eventid": "1001273",
            "eventtype": "TC",
            "iso3": "",
            "iso3_list": [],
            "country": "",
            "fromdate": dt.date(2024, 3, 5),
            "todate": dt.date(2024, 3, 9),
            "alertlevel": "Orange",
            "population": 250_000.0,
        }
        event.update(overrides)
        return event

    def test_a_point_inside_a_country_resolves_to_it(self, monkeypatch):
        from resolver.hazard_resolution import gdacs as gdacs_mod
        from resolver.hazard_resolution.geometry import load_country_geometries

        monkeypatch.setattr(
            "resolver.hazard_resolution.geometry.load_country_geometries",
            lambda *a, **k: load_country_geometries(SYNTHETIC_COUNTRIES_GEOJSON),
        )
        # Squareland spans lon 10..15, lat -2..2.
        record = gdacs_mod._event_record(self._event(lat=0.0, lon=12.0), "TC")

        assert record is not None
        assert record.payload["iso3_list"] == ["AAA"]
        assert record.iso3 == "AAA"
        # Provenance: a derived attribution is a weaker claim than a stated
        # one, and the row says which this was.
        assert record.payload["iso3_from_geometry"] == ["AAA"]

    def test_an_offshore_point_still_resolves_to_the_nearby_coast(self, monkeypatch):
        """A cyclone eye sits well off the coast it is about to hit."""

        from resolver.hazard_resolution import gdacs as gdacs_mod
        from resolver.hazard_resolution.geometry import load_country_geometries

        monkeypatch.setattr(
            "resolver.hazard_resolution.geometry.load_country_geometries",
            lambda *a, **k: load_country_geometries(SYNTHETIC_COUNTRIES_GEOJSON),
        )
        # ~2 degrees east of Squareland's edge, roughly 220 km at the equator.
        record = gdacs_mod._event_record(self._event(lat=0.0, lon=17.0), "TC")

        assert record.payload["iso3_list"] == ["AAA"]

    def test_a_point_in_the_middle_of_an_ocean_resolves_to_nothing(self, monkeypatch):
        """Failing to place an event leaves it exactly as it was.

        Attribution by proximity must not become attribution by guesswork.
        """

        from resolver.hazard_resolution import gdacs as gdacs_mod
        from resolver.hazard_resolution.geometry import load_country_geometries

        monkeypatch.setattr(
            "resolver.hazard_resolution.geometry.load_country_geometries",
            lambda *a, **k: load_country_geometries(SYNTHETIC_COUNTRIES_GEOJSON),
        )
        record = gdacs_mod._event_record(self._event(lat=-40.0, lon=-140.0), "TC")

        assert record.payload["iso3_list"] == []
        assert record.payload["iso3_from_geometry"] == []

    def test_a_stated_country_is_never_overridden_by_geometry(self, monkeypatch):
        """GDACS's own attribution outranks anything derived here."""

        from resolver.hazard_resolution import gdacs as gdacs_mod

        def explode(*_a, **_k):  # pragma: no cover - must not be reached
            raise AssertionError("geometry must not run when GDACS named a country")

        monkeypatch.setattr(gdacs_mod, "_iso3s_near_event", explode)
        record = gdacs_mod._event_record(
            self._event(iso3="PHL", iso3_list=["PHL"], lat=0.0, lon=12.0), "TC"
        )

        assert record.payload["iso3_list"] == ["PHL"]
        assert record.payload["iso3_from_geometry"] == []

    def test_an_event_with_no_position_is_not_a_crash(self):
        from resolver.hazard_resolution import gdacs as gdacs_mod

        record = gdacs_mod._event_record(self._event(), "TC")
        assert record.payload["iso3_list"] == []
