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

import duckdb
import pytest

from resolver.hazard_resolution import emdat as emdat_mod
from resolver.hazard_resolution import gdacs as gdacs_mod
from resolver.hazard_resolution import idmc_idu as idu_mod
from resolver.hazard_resolution import ifrc_go as go_mod
from resolver.hazard_resolution.schema import ensure_haz_schema
from resolver.hazard_resolution.sources import (
    RawRecord,
    fetch_window,
    load_raw_records,
    parse_date,
    parse_number,
    shift_month,
    store_raw_records,
)
from resolver.tests.hazard_resolution_utils import make_rulebook


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
