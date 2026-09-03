# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The instrumentation the debug bundle reads: redaction, HTTP, ledgers.

The bundle can only report what the code records, so these tests guard the
recording rather than the reporting: that a URL keeps its route and loses its
key, that the HTTP wrapper never changes what a caller gets back, and that
every stream stays silent unless this run is collecting evidence.
"""

from __future__ import annotations

import json

import pytest
import requests

from resolver.diagnostics import http_recorder, redaction, run_log
from resolver.hazard_resolution import cell_ledger


@pytest.fixture()
def recording(tmp_path, monkeypatch):
    monkeypatch.setenv(run_log.ENV_DIR, str(tmp_path / "runlog"))
    run_log.reset_for_tests()
    yield tmp_path / "runlog"
    run_log.reset_for_tests()


# --------------------------------------------------------------------------
# Redaction
# --------------------------------------------------------------------------


def test_a_url_keeps_its_route_and_loses_its_key():
    """"Which URL was called" is the question; the answer is worthless with
    the route removed and dangerous with the key kept."""

    url = "https://api.acleddata.com/cast/read?key=SUPERSECRET-0123&limit=5000&year=2026"
    out = redaction.redact_url(url, ["SUPERSECRET-0123"])
    assert "api.acleddata.com/cast/read" in out
    assert "limit=5000" in out and "year=2026" in out
    assert "SUPERSECRET-0123" not in out
    assert redaction.fingerprint("SUPERSECRET-0123") in out


def test_credentials_are_fingerprinted_never_blanked():
    """A constant mask cannot answer 'is this the key the last good run used'."""

    one = redaction.fingerprint("key-one-value-here")
    two = redaction.fingerprint("key-two-value-here")
    assert one != two
    assert one == redaction.fingerprint("key-one-value-here")
    assert one.startswith("<redacted:sha256:")


def test_user_info_in_a_url_is_removed_whole():
    out = redaction.redact_url("https://user:pw@example.org/path?a=1", [])
    assert "user:pw" not in out
    assert "example.org/path" in out and "a=1" in out


def test_a_config_name_that_merely_matches_the_pattern_is_not_a_credential():
    """RELIEFWEB_APPNAME matches on 'NAME' but is configuration, and a reader
    who cannot see which appname was sent cannot diagnose a refusal."""

    assert not redaction.is_secret_name("RELIEFWEB_APPNAME")
    assert redaction.is_secret_name("ACLED_ACCESS_KEY")
    assert redaction.is_secret_name("IDMC_API_TOKEN")
    assert redaction.is_secret_name("acaps_password")


def test_short_values_are_redacted_by_name_only():
    """Substring-replacing a four-character token would mangle half the bundle."""

    values = redaction.secret_values({"X_KEY": "abc", "Y_TOKEN": "long-enough-value"})
    assert "abc" not in values
    assert "long-enough-value" in values


def test_nested_structures_are_redacted_by_key_and_by_value():
    payload = {"auth": {"api_key": "hunter2-and-then-some"},
               "note": "sent hunter2-and-then-some upstream", "n": 3}
    out = redaction.redact_obj(payload, ["hunter2-and-then-some"])
    assert out["auth"]["api_key"].startswith("<redacted:")
    assert "hunter2-and-then-some" not in out["note"]
    assert out["n"] == 3


def test_a_leak_report_names_the_fingerprint_not_the_value():
    hits = redaction.find_secrets("token=abcdefghijkl here", ["abcdefghijkl"])
    assert hits == [redaction.fingerprint("abcdefghijkl")]
    assert "abcdefghijkl" not in hits[0]


# --------------------------------------------------------------------------
# The run-scoped streams
# --------------------------------------------------------------------------


def test_recording_is_off_unless_this_run_is_collecting_evidence(monkeypatch):
    monkeypatch.delenv(run_log.ENV_DIR, raising=False)
    run_log.reset_for_tests()
    assert not run_log.enabled()
    assert run_log.stream_path(run_log.STREAM_HTTP) is None
    run_log.record(run_log.STREAM_HTTP, {"x": 1})  # a no-op, not an error


def test_a_write_failure_disables_the_stream_and_never_raises(tmp_path, monkeypatch):
    """A machine that fails because its ledger could not be written is worse."""

    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv(run_log.ENV_DIR, str(blocker))
    run_log.reset_for_tests()
    run_log.record(run_log.STREAM_CELLS, {"iso3": "PHL"})  # must not raise
    assert not run_log.enabled(), "the stream disables itself after a failure"
    run_log.reset_for_tests()


def test_a_truncated_final_line_costs_one_record(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}\n{"a": 3', encoding="utf-8")
    assert [r["a"] for r in run_log.read_stream(path)] == [1, 2]


# --------------------------------------------------------------------------
# The HTTP recorder
# --------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.headers = {"Content-Type": "application/json"}
        self.content = json.dumps(payload).encode()
        self.history: list = []

    def json(self):
        return self._payload


def test_the_recorder_captures_the_envelope_the_connector_discards(recording, monkeypatch):
    """ACLED CAST's envelope says whether a stalled vintage is a quota or an
    upstream stop. The connector reads `data` and drops the rest."""

    payload = {
        "status": 200, "success": True, "count": 9879,
        "last_update": "2025-12-01", "messages": "",
        "data_query_restrictions": "free tier: 1 year of history",
        "data": [{"country": "Somalia", "year": 2025, "month": 12,
                  "timestamp": "2025-12-01T00:00:00"}],
    }
    monkeypatch.setattr(
        requests.Session, "request",
        lambda self, method, url, *a, **k: _FakeResponse(payload),
    )
    assert http_recorder.install()
    try:
        requests.Session().get("https://api.acleddata.com/cast/read?key=SECRET-VALUE-1234")
    finally:
        http_recorder.uninstall()

    envelopes = list(run_log.read_stream(recording / f"{run_log.STREAM_ENVELOPE}.jsonl"))
    assert len(envelopes) == 1
    top = envelopes[0]["envelope"]["top_level"]
    for field in ("status", "success", "count", "last_update",
                  "messages", "data_query_restrictions"):
        assert field in top, f"{field} is what tells a quota from an outage"
    # The arithmetic behind a vintage claim, not just the claim.
    assert envelopes[0]["envelope"]["max_by_date_column"]["timestamp"] == "2025-12-01T00:00:00"
    assert envelopes[0]["envelope"]["columns"] == ["country", "month", "timestamp", "year"]


def test_the_recorder_redacts_the_url_at_the_point_of_capture(recording, monkeypatch):
    monkeypatch.setenv("ACLED_ACCESS_KEY", "SECRET-VALUE-1234")
    monkeypatch.setattr(
        requests.Session, "request",
        lambda self, method, url, *a, **k: _FakeResponse({"data": []}),
    )
    assert http_recorder.install()
    try:
        requests.Session().get("https://api.acleddata.com/cast/read?key=SECRET-VALUE-1234")
    finally:
        http_recorder.uninstall()

    text = (recording / f"{run_log.STREAM_HTTP}.jsonl").read_text(encoding="utf-8")
    assert "SECRET-VALUE-1234" not in text
    assert "api.acleddata.com/cast/read" in text


def test_the_recorder_never_changes_what_the_caller_gets_back(recording, monkeypatch):
    """A recorder that can break an ingest is worse than no recorder."""

    sentinel = _FakeResponse({"data": [1]})
    monkeypatch.setattr(
        requests.Session, "request", lambda self, method, url, *a, **k: sentinel
    )
    http_recorder.install()
    try:
        assert requests.Session().get("https://example.org/x") is sentinel
    finally:
        http_recorder.uninstall()


def test_a_transport_failure_is_recorded_and_re_raised(recording, monkeypatch):
    def boom(self, method, url, *a, **k):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests.Session, "request", boom)
    http_recorder.install()
    try:
        with pytest.raises(requests.ConnectionError):
            requests.Session().get("https://api.emdat.be/v1")
    finally:
        http_recorder.uninstall()

    records = list(run_log.read_stream(recording / f"{run_log.STREAM_HTTP}.jsonl"))
    assert records[0]["status"] is None
    assert "connection refused" in records[0]["error"]


def test_a_broken_recorder_does_not_break_the_request(recording, monkeypatch):
    monkeypatch.setattr(
        requests.Session, "request",
        lambda self, method, url, *a, **k: _FakeResponse({"data": []}),
    )
    monkeypatch.setattr(
        http_recorder, "_record",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("recorder is broken")),
    )
    http_recorder.install()
    try:
        assert requests.Session().get("https://example.org/x").status_code == 200
    finally:
        http_recorder.uninstall()


def test_the_recorder_is_off_without_the_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv(run_log.ENV_DIR, raising=False)
    run_log.reset_for_tests()
    assert not http_recorder.maybe_install_from_env()


# --------------------------------------------------------------------------
# The cell + figure ledgers
# --------------------------------------------------------------------------


def test_a_cell_records_the_reason_it_produced_no_row(recording):
    cell_ledger.record_cell(
        stage=cell_ledger.STAGE_LADDER, iso3="vnm", hazard="TC", ym="2026-08",
        triggered=True, write_outcome="pending_skip",
        reason_code=cell_ledger.REASON_PENDING, rungs_unavailable=["emdat"],
    )
    records = list(run_log.read_stream(recording / f"{run_log.STREAM_CELLS}.jsonl"))
    assert records[0]["iso3"] == "VNM"
    assert records[0]["reason_code"] == "pending_before_freeze"
    assert records[0]["rungs_unavailable"] == ["emdat"]


def test_a_rejected_figure_records_the_ceiling_and_where_it_came_from(recording):
    cell_ledger.record_figure(
        iso3="PHL", hazard="TC", ym="2026-08", outcome="rejected",
        doc_id="rw-1", value=120000.0, reason="exceeds_gdacs_exposure_ceiling",
        ceiling=2.0, ceiling_multiplier=3.0, ceiling_source="gdacs",
        ceiling_source_ref="TC-1001273", ceiling_field="gdacs.population",
    )
    record = next(iter(run_log.read_stream(recording / f"{run_log.STREAM_FIGURES}.jsonl")))
    # A ceiling of 2 against a reported 120,000 is an enrichment failure, not a
    # mis-transcription — and only these three fields say which.
    assert record["ceiling"] == 2.0
    assert record["ceiling_source_ref"] == "TC-1001273"
    assert record["ceiling_field"] == "gdacs.population"


def test_the_ledgers_are_silent_when_recording_is_off(monkeypatch, tmp_path):
    monkeypatch.delenv(run_log.ENV_DIR, raising=False)
    run_log.reset_for_tests()
    assert not cell_ledger.enabled()
    cell_ledger.record_cell(stage="ladder", iso3="PHL", hazard="TC", ym="2026-08")
    cell_ledger.record_figure(iso3="PHL", hazard="TC", ym="2026-08", outcome="accepted")
    assert not list(tmp_path.glob("*.jsonl"))
