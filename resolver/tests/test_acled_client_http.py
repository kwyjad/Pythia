# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List
from urllib.parse import urlencode

import pytest

from resolver.ingestion import acled_client


class StubResponse:
    def __init__(self, url: str, params: dict[str, Any], data: Iterable[dict[str, Any]] | None, *, status: int = 200) -> None:
        self.status_code = status
        self.headers: dict[str, str] = {}
        self._url = f"{url}?{urlencode(params)}"
        if status == 200 and data is not None:
            payload = {"status": 200, "data": list(data)}
        else:
            payload = {"status": status}
        self._payload = payload
        self.text = json.dumps(payload)

    @property
    def url(self) -> str:
        return self._url

    def json(self) -> dict[str, Any]:
        return self._payload


class StubSession:
    def __init__(self, responses: List[Any]) -> None:
        self._responses = responses
        self.calls: List[dict[str, str]] = []

    def get(self, url: str, params: dict[str, Any], headers: dict[str, str], timeout: int) -> StubResponse:
        self.calls.append(headers)
        index = len(self.calls) - 1
        if index < len(self._responses):
            payload = self._responses[index]
        else:
            payload = []
        if isinstance(payload, tuple):
            data, status = payload
        else:
            data, status = payload, 200
        return StubResponse(url, params, data, status=status)


def _patch_acled_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    diag_root = tmp_path / "diagnostics" / "ingestion"
    monkeypatch.setattr(acled_client, "ACLED_DIAGNOSTICS", diag_root / "acled")
    monkeypatch.setattr(acled_client, "ACLED_RUN_PATH", diag_root / "acled_client" / "acled_client_run.json")
    monkeypatch.setattr(acled_client, "ACLED_HTTP_DIAG_PATH", diag_root / "acled" / "http_diag.json")
    monkeypatch.setattr(acled_client, "OUT_PATH", tmp_path / "acled.csv")


def test_fetch_events_creates_diag_and_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _patch_acled_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(acled_client.acled_auth, "get_access_token", lambda: "ACCESS-TOKEN")

    payloads = [
        [
            {
                "event_date": "2024-01-01",
                "iso3": "KEN",
                "country": "Kenya",
                "fatalities": "1",
                "event_type": "Battles",
                "notes": "",
            },
            {
                "event_date": "2024-01-02",
                "iso3": "ETH",
                "country": "Ethiopia",
                "fatalities": "2",
                "event_type": "Protests",
                "notes": "",
            },
        ],
        [],
    ]
    sessions: List[StubSession] = []

    def _session_factory() -> StubSession:
        session = StubSession(list(payloads))
        sessions.append(session)
        return session

    monkeypatch.setattr(acled_client.requests, "Session", _session_factory)

    records, source_url, meta = acled_client.fetch_events({})

    assert len(records) == 2
    assert "acled" in source_url
    assert meta["http_status"] == 200
    assert sessions[0].calls
    for headers in sessions[0].calls:
        assert headers["Authorization"] == "Bearer ACCESS-TOKEN"

    config = acled_client.load_config()
    countries, shocks = acled_client.load_registries()
    publication_date = "2024-02-01"
    ingested_at = "2024-02-02T00:00:00Z"
    rows = acled_client._build_rows(records, config, countries, shocks, source_url, publication_date, ingested_at)
    assert rows, "expected normalized rows"

    acled_client._write_rows(rows, acled_client.OUT_PATH)
    csv_path = Path(acled_client.OUT_PATH)
    assert csv_path.is_file()
    contents = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) > 1, "staging CSV should have data rows"

    http_diag = json.loads((tmp_path / "diagnostics" / "ingestion" / "acled" / "http_diag.json").read_text(encoding="utf-8"))
    assert http_diag["status"] == 200
    assert "https://" in http_diag["url"]


class _TimeoutThenSession:
    """Raise a Timeout on the first N ``get`` calls, then serve real responses."""

    def __init__(self, responses: List[Any], *, fail_times: int) -> None:
        self._delegate = StubSession(responses)
        self._remaining_failures = fail_times
        self.timeout_raises = 0

    def get(self, url: str, params: dict[str, Any], headers: dict[str, str], timeout: int):
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            self.timeout_raises += 1
            raise acled_client.requests.exceptions.Timeout("simulated read timeout")
        return self._delegate.get(url, params=params, headers=headers, timeout=timeout)


def test_fetch_events_retries_on_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient page timeout is retried, not fatal (regression for the 900s SIGKILL)."""
    monkeypatch.chdir(tmp_path)
    _patch_acled_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(acled_client.acled_auth, "get_access_token", lambda: "ACCESS-TOKEN")
    monkeypatch.setattr(acled_client.time, "sleep", lambda *_a, **_k: None)

    payloads = [
        [
            {
                "event_date": "2024-01-01",
                "iso3": "KEN",
                "country": "Kenya",
                "fatalities": "1",
                "event_type": "Battles",
                "notes": "",
            }
        ],
        [],
    ]
    sessions: List[_TimeoutThenSession] = []

    def _session_factory() -> _TimeoutThenSession:
        session = _TimeoutThenSession(list(payloads), fail_times=1)
        sessions.append(session)
        return session

    monkeypatch.setattr(acled_client.requests, "Session", _session_factory)

    records, _source_url, meta = acled_client.fetch_events({})

    assert sessions[0].timeout_raises == 1, "expected exactly one timeout retry"
    assert len(records) == 1, "records should be fetched after the retry succeeds"
    assert meta["http_status"] == 200


def test_fetch_events_reraises_timeout_after_max_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Persistent timeouts still surface once retries are exhausted."""
    monkeypatch.chdir(tmp_path)
    _patch_acled_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(acled_client.acled_auth, "get_access_token", lambda: "ACCESS-TOKEN")
    monkeypatch.setattr(acled_client.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setenv("ACLED_MAX_RETRIES", "3")

    def _session_factory() -> _TimeoutThenSession:
        return _TimeoutThenSession([], fail_times=99)

    monkeypatch.setattr(acled_client.requests, "Session", _session_factory)

    with pytest.raises(acled_client.requests.exceptions.Timeout):
        acled_client.fetch_events({})


def test_fetch_events_forwards_fields_param(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A `query.fields` config value is sent to ACLED (limits payload size)."""
    monkeypatch.chdir(tmp_path)
    _patch_acled_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(acled_client.acled_auth, "get_access_token", lambda: "ACCESS-TOKEN")

    captured: List[dict[str, Any]] = []

    class _CapturingSession:
        def get(self, url: str, params: dict[str, Any], headers: dict[str, str], timeout: int) -> StubResponse:
            captured.append(dict(params))
            return StubResponse(url, params, [])  # empty → single page

    monkeypatch.setattr(acled_client.requests, "Session", lambda: _CapturingSession())

    acled_client.fetch_events({"query": {"fields": "event_date|iso3|fatalities"}})

    assert captured, "expected at least one request"
    assert captured[0].get("fields") == "event_date|iso3|fatalities"


def test_fetch_events_respects_runtime_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The wall-clock deadline stops pagination gracefully (regression for SIGKILL)."""
    monkeypatch.chdir(tmp_path)
    _patch_acled_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(acled_client.acled_auth, "get_access_token", lambda: "ACCESS-TOKEN")
    monkeypatch.setenv("ACLED_MAX_RUNTIME_SEC", "5")

    # Advancing monotonic clock: first call sets the deadline (t=0 → deadline=5),
    # the next loop-top check is already past it (t=1000) → break before any fetch.
    ticks = iter([0.0, 1000.0])
    monkeypatch.setattr(acled_client.time, "monotonic", lambda: next(ticks, 1000.0))

    class _NeverCalledSession:
        def get(self, *_a: Any, **_k: Any) -> StubResponse:  # pragma: no cover - must not run
            raise AssertionError("deadline should have stopped pagination before any fetch")

    monkeypatch.setattr(acled_client.requests, "Session", lambda: _NeverCalledSession())

    records, _source_url, meta = acled_client.fetch_events({})

    assert records == []
    assert meta.get("truncated_by_deadline") is True


def test_fetch_events_error_raises_and_records_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _patch_acled_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(acled_client.acled_auth, "get_access_token", lambda: "ACCESS-TOKEN")

    payloads = [([], 401)]

    def _session_factory() -> StubSession:
        return StubSession(list(payloads))

    monkeypatch.setattr(acled_client.requests, "Session", _session_factory)

    with pytest.raises(RuntimeError, match="HTTP 401"):
        acled_client.fetch_events({})

    http_diag_path = tmp_path / "diagnostics" / "ingestion" / "acled" / "http_diag.json"
    assert http_diag_path.is_file(), "expected HTTP diagnostic file"
    http_diag = json.loads(http_diag_path.read_text(encoding="utf-8"))
    assert http_diag["status"] == 401
