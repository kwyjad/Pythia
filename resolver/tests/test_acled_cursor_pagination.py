# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""ACLED cursor pagination (ACLED notice, Sept 2026; ``page`` deprecated 27 Oct 2026).

A walk starts at ``cursor=0`` and follows ``next_cursor`` until it is null.
Every ACLED pagination loop in this repository goes through
``acled_auth.advance_cursor``; these tests pin the three loops to it and pin
the helper's refusal to read a truncated walk as complete.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from resolver.connectors import acled_cast
from resolver.ingestion import acled_auth, acled_client


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------


class _Resp:
    def __init__(self, payload: Any = None, headers: Dict[str, str] | None = None,
                 status: int = 200, url: str = "https://acleddata.com/api/acled/read") -> None:
        self._payload = payload
        self.headers = headers or {"Content-Type": "application/json"}
        self.status_code = status
        self.url = url
        self.text = json.dumps(payload) if payload is not None else ""

    def json(self) -> Any:
        return self._payload


def test_a_stated_null_cursor_ends_the_walk_even_on_a_full_page() -> None:
    assert acled_auth.advance_cursor(0, {"next_cursor": None}, page_rows=5000, limit=5000) is None


def test_a_stated_cursor_is_followed() -> None:
    assert acled_auth.advance_cursor(0, {"next_cursor": 48213}, page_rows=5000, limit=5000) == 48213


def test_the_csv_header_form_is_read() -> None:
    resp = _Resp(headers={"X-Next-Cursor": "96543"})
    assert acled_auth.advance_cursor(48213, None, page_rows=5000, limit=5000, resp=resp) == "96543"
    resp_null = _Resp(headers={"X-Next-Cursor": "null"})
    assert acled_auth.advance_cursor(48213, None, page_rows=10, limit=5000, resp=resp_null) is None


def test_a_short_page_with_no_cursor_is_the_last_page() -> None:
    assert acled_auth.advance_cursor(0, {"data": []}, page_rows=12, limit=5000) is None


def test_a_full_page_with_no_cursor_is_refused_not_read_as_complete() -> None:
    with pytest.raises(acled_auth.AcledPaginationError, match="next_cursor"):
        acled_auth.advance_cursor(0, {"data": []}, page_rows=5000, limit=5000)


def test_a_cursor_that_does_not_move_is_refused() -> None:
    with pytest.raises(acled_auth.AcledPaginationError, match="loop"):
        acled_auth.advance_cursor(48213, {"next_cursor": "48213"}, page_rows=5000, limit=5000)


# ---------------------------------------------------------------------------
# The connector path: acled_client.fetch_events
# ---------------------------------------------------------------------------


def _patch_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    diag = tmp_path / "diagnostics" / "ingestion"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(acled_client, "ACLED_DIAGNOSTICS", diag / "acled")
    monkeypatch.setattr(acled_client, "ACLED_RUN_PATH", diag / "acled_client" / "acled_client_run.json")
    monkeypatch.setattr(acled_client, "ACLED_HTTP_DIAG_PATH", diag / "acled" / "http_diag.json")
    monkeypatch.setattr(acled_client, "OUT_PATH", tmp_path / "acled.csv")
    monkeypatch.setattr(acled_client.acled_auth, "get_access_token", lambda: "TOKEN")


def _event(i: int) -> Dict[str, Any]:
    return {"event_date": "2026-08-01", "iso3": "KEN", "country": "Kenya",
            "fatalities": "1", "event_type": "Battles", "notes": f"e{i}"}


class _CursorSession:
    """Serves pages keyed by the cursor it is asked with, as ACLED does."""

    def __init__(self, pages: Dict[str, Dict[str, Any]]) -> None:
        self.pages = pages
        self.params: List[Dict[str, Any]] = []

    def get(self, url: str, params: Dict[str, Any], headers: Dict[str, str], timeout: int) -> _Resp:
        self.params.append(dict(params))
        return _Resp({"status": 200, **self.pages[str(params["cursor"])]}, url=url)


def test_fetch_events_follows_next_cursor_and_never_sends_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_paths(tmp_path, monkeypatch)
    session = _CursorSession({
        "0": {"data": [_event(1), _event(2)], "next_cursor": 48213},
        "48213": {"data": [_event(3), _event(4)], "next_cursor": 96543},
        "96543": {"data": [_event(5)], "next_cursor": None},
    })
    monkeypatch.setattr(acled_client.requests, "Session", lambda: session)

    records, _url, _meta = acled_client.fetch_events({"limit": 2, "query": {"page": 7}})

    assert len(records) == 5
    assert [p["cursor"] for p in session.params] == [0, 48213, 96543]
    assert all("page" not in p for p in session.params), "a config that names page must not send it"


def test_fetch_events_refuses_a_full_page_with_no_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_paths(tmp_path, monkeypatch)
    session = _CursorSession({"0": {"data": [_event(1), _event(2)]}})
    monkeypatch.setattr(acled_client.requests, "Session", lambda: session)

    with pytest.raises(acled_auth.AcledPaginationError):
        acled_client.fetch_events({"limit": 2})


# ---------------------------------------------------------------------------
# The monthly-fatalities path: ACLEDClient.fetch_events
# ---------------------------------------------------------------------------


def test_monthly_client_walks_by_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(acled_client.acled_auth, "get_access_token", lambda: "TOKEN")
    pages = {
        "0": {"data": [{"event_date": "2026-08-01", "iso3": "KEN", "country": "Kenya", "fatalities": "1"}] * 2,
              "next_cursor": "abc"},
        "abc": {"data": [{"event_date": "2026-08-02", "iso3": "KEN", "country": "Kenya", "fatalities": "2"}],
                "next_cursor": None},
    }
    seen: List[Dict[str, Any]] = []

    def _fake_fetch(self: acled_client.ACLEDClient, params: Dict[str, Any]) -> Dict[str, Any]:
        seen.append(dict(params))
        return pages[str(params["cursor"])]

    monkeypatch.setattr(acled_client.ACLEDClient, "_fetch_page", _fake_fetch)
    client = acled_client.ACLEDClient()
    client.page_size = 2
    frame = client.fetch_events("2026-08-01", "2026-08-31")

    assert len(frame) == 3
    assert [p["cursor"] for p in seen] == [0, "abc"]
    assert all("page" not in p for p in seen)


# ---------------------------------------------------------------------------
# The CAST path: AcledCastConnector._fetch_all_records
# ---------------------------------------------------------------------------


def test_cast_walks_by_cursor() -> None:
    calls: List[Dict[str, Any]] = []
    pages = {
        "0": {"data": [{"country": "Kenya"}] * acled_cast._PAGE_SIZE, "next_cursor": 11},
        "11": {"data": [{"country": "Kenya"}], "next_cursor": None},
    }

    def _get(url: str, params: Dict[str, Any], headers: Dict[str, str], timeout: int) -> _Resp:
        calls.append(dict(params))
        return _Resp(pages[str(params["cursor"])], url=url)

    with patch("resolver.ingestion.acled_auth.get_auth_header", return_value={}), \
            patch.object(acled_cast.requests, "get", side_effect=_get), \
            patch.object(acled_cast.time, "sleep"):
        records = acled_cast.AcledCastConnector()._fetch_all_records(year=2026)

    assert len(records) == acled_cast._PAGE_SIZE + 1
    assert [c["cursor"] for c in calls] == [0, 11]
    assert all("page" not in c for c in calls)
    assert all(c["year"] == 2026 for c in calls)
