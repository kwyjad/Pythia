from __future__ import annotations

import json
from typing import Any, Iterable
from urllib.parse import urlencode

import pandas as pd
import pytest

from resolver.ingestion import acled_client


class FakeResponse:
    def __init__(self, url: str, params: dict[str, Any], data: Iterable[dict[str, Any]] | None, *, status: int = 200) -> None:
        self.status_code = status
        self.headers: dict[str, str] = {}
        self._payload = {"status": status, "data": list(data) if data is not None else []}
        self._text = json.dumps(self._payload)
        self._url = f"{url}?{urlencode(params)}"

    @property
    def url(self) -> str:
        return self._url

    @property
    def text(self) -> str:
        return self._text

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self._payloads = payloads
        self.calls: list[dict[str, Any]] = []
        self._index = 0

    def get(self, url: str, params: dict[str, Any], headers: dict[str, str], timeout: int) -> FakeResponse:
        self.calls.append({"url": url, "params": dict(params), "headers": dict(headers)})
        if self._index < len(self._payloads):
            payload = self._payloads[self._index]
        else:
            payload = self._payloads[-1]
        self._index += 1
        data = payload.get("data")
        status = int(payload.get("status", 200))
        return FakeResponse(url, params, data, status=status)


def _make_client(monkeypatch: pytest.MonkeyPatch, payloads: list[dict[str, Any]]) -> tuple[acled_client.ACLEDClient, FakeSession]:
    monkeypatch.setattr(acled_client.acled_auth, "get_access_token", lambda: "token")
    session = FakeSession(payloads)
    client = acled_client.ACLEDClient(session=session)
    return client, session


def test_fetch_events_uses_pipe_delimiter(monkeypatch: pytest.MonkeyPatch) -> None:
    client, session = _make_client(
        monkeypatch,
        [
            {
                "data": [
                    {
                        "event_date": "2025-11-01",
                        "iso3": "AFG",
                        "country": "Afghanistan",
                        "fatalities": 1,
                    }
                ]
            }
        ],
    )

    frame = client.fetch_events("2025-11-01", "2025-11-30")

    assert not frame.empty
    assert session.calls
    params = session.calls[0]["params"]
    assert params["fields"] == "event_date|iso3|country|fatalities"
    assert "|" in params["fields"] and "," not in params["fields"]


def test_fetch_events_retries_without_fields_on_empty_page(monkeypatch: pytest.MonkeyPatch) -> None:
    client, session = _make_client(
        monkeypatch,
        [
            {"data": []},
            {
                "data": [
                    {
                        "event_date": "2025-11-15",
                        "iso3": "AFG",
                        "country": "Afghanistan",
                        "fatalities": 2,
                    }
                ]
            },
        ],
    )

    frame = client.fetch_events("2025-11-01", "2025-11-30")

    assert len(session.calls) == 2
    assert session.calls[0]["params"].get("fields")
    assert "fields" not in session.calls[1]["params"]
    assert frame.shape[0] == 1


def test_monthly_fatalities_sums_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(acled_client.acled_auth, "get_access_token", lambda: "token")

    def fake_fetch_events(
        self: acled_client.ACLEDClient,
        start_date: str,
        end_date: str,
        *,
        countries: Iterable[str] | None = None,
        fields: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"event_date": "2025-01-05", "iso3": "AFG", "country": "Afghanistan", "fatalities": 1},
                {"event_date": "2025-01-25", "iso3": "AFG", "country": "Afghanistan", "fatalities": 2},
            ]
        )

    monkeypatch.setattr(acled_client.ACLEDClient, "fetch_events", fake_fetch_events)
    client = acled_client.ACLEDClient(session=FakeSession([{"data": []}]))

    result = client.monthly_fatalities("2025-01-01", "2025-01-31")

    assert list(result.columns) == ["iso3", "month", "fatalities", "source", "updated_at"]
    assert result.loc[0, "iso3"] == "AFG"
    assert int(result.loc[0, "fatalities"]) == 3
    assert result.loc[0, "source"] == "ACLED"
    assert pd.to_datetime(result.loc[0, "month"]).strftime("%Y-%m-%d") == "2025-01-01"
