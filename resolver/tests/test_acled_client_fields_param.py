"""Tests for ACLEDClient field parameter behavior."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import pytest

from resolver.ingestion import acled_client


@pytest.fixture(autouse=True)
def _mock_access_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(acled_client.acled_auth, "get_access_token", lambda: "token")


def _fake_payload() -> Dict[str, Any]:
    return {
        "data": [
            {
                "event_date": "2025-01-05",
                "iso3": "AFG",
                "country": "Afghanistan",
                "fatalities": 1,
            }
        ]
    }


def _patch_fetch_page(monkeypatch: pytest.MonkeyPatch, params_store: List[Dict[str, Any]]) -> None:
    def _fake_fetch(self: acled_client.ACLEDClient, params: Dict[str, Any]) -> Dict[str, Any]:
        params_store.append(dict(params))
        return _fake_payload()

    monkeypatch.setattr(acled_client.ACLEDClient, "_fetch_page", _fake_fetch)


def test_fields_omitted_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    params_store: List[Dict[str, Any]] = []
    _patch_fetch_page(monkeypatch, params_store)
    client = acled_client.ACLEDClient()

    frame = client.fetch_events("2025-01-01", "2025-01-31")

    assert not frame.empty
    assert params_store, "fetch should have been invoked"
    assert "fields" not in params_store[0]


def test_fields_forced_uses_pipe_delimiter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACLED_FORCE_FIELDS", "1")
    params_store: List[Dict[str, Any]] = []
    _patch_fetch_page(monkeypatch, params_store)
    client = acled_client.ACLEDClient()

    frame = client.fetch_events("2025-01-01", "2025-01-31")

    assert not frame.empty
    assert params_store, "fetch should have been invoked"
    fields_value = params_store[0].get("fields", "")
    assert fields_value
    assert "|" in fields_value
    assert "," not in fields_value


def test_monthly_fatalities_sums_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_fetch_events(
        self: acled_client.ACLEDClient,
        start_date: str,
        end_date: str,
        *,
        countries: Any | None = None,
        fields: Any | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"event_date": "2025-01-05", "iso3": "AFG", "country": "Afghanistan", "fatalities": 1},
                {"event_date": "2025-01-25", "iso3": "AFG", "country": "Afghanistan", "fatalities": 2},
            ]
        )

    monkeypatch.setattr(acled_client.ACLEDClient, "fetch_events", fake_fetch_events)
    client = acled_client.ACLEDClient()

    result = client.monthly_fatalities("2025-01-01", "2025-01-31")

    assert list(result.columns) == ["iso3", "month", "fatalities", "source", "updated_at"]
    assert result.loc[0, "iso3"] == "AFG"
    assert int(result.loc[0, "fatalities"]) == 3
    assert result.loc[0, "source"] == "ACLED"
    assert pd.to_datetime(result.loc[0, "month"]).strftime("%Y-%m-%d") == "2025-01-01"
