"""Unit tests for the thin DTMApiClient wrapper."""

from __future__ import annotations

from types import SimpleNamespace
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pandas as pd
import pytest

from resolver.ingestion import dtm_client
from resolver.ingestion.dtm_client import DTMApiClient, DTMHttpError


class DummyDTMApi:
    def __init__(self, *, subscription_key: str) -> None:
        self.subscription_key = subscription_key
        self.calls: Dict[str, Dict[str, Any]] = {}

    def get_all_countries(self) -> pd.DataFrame:
        self.calls.setdefault("countries", {})
        return pd.DataFrame([{"CountryName": "Kenya", "ISO3": "KEN"}])

    def get_idp_admin0_data(self, **params: Any) -> pd.DataFrame:
        self.calls.setdefault("admin0", params)
        return pd.DataFrame(
            {"CountryName": ["Kenya"], "ReportingDate": ["2024-01-01"], "TotalIDPs": [100]}
        )

    def get_idp_admin1_data(self, **params: Any) -> pd.DataFrame:
        self.calls.setdefault("admin1", params)
        return pd.DataFrame()

    def get_idp_admin2_data(self, **params: Any) -> pd.DataFrame:
        self.calls.setdefault("admin2", params)
        return pd.DataFrame()


@pytest.fixture(autouse=True)
def stub_dtmapi(monkeypatch: pytest.MonkeyPatch) -> DummyDTMApi:
    dummy = DummyDTMApi(subscription_key="primary")
    module = SimpleNamespace(DTMApi=lambda subscription_key=None: DummyDTMApi(subscription_key=subscription_key))
    monkeypatch.setitem(sys.modules, "dtmapi", module)
    return dummy


import sys  # placed after fixture to appease linter


@pytest.fixture()
def config() -> dict:
    return {"api": {"timeout": 1, "rate_limit_delay": 0}}


def test_init_requires_key(monkeypatch: pytest.MonkeyPatch, config: dict) -> None:
    monkeypatch.delenv("DTM_API_KEY", raising=False)
    with pytest.raises(ValueError):
        DTMApiClient(config)


def test_get_countries(monkeypatch: pytest.MonkeyPatch, config: dict) -> None:
    monkeypatch.setenv("DTM_API_KEY", "primary")
    client = DTMApiClient(config)
    df = client.get_countries()
    assert list(df["CountryName"]) == ["Kenya"]


def test_get_idp_admin0_updates_http_counts(monkeypatch: pytest.MonkeyPatch, config: dict) -> None:
    monkeypatch.setenv("DTM_API_KEY", "primary")
    client = DTMApiClient(config)
    http_counts = {"2xx": 0, "4xx": 0, "5xx": 0, "timeout": 0, "error": 0, "last_status": None}
    df = client.get_idp_admin0(country="Kenya", http_counts=http_counts)
    assert not df.empty
    assert http_counts["2xx"] == 1
    assert http_counts["last_status"] == 200


def test_http_errors_raise(monkeypatch: pytest.MonkeyPatch, config: dict) -> None:
    class ErrorDTMApi(DummyDTMApi):
        def get_idp_admin0_data(self, **params: Any) -> pd.DataFrame:
            raise dtm_client.DTMUnauthorizedError(401, "unauthorized")  # type: ignore[name-defined]

    monkeypatch.setitem(sys.modules, "dtmapi", SimpleNamespace(DTMApi=lambda subscription_key=None: ErrorDTMApi(subscription_key=subscription_key)))
    monkeypatch.setenv("DTM_API_KEY", "primary")
    client = DTMApiClient(config)
    with pytest.raises(DTMHttpError):
        client.get_idp_admin0(country="Kenya")


def test_admin_methods_use_sdk_parameter_names(
    monkeypatch: pytest.MonkeyPatch, config: dict
) -> None:
    monkeypatch.setenv("DTM_API_KEY", "primary")
    client = DTMApiClient(config)
    client.get_idp_admin0(country="Kenya")
    calls = client.client.calls
    assert calls["admin0"]["CountryName"] == "Kenya"
    assert "FromReportingDate" in calls["admin0"]
    client.get_idp_admin1(country="Kenya")
    assert calls["admin1"]["CountryName"] == "Kenya"
    client.get_idp_admin2(country="Kenya", operation=None)
    assert calls["admin2"]["CountryName"] == "Kenya"
    assert "Operation" not in calls["admin2"]


def test_discover_all_countries_normalizes(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame(
        [
            {"CountryName": "Kenya"},
            {"CountryName": " Somalia "},
            {"CountryName": None},
            {"CountryName": "Kenya"},
        ]
    )
    api = SimpleNamespace(get_all_countries=lambda: frame)
    result = dtm_client._discover_all_countries(api)
    assert result == ["Kenya", "Somalia"]


def test_discover_all_countries_uses_client_http_counts() -> None:
    called = {}

    class FakeClient:
        def get_countries(self, http_counts: Optional[dict] = None) -> pd.DataFrame:  # type: ignore[override]
            called["http_counts"] = http_counts
            return pd.DataFrame([
                {"CountryName": "Ethiopia"},
                {"CountryName": "Somalia"},
            ])

    names = dtm_client._discover_all_countries(FakeClient(), http_counts={})
    assert names == ["Ethiopia", "Somalia"]
    assert called["http_counts"] == {}
