"""Unit tests for the thin DTMApiClient wrapper."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pandas as pd
import pytest
import sys

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
def patch_discovery_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    diagnostics_dir = tmp_path / "diagnostics"
    monkeypatch.setattr(dtm_client, "DTM_DIAGNOSTICS_DIR", diagnostics_dir)
    monkeypatch.setattr(dtm_client, "DISCOVERY_SNAPSHOT_PATH", diagnostics_dir / "discovery_countries.csv")
    monkeypatch.setattr(dtm_client, "DISCOVERY_FAIL_PATH", diagnostics_dir / "discovery_fail.json")
    monkeypatch.setattr(dtm_client, "DTM_HTTP_LOG_PATH", diagnostics_dir / "dtm_http.ndjson")
    return diagnostics_dir


@pytest.fixture(autouse=True)
def stub_dtmapi(monkeypatch: pytest.MonkeyPatch) -> DummyDTMApi:
    dummy = DummyDTMApi(subscription_key="primary")
    module = SimpleNamespace(DTMApi=lambda subscription_key=None: DummyDTMApi(subscription_key=subscription_key))
    monkeypatch.setitem(sys.modules, "dtmapi", module)
    return dummy


@pytest.fixture()
def config() -> dict:
    return {"api": {"timeout": 1, "rate_limit_delay": 0}}


def test_init_requires_key(monkeypatch: pytest.MonkeyPatch, config: dict) -> None:
    monkeypatch.delenv("DTM_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        DTMApiClient(config)
    assert "Missing DTM_API_KEY" in str(excinfo.value)


def test_get_countries(monkeypatch: pytest.MonkeyPatch, config: dict) -> None:
    monkeypatch.setenv("DTM_API_KEY", "primary")
    client = DTMApiClient(config)
    df = client.get_countries()
    assert list(df["CountryName"]) == ["Kenya"]
    assert client.client.subscription_key == "primary"


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


def test_discover_all_countries_writes_snapshot(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {"CountryName": "Kenya"},
            {"CountryName": "Somalia"},
        ]
    )
    api = SimpleNamespace(get_all_countries=lambda: frame)
    result = dtm_client._discover_all_countries(api)
    assert result == ["Kenya", "Somalia"]
    snapshot = dtm_client.DISCOVERY_SNAPSHOT_PATH
    assert snapshot.exists()
    saved = pd.read_csv(snapshot)
    assert set(saved["CountryName"]) == {"Kenya", "Somalia"}


def test_discover_all_countries_empty_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    api = SimpleNamespace(get_all_countries=lambda: pd.DataFrame(columns=["CountryName"]))
    with pytest.raises(RuntimeError) as excinfo:
        dtm_client._discover_all_countries(api)
    assert "0 countries" in str(excinfo.value)
    fail_path = dtm_client.DISCOVERY_FAIL_PATH
    assert fail_path.exists()
    payload = json.loads(fail_path.read_text())
    assert payload["reason"] == "empty_discovery"


def test_discover_all_countries_invokes_sdk_catalog() -> None:
    class FakeApi:
        def __init__(self) -> None:
            self.calls = 0

        def get_all_countries(self) -> pd.DataFrame:
            self.calls += 1
            return pd.DataFrame(
                [
                    {"CountryName": "Ethiopia"},
                    {"CountryName": "Somalia"},
                ]
            )

    api = FakeApi()
    names = dtm_client._discover_all_countries(api)
    assert names == ["Ethiopia", "Somalia"]
    assert api.calls == 1
