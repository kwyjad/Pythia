from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

from resolver.ingestion import dtm_client
import resolver.ingestion.dtm_auth as dtm_auth


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    monkeypatch.delenv("DTM_API_KEY", raising=False)
    monkeypatch.delenv("DTM_API_SECONDARY_KEY", raising=False)


def _configure_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Dict[str, Path]:
    out_path = tmp_path / "dtm.csv"
    monkeypatch.setattr(dtm_client, "OUT_PATH", out_path)
    monkeypatch.setattr(dtm_client, "OUTPUT_PATH", out_path)
    monkeypatch.setattr(dtm_client, "META_PATH", tmp_path / "dtm.meta.json")
    monkeypatch.setattr(dtm_client, "RUN_DETAILS_PATH", tmp_path / "dtm_run.json")
    monkeypatch.setattr(dtm_client, "API_REQUEST_PATH", tmp_path / "dtm_api_request.json")
    monkeypatch.setattr(
        dtm_client, "API_RESPONSE_SAMPLE_PATH", tmp_path / "dtm_api_response_sample.json"
    )
    return {
        "out": out_path,
        "run": dtm_client.RUN_DETAILS_PATH,
        "request": dtm_client.API_REQUEST_PATH,
        "sample": dtm_client.API_RESPONSE_SAMPLE_PATH,
    }


def test_build_rows_requires_api_config():
    with pytest.raises(ValueError) as excinfo:
        dtm_client.build_rows({})
    assert "DTM is API-only" in str(excinfo.value)


def test_main_writes_api_outputs_and_diagnostics(monkeypatch, tmp_path):
    paths = _configure_paths(tmp_path, monkeypatch)
    monkeypatch.setenv("DTM_API_KEY", "primary")
    monkeypatch.setattr(dtm_auth, "check_api_key_configured", lambda: True)

    class DummyClient:
        def __init__(self, _cfg: dict, *, subscription_key: Optional[str] = None):
            self.subscription_key = subscription_key
            self.rate_limit_delay = 0
            self.timeout = 60

        def get_idp_admin0(
            self,
            country: Optional[str] = None,
            from_date: Optional[str] = None,
            to_date: Optional[str] = None,
            http_counts: Optional[Dict[str, int]] = None,
            **_: Any,
        ) -> pd.DataFrame:
            if http_counts is not None:
                http_counts["2xx"] = http_counts.get("2xx", 0) + 1
            return pd.DataFrame(
                {
                    "CountryName": ["Kenya"],
                    "ReportingDate": ["2024-01-15"],
                    "TotalIDPs": [120],
                }
            )

        def get_idp_admin1(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
            return pd.DataFrame()

        def get_idp_admin2(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(dtm_client, "DTMApiClient", DummyClient)
    monkeypatch.setattr(dtm_client, "diagnostics_start_run", lambda *_, **__: object())
    monkeypatch.setattr(dtm_client, "diagnostics_finalize_run", lambda *_, **__: {})
    monkeypatch.setattr(dtm_client, "diagnostics_append_jsonl", lambda *_, **__: None)
    monkeypatch.setattr(dtm_client, "load_config", lambda: {"enabled": True, "api": {}})
    monkeypatch.setattr(dtm_client, "resolve_ingestion_window", lambda: (None, None))

    rc = dtm_client.main([])
    assert rc == 0
    csv_path = paths["out"]
    assert csv_path.exists()
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # header + row

    run_payload = json.loads(paths["run"].read_text(encoding="utf-8"))
    assert run_payload["mode"] == "api"
    assert run_payload["row_counts"]["total"] == 1
    request_payload = json.loads(paths["request"].read_text(encoding="utf-8"))
    assert request_payload["admin_levels"] == ["admin0", "admin1", "admin2"]
    sample_rows = json.loads(paths["sample"].read_text(encoding="utf-8"))
    assert len(sample_rows) == 1


def test_main_strict_empty_exits_nonzero(monkeypatch, tmp_path):
    _configure_paths(tmp_path, monkeypatch)
    monkeypatch.setenv("DTM_API_KEY", "primary")
    monkeypatch.setattr(dtm_auth, "check_api_key_configured", lambda: True)

    class EmptyClient:
        def __init__(self, *_: Any, **__: Any):
            self.rate_limit_delay = 0
            self.timeout = 60

        def get_idp_admin0(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
            return pd.DataFrame(columns=["CountryName", "ReportingDate", "TotalIDPs"])

        def get_idp_admin1(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
            return pd.DataFrame()

        def get_idp_admin2(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(dtm_client, "DTMApiClient", EmptyClient)
    monkeypatch.setattr(dtm_client, "diagnostics_start_run", lambda *_, **__: object())
    monkeypatch.setattr(dtm_client, "diagnostics_finalize_run", lambda *_, **__: {})
    monkeypatch.setattr(dtm_client, "diagnostics_append_jsonl", lambda *_, **__: None)
    monkeypatch.setattr(dtm_client, "load_config", lambda: {"enabled": True, "api": {}})
    monkeypatch.setattr(dtm_client, "resolve_ingestion_window", lambda: (None, None))

    rc = dtm_client.main(["--strict-empty"])
    assert rc == 3


def test_fetch_retries_with_secondary_key(monkeypatch):
    monkeypatch.setenv("DTM_API_KEY", "primary")
    monkeypatch.setenv("DTM_API_SECONDARY_KEY", "secondary")
    monkeypatch.setattr(dtm_auth, "check_api_key_configured", lambda: True)

    call_state = {"attempts": 0}

    class FailingClient:
        def __init__(self, *_: Any, subscription_key: Optional[str] = None):
            self.subscription_key = subscription_key
            self.rate_limit_delay = 0
            self.timeout = 0

        def get_idp_admin0(
            self,
            *,
            http_counts: Optional[Dict[str, int]] = None,
            **__: Any,
        ) -> pd.DataFrame:
            call_state["attempts"] += 1
            if call_state["attempts"] == 1:
                _record = dtm_client.DTMUnauthorizedError(401, "unauthorized")
                raise _record
            if http_counts is not None:
                http_counts["2xx"] = http_counts.get("2xx", 0) + 1
            return pd.DataFrame(
                {
                    "CountryName": ["Kenya"],
                    "ReportingDate": ["2024-01-15"],
                    "TotalIDPs": [50],
                }
            )

        def get_idp_admin1(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
            return pd.DataFrame()

        def get_idp_admin2(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(dtm_client, "DTMApiClient", FailingClient)
    records, summary = dtm_client._fetch_api_data({"api": {}, "admin_levels": ["admin0"]})
    assert call_state["attempts"] == 2
    assert summary["row_counts"]["total"] == 1
    assert records[0]["value"] == 50.0


def test_fetch_accumulates_multiple_levels(monkeypatch):
    monkeypatch.setenv("DTM_API_KEY", "primary")
    monkeypatch.setattr(dtm_auth, "check_api_key_configured", lambda: True)

    class MultiClient:
        def __init__(self, *_: Any, **__: Any):
            self.rate_limit_delay = 0
            self.timeout = 0

        def get_idp_admin0(
            self,
            *,
            http_counts: Optional[Dict[str, int]] = None,
            **__: Any,
        ) -> pd.DataFrame:
            if http_counts is not None:
                http_counts["2xx"] = http_counts.get("2xx", 0) + 1
            return pd.DataFrame(
                {
                    "CountryName": ["Kenya"],
                    "ReportingDate": ["2024-01-15"],
                    "TotalIDPs": [100],
                }
            )

        def get_idp_admin1(
            self,
            *,
            http_counts: Optional[Dict[str, int]] = None,
            **__: Any,
        ) -> pd.DataFrame:
            if http_counts is not None:
                http_counts["2xx"] = http_counts.get("2xx", 0) + 1
            return pd.DataFrame(
                {
                    "CountryName": ["Kenya"],
                    "Admin1Name": ["Nairobi"],
                    "ReportingDate": ["2024-01-15"],
                    "TotalIDPs": [25],
                }
            )

        def get_idp_admin2(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(dtm_client, "DTMApiClient", MultiClient)
    records, summary = dtm_client._fetch_api_data({"api": {}, "admin_levels": ["admin0", "admin1"]})
    assert summary["row_counts"]["admin0"] == 1
    assert summary["row_counts"]["admin1"] == 1
    assert summary["row_counts"]["total"] == 2
    assert {record["source_id"] for record in records} == {"dtm_api_admin0", "dtm_api_admin1"}
