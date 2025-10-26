import json
from pathlib import Path

import pandas as pd
import pytest
import requests

from resolver.ingestion import dtm_client as dtm


def _patch_diagnostics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "diagnostics"
    mapping = {
        "DIAGNOSTICS_DIR": root,
        "DTM_DIAGNOSTICS_DIR": root / "dtm",
        "DTM_RAW_DIR": root / "dtm" / "raw",
        "DTM_METRICS_DIR": root / "dtm" / "metrics",
        "DTM_SAMPLES_DIR": root / "dtm" / "samples",
        "DTM_LOG_DIR": root / "dtm" / "logs",
        "CONNECTORS_REPORT": root / "connectors_report.jsonl",
        "RUN_DETAILS_PATH": root / "dtm" / "dtm_run.json",
        "DISCOVERY_SNAPSHOT_PATH": root / "dtm" / "discovery_snapshot.csv",
        "DISCOVERY_FAIL_PATH": root / "dtm" / "discovery_fail.json",
        "DTM_HTTP_LOG_PATH": root / "dtm" / "dtm_http.ndjson",
        "DISCOVERY_RAW_JSON_PATH": root / "dtm" / "raw" / "dtm_countries.json",
        "PER_COUNTRY_METRICS_PATH": root / "dtm" / "metrics" / "dtm_per_country.jsonl",
        "SAMPLE_ROWS_PATH": root / "dtm" / "samples" / "dtm_sample.csv",
        "DTM_CLIENT_LOG_PATH": root / "dtm" / "logs" / "dtm_client.log",
        "METRICS_SUMMARY_PATH": root / "dtm" / "metrics" / "metrics.json",
        "SAMPLE_ADMIN0_PATH": root / "dtm" / "samples" / "admin0_head.csv",
    }
    for name, value in mapping.items():
        monkeypatch.setattr(dtm, name, value, raising=False)
    dtm._reset_admin0_sample_counter()
    dtm._clear_discovery_error_log()


def test_sdk_discovery_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_diagnostics(monkeypatch, tmp_path)
    monkeypatch.setenv("DTM_API_KEY", "sdk-key")

    class StubClient:
        def get_countries(self, http_counts: dict | None = None) -> pd.DataFrame:
            if http_counts is not None:
                http_counts["last_status"] = 200
            return pd.DataFrame([
                {"admin0Name": "Kenya", "admin0Pcode": "KEN"},
                {"admin0Name": "Somalia", "admin0Pcode": "SOM"},
            ])

    cfg = {"api": {"countries": []}}
    metrics = dtm._init_metrics_summary()
    stub = StubClient()
    result = dtm._perform_discovery(cfg, metrics=metrics, api_key="sdk-key", client=stub)

    assert result.stage_used == "sdk"
    assert metrics["stage_used"] == "sdk"
    snapshot = pd.read_csv(dtm.DISCOVERY_SNAPSHOT_PATH)
    assert not snapshot.empty
    assert set(snapshot.columns) >= {"admin0Name", "admin0Pcode"}
    payload = json.loads(dtm.DISCOVERY_FAIL_PATH.read_text(encoding="utf-8"))
    assert payload.get("used_stage") == "sdk"


def test_http_discovery_uses_alternate_header(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_diagnostics(monkeypatch, tmp_path)
    monkeypatch.setenv("DTM_API_KEY", "http-key")

    class DenySDK:
        def get_countries(self, http_counts: dict | None = None) -> pd.DataFrame:
            raise dtm.DTMUnauthorizedError(403, "sdk forbidden")

    calls: list[tuple[str, str]] = []

    def fake_http(
        path: str,
        key: str,
        *,
        headers_variants: list[dict[str, str]] | None = None,
        **_: object,
    ) -> pd.DataFrame:
        headers = headers_variants[0] if headers_variants else {"Ocp-Apim-Subscription-Key": key}
        label = next(iter(headers))
        calls.append((path, label))
        if "X-API-Key" not in headers:
            dtm._get_country_list_via_http.last_attempts = 1
            dtm._get_country_list_via_http.last_status = 403
            dtm._get_country_list_via_http.last_error_payload = "{\"detail\":\"forbidden\"}"
            raise requests.exceptions.HTTPError("403 forbidden")
        dtm._get_country_list_via_http.last_attempts = 1
        dtm._get_country_list_via_http.last_status = 200
        dtm._get_country_list_via_http.last_error_payload = None
        return pd.DataFrame([
            {"admin0Name": "Kenya", "admin0Pcode": "KEN"},
            {"admin0Name": "Somalia", "admin0Pcode": "SOM"},
        ])

    monkeypatch.setattr(dtm, "_get_country_list_via_http", fake_http)

    cfg = {"api": {"countries": []}}
    metrics = dtm._init_metrics_summary()
    result = dtm._perform_discovery(cfg, metrics=metrics, api_key="http-key", client=DenySDK())

    assert result.stage_used == "http_country_x_api"
    assert any(header == "Ocp-Apim-Subscription-Key" for _, header in calls)
    assert any(header == "X-API-Key" for _, header in calls)
    fail_payload = json.loads(dtm.DISCOVERY_FAIL_PATH.read_text(encoding="utf-8"))
    errors = fail_payload.get("errors", [])
    assert any("403" in json.dumps(entry) for entry in errors)


def test_discovery_fallback_to_minimal_allowlist(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_diagnostics(monkeypatch, tmp_path)
    monkeypatch.setenv("DTM_API_KEY", "fallback-key")

    class DenySDK:
        def get_countries(self, http_counts: dict | None = None) -> pd.DataFrame:
            raise dtm.DTMUnauthorizedError(403, "sdk forbidden")

    def always_forbidden(
        path: str,
        key: str,
        *,
        headers_variants: list[dict[str, str]] | None = None,
        **_: object,
    ) -> pd.DataFrame:
        dtm._get_country_list_via_http.last_attempts = 1
        dtm._get_country_list_via_http.last_status = 403
        dtm._get_country_list_via_http.last_error_payload = "{\"detail\":\"forbidden\"}"
        raise requests.exceptions.HTTPError(f"403 forbidden {path}")

    monkeypatch.setattr(dtm, "_get_country_list_via_http", always_forbidden)

    cfg = {"api": {"countries": []}}
    metrics = dtm._init_metrics_summary()
    result = dtm._perform_discovery(cfg, metrics=metrics, api_key="fallback-key", client=DenySDK())

    assert result.stage_used == "static_iso3_minimal"
    assert result.report.get("reason") == "static_iso3_minimal_fallback"
    snapshot = pd.read_csv(dtm.DISCOVERY_SNAPSHOT_PATH)
    assert set(snapshot["source"]) == {"static_iso3_minimal"}
    fail_payload = json.loads(dtm.DISCOVERY_FAIL_PATH.read_text(encoding="utf-8"))
    assert fail_payload.get("used_stage") == "static_iso3_minimal"
    assert any("403" in json.dumps(entry) for entry in fail_payload.get("errors", []))
