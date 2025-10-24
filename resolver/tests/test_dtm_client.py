"""High-level dtm_client tests covering CLI behaviours."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pytest

from resolver.ingestion import dtm_client


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DTM_API_KEY", raising=False)
    monkeypatch.delenv("DTM_API_PRIMARY_KEY", raising=False)
    monkeypatch.delenv("DTM_API_SECONDARY_KEY", raising=False)
    monkeypatch.delenv("RESOLVER_START_ISO", raising=False)
    monkeypatch.delenv("RESOLVER_END_ISO", raising=False)


@pytest.fixture()
def patch_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Dict[str, Path]:
    diagnostics_dir = tmp_path / "diagnostics" / "ingestion"
    out_path = tmp_path / "outputs" / "dtm.csv"
    mappings = {
        "OUT_PATH": out_path,
        "OUT_DIR": out_path.parent,
        "OUTPUT_PATH": out_path,
        "DEFAULT_OUTPUT": out_path,
        "META_PATH": out_path.with_suffix(out_path.suffix + ".meta.json"),
        "DIAGNOSTICS_DIR": diagnostics_dir,
        "CONNECTORS_REPORT": diagnostics_dir / "connectors_report.jsonl",
        "RUN_DETAILS_PATH": diagnostics_dir / "dtm_run.json",
        "API_REQUEST_PATH": diagnostics_dir / "dtm_api_request.json",
        "API_SAMPLE_PATH": diagnostics_dir / "dtm_api_sample.json",
    }
    for name, value in mappings.items():
        monkeypatch.setattr(dtm_client, name, value)
    return mappings


def test_preflight_missing_dep_marks_error(monkeypatch: pytest.MonkeyPatch, patch_paths: Dict[str, Path]) -> None:
    original_import = importlib.import_module

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "dtmapi":
            raise ModuleNotFoundError("No module named 'dtmapi'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    captured: Dict[str, Any] = {}
    original_writer = dtm_client._write_connector_report

    def spy_writer(**kwargs: Any) -> None:
        captured.update(kwargs)
        original_writer(**kwargs)

    monkeypatch.setattr(dtm_client, "_write_connector_report", spy_writer)

    rc = dtm_client.main([])
    assert rc == 1

    assert captured["status"] == "error"
    assert "dependency-missing" in captured["reason"]

    report_lines = patch_paths["CONNECTORS_REPORT"].read_text(encoding="utf-8").strip().splitlines()
    assert len(report_lines) == 1
    report_payload = json.loads(report_lines[0])
    assert report_payload["status"] == "error"
    assert report_payload["reason"].startswith("dependency-missing: dtmapi")
    assert report_payload["extras"]["exit_code"] == 1

    summary_path = patch_paths["DIAGNOSTICS_DIR"] / "summary.md"
    assert summary_path.exists()
    assert "dependency-missing" in summary_path.read_text(encoding="utf-8")

    run_payload = json.loads(patch_paths["RUN_DETAILS_PATH"].read_text(encoding="utf-8"))
    assert run_payload["status"] == "error"
    assert run_payload["reason"].startswith("dependency-missing: dtmapi")
    assert "deps" in run_payload["extras"]

    meta_manifest = json.loads(patch_paths["META_PATH"].read_text(encoding="utf-8"))
    assert "deps" not in meta_manifest


def test_canonical_headers_exported() -> None:
    assert dtm_client.CANONICAL_HEADERS == [
        "source",
        "country_iso3",
        "admin1",
        "event_id",
        "as_of",
        "month_start",
        "value_type",
        "value",
        "unit",
        "method",
        "confidence",
        "raw_event_id",
        "raw_fields_json",
    ]


def test_build_rows_requires_api_config() -> None:
    with pytest.raises(ValueError) as excinfo:
        dtm_client.build_rows({}, no_date_filter=False, window_start=None, window_end=None)
    assert "DTM is API-only" in str(excinfo.value)


def test_main_writes_outputs_and_diagnostics(
    monkeypatch: pytest.MonkeyPatch, patch_paths: Dict[str, Path], tmp_path: Path
) -> None:
    monkeypatch.setenv("DTM_API_KEY", "primary")

    class DummyClient:
        def __init__(self, *_: Any, **__: Any) -> None:
            self.rate_limit_delay = 0
            self.timeout = 0

        def get_countries(self, *_: Any, **__: Any) -> pd.DataFrame:
            return pd.DataFrame([{"CountryName": "Kenya", "ISO3": "KEN"}])

        def get_idp_admin0(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "CountryName": ["Kenya"],
                    "ReportingDate": ["2024-05-15"],
                    "TotalIDPs": [120],
                }
            )

        def get_idp_admin1(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame()

        def get_idp_admin2(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(dtm_client, "DTMApiClient", DummyClient)
    monkeypatch.setattr(dtm_client, "resolve_accept_names", lambda *_: ["Kenya"])
    monkeypatch.setattr(dtm_client, "load_config", lambda: {"enabled": True, "api": {"admin_levels": ["admin0"]}})
    monkeypatch.setattr(dtm_client, "diagnostics_start_run", lambda *_, **__: object())
    def _finalize(*_, status: str, reason: str, http: Dict[str, Any], counts: Dict[str, Any], extras: Dict[str, Any]):
        payload = {"status": status, "reason": reason, "http": http, "counts": counts, "extras": extras}
        return payload

    def _append(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    monkeypatch.setattr(dtm_client, "diagnostics_finalize_run", _finalize)
    monkeypatch.setattr(dtm_client, "diagnostics_append_jsonl", _append)
    monkeypatch.setattr(dtm_client, "resolve_ingestion_window", lambda: (None, None))

    rc = dtm_client.main([])
    assert rc == 0
    csv_lines = patch_paths["OUT_PATH"].read_text(encoding="utf-8").strip().splitlines()
    assert len(csv_lines) == 3
    run_payload = json.loads(patch_paths["RUN_DETAILS_PATH"].read_text(encoding="utf-8"))
    assert run_payload["rows"]["written"] == 2
    assert run_payload["rows"]["fetched"] == 1
    assert run_payload["totals"]["rows_written"] == 2
    assert run_payload["args"]["strict_empty"] is False
    request = json.loads(patch_paths["API_REQUEST_PATH"].read_text(encoding="utf-8"))
    assert request["admin_levels"] == ["admin0"]
    meta_payload = json.loads(patch_paths["META_PATH"].read_text(encoding="utf-8"))
    assert "deps" in meta_payload
    assert "effective_params" in meta_payload
    assert "http_counters" in meta_payload
    assert "timings_ms" in meta_payload
    assert "timings_ms" in run_payload["extras"]
    assert "effective_params" in run_payload["extras"]
    assert "deps" in run_payload["extras"]


def test_main_strict_empty_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, patch_paths: Dict[str, Path], tmp_path: Path
) -> None:
    monkeypatch.setenv("DTM_API_KEY", "primary")

    class EmptyClient:
        def __init__(self, *_: Any, **__: Any) -> None:
            self.rate_limit_delay = 0
            self.timeout = 0

        def get_countries(self, *_: Any, **__: Any) -> pd.DataFrame:
            return pd.DataFrame([{"CountryName": "Kenya", "ISO3": "KEN"}])

        def get_idp_admin0(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame(columns=["CountryName", "ReportingDate", "TotalIDPs"])

        def get_idp_admin1(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame()

        def get_idp_admin2(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(dtm_client, "DTMApiClient", EmptyClient)
    monkeypatch.setattr(dtm_client, "resolve_accept_names", lambda *_: ["Kenya"])
    monkeypatch.setattr(dtm_client, "load_config", lambda: {"enabled": True, "api": {"admin_levels": ["admin0"]}})
    monkeypatch.setattr(dtm_client, "diagnostics_start_run", lambda *_, **__: object())
    monkeypatch.setattr(dtm_client, "diagnostics_finalize_run", lambda *_, **__: {})
    monkeypatch.setattr(dtm_client, "diagnostics_append_jsonl", lambda *_, **__: None)
    monkeypatch.setattr(dtm_client, "resolve_ingestion_window", lambda: (None, None))

    rc = dtm_client.main(["--strict-empty"])
    assert rc == 3


def test_main_zero_rows_reason_and_totals(
    monkeypatch: pytest.MonkeyPatch, patch_paths: Dict[str, Path], tmp_path: Path
) -> None:
    monkeypatch.setenv("DTM_API_KEY", "primary")

    class EmptyClient:
        def __init__(self, *_: Any, **__: Any) -> None:
            self.rate_limit_delay = 0
            self.timeout = 0

        def get_countries(self, *_: Any, **__: Any) -> pd.DataFrame:
            return pd.DataFrame([{"CountryName": "Kenya", "ISO3": "KEN"}])

        def get_idp_admin0(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame(columns=["CountryName", "ReportingDate", "TotalIDPs"])

        def get_idp_admin1(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame()

        def get_idp_admin2(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(dtm_client, "DTMApiClient", EmptyClient)
    monkeypatch.setattr(dtm_client, "resolve_accept_names", lambda *_: ["Kenya"])
    monkeypatch.setattr(
        dtm_client,
        "load_config",
        lambda: {"enabled": True, "api": {"admin_levels": ["admin0"]}},
    )
    monkeypatch.setattr(dtm_client, "diagnostics_start_run", lambda *_, **__: object())

    def _finalize(
        *_: Any,
        status: str,
        reason: str,
        http: Dict[str, Any],
        counts: Dict[str, Any],
        extras: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "status": status,
            "reason": reason,
            "http": http,
            "counts": counts,
            "extras": extras,
        }

    def _append(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    monkeypatch.setattr(dtm_client, "diagnostics_finalize_run", _finalize)
    monkeypatch.setattr(dtm_client, "diagnostics_append_jsonl", _append)
    monkeypatch.setattr(dtm_client, "resolve_ingestion_window", lambda: (None, None))

    rc = dtm_client.main([])
    assert rc == 0
    run_payload = json.loads(patch_paths["RUN_DETAILS_PATH"].read_text(encoding="utf-8"))
    assert run_payload["status"] == "ok-empty"
    assert run_payload["reason"] == "header-only (0 rows)"
    assert run_payload["totals"]["rows_written"] == 0
    assert run_payload["totals"]["rows_fetched"] == 0
    assert run_payload["args"]["strict_empty"] is False
    report_lines = patch_paths["CONNECTORS_REPORT"].read_text(encoding="utf-8").strip().splitlines()
    assert report_lines
    report = json.loads(report_lines[-1])
    assert report["status"] == "ok-empty"
    assert report["reason"] == "header-only (0 rows)"


def test_main_retries_with_secondary_key(
    monkeypatch: pytest.MonkeyPatch, patch_paths: Dict[str, Path], tmp_path: Path
) -> None:
    monkeypatch.setenv("DTM_API_PRIMARY_KEY", "primary")
    monkeypatch.setenv("DTM_API_SECONDARY_KEY", "secondary")

    attempt = {"count": 0}

    class FailingClient:
        def __init__(self, *_: Any, subscription_key: Optional[str] = None) -> None:
            self.subscription_key = subscription_key
            self.rate_limit_delay = 0
            self.timeout = 0

        def get_countries(self, *_: Any, **__: Any) -> pd.DataFrame:
            return pd.DataFrame([{"CountryName": "Kenya", "ISO3": "KEN"}])

        def get_idp_admin0(self, **_: Any) -> pd.DataFrame:
            attempt["count"] += 1
            if self.subscription_key == "primary":
                raise dtm_client.DTMUnauthorizedError(401, "unauthorized")
            return pd.DataFrame(
                {
                    "CountryName": ["Kenya"],
                    "ReportingDate": ["2024-06-01"],
                    "TotalIDPs": [25],
                }
            )

        def get_idp_admin1(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame()

        def get_idp_admin2(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(dtm_client, "DTMApiClient", FailingClient)
    monkeypatch.setattr(dtm_client, "resolve_accept_names", lambda *_: ["Kenya"])
    monkeypatch.setattr(dtm_client, "load_config", lambda: {"enabled": True, "api": {"admin_levels": ["admin0"]}})
    monkeypatch.setattr(dtm_client, "diagnostics_start_run", lambda *_, **__: object())
    monkeypatch.setattr(dtm_client, "diagnostics_finalize_run", lambda *_, **__: {})
    monkeypatch.setattr(dtm_client, "diagnostics_append_jsonl", lambda *_, **__: None)
    monkeypatch.setattr(dtm_client, "resolve_ingestion_window", lambda: (None, None))

    rc = dtm_client.main([])
    assert rc == 0
    assert attempt["count"] == 2
    run_payload = json.loads(patch_paths["RUN_DETAILS_PATH"].read_text(encoding="utf-8"))
    assert run_payload["http"]["retries"] == 1
