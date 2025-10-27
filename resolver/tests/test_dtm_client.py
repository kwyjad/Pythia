"""High-level dtm_client tests covering CLI behaviours."""

from __future__ import annotations

import csv
import logging
import importlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pytest
import requests

from resolver.ingestion import dtm_client


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DTM_API_KEY", raising=False)
    monkeypatch.delenv("RESOLVER_START_ISO", raising=False)
    monkeypatch.delenv("RESOLVER_END_ISO", raising=False)
    monkeypatch.delenv("RESOLVER_SKIP_DTM", raising=False)


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
        "DTM_DIAGNOSTICS_DIR": diagnostics_dir / "dtm",
        "DTM_RAW_DIR": diagnostics_dir / "dtm" / "raw",
        "DTM_METRICS_DIR": diagnostics_dir / "dtm" / "metrics",
        "DTM_SAMPLES_DIR": diagnostics_dir / "dtm" / "samples",
        "DTM_LOG_DIR": diagnostics_dir / "dtm" / "logs",
        "DIAGNOSTICS_RAW_DIR": diagnostics_dir / "raw",
        "DIAGNOSTICS_METRICS_DIR": diagnostics_dir / "metrics",
        "DIAGNOSTICS_SAMPLES_DIR": diagnostics_dir / "samples",
        "DIAGNOSTICS_LOG_DIR": diagnostics_dir / "logs",
        "CONNECTORS_REPORT": diagnostics_dir / "connectors_report.jsonl",
        "RUN_DETAILS_PATH": diagnostics_dir / "dtm" / "dtm_run.json",
        "API_REQUEST_PATH": diagnostics_dir / "dtm_api_request.json",
        "API_SAMPLE_PATH": diagnostics_dir / "dtm_api_sample.json",
        "DISCOVERY_SNAPSHOT_PATH": diagnostics_dir / "dtm" / "discovery_countries.csv",
        "DISCOVERY_FAIL_PATH": diagnostics_dir / "dtm" / "discovery_fail.json",
        "DTM_HTTP_LOG_PATH": diagnostics_dir / "dtm" / "dtm_http.ndjson",
        "DISCOVERY_RAW_JSON_PATH": diagnostics_dir / "raw" / "dtm_countries.json",
        "PER_COUNTRY_METRICS_PATH": diagnostics_dir / "metrics" / "dtm_per_country.jsonl",
        "SAMPLE_ROWS_PATH": diagnostics_dir / "samples" / "dtm_sample.csv",
        "DTM_CLIENT_LOG_PATH": diagnostics_dir / "logs" / "dtm_client.log",
    }
    for name, value in mappings.items():
        monkeypatch.setattr(dtm_client, name, value)
    # Reset file logging so tests capture the patched log path.
    for handler in list(dtm_client.LOG.handlers):
        if isinstance(handler, logging.FileHandler):
            dtm_client.LOG.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
    monkeypatch.setattr(dtm_client, "_FILE_LOGGING_INITIALIZED", False)
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

        def get_all_countries(self) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {"CountryName": "Ethiopia", "ISO3": "ETH"},
                    {"CountryName": "Somalia", "ISO3": "SOM"},
                ]
            )

        def get_idp_admin0(self, **_: Any) -> pd.DataFrame:
            country = _.get("country")
            if country in {"Ethiopia", "ETH"}:
                return pd.DataFrame(
                    {
                        "CountryName": ["Ethiopia"],
                        "ReportingDate": ["2024-05-15"],
                        "numPresentIdpInd": [120],
                    }
                )
            if country in {"Somalia", "SOM"}:
                return pd.DataFrame(
                    {
                        "CountryName": ["Somalia"],
                        "ReportingDate": ["2024-05-15"],
                        "total_people_idp": [95],
                    }
                )
            if country in {"Kenya", "KEN"}:
                return pd.DataFrame(
                    {
                        "CountryName": ["Kenya"],
                        "ReportingDate": ["2024-05-15"],
                        "TotalIDPs": [135],
                    }
                )
            return pd.DataFrame()

        def get_idp_admin1(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame()

        def get_idp_admin2(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(dtm_client, "DTMApiClient", DummyClient)
    monkeypatch.setattr(
        dtm_client,
        "load_config",
        lambda: {"enabled": True, "api": {"admin_levels": ["admin0"], "countries": ["Kenya"]}},
    )
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
    assert len(csv_lines) > 1
    with patch_paths["OUT_PATH"].open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        value_samples = {
            (
                lambda payload: payload.get("value", payload.get("total_value"))
            )(json.loads(row["raw_fields_json"]))
            if row.get("raw_fields_json")
            else None
            for row in reader
        }
    assert 135 in value_samples
    run_payload = json.loads(patch_paths["RUN_DETAILS_PATH"].read_text(encoding="utf-8"))
    assert run_payload["rows"]["written"] > 0
    assert run_payload["rows"]["fetched"] > 0
    assert run_payload["totals"]["rows_written"] == run_payload["rows"]["written"]
    assert run_payload["args"]["strict_empty"] is False
    request = json.loads(patch_paths["API_REQUEST_PATH"].read_text(encoding="utf-8"))
    assert request["admin_levels"] == ["admin0"]
    assert request["country_mode"] == "explicit_config"
    meta_payload = json.loads(patch_paths["META_PATH"].read_text(encoding="utf-8"))
    assert "deps" in meta_payload
    assert "effective_params" in meta_payload
    assert "http_counters" in meta_payload
    assert "timings_ms" in meta_payload
    assert "per_country_counts" in meta_payload
    assert isinstance(meta_payload["per_country_counts"], list)
    assert "failures" in meta_payload
    assert isinstance(meta_payload["failures"], list)
    assert meta_payload["effective_params"]["country_mode"] == "explicit_config"
    assert meta_payload["effective_params"]["discovered_countries_count"] == 1
    assert meta_payload["effective_params"]["countries_requested"] == ["Kenya"]
    assert meta_payload["effective_params"]["idp_aliases"]
    assert "discovery" in meta_payload
    assert meta_payload["discovery"]["total_countries"] == 1
    assert meta_payload["discovery"]["source"] == "explicit_config"
    assert "diagnostics" in meta_payload
    assert meta_payload["diagnostics"]["http_trace"] == str(patch_paths["DTM_HTTP_LOG_PATH"])
    assert meta_payload["diagnostics"]["raw_countries"] == str(patch_paths["DISCOVERY_RAW_JSON_PATH"])
    assert meta_payload["diagnostics"]["metrics"] == str(patch_paths["PER_COUNTRY_METRICS_PATH"])
    assert meta_payload["diagnostics"]["sample"] == str(patch_paths["SAMPLE_ROWS_PATH"])
    assert meta_payload["diagnostics"]["log"] == str(patch_paths["DTM_CLIENT_LOG_PATH"])
    assert any(
        str(path).endswith("sample_admin0.csv")
        for path in meta_payload["diagnostics"].get("samples", [])
    )
    raw_dir = patch_paths["DTM_DIAGNOSTICS_DIR"] / "raw"
    assert raw_dir.exists()
    assert any(child.name.startswith("admin0.") for child in raw_dir.iterdir())
    discovery_snapshot = patch_paths["DISCOVERY_SNAPSHOT_PATH"]
    assert discovery_snapshot.exists()
    snapshot_df = pd.read_csv(discovery_snapshot)
    assert set(snapshot_df["country_label"]) == {"Kenya"}
    discovery_raw = patch_paths["DISCOVERY_RAW_JSON_PATH"]
    assert discovery_raw.exists()
    raw_payload = json.loads(discovery_raw.read_text(encoding="utf-8"))
    assert isinstance(raw_payload, list)
    assert any(entry.get("admin0Name") == "Kenya" for entry in raw_payload)
    metrics_path = patch_paths["PER_COUNTRY_METRICS_PATH"]
    assert metrics_path.exists()
    metrics_lines = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert metrics_lines
    assert any(line.get("rows", 0) > 0 for line in metrics_lines)
    http_trace = patch_paths["DTM_HTTP_LOG_PATH"]
    assert http_trace.exists()
    request_entries = [
        json.loads(line)
        for line in http_trace.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert request_entries
    assert all("country" in entry for entry in request_entries)
    sample_path = patch_paths["DTM_DIAGNOSTICS_DIR"] / "sample_admin0.csv"
    assert sample_path.exists()
    sample_rows_path = patch_paths["SAMPLE_ROWS_PATH"]
    assert sample_rows_path.exists()
    log_path = patch_paths["DTM_CLIENT_LOG_PATH"]
    assert log_path.exists()
    assert "timings_ms" in run_payload["extras"]
    assert "effective_params" in run_payload["extras"]
    assert "deps" in run_payload["extras"]
    assert "per_country_counts" in run_payload["extras"]
    assert "failures" in run_payload["extras"]
    assert "discovery" in run_payload["extras"]
    assert run_payload["extras"]["discovery"]["total_countries"] == 1
    assert run_payload["extras"]["discovery"]["source"] == "explicit_config"
    assert "diagnostics" in run_payload["extras"]
    assert run_payload["extras"]["diagnostics"]["http_trace"] == str(http_trace)
    assert run_payload["extras"]["diagnostics"]["log"] == str(log_path)
    assert run_payload["extras"]["diagnostics"]["sample"] == str(sample_rows_path)
    assert run_payload["extras"]["diagnostics"]["metrics"] == str(metrics_path)
    assert run_payload["extras"]["diagnostics"]["raw_countries"] == str(discovery_raw)


def test_main_strict_empty_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, patch_paths: Dict[str, Path], tmp_path: Path
) -> None:
    monkeypatch.setenv("DTM_API_KEY", "primary")

    class EmptyClient:
        def __init__(self, *_: Any, **__: Any) -> None:
            self.rate_limit_delay = 0
            self.timeout = 0

        def get_all_countries(self) -> pd.DataFrame:
            return pd.DataFrame([{"CountryName": "Kenya", "ISO3": "KEN"}])

        def get_idp_admin0(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame(columns=["CountryName", "ReportingDate", "TotalIDPs"])

        def get_idp_admin1(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame()

        def get_idp_admin2(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(dtm_client, "DTMApiClient", EmptyClient)
    monkeypatch.setattr(dtm_client, "load_config", lambda: {"enabled": True, "api": {"admin_levels": ["admin0"]}})
    monkeypatch.setattr(dtm_client, "diagnostics_start_run", lambda *_, **__: object())
    monkeypatch.setattr(dtm_client, "diagnostics_finalize_run", lambda *_, **__: {})
    monkeypatch.setattr(dtm_client, "diagnostics_append_jsonl", lambda *_, **__: None)
    monkeypatch.setattr(dtm_client, "resolve_ingestion_window", lambda: (None, None))

    rc = dtm_client.main(["--strict-empty"])
    assert rc == 2
    run_payload = json.loads(patch_paths["RUN_DETAILS_PATH"].read_text(encoding="utf-8"))
    assert run_payload["status"] == "ok-empty"
    assert run_payload["reason"] == "header-only; kept=0"
    assert run_payload["extras"]["zero_rows_reason"] == "filter_excluded_all"
    assert run_payload["extras"]["rows_written"] == 0
    assert run_payload["extras"]["exit_code"] == 2


def test_main_zero_rows_reason_and_totals(
    monkeypatch: pytest.MonkeyPatch, patch_paths: Dict[str, Path], tmp_path: Path
) -> None:
    monkeypatch.setenv("DTM_API_KEY", "primary")

    class EmptyClient:
        def __init__(self, *_: Any, **__: Any) -> None:
            self.rate_limit_delay = 0
            self.timeout = 0

        def get_all_countries(self) -> pd.DataFrame:
            return pd.DataFrame([{"CountryName": "Kenya", "ISO3": "KEN"}])

        def get_idp_admin0(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame(columns=["CountryName", "ReportingDate", "TotalIDPs"])

        def get_idp_admin1(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame()

        def get_idp_admin2(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(dtm_client, "DTMApiClient", EmptyClient)
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
    assert run_payload["reason"] == "header-only; kept=0"
    assert run_payload["extras"]["zero_rows_reason"] == "filter_excluded_all"
    report_lines = patch_paths["CONNECTORS_REPORT"].read_text(encoding="utf-8").strip().splitlines()
    assert report_lines
    report = json.loads(report_lines[-1])
    assert report["status"] == "ok-empty"
    assert report["reason"] == "header-only; kept=0"
    assert report["extras"]["zero_rows_reason"] == "filter_excluded_all"
    assert report["extras"]["exit_code"] == 0
    assert report["extras"]["rows_written"] == 0


def test_main_invalid_key_aborts(
    monkeypatch: pytest.MonkeyPatch, patch_paths: Dict[str, Path], tmp_path: Path
) -> None:
    monkeypatch.setenv("DTM_API_KEY", "invalid")

    attempt = {"count": 0}

    class UnauthorizedClient:
        def __init__(self, *_: Any, subscription_key: Optional[str] = None) -> None:
            self.subscription_key = subscription_key
            self.rate_limit_delay = 0
            self.timeout = 0

        def get_all_countries(self) -> pd.DataFrame:
            return pd.DataFrame([{"CountryName": "Kenya", "ISO3": "KEN"}])

        def get_idp_admin0(self, **_: Any) -> pd.DataFrame:
            attempt["count"] += 1
            raise dtm_client.DTMUnauthorizedError(401, "unauthorized")

        def get_idp_admin1(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame()

        def get_idp_admin2(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(dtm_client, "DTMApiClient", UnauthorizedClient)
    monkeypatch.setattr(dtm_client, "load_config", lambda: {"enabled": True, "api": {"admin_levels": ["admin0"]}})
    monkeypatch.setattr(dtm_client, "diagnostics_start_run", lambda *_, **__: object())
    monkeypatch.setattr(dtm_client, "diagnostics_finalize_run", lambda *_, **__: {})
    monkeypatch.setattr(dtm_client, "diagnostics_append_jsonl", lambda *_, **__: None)
    monkeypatch.setattr(dtm_client, "resolve_ingestion_window", lambda: (None, None))

    rc = dtm_client.main([])
    assert rc == 1
    assert attempt["count"] == 1
    fail_path = patch_paths["DISCOVERY_FAIL_PATH"]
    assert fail_path.exists()
    payload = json.loads(fail_path.read_text(encoding="utf-8"))
    assert payload["reason"] == "invalid_key"
    run_payload = json.loads(patch_paths["RUN_DETAILS_PATH"].read_text(encoding="utf-8"))
    assert run_payload["status"] == "error"


def test_discovery_empty_hard_fail(
    monkeypatch: pytest.MonkeyPatch, patch_paths: Dict[str, Path], tmp_path: Path
) -> None:
    monkeypatch.setenv("DTM_API_KEY", "primary")

    fetch_calls: list[str] = []

    class DiscoveryClient:
        def __init__(self, *_: Any, **__: Any) -> None:
            self.rate_limit_delay = 0
            self.timeout = 0

        def get_all_countries(self) -> pd.DataFrame:
            return pd.DataFrame(columns=["CountryName", "ISO3"])

        def get_idp_admin0(self, **_: Any) -> pd.DataFrame:
            fetch_calls.append(str(_.get("country")))
            return pd.DataFrame()

        def get_idp_admin1(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame()

        def get_idp_admin2(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(dtm_client, "DTMApiClient", DiscoveryClient)
    monkeypatch.setattr(
        dtm_client,
        "load_config",
        lambda: {"enabled": True, "api": {"admin_levels": ["admin0"]}},
    )
    monkeypatch.setattr(dtm_client, "diagnostics_start_run", lambda *_, **__: object())
    monkeypatch.setattr(dtm_client, "diagnostics_finalize_run", lambda *_, **__: {})
    monkeypatch.setattr(dtm_client, "diagnostics_append_jsonl", lambda *_, **__: None)
    monkeypatch.setattr(dtm_client, "resolve_ingestion_window", lambda: (None, None))
    monkeypatch.setattr(
        dtm_client,
        "_get_country_list_via_http",
        lambda *_, **__: (_ for _ in ()).throw(requests.exceptions.Timeout("discovery timeout")),
    )
    monkeypatch.setattr(
        dtm_client,
        "_load_static_iso3",
        lambda *_: pd.DataFrame(columns=["admin0Name", "admin0Pcode"]),
    )

    rc = dtm_client.main([])
    assert rc == 0
    run_payload = json.loads(patch_paths["RUN_DETAILS_PATH"].read_text(encoding="utf-8"))
    assert run_payload["status"] in {"ok", "ok-empty"}
    assert run_payload["extras"].get("zero_rows_reason")
    assert fetch_calls  # minimal fallback attempted fetches
    assert set(fetch_calls).issuperset({name for name, _ in dtm_client.STATIC_MINIMAL_FALLBACK})

def test_http_trace_written(
    monkeypatch: pytest.MonkeyPatch, patch_paths: Dict[str, Path], tmp_path: Path
) -> None:
    monkeypatch.setenv("DTM_API_KEY", "primary")

    class DummyClient:
        def __init__(self, *_: Any, **__: Any) -> None:
            self.rate_limit_delay = 0
            self.timeout = 0

        def get_all_countries(self) -> pd.DataFrame:
            return pd.DataFrame([{"CountryName": "Kenya"}])

        def get_idp_admin0(self, **_: Any) -> pd.DataFrame:
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

    monkeypatch.setattr(dtm_client, "DTMApiClient", DummyClient)
    monkeypatch.setattr(dtm_client, "load_config", lambda: {"enabled": True, "api": {"admin_levels": ["admin0"]}})
    monkeypatch.setattr(dtm_client, "diagnostics_start_run", lambda *_, **__: object())
    monkeypatch.setattr(dtm_client, "diagnostics_finalize_run", lambda *_, **__: {})
    monkeypatch.setattr(dtm_client, "diagnostics_append_jsonl", lambda *_, **__: None)
    monkeypatch.setattr(dtm_client, "resolve_ingestion_window", lambda: (None, None))

    rc = dtm_client.main([])
    assert rc == 0
    log_file = patch_paths["DTM_HTTP_LOG_PATH"]
    assert log_file.exists()
    lines = [
        json.loads(line)
        for line in log_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert lines
    assert all("country" in entry for entry in lines)
