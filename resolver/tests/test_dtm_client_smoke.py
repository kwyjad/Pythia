# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import csv
import json
from pathlib import Path

import pytest

from resolver.ingestion import dtm_client


@pytest.fixture(autouse=True)
def clear_dtm_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_name in (
        "DTM_API_KEY",
        "DTM_SUBSCRIPTION_KEY",
        "DTM_API_PRIMARY_KEY",
        "DTM_API_SECONDARY_KEY",
    ):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.delenv("RESOLVER_SKIP_DTM", raising=False)


def _override_dtm_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    resolver_root = repo_root / "resolver"
    diagnostics_root = repo_root / "diagnostics"
    ingestion_diagnostics = diagnostics_root / "ingestion"
    dtm_dir = ingestion_diagnostics / "dtm"
    staging_dir = resolver_root / "staging"
    out_path = staging_dir / "dtm_displacement.csv"

    path_overrides = {
        "_REPO_ROOT": repo_root,
        "REPO_ROOT": repo_root,
        "RESOLVER_ROOT": resolver_root,
        "_LEGACY_DIAGNOSTICS_DIR": diagnostics_root / "legacy",
        "CONNECTORS_REPORT": ingestion_diagnostics / "connectors_report.jsonl",
        "LEGACY_CONNECTORS_REPORT": diagnostics_root / "connectors_report.jsonl",
        "RUN_DETAILS_PATH": dtm_dir / "dtm_run.json",
        "DTM_HTTP_LOG_PATH": dtm_dir / "dtm_http.ndjson",
        "HTTP_TRACE_PATH": dtm_dir / "dtm_http.ndjson",
        "DTM_DIAGNOSTICS_DIR": dtm_dir,
        "DIAGNOSTICS_DIR": ingestion_diagnostics,
        "DIAGNOSTICS_ROOT": ingestion_diagnostics,
        "DTM_RAW_DIR": dtm_dir / "raw",
        "DTM_METRICS_DIR": dtm_dir / "metrics",
        "DTM_SAMPLES_DIR": dtm_dir / "samples",
        "DTM_LOG_DIR": dtm_dir / "logs",
        "DIAGNOSTICS_RAW_DIR": ingestion_diagnostics / "raw",
        "DIAGNOSTICS_METRICS_DIR": ingestion_diagnostics / "metrics",
        "DIAGNOSTICS_SAMPLES_DIR": ingestion_diagnostics / "samples",
        "DIAGNOSTICS_LOG_DIR": ingestion_diagnostics / "logs",
        "OUT_PATH": out_path,
        "OUT_DIR": staging_dir,
        "OUTPUT_PATH": out_path,
        "DEFAULT_OUTPUT": out_path,
        "META_PATH": out_path.with_suffix(out_path.suffix + ".meta.json"),
        "API_RESPONSE_SAMPLE_PATH": dtm_dir / "dtm_api_response_sample.json",
        "API_SAMPLE_PATH": dtm_dir / "dtm_api_sample.json",
        "API_REQUEST_PATH": dtm_dir / "dtm_api_request.json",
        "DISCOVERY_SNAPSHOT_PATH": dtm_dir / "discovery_countries.csv",
        "DISCOVERY_FAIL_PATH": dtm_dir / "discovery_fail.json",
        "DISCOVERY_RAW_JSON_PATH": dtm_dir / "raw" / "dtm_countries.json",
        "PER_COUNTRY_METRICS_PATH": dtm_dir / "metrics" / "dtm_per_country.jsonl",
        "SAMPLE_ROWS_PATH": dtm_dir / "dtm_sample.csv",
        "DTM_CLIENT_LOG_PATH": dtm_dir / "logs" / "dtm_client.log",
        "RESCUE_PROBE_PATH": dtm_dir / "rescue_probe.json",
        "METRICS_SUMMARY_PATH": dtm_dir / "metrics" / "metrics.json",
        "INGESTION_SUMMARY_PATH": dtm_dir / "summary.json",
    }

    for name, value in path_overrides.items():
        monkeypatch.setattr(dtm_client, name, value, raising=False)
        if isinstance(value, Path):
            value.parent.mkdir(parents=True, exist_ok=True)

    return out_path


def test_dtm_client_without_credentials_produces_header_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out_path = _override_dtm_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(dtm_client, "OFFLINE", False, raising=False)

    exit_code = dtm_client.main([])
    assert exit_code == 0

    assert out_path.exists()
    with out_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0] == dtm_client.COLUMNS

    summary_path = dtm_client.INGESTION_SUMMARY_PATH
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "skipped"
    assert summary["reason"] == "auth_missing"
    assert summary["mode"] == "auth_missing"
    assert summary["rows_out"] == 0
    assert Path(summary["output_path"]) == out_path

    connectors_new = dtm_client.CONNECTORS_REPORT
    connectors_legacy = dtm_client.LEGACY_CONNECTORS_REPORT
    for report_path in (connectors_new, connectors_legacy):
        assert report_path.exists()
        lines = [line for line in report_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert lines, f"no records written to {report_path}"
        record = json.loads(lines[-1])
        assert record["status"] == "skipped"
        assert record["reason"] == "auth_missing"
        assert record.get("mode") in {"auth_missing", "skip"}
        counts = record.get("counts", {})
        assert counts.get("written") == 0
