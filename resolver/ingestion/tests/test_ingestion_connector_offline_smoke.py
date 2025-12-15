# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Offline smoke tests for the DTM connector CLI."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from resolver.ingestion import dtm_client


@pytest.fixture()
def offline_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Path]:
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
        "DIAGNOSTICS_RAW_DIR": diagnostics_dir / "raw",
        "DIAGNOSTICS_METRICS_DIR": diagnostics_dir / "metrics",
        "DIAGNOSTICS_SAMPLES_DIR": diagnostics_dir / "samples",
        "DIAGNOSTICS_LOG_DIR": diagnostics_dir / "logs",
        "CONNECTORS_REPORT": diagnostics_dir / "connectors_report.jsonl",
        "RUN_DETAILS_PATH": diagnostics_dir / "dtm_run.json",
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
    for handler in list(dtm_client.LOG.handlers):
        if isinstance(handler, logging.FileHandler):
            dtm_client.LOG.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
    # reset logging guard so the patched path is used
    monkeypatch.setattr(dtm_client, "_FILE_LOGGING_INITIALIZED", False)
    return mappings


def test_offline_smoke_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
    offline_paths: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("DTM_API_KEY", raising=False)
    exit_code = dtm_client.main(["--offline-smoke"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "offline-smoke mode active" in captured.out

    csv_path = offline_paths["OUT_PATH"]
    assert csv_path.exists()
    content = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert content == [",".join(dtm_client.CANONICAL_HEADERS)]

    meta_path = offline_paths["META_PATH"]
    assert meta_path.exists()
    meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta_payload["diagnostics"]["mode"] == "offline-smoke"

    run_details = json.loads(offline_paths["RUN_DETAILS_PATH"].read_text(encoding="utf-8"))
    assert run_details["status"] == "offline"
    assert run_details["reason"] == "offline-smoke mode"
    assert run_details["extras"]["mode"] == "offline-smoke"

    report_lines = offline_paths["CONNECTORS_REPORT"].read_text(encoding="utf-8").strip().splitlines()
    assert report_lines
    report_payload = json.loads(report_lines[-1])
    assert report_payload["status"] == "skipped"
    assert report_payload["reason"] == "offline-smoke mode"
