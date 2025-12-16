# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Validate public path constants for the DTM connector."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _import_module():
    return importlib.import_module("resolver.ingestion.dtm_client")


def test_path_constants_exposed():
    module = _import_module()
    expected_names = [
        "REPO_ROOT",
        "DIAGNOSTICS_DIR",
        "DTM_DIAGNOSTICS_DIR",
        "DIAGNOSTICS_RAW_DIR",
        "DIAGNOSTICS_METRICS_DIR",
        "DIAGNOSTICS_SAMPLES_DIR",
        "DIAGNOSTICS_LOG_DIR",
        "CONNECTORS_REPORT",
        "RUN_DETAILS_PATH",
        "DTM_HTTP_LOG_PATH",
        "DISCOVERY_SNAPSHOT_PATH",
        "DISCOVERY_FAIL_PATH",
        "DISCOVERY_RAW_JSON_PATH",
        "PER_COUNTRY_METRICS_PATH",
        "SAMPLE_ROWS_PATH",
        "DTM_CLIENT_LOG_PATH",
        "API_REQUEST_PATH",
        "API_SAMPLE_PATH",
        "API_RESPONSE_SAMPLE_PATH",
        "OUT_DIR",
        "OUT_PATH",
        "OUTPUT_PATH",
        "DEFAULT_OUTPUT",
        "META_PATH",
        "HTTP_TRACE_PATH",
    ]

    for name in expected_names:
        assert hasattr(module, name), f"missing constant: {name}"
        value = getattr(module, name)
        assert isinstance(
            value, Path
        ), f"constant {name} should be a pathlib.Path (got {type(value)!r})"


def test_paths_can_be_monkeypatched(tmp_path, monkeypatch):
    module = _import_module()

    diagnostics_root = tmp_path / "diagnostics" / "ingestion"
    dtm_diag = diagnostics_root / "dtm"
    staging_dir = tmp_path / "resolver" / "staging"

    overrides = {
        "DIAGNOSTICS_DIR": diagnostics_root,
        "DTM_DIAGNOSTICS_DIR": dtm_diag,
        "DIAGNOSTICS_RAW_DIR": diagnostics_root / "raw",
        "DIAGNOSTICS_METRICS_DIR": diagnostics_root / "metrics",
        "DIAGNOSTICS_SAMPLES_DIR": diagnostics_root / "samples",
        "DIAGNOSTICS_LOG_DIR": diagnostics_root / "logs",
        "DTM_HTTP_LOG_PATH": dtm_diag / "dtm_http.ndjson",
        "HTTP_TRACE_PATH": dtm_diag / "dtm_http.ndjson",
        "RUN_DETAILS_PATH": dtm_diag / "dtm_run.json",
        "DISCOVERY_SNAPSHOT_PATH": dtm_diag / "discovery_countries.csv",
        "DISCOVERY_FAIL_PATH": dtm_diag / "discovery_fail.json",
        "DISCOVERY_RAW_JSON_PATH": diagnostics_root / "raw" / "dtm_countries.json",
        "PER_COUNTRY_METRICS_PATH": diagnostics_root / "metrics" / "dtm_per_country.jsonl",
        "SAMPLE_ROWS_PATH": diagnostics_root / "samples" / "dtm_sample.csv",
        "DTM_CLIENT_LOG_PATH": diagnostics_root / "logs" / "dtm_client.log",
        "API_REQUEST_PATH": diagnostics_root / "dtm_api_request.json",
        "API_SAMPLE_PATH": diagnostics_root / "dtm_api_sample.json",
        "API_RESPONSE_SAMPLE_PATH": diagnostics_root / "dtm_api_response_sample.json",
        "OUT_DIR": staging_dir,
        "OUT_PATH": staging_dir / "dtm_displacement.csv",
        "OUTPUT_PATH": staging_dir / "dtm_displacement.csv",
        "DEFAULT_OUTPUT": staging_dir / "dtm_displacement.csv",
        "META_PATH": staging_dir / "dtm_displacement.csv.meta.json",
        "CONNECTORS_REPORT": diagnostics_root / "connectors_report.jsonl",
        "RESCUE_PROBE_PATH": dtm_diag / "rescue_probe.json",
    }

    for name, path in overrides.items():
        monkeypatch.setattr(module, name, path)

    module.ensure_zero_row_outputs(offline=True)

    csv_path = staging_dir / "dtm_displacement.csv"
    assert csv_path.exists()
    assert csv_path.read_text(encoding="utf-8").strip(), "header-only CSV should not be empty"

    assert (dtm_diag / "dtm_http.ndjson").exists()
