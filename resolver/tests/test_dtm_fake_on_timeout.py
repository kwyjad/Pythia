"""Tests for the DTM fake-on-timeout rescue mode."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest
import requests

from resolver.ingestion import dtm_client


@pytest.fixture(autouse=True)
def reset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "DTM_API_KEY",
        "DTM_CONFIG_PATH",
        "RESOLVER_SKIP_DTM",
        "DTM_OFFLINE_SMOKE",
        "DTM_FAKE_ON_TIMEOUT",
        "DTM_FORCE_FAKE",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def patch_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Path]:
    diagnostics_dir = tmp_path / "diagnostics" / "ingestion"
    dtm_dir = diagnostics_dir / "dtm"
    out_path = tmp_path / "outputs" / "dtm_displacement.csv"
    mappings = {
        "OUT_PATH": out_path,
        "OUT_DIR": out_path.parent,
        "OUTPUT_PATH": out_path,
        "DEFAULT_OUTPUT": out_path,
        "META_PATH": out_path.with_suffix(out_path.suffix + ".meta.json"),
        "DIAGNOSTICS_DIR": diagnostics_dir,
        "DTM_DIAGNOSTICS_DIR": dtm_dir,
        "CONNECTORS_REPORT": diagnostics_dir / "connectors_report.jsonl",
        "RUN_DETAILS_PATH": dtm_dir / "dtm_run.json",
        "HTTP_TRACE_PATH": dtm_dir / "dtm_http.ndjson",
        "DTM_HTTP_LOG_PATH": dtm_dir / "dtm_http.ndjson",
    }
    for name, value in mappings.items():
        monkeypatch.setattr(dtm_client, name, value)
    return mappings


def _fake_config(path: str) -> dtm_client.ConfigDict:
    cfg = dtm_client.ConfigDict(
        {
            "enabled": True,
            "api": {
                "countries": ["Sudan", "DR Congo"],
                "admin_levels": ["admin0"],
            },
        }
    )
    cfg._source_path = path
    cfg._source_exists = True
    cfg._source_sha256 = "deadbeefcafecafe"
    cfg._config_parse = {
        "countries": ["Sudan", "DR Congo"],
        "admin_levels": ["admin0"],
        "countries_count": 2,
        "countries_mode": "explicit_config",
    }
    return cfg


def test_fake_on_timeout_writes_staging(
    monkeypatch: pytest.MonkeyPatch,
    patch_paths: dict[str, Path],
) -> None:
    monkeypatch.setenv("DTM_API_KEY", "fake")
    monkeypatch.setenv("DTM_FAKE_ON_TIMEOUT", "1")
    monkeypatch.setattr(dtm_client, "load_config", lambda: _fake_config("inline.yml"))
    monkeypatch.setattr(
        dtm_client,
        "_preflight_dependencies",
        lambda: ({"python": sys.version.split()[0], "executable": sys.executable}, True),
    )
    monkeypatch.setattr(
        dtm_client,
        "resolve_ingestion_window",
        lambda: (date(2025, 5, 1), date(2025, 10, 31)),
    )

    def raise_connect_timeout(*_: object, **__: object) -> None:
        raise requests.exceptions.ConnectTimeout("connect")

    monkeypatch.setattr(dtm_client, "build_rows", raise_connect_timeout)

    exit_code = dtm_client.main([])
    assert exit_code == 0

    csv_path = patch_paths["OUT_PATH"]
    meta_path = patch_paths["META_PATH"]
    run_details_path = patch_paths["RUN_DETAILS_PATH"]

    assert csv_path.exists()
    csv_lines = [line for line in csv_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert csv_lines == ["CountryISO3,ReportingDate,idp_count"]

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta.get("row_count") == 0
    assert meta.get("status") == "ok"
    assert meta.get("zero_rows_reason") == "connect_timeout"
    assert meta.get("http", {}).get("timeout") == 1
    assert meta.get("fake_data") is None

    run_payload = json.loads(run_details_path.read_text(encoding="utf-8"))
    assert run_payload["status"] == "ok"
    assert run_payload.get("reason") == "connect_timeout"
    extras = run_payload.get("extras", {})
    assert extras.get("soft_timeout_applied") is True
    assert extras.get("zero_rows_reason") == "connect_timeout"
    assert extras.get("http", {}).get("timeouts") == 1
