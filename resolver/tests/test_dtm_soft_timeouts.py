# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import requests

from resolver.ingestion import dtm_client


@pytest.fixture(autouse=True)
def reset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DTM_API_KEY", raising=False)
    monkeypatch.delenv("DTM_CONFIG_PATH", raising=False)
    monkeypatch.delenv("RESOLVER_SKIP_DTM", raising=False)
    monkeypatch.delenv("DTM_OFFLINE_SMOKE", raising=False)


@pytest.fixture()
def patch_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Path]:
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
        "CONNECTORS_REPORT": diagnostics_dir / "connectors_report.jsonl",
        "RUN_DETAILS_PATH": diagnostics_dir / "dtm" / "dtm_run.json",
    }
    for name, value in mappings.items():
        monkeypatch.setattr(dtm_client, name, value)
    return mappings


def _fake_config(path: str) -> dtm_client.ConfigDict:
    cfg = dtm_client.ConfigDict({"enabled": True, "api": {"endpoint": "dtmapi"}})
    cfg._source_path = path
    cfg._source_exists = True
    cfg._source_sha256 = "deadbeefcafe"
    return cfg


def test_soft_timeouts_yield_ok_empty(monkeypatch: pytest.MonkeyPatch, patch_paths: dict[str, Path]) -> None:
    monkeypatch.setenv("DTM_API_KEY", "fake")
    monkeypatch.setattr(dtm_client, "load_config", lambda: _fake_config("custom.yml"))
    monkeypatch.setattr(
        dtm_client,
        "_preflight_dependencies",
        lambda: ({"python": sys.version.split()[0], "executable": sys.executable}, True),
    )

    def raise_connect_timeout(*_: object, **__: object):
        raise requests.exceptions.ConnectTimeout("connect")

    monkeypatch.setattr(dtm_client, "build_rows", raise_connect_timeout)

    rc = dtm_client.main(["--soft-timeouts"])
    assert rc == 0

    report_payloads = [
        json.loads(line)
        for line in patch_paths["CONNECTORS_REPORT"].read_text(encoding="utf-8").strip().splitlines()
        if line.strip()
    ]
    assert report_payloads
    zero_reasons = {entry.get("extras", {}).get("zero_rows_reason") for entry in report_payloads}
    assert "connect_timeout" in zero_reasons
    assert any(entry.get("status") in {"ok", "ok-empty"} for entry in report_payloads)
    assert any(entry.get("extras", {}).get("status_raw") in {"ok", "ok-empty"} for entry in report_payloads)
    assert all(entry.get("extras", {}).get("exit_code", 0) == 0 for entry in report_payloads)

    csv_text = patch_paths["OUT_PATH"].read_text(encoding="utf-8").splitlines()
    assert csv_text == ["CountryISO3,ReportingDate,idp_count"]

    meta_payload = json.loads(patch_paths["META_PATH"].read_text(encoding="utf-8"))
    assert meta_payload["row_count"] == 0
    assert meta_payload.get("status") == "ok"
    assert meta_payload.get("zero_rows_reason") == "connect_timeout"
    assert meta_payload.get("http", {}).get("timeout") == 1

    run_payload = json.loads(patch_paths["RUN_DETAILS_PATH"].read_text(encoding="utf-8"))
    assert run_payload["status"] == "ok"
    assert run_payload["reason"] == "connect_timeout"
    assert run_payload["extras"].get("zero_rows_reason") == "connect_timeout"
    assert run_payload["extras"].get("http", {}).get("timeouts") == 1



def test_connect_timeout_without_soft_flag_errors(monkeypatch: pytest.MonkeyPatch, patch_paths: dict[str, Path]) -> None:
    monkeypatch.setenv("DTM_API_KEY", "fake")
    monkeypatch.setattr(dtm_client, "load_config", lambda: _fake_config("custom.yml"))
    monkeypatch.setattr(
        dtm_client,
        "_preflight_dependencies",
        lambda: ({"python": sys.version.split()[0], "executable": sys.executable}, True),
    )

    def raise_connect_timeout(*_: object, **__: object):
        raise requests.exceptions.ConnectTimeout("connect")

    monkeypatch.setattr(dtm_client, "build_rows", raise_connect_timeout)

    rc = dtm_client.main([])
    assert rc == 1

    report_lines = patch_paths["CONNECTORS_REPORT"].read_text(encoding="utf-8").strip().splitlines()
    assert len(report_lines) == 1
    payload = json.loads(report_lines[0])
    assert payload["status"] == "error"
    assert payload["extras"].get("status_raw") == "error"
    assert payload["extras"].get("zero_rows_reason") is None
