"""DTM configuration validation tests for API-only mode."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pytest
import yaml

from resolver.ingestion import dtm_client


@pytest.fixture()
def patched_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Dict[str, Path]:
    diagnostics_dir = tmp_path / "diagnostics" / "ingestion"
    out_path = tmp_path / "outputs" / "dtm.csv"
    api_request = diagnostics_dir / "dtm_api_request.json"
    run_path = diagnostics_dir / "dtm_run.json"
    sample_path = diagnostics_dir / "dtm_api_sample.json"
    mappings = {
        "CONFIG_PATH": tmp_path / "dtm.yml",
        "OUT_PATH": out_path,
        "OUT_DIR": out_path.parent,
        "OUTPUT_PATH": out_path,
        "DEFAULT_OUTPUT": out_path,
        "META_PATH": out_path.with_suffix(out_path.suffix + ".meta.json"),
        "DIAGNOSTICS_DIR": diagnostics_dir,
        "CONNECTORS_REPORT": diagnostics_dir / "connectors_report.jsonl",
        "RUN_DETAILS_PATH": run_path,
        "API_REQUEST_PATH": api_request,
        "API_SAMPLE_PATH": sample_path,
    }
    for name, value in mappings.items():
        monkeypatch.setattr(dtm_client, name, value)
    return mappings


def write_config(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def read_report(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines, "expected connectors_report.jsonl to have at least one entry"
    return json.loads(lines[-1])


def test_missing_api_block_aborts(patched_paths: Dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    write_config(patched_paths["CONFIG_PATH"], {"enabled": True})
    monkeypatch.delenv("DTM_API_KEY", raising=False)

    exit_code = dtm_client.main([])

    assert exit_code == 2
    report = read_report(patched_paths["CONNECTORS_REPORT"])
    assert report["status"] == "error"
    assert "API-only" in report["reason"]
    run_payload = json.loads(patched_paths["RUN_DETAILS_PATH"].read_text(encoding="utf-8"))
    assert run_payload["status"] == "error"
    assert run_payload["rows"]["written"] == 0


def test_disabled_configuration_skips(patched_paths: Dict[str, Path]) -> None:
    write_config(patched_paths["CONFIG_PATH"], {"enabled": False, "api": {}})

    exit_code = dtm_client.main([])

    assert exit_code == 0
    report = read_report(patched_paths["CONNECTORS_REPORT"])
    assert report["status"] == "skipped"
    assert report["reason"] == "disabled via config"
    csv_path = patched_paths["OUT_PATH"]
    assert csv_path.exists()
    content = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 1, "header-only output expected"


def test_env_skip_overrides(patched_paths: Dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    write_config(patched_paths["CONFIG_PATH"], {"enabled": True, "api": {}})
    monkeypatch.setenv("RESOLVER_SKIP_DTM", "1")

    exit_code = dtm_client.main([])

    assert exit_code == 0
    report = read_report(patched_paths["CONNECTORS_REPORT"])
    assert report["status"] == "skipped"
    assert report["reason"] == "disabled via RESOLVER_SKIP_DTM"
