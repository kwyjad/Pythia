"""Ensure diagnostics are emitted even when the connector fails."""

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
    mappings = {
        "CONFIG_PATH": tmp_path / "dtm.yml",
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


def write_min_config(path: Path) -> None:
    payload = {"enabled": True, "api": {}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def read_report(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines, "expected report lines"
    return json.loads(lines[-1])


def test_report_line_written_on_exception(patched_paths: Dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    write_min_config(patched_paths["CONFIG_PATH"])

    def explode(*_: object, **__: object) -> tuple[list[list[object]], dict]:
        raise RuntimeError("kaboom")

    monkeypatch.setenv("DTM_API_KEY", "dummy")
    monkeypatch.setattr(dtm_client, "build_rows", explode)

    exit_code = dtm_client.main([])

    assert exit_code == 1
    report = read_report(patched_paths["CONNECTORS_REPORT"])
    assert report["status"] == "error"
    assert report["reason"] == "exception: RuntimeError"
    extras = report["extras"]
    assert extras["rows_total"] == 0
    run_payload = json.loads(patched_paths["RUN_DETAILS_PATH"].read_text(encoding="utf-8"))
    assert run_payload["status"] == "error"
    assert run_payload["reason"] == "exception: RuntimeError"
    assert run_payload["rows"]["written"] == 0
