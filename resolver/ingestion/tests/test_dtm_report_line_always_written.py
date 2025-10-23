from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from resolver.ingestion import dtm_client


def _prepare_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Path]:
    out_path = tmp_path / "dtm.csv"
    diagnostics_dir = tmp_path / "diagnostics" / "ingestion"
    mapping = {
        "CONFIG_PATH": tmp_path / "dtm.yml",
        "OUT_PATH": out_path,
        "OUT_DIR": out_path.parent,
        "OUTPUT_PATH": out_path,
        "DEFAULT_OUTPUT": out_path,
        "DIAGNOSTICS_DIR": diagnostics_dir,
        "CONNECTORS_REPORT": diagnostics_dir / "connectors_report.jsonl",
        "CONFIG_ISSUES_PATH": diagnostics_dir / "dtm_config_issues.json",
        "RESOLVED_SOURCES_PATH": diagnostics_dir / "dtm_sources_resolved.json",
    }
    for name, value in mapping.items():
        monkeypatch.setattr(dtm_client, name, value)
    return mapping


def _write_config(path: Path, *, source_path: Path) -> None:
    payload = {
        "enabled": True,
        "sources": [{"name": "broken", "id_or_path": str(source_path)}],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_diagnostics_written_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _prepare_paths(monkeypatch, tmp_path)
    missing_file = tmp_path / "does-not-exist.csv"
    _write_config(paths["CONFIG_PATH"], source_path=missing_file)
    monkeypatch.delenv("DTM_STRICT", raising=False)

    exit_code = dtm_client.main([])

    assert exit_code == 1
    report_path = paths["CONNECTORS_REPORT"]
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8").strip().splitlines()
    assert content
    payload = json.loads(content[-1])
    assert payload["status"] == "error"
    assert "missing id_or_path" not in payload["reason"]
    assert "DTM source not found" in payload["reason"]
    extras = payload["extras"]
    assert extras["rows_total"] == 0
    issues = json.loads(paths["CONFIG_ISSUES_PATH"].read_text(encoding="utf-8"))
    assert issues["summary"]["invalid"] == 0
