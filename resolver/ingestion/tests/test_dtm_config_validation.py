from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from resolver.ingestion import dtm_client


def _prepare_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Path]:
    out_path = tmp_path / "dtm.csv"
    diagnostics_dir = tmp_path / "diagnostics" / "ingestion"
    paths = {
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
    for name, value in paths.items():
        monkeypatch.setattr(dtm_client, name, value)
    return paths


def _write_config(path: Path, sources: list[dict[str, object]]) -> None:
    payload = {"enabled": True, "sources": sources}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _read_report(path: Path) -> dict:
    content = path.read_text(encoding="utf-8").strip().splitlines()
    assert content, "expected at least one diagnostics record"
    return json.loads(content[-1])


def test_dtm_config_preflight_skips_invalid_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _prepare_paths(monkeypatch, tmp_path)
    valid_csv = tmp_path / "valid.csv"
    valid_csv.write_text("country_iso3,date,value\nKEN,2023-01-15,10\n", encoding="utf-8")
    sources = [
        {"name": "missing"},
        {"name": "blank", "id_or_path": ""},
        {"name": "valid", "id_or_path": str(valid_csv)},
    ]
    _write_config(paths["CONFIG_PATH"], sources)
    monkeypatch.delenv("DTM_STRICT", raising=False)

    exit_code = dtm_client.main([])

    assert exit_code == 0
    issues_path = paths["CONFIG_ISSUES_PATH"]
    assert issues_path.exists()
    issues = json.loads(issues_path.read_text(encoding="utf-8"))
    assert issues["summary"]["invalid"] == 2
    assert all(item["error"] == "missing id_or_path" for item in issues["invalid"])

    report = _read_report(paths["CONNECTORS_REPORT"])
    assert report["status"] == "ok"
    assert report["reason"] == "missing id_or_path"
    extras = report["extras"]
    assert extras["invalid_sources"] == 2
    assert extras["valid_sources"] == 1
    assert extras["rows_total"] > 0
    assert extras["config_issues_path"] == str(issues_path)


def test_dtm_strict_mode_exits_with_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _prepare_paths(monkeypatch, tmp_path)
    valid_csv = tmp_path / "valid.csv"
    valid_csv.write_text("country_iso3,date,value\nKEN,2023-01-15,10\n", encoding="utf-8")
    sources = [
        {"name": "missing"},
        {"name": "blank", "id_or_path": ""},
        {"name": "valid", "id_or_path": str(valid_csv)},
    ]
    _write_config(paths["CONFIG_PATH"], sources)
    monkeypatch.setenv("DTM_STRICT", "1")

    exit_code = dtm_client.main([])

    assert exit_code == 2
    issues_path = paths["CONFIG_ISSUES_PATH"]
    assert issues_path.exists()
    issues = json.loads(issues_path.read_text(encoding="utf-8"))
    assert issues["summary"]["invalid"] == 2

    report = _read_report(paths["CONNECTORS_REPORT"])
    assert report["status"] == "error"
    assert report["reason"] == "missing id_or_path"
    extras = report["extras"]
    assert extras["invalid_sources"] == 2
    assert extras["rows_total"] == 0
    assert extras["config_issues_path"] == str(issues_path)
