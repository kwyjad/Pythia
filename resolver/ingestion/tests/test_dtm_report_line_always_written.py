"""Ensure the DTM connector records diagnostics even when individual countries fail."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pytest
import yaml

from resolver.ingestion import dtm_client


@pytest.fixture()
def patched_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Dict[str, Path]:
    diagnostics_dir = tmp_path / "diagnostics" / "ingestion"
    out_path = tmp_path / "outputs" / "dtm.csv"
    mappings = {
        "CONFIG_PATH": tmp_path / "dtm.yml",
        "LEGACY_CONFIG_PATH": tmp_path / "dtm.yml",
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
    monkeypatch.setenv("DTM_CONFIG_PATH", str(mappings["CONFIG_PATH"]))
    return mappings


def write_min_config(path: Path) -> None:
    payload = {"enabled": True, "api": {"admin_levels": ["admin0"]}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def read_report(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines, "expected report lines"
    return json.loads(lines[-1])


def test_partial_failures_recorded(patched_paths: Dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    write_min_config(patched_paths["CONFIG_PATH"])

    row: List[object] = [
        "dtm",
        "SOM",
        "",
        "evt",
        "2024-01-15",
        "2024-01-01",
        "flow",
        25,
        "people",
        "method",
        "medium",
        "raw",
        "{}",
    ]

    def fake_build_rows(*_: object, **__: object) -> tuple[list[List[object]], dict]:
        summary = {
            "extras": {
                "effective_params": {
                    "resource": "dtmapi",
                    "admin_levels": ["admin0"],
                    "countries_requested": [],
                    "countries": ["Somalia"],
                    "operations": None,
                    "from": "2024-01-01",
                    "to": "2024-02-01",
                    "no_date_filter": False,
                    "per_page": None,
                    "max_pages": None,
                    "country_mode": "ALL",
                    "discovered_countries_count": 1,
                    "idp_aliases": ["TotalIDPs", "IDPTotal", "numPresentIdpInd"],
                },
                "per_country_counts": [
                    {"country": "Somalia", "level": "admin0", "rows": 1, "window": "2024-01-01->2024-02-01"}
                ],
                "failures": [
                    {"country": "Kenya", "level": "admin0", "operation": None, "error": "RuntimeError"}
                ],
            },
            "rows": {"fetched": 1, "normalized": 1, "written": 1},
            "countries": {"requested": [], "resolved": ["Somalia"]},
        }
        return [row], summary

    monkeypatch.setenv("DTM_API_KEY", "dummy")
    monkeypatch.setattr(dtm_client, "build_rows", fake_build_rows)

    exit_code = dtm_client.main([])

    assert exit_code == 0
    report = read_report(patched_paths["CONNECTORS_REPORT"])
    assert report["status"].startswith("ok")
    assert report["extras"]["failures"] == [
        {"country": "Kenya", "level": "admin0", "operation": None, "error": "RuntimeError"}
    ]
    meta_payload = json.loads(patched_paths["META_PATH"].read_text(encoding="utf-8"))
    assert meta_payload["failures"] == [
        {"country": "Kenya", "level": "admin0", "operation": None, "error": "RuntimeError"}
    ]
    assert meta_payload["per_country_counts"][0]["country"] == "Somalia"
