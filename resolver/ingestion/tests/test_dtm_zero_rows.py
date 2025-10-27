"""Zero-row handling for the DTM connector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd
import pytest
import yaml

from resolver.ingestion import dtm_client


class DummyClient:
    def __init__(self, config: dict, *, subscription_key: str | None = None) -> None:
        self.config = config

    def get_all_countries(self, *_: object, **__: object) -> pd.DataFrame:
        return pd.DataFrame([{"CountryName": "Kenya", "ISO3": "KEN"}])

    def get_idp_admin0(self, **_: object) -> pd.DataFrame:
        return pd.DataFrame()

    def get_idp_admin1(self, **_: object) -> pd.DataFrame:
        return pd.DataFrame()

    def get_idp_admin2(self, **_: object) -> pd.DataFrame:
        return pd.DataFrame()


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
        "API_RESPONSE_SAMPLE_PATH": diagnostics_dir / "dtm_api_response_sample.json",
    }
    for name, value in mappings.items():
        monkeypatch.setattr(dtm_client, name, value)
    monkeypatch.setenv("DTM_CONFIG_PATH", str(mappings["CONFIG_PATH"]))
    return mappings


def write_config(path: Path) -> None:
    payload = {"enabled": True, "api": {"countries": ["KEN"], "admin_levels": ["admin0"]}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def read_report(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    return json.loads(lines[-1])


def run_zero_row(monkeypatch: pytest.MonkeyPatch, patched_paths: Dict[str, Path], strict: bool = False) -> int:
    write_config(patched_paths["CONFIG_PATH"])
    monkeypatch.setenv("DTM_API_KEY", "dummy")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setattr(dtm_client, "DTMApiClient", DummyClient)
    args = ["--strict-empty"] if strict else []
    return dtm_client.main(args)


def test_zero_rows_ok_empty(monkeypatch: pytest.MonkeyPatch, patched_paths: Dict[str, Path]) -> None:
    exit_code = run_zero_row(monkeypatch, patched_paths, strict=False)

    assert exit_code == 0
    report = read_report(patched_paths["CONNECTORS_REPORT"])
    assert report["status"] == "ok"
    assert patched_paths["API_SAMPLE_PATH"].exists()
    run_payload = json.loads(patched_paths["RUN_DETAILS_PATH"].read_text(encoding="utf-8"))
    assert run_payload["status"] == "ok-empty"
    assert run_payload["rows"]["written"] == 0


def test_zero_rows_strict_empty(monkeypatch: pytest.MonkeyPatch, patched_paths: Dict[str, Path]) -> None:
    exit_code = run_zero_row(monkeypatch, patched_paths, strict=True)

    assert exit_code == 3
    report = read_report(patched_paths["CONNECTORS_REPORT"])
    assert report["status"] == "ok"
    assert "strict" in report["reason"].lower() or report["extras"].get("strict_empty")
