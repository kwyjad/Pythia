"""Authentication retry behaviour tests for the DTM connector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd
import pytest
import yaml

from resolver.ingestion import dtm_client


class UnauthorizedClient:
    def __init__(self, config: dict, *, subscription_key: str | None = None) -> None:
        self.config = config
        self.subscription_key = subscription_key

    def get_all_countries(self, *_: object, **__: object) -> pd.DataFrame:
        return pd.DataFrame([{"CountryName": "Kenya", "ISO3": "KEN"}])

    def get_idp_admin0(self, **_: object) -> pd.DataFrame:
        raise dtm_client.DTMUnauthorizedError(401, "unauthorized")

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
        "DTM_DIAGNOSTICS_DIR": diagnostics_dir / "dtm",
        "DISCOVERY_FAIL_PATH": diagnostics_dir / "dtm" / "discovery_fail.json",
        "DTM_HTTP_LOG_PATH": diagnostics_dir / "dtm" / "dtm_http.ndjson",
    }
    for name, value in mappings.items():
        monkeypatch.setattr(dtm_client, name, value)
    return mappings


def write_config(path: Path) -> None:
    payload = {
        "enabled": True,
        "api": {"countries": ["KEN"], "admin_levels": ["admin0"]},
        "output": {"measure": "flow"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_invalid_key_records_failure(monkeypatch: pytest.MonkeyPatch, patched_paths: Dict[str, Path]) -> None:
    write_config(patched_paths["CONFIG_PATH"])
    monkeypatch.setenv("DTM_API_KEY", "invalid")
    monkeypatch.setattr(dtm_client, "DTMApiClient", UnauthorizedClient)

    exit_code = dtm_client.main([])

    assert exit_code == 1
    fail_path = patched_paths["DISCOVERY_FAIL_PATH"]
    assert fail_path.exists()
    payload = json.loads(fail_path.read_text(encoding="utf-8"))
    assert payload["reason"] == "invalid_key"
    report_lines = patched_paths["CONNECTORS_REPORT"].read_text(encoding="utf-8").strip().splitlines()
    report = json.loads(report_lines[-1])
    assert report["status"] == "error"
