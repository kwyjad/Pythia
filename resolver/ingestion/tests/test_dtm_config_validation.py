"""DTM configuration validation tests for API-only mode."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import pandas as pd
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
        "LEGACY_CONFIG_PATH": tmp_path / "dtm.yml",
        "OUT_PATH": out_path,
        "OUT_DIR": out_path.parent,
        "OUTPUT_PATH": out_path,
        "DEFAULT_OUTPUT": out_path,
        "META_PATH": out_path.with_suffix(out_path.suffix + ".meta.json"),
        "DIAGNOSTICS_DIR": diagnostics_dir,
        "DTM_DIAGNOSTICS_DIR": diagnostics_dir / "dtm",
        "CONNECTORS_REPORT": diagnostics_dir / "connectors_report.jsonl",
        "RUN_DETAILS_PATH": run_path,
        "API_REQUEST_PATH": api_request,
        "API_SAMPLE_PATH": sample_path,
        "DISCOVERY_SNAPSHOT_PATH": diagnostics_dir / "dtm" / "discovery_countries.csv",
        "DISCOVERY_FAIL_PATH": diagnostics_dir / "dtm" / "discovery_fail.json",
        "DTM_HTTP_LOG_PATH": diagnostics_dir / "dtm" / "dtm_http.ndjson",
    }
    for name, value in mappings.items():
        monkeypatch.setattr(dtm_client, name, value)
    monkeypatch.setenv("DTM_CONFIG_PATH", str(mappings["CONFIG_PATH"]))
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


def test_config_country_list_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    config = {
        "enabled": True,
        "api": {"admin_levels": ["admin0"], "countries": ["Kenya"]},
    }

    class DiscoveryClient:
        def __init__(self, *_: object, **__: object) -> None:
            self.rate_limit_delay = 0
            self.timeout = 0
            self.client = SimpleNamespace(
                get_all_countries=lambda: pd.DataFrame(
                    [
                        {"CountryName": "Somalia"},
                        {"CountryName": "Ethiopia"},
                        {"CountryName": "Somalia"},
                    ]
                )
            )

        def get_countries(self, *_: object, **__: object) -> pd.DataFrame:
            return pd.DataFrame([{"CountryName": "Fallback"}])

        def get_idp_admin0(self, **__: object) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "CountryName": ["Somalia"],
                    "ReportingDate": ["2024-01-15"],
                    "TotalIDPs": [50],
                }
            )

        def get_idp_admin1(self, **__: object) -> pd.DataFrame:
            return pd.DataFrame()

        def get_idp_admin2(self, **__: object) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setenv("DTM_API_KEY", "primary")
    monkeypatch.setattr(dtm_client, "DTMApiClient", DiscoveryClient)

    rows, summary = dtm_client.build_rows(
        config,
        no_date_filter=False,
        window_start="2024-01-01",
        window_end="2024-02-01",
        http_counts={},
    )

    assert rows, "expected at least one row"
    assert summary["countries"]["requested"] == ["Kenya"]
    assert summary["countries"]["resolved"] == ["Ethiopia", "Somalia"]
    assert summary["api"]["query_params"]["country_mode"] == "ALL"
    effective = summary["extras"]["effective_params"]
    assert effective["country_mode"] == "ALL"
    assert effective["discovered_countries_count"] == 2
    assert effective["countries_requested"] == ["Kenya"]
    assert effective["countries"] == ["Ethiopia", "Somalia"]
    per_country = summary["extras"]["per_country_counts"]
    assert {entry["country"] for entry in per_country} == {"Ethiopia", "Somalia"}
    assert summary["extras"]["discovery"]["total_countries"] == 2
    diagnostics_payload = summary["extras"].get("diagnostics", {})
    assert diagnostics_payload.get("http_trace")


def test_discovery_empty_failfast(
    monkeypatch: pytest.MonkeyPatch, patched_paths: Dict[str, Path]
) -> None:
    monkeypatch.setenv("DTM_API_KEY", "primary")

    class EmptyDiscoveryClient:
        def __init__(self, *_: object, **__: object) -> None:
            self.rate_limit_delay = 0
            self.timeout = 0
            self.client = SimpleNamespace(get_all_countries=lambda: pd.DataFrame(columns=["CountryName"]))

        def get_idp_admin0(self, **__: object) -> pd.DataFrame:
            return pd.DataFrame()

        def get_idp_admin1(self, **__: object) -> pd.DataFrame:
            return pd.DataFrame()

        def get_idp_admin2(self, **__: object) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(dtm_client, "DTMApiClient", EmptyDiscoveryClient)

    with pytest.raises(SystemExit) as excinfo:
        dtm_client.build_rows(
            {"enabled": True, "api": {"admin_levels": ["admin0"]}},
            no_date_filter=False,
            window_start=None,
            window_end=None,
            http_counts={},
        )

    assert excinfo.value.code == 2
    assert patched_paths["DISCOVERY_FAIL_PATH"].exists()
