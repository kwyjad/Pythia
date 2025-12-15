# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import json
import socket
import sys
from pathlib import Path

import pytest

from resolver.ingestion import dtm_client


@pytest.fixture(autouse=True)
def reset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DTM_API_KEY", raising=False)
    monkeypatch.delenv("DTM_CONFIG_PATH", raising=False)
    monkeypatch.delenv("RESOLVER_SKIP_DTM", raising=False)
    monkeypatch.delenv("DTM_OFFLINE_SMOKE", raising=False)
    monkeypatch.delenv("EMPTY_POLICY", raising=False)


@pytest.fixture()
def patch_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Path]:
    staging_dir = tmp_path / "resolver" / "staging"
    out_path = staging_dir / "dtm_displacement.csv"
    diagnostics_dir = tmp_path / "diagnostics" / "ingestion"
    mappings = {
        "OUT_PATH": out_path,
        "OUT_DIR": staging_dir,
        "OUTPUT_PATH": out_path,
        "DEFAULT_OUTPUT": out_path,
        "META_PATH": out_path.with_suffix(out_path.suffix + ".meta.json"),
        "DIAGNOSTICS_DIR": diagnostics_dir,
        "DTM_DIAGNOSTICS_DIR": diagnostics_dir / "dtm",
        "CONNECTORS_REPORT": diagnostics_dir / "connectors_report.jsonl",
        "RUN_DETAILS_PATH": diagnostics_dir / "dtm" / "dtm_run.json",
        "RESCUE_PROBE_PATH": diagnostics_dir / "dtm" / "rescue_probe.json",
    }
    for name, value in mappings.items():
        monkeypatch.setattr(dtm_client, name, value)
    return mappings


def _config_with_countries(path: str) -> dtm_client.ConfigDict:
    cfg = dtm_client.ConfigDict(
        {
            "enabled": True,
            "api": {"endpoint": "https://dtmapi.iom.int", "countries": ["Sudan", "DR Congo"]},
        }
    )
    cfg._source_path = path
    cfg._source_exists = True
    cfg._source_sha256 = "feedbeefcafe"
    cfg._config_parse = {
        "countries": ["Sudan", "DR Congo"],
        "countries_count": 2,
        "countries_preview": ["Sudan", "DR Congo"],
        "countries_mode": "explicit_config",
    }
    return cfg


def test_preflight_connect_timeout_soft_rescue(
    monkeypatch: pytest.MonkeyPatch, patch_paths: dict[str, Path]
) -> None:
    monkeypatch.setenv("DTM_API_KEY", "fake")
    monkeypatch.setenv("EMPTY_POLICY", "allow")
    monkeypatch.setattr(dtm_client, "load_config", lambda: _config_with_countries("resolver/config/dtm.yml"))
    monkeypatch.setattr(
        dtm_client,
        "_preflight_dependencies",
        lambda: (
            {"python": sys.version.split()[0], "executable": sys.executable, "packages": [], "missing": []},
            True,
        ),
    )

    def raise_timeout(*_: object, **__: object) -> None:
        raise socket.timeout("connect timed out")

    monkeypatch.setattr(dtm_client.socket, "create_connection", raise_timeout)

    rc = dtm_client.main(["--soft-timeouts"])
    assert rc == 0

    csv_lines = patch_paths["OUT_PATH"].read_text(encoding="utf-8").strip().splitlines()
    assert csv_lines == ["CountryISO3,ReportingDate,idp_count"]

    meta_payload = json.loads(patch_paths["META_PATH"].read_text(encoding="utf-8"))
    assert meta_payload["row_count"] == 0
    assert meta_payload.get("status") == "ok"
    assert meta_payload.get("zero_rows_reason") == "connect_timeout"
    assert meta_payload.get("http", {}).get("timeout") == 1
    assert meta_payload.get("effective_params", {}).get("countries") == ["Sudan", "DR Congo"]

    run_payload = json.loads(patch_paths["RUN_DETAILS_PATH"].read_text(encoding="utf-8"))
    assert run_payload["status"] == "ok"
    assert run_payload["reason"] == "connect_timeout"
    extras = run_payload["extras"]
    assert extras.get("zero_rows_reason") == "connect_timeout"
    discovery = extras.get("discovery") or {}
    assert discovery.get("used_stage") == "explicit_config"
    assert discovery.get("configured_labels") == ["Sudan", "DR Congo"]
    assert extras.get("soft_timeouts") is True

    connectors_report = [
        json.loads(line)
        for line in patch_paths["CONNECTORS_REPORT"].read_text(encoding="utf-8").strip().splitlines()
        if line.strip()
    ]
    assert connectors_report, "expected connectors report entries"
    zero_reasons = {entry.get("extras", {}).get("zero_rows_reason") for entry in connectors_report}
    assert "connect_timeout" in zero_reasons

    rescue_probe = json.loads(patch_paths["RESCUE_PROBE_PATH"].read_text(encoding="utf-8"))
    assert rescue_probe.get("stage") == "preflight_tls"
