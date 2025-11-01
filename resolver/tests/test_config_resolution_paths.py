from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

import pytest
import yaml

from resolver.ingestion import dtm_client
from resolver.ingestion._shared import config_loader
from resolver.ingestion._shared.run_io import attach_config_source
from scripts.ci import summarize_connectors


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


@pytest.fixture()
def loader_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Tuple[Path, Path, Path]:
    repo_root = tmp_path / "repo"
    resolver_root = repo_root / "resolver"
    ingestion_dir = resolver_root / "ingestion" / "config"
    fallback_dir = resolver_root / "config"
    monkeypatch.setattr(config_loader, "INGESTION_CONFIG_ROOT", ingestion_dir)
    monkeypatch.setattr(config_loader, "LEGACY_CONFIG_ROOT", fallback_dir)
    monkeypatch.setattr(config_loader, "RESOLVER_ROOT", resolver_root)
    monkeypatch.setattr(config_loader, "REPO_ROOT", repo_root)
    monkeypatch.setattr(config_loader, "_LAST_RESULTS", {})
    return resolver_root, ingestion_dir, fallback_dir


def test_loader_prefers_resolver(loader_environment: Tuple[Path, Path, Path]) -> None:
    _, _, legacy_dir = loader_environment
    data = {"enabled": False, "api": {"token_env": "EXAMPLE"}}
    _write_yaml(legacy_dir / "idmc.yml", data)

    cfg, source, chosen_path, warnings = config_loader.load_connector_config("idmc")
    assert source == "resolver"
    assert cfg == data
    assert chosen_path == (legacy_dir / "idmc.yml").resolve()
    assert not warnings

    details = config_loader.get_config_details("idmc")
    assert details is not None
    assert details.path == (legacy_dir / "idmc.yml").resolve()
    assert not details.warnings


def test_loader_ingestion_fallback_warns(
    loader_environment: Tuple[Path, Path, Path]
) -> None:
    _, ingestion_dir, legacy_dir = loader_environment
    fallback_payload = {"enabled": True}
    _write_yaml(ingestion_dir / "idmc.yml", fallback_payload)

    cfg, source, chosen_path, warnings = config_loader.load_connector_config("idmc")
    assert source == "ingestion"
    assert cfg == fallback_payload
    assert chosen_path == (ingestion_dir / "idmc.yml").resolve()
    assert warnings
    assert any("resolver/ingestion/config/idmc.yml" in message for message in warnings)

    diagnostics = attach_config_source({}, "idmc")
    assert diagnostics["config_source"] == "ingestion"
    assert any(
        "resolver/ingestion/config/idmc.yml" in message
        for message in diagnostics.get("warnings", [])
    )
    assert diagnostics["config"]["config_path_used"].endswith(
        "resolver/ingestion/config/idmc.yml"
    )
    assert diagnostics["config"]["config_source_label"] == "ingestion"
    # Ensure helper respected canonical paths from loader
    assert diagnostics["config"]["legacy_config_path"].endswith(
        "resolver/config/idmc.yml"
    )
    assert diagnostics["config"]["ingestion_config_path"].endswith(
        "resolver/ingestion/config/idmc.yml"
    )


def test_loader_duplicate_equal(loader_environment: Tuple[Path, Path, Path]) -> None:
    _, ingestion_dir, legacy_dir = loader_environment
    payload = {"enabled": True, "cache": {"dir": ".cache"}}
    _write_yaml(ingestion_dir / "idmc.yml", payload)
    _write_yaml(legacy_dir / "idmc.yml", payload)

    cfg, source, chosen_path, warnings = config_loader.load_connector_config("idmc")
    assert source == "resolver (dup-equal)"
    assert cfg == payload
    assert chosen_path == (legacy_dir / "idmc.yml").resolve()
    assert not config_loader.get_config_warnings("idmc")
    assert not warnings


def test_loader_duplicate_mismatch(loader_environment: Tuple[Path, Path, Path]) -> None:
    _, ingestion_dir, legacy_dir = loader_environment
    _write_yaml(ingestion_dir / "idmc.yml", {"enabled": True, "api": {"countries": ["AAA"]}})
    legacy_payload = {"enabled": False, "cache": {"ttl_seconds": 3600}}
    _write_yaml(legacy_dir / "idmc.yml", legacy_payload)

    cfg, source, chosen_path, warnings = config_loader.load_connector_config("idmc")
    assert source == "resolver"
    assert cfg == legacy_payload
    assert chosen_path == (legacy_dir / "idmc.yml").resolve()
    mismatch_warnings = config_loader.get_config_warnings("idmc")
    assert mismatch_warnings and "duplicate-mismatch" in mismatch_warnings[0]
    assert warnings and "duplicate-mismatch" in warnings[0]
    with pytest.raises(ValueError) as excinfo:
        config_loader.load_connector_config("idmc", strict_mismatch=True)
    message = str(excinfo.value)
    assert "resolver/config/idmc.yml" in message
    assert "resolver/ingestion/config/idmc.yml" in message


def test_dtm_offline_smoke_records_config_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    diagnostics_root = tmp_path / "diagnostics"
    ingestion_diag = diagnostics_root / "ingestion"
    out_path = tmp_path / "outputs" / "dtm.csv"
    mappings = {
        "OUT_PATH": out_path,
        "OUT_DIR": out_path.parent,
        "OUTPUT_PATH": out_path,
        "DEFAULT_OUTPUT": out_path,
        "META_PATH": out_path.with_suffix(out_path.suffix + ".meta.json"),
        "DIAGNOSTICS_DIR": ingestion_diag,
        "DTM_DIAGNOSTICS_DIR": ingestion_diag / "dtm",
        "DIAGNOSTICS_RAW_DIR": ingestion_diag / "raw",
        "DIAGNOSTICS_METRICS_DIR": ingestion_diag / "metrics",
        "DIAGNOSTICS_SAMPLES_DIR": ingestion_diag / "samples",
        "DIAGNOSTICS_LOG_DIR": ingestion_diag / "logs",
        "CONNECTORS_REPORT": ingestion_diag / "connectors_report.jsonl",
        "RUN_DETAILS_PATH": ingestion_diag / "dtm_run.json",
        "API_REQUEST_PATH": ingestion_diag / "dtm_api_request.json",
        "API_SAMPLE_PATH": ingestion_diag / "dtm_api_sample.json",
        "DISCOVERY_SNAPSHOT_PATH": ingestion_diag / "dtm" / "discovery_countries.csv",
        "DISCOVERY_FAIL_PATH": ingestion_diag / "dtm" / "discovery_fail.json",
        "DTM_HTTP_LOG_PATH": ingestion_diag / "dtm" / "dtm_http.ndjson",
        "DISCOVERY_RAW_JSON_PATH": ingestion_diag / "raw" / "dtm_countries.json",
        "PER_COUNTRY_METRICS_PATH": ingestion_diag / "metrics" / "dtm_per_country.jsonl",
        "SAMPLE_ROWS_PATH": ingestion_diag / "samples" / "dtm_sample.csv",
        "DTM_CLIENT_LOG_PATH": ingestion_diag / "logs" / "dtm_client.log",
    }
    for name, value in mappings.items():
        monkeypatch.setattr(dtm_client, name, value)

    for handler in list(dtm_client.LOG.handlers):
        if isinstance(handler, logging.FileHandler):
            dtm_client.LOG.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # pragma: no cover - defensive cleanup
                pass
    monkeypatch.setattr(dtm_client, "_FILE_LOGGING_INITIALIZED", False)
    monkeypatch.delenv("DTM_CONFIG_PATH", raising=False)

    exit_code = dtm_client.main(["--offline-smoke"])
    assert exit_code == 0

    meta_payload = json.loads(mappings["META_PATH"].read_text(encoding="utf-8"))
    diagnostics_block = meta_payload["diagnostics"]
    assert diagnostics_block["config_source"] == "resolver"
    assert diagnostics_block["config"]["config_source_label"] == "resolver"
    assert "resolver/config/dtm.yml" in diagnostics_block["config"]["config_path_used"]

    report_lines = mappings["CONNECTORS_REPORT"].read_text(encoding="utf-8").strip().splitlines()
    assert report_lines
    report_payload = json.loads(report_lines[-1])
    config_payload = report_payload["extras"]["config"]
    assert config_payload["config_source_label"] == "resolver"
    warnings = config_payload.get("config_warnings") or []
    assert any("resolver/config/dtm.yml" in message for message in warnings)

    entries = summarize_connectors.load_report(mappings["CONNECTORS_REPORT"])
    markdown = summarize_connectors.build_markdown(entries)
    assert "Config source:" in markdown
    assert "Config: resolver/config/dtm.yml" in markdown
