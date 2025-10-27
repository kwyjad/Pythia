from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from resolver.ingestion import dtm_client


@pytest.fixture(autouse=True)
def clear_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DTM_CONFIG_PATH", raising=False)


def _prepare_repo(monkeypatch: pytest.MonkeyPatch, repo_root: Path) -> None:
    resolver_root = repo_root / "resolver"
    ingestion_config = resolver_root / "ingestion" / "config"
    ingestion_config.mkdir(parents=True, exist_ok=True)
    (resolver_root / "config").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(dtm_client, "_REPO_ROOT", repo_root, raising=False)
    monkeypatch.setattr(dtm_client, "REPO_ROOT", repo_root, raising=False)
    monkeypatch.setattr(dtm_client, "RESOLVER_ROOT", resolver_root, raising=False)
    ingestion_default = (ingestion_config / "dtm.yml").resolve()
    monkeypatch.setattr(dtm_client, "LEGACY_CONFIG_PATH", ingestion_default, raising=False)
    monkeypatch.setattr(
        dtm_client,
        "REPO_CONFIG_PATH",
        (resolver_root / "config" / "dtm.yml").resolve(),
        raising=False,
    )
    monkeypatch.setattr(dtm_client, "CONFIG_PATH", ingestion_default, raising=False)


def test_load_config_honours_search_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _prepare_repo(monkeypatch, repo_root)

    custom_config = tmp_path / "custom" / "dtm.yml"
    custom_config.parent.mkdir(parents=True, exist_ok=True)
    custom_config.write_text("enabled: false\napi:\n  countries: []\n", encoding="utf-8")

    resolver_config = repo_root / "resolver" / "config" / "dtm.yml"
    resolver_config.write_text("enabled: true\n", encoding="utf-8")

    ingestion_config = repo_root / "resolver" / "ingestion" / "config" / "dtm.yml"
    ingestion_config.write_text("enabled: true\napi:\n  countries: []\n", encoding="utf-8")

    monkeypatch.setenv("DTM_CONFIG_PATH", str(custom_config))
    cfg = dtm_client.load_config()
    assert Path(getattr(cfg, "_source_path")).resolve() == custom_config.resolve()
    assert getattr(cfg, "_source_exists") is True
    expected_sha = hashlib.sha256(custom_config.read_bytes()).hexdigest()[:12]
    assert getattr(cfg, "_source_sha256") == expected_sha
    assert cfg.get("enabled") is False

    monkeypatch.delenv("DTM_CONFIG_PATH", raising=False)
    cfg = dtm_client.load_config()
    assert Path(getattr(cfg, "_source_path")).resolve() == resolver_config.resolve()
    assert getattr(cfg, "_source_exists") is True

    resolver_config.unlink()
    cfg = dtm_client.load_config()
    assert Path(getattr(cfg, "_source_path")).resolve() == ingestion_config.resolve()
    assert getattr(cfg, "_source_exists") is True

    ingestion_config.unlink()
    cfg = dtm_client.load_config()
    assert Path(getattr(cfg, "_source_path")).resolve() == ingestion_config.resolve()
    assert getattr(cfg, "_source_exists") is False
    assert cfg.get("api") == {}


def test_connector_report_includes_config_extras(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    _prepare_repo(monkeypatch, repo_root)

    resolver_config = repo_root / "resolver" / "config" / "dtm.yml"
    resolver_config.write_text(
        "\n".join(
            [
                "enabled: true",
                "api:",
                "  countries: [SSD, ETH, SOM]",
                "  admin_levels: [admin0, admin1]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    diagnostics_dir = tmp_path / "diagnostics" / "ingestion"
    dtm_dir = diagnostics_dir / "dtm"
    out_path = repo_root / "resolver" / "staging" / "dtm_displacement.csv"

    path_overrides = {
        "CONNECTORS_REPORT": diagnostics_dir / "connectors_report.jsonl",
        "RUN_DETAILS_PATH": dtm_dir / "dtm_run.json",
        "DTM_HTTP_LOG_PATH": dtm_dir / "dtm_http.ndjson",
        "DTM_DIAGNOSTICS_DIR": dtm_dir,
        "DIAGNOSTICS_DIR": diagnostics_dir,
        "DIAGNOSTICS_ROOT": diagnostics_dir,
        "DTM_RAW_DIR": dtm_dir / "raw",
        "DTM_METRICS_DIR": dtm_dir / "metrics",
        "DTM_SAMPLES_DIR": dtm_dir / "samples",
        "DTM_LOG_DIR": dtm_dir / "logs",
        "DIAGNOSTICS_RAW_DIR": diagnostics_dir / "raw",
        "DIAGNOSTICS_METRICS_DIR": diagnostics_dir / "metrics",
        "DIAGNOSTICS_SAMPLES_DIR": diagnostics_dir / "samples",
        "DIAGNOSTICS_LOG_DIR": diagnostics_dir / "logs",
        "OUT_PATH": out_path,
        "OUT_DIR": out_path.parent,
        "OUTPUT_PATH": out_path,
        "DEFAULT_OUTPUT": out_path,
        "META_PATH": out_path.with_suffix(out_path.suffix + ".meta.json"),
        "HTTP_TRACE_PATH": dtm_dir / "dtm_http.ndjson",
        "API_RESPONSE_SAMPLE_PATH": dtm_dir / "dtm_api_response_sample.json",
        "API_SAMPLE_PATH": dtm_dir / "dtm_api_sample.json",
        "API_REQUEST_PATH": dtm_dir / "dtm_api_request.json",
        "DISCOVERY_SNAPSHOT_PATH": dtm_dir / "discovery_countries.csv",
        "DISCOVERY_FAIL_PATH": dtm_dir / "discovery_fail.json",
        "DISCOVERY_RAW_JSON_PATH": dtm_dir / "raw" / "dtm_countries.json",
        "PER_COUNTRY_METRICS_PATH": dtm_dir / "metrics" / "dtm_per_country.jsonl",
        "SAMPLE_ROWS_PATH": dtm_dir / "dtm_sample.csv",
        "DTM_CLIENT_LOG_PATH": dtm_dir / "logs" / "dtm_client.log",
        "RESCUE_PROBE_PATH": dtm_dir / "rescue_probe.json",
        "METRICS_SUMMARY_PATH": dtm_dir / "metrics" / "metrics.json",
    }
    for name, value in path_overrides.items():
        monkeypatch.setattr(dtm_client, name, value)

    for directory in path_overrides.values():
        if isinstance(directory, Path):
            directory.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("DTM_API_KEY", "dummy")

    monkeypatch.setattr(
        dtm_client,
        "_preflight_dependencies",
        lambda: ({"python": "3.11", "packages": [], "missing": []}, True),
    )
    monkeypatch.setattr(dtm_client, "_log_dependency_snapshot", lambda _: None)
    monkeypatch.setattr(dtm_client, "_dtm_sdk_version", lambda: "0.0-test")
    monkeypatch.setattr(
        dtm_client,
        "resolve_ingestion_window",
        lambda: (None, None),
    )

    summary_payload = {
        "rows": {"fetched": 0, "normalized": 0, "kept": 0, "dropped": 0, "parse_errors": 0},
        "extras": {
            "effective_params": {"admin_levels": ["admin0", "admin1"]},
            "per_country_counts": [
                {"country": "SSD", "rows": 0, "level": "admin0"},
                {"country": "ETH", "rows": 5, "level": "admin0"},
            ],
            "discovery": {
                "report": {
                    "used_stage": "explicit_config",
                    "stages": [],
                    "reason": None,
                    "configured_labels": [],
                    "unresolved_labels": [],
                },
                "total_countries": 3,
                "source": "explicit_config",
            },
            "fetch": {"pages": 1, "total_received": 5},
            "normalize": {"drop_reasons": {"no_country_match": 2}},
            "drop_reasons_counter": {"no_country_match": 2},
            "value_column_usage": {},
        },
        "countries": {"resolved": ["SSD", "ETH", "SOM"]},
        "paging": {"pages": 1, "page_size": 100, "total_received": 5},
    }

    monkeypatch.setattr(dtm_client, "build_rows", lambda *_, **__: ([], summary_payload))
    monkeypatch.setattr(dtm_client, "write_rows", lambda *_, **__: None)

    rc = dtm_client.main([])
    assert rc == 0

    report_path = path_overrides["CONNECTORS_REPORT"]
    report_text = report_path.read_text(encoding="utf-8").strip()
    assert report_text
    payload = json.loads(report_text.splitlines()[0])
    config = payload["extras"]["config"]
    assert Path(config["config_path_used"]).resolve() == resolver_config.resolve()
    assert config["config_exists"] is True
    assert config["config_sha256"] == hashlib.sha256(resolver_config.read_bytes()).hexdigest()[:12]
    assert config["countries_count"] == 3
