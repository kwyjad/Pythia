from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.ci import summarize_connectors


def _write_report(path: Path) -> None:
    payloads = [
        {
            "connector_id": "dtm_client",
            "mode": "real",
            "status": "error",
            "reason": "first attempt",
            "started_at_utc": "2024-01-01T00:00:00Z",
            "duration_ms": 100,
            "http": {"2xx": 0, "4xx": 0, "5xx": 0, "retries": 0, "rate_limit_remaining": None, "last_status": None},
            "counts": {"fetched": 0, "normalized": 0, "written": 0},
            "extras": {
                "status_raw": "error",
                "exit_code": 1,
                "config": {
                    "config_path_used": "custom.yml",
                    "config_exists": True,
                    "config_sha256": "abc123deadbeef",
                    "admin_levels": ["admin0"],
                    "countries_mode": "explicit_config",
                    "countries_count": 2,
                    "countries_preview": ["SSD", "ETH"],
                    "no_date_filter": 0,
                },
                "per_country_counts": [
                    {"country": "SSD", "level": "admin0", "operation": "stock", "rows": 0},
                    {"country": "ETH", "level": "admin0", "operation": "stock", "rows": 12},
                ],
            },
        },
        {
            "connector_id": "dtm_client",
            "mode": "real",
            "status": "ok",
            "reason": "second attempt",
            "started_at_utc": "2024-01-01T01:00:00Z",
            "duration_ms": 200,
            "http": {"2xx": 3, "4xx": 0, "5xx": 0, "retries": 1, "rate_limit_remaining": 50, "last_status": 200},
            "counts": {"fetched": 100, "normalized": 90, "written": 80},
            "extras": {
                "status_raw": "ok",
                "exit_code": 0,
                "rows_written": 80,
                "config": {
                    "config_path_used": "custom.yml",
                    "config_exists": True,
                    "config_sha256": "abc123deadbeef",
                    "admin_levels": ["admin0", "admin1"],
                    "countries_mode": "discovered",
                    "countries_count": 4,
                    "countries_preview": ["SSD", "ETH", "NGA", "SOM"],
                    "no_date_filter": 0,
                },
                "per_country_counts": [
                    {"country": "SSD", "level": "admin0", "operation": "stock", "rows": 12},
                    {"country": "ETH", "level": "admin1", "operation": "flow", "rows": 34},
                    {"country": "NGA", "level": "admin0", "operation": "stock", "rows": 0},
                    {"country": "SOM", "level": "admin1", "operation": "flow", "rows": 10},
                ],
            },
        },
        {
            "connector_id": "other_client",
            "mode": "real",
            "status": "ok",
            "reason": "done",
            "started_at_utc": "2024-01-01T02:00:00Z",
            "duration_ms": 150,
            "http": {"2xx": 5, "4xx": 0, "5xx": 0, "retries": 0, "rate_limit_remaining": None, "last_status": 200},
            "counts": {"fetched": 10, "normalized": 10, "written": 10},
            "extras": {"status_raw": "ok", "exit_code": 0},
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload))
            handle.write("\n")


def _write_reachability(path: Path) -> None:
    payload = {
        "generated_at": "2024-01-01T00:00:00Z",
        "completed_at": "2024-01-01T00:00:02Z",
        "target_host": "dtmapi.iom.int",
        "target_port": 443,
        "python_version": "3.11",
        "requests_version": "2.31.0",
        "ca_bundle": "/tmp/cacert.pem",
        "dns": {"records": [{"address": "1.2.3.4", "family": "AF_INET"}], "elapsed_ms": 5},
        "tcp": {"ok": True, "elapsed_ms": 20, "peer": ["1.2.3.4", 443]},
        "tls": {"ok": True, "elapsed_ms": 30, "subject": ((("commonName", "dtmapi.iom.int"),),), "issuer": ((("commonName", "IOM"),),), "not_after": "Mar  1 00:00:00 2025 GMT"},
        "curl_head": {"exit_code": 0, "status_line": "HTTP/1.1 200 OK"},
        "egress": {
            "ifconfig_me": {"status_code": 200, "text": "203.0.113.10"},
            "api_ipify_org": {"status_code": 200, "text": "203.0.113.11"},
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_summary_includes_config_reachability_and_dedupe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    report_path = tmp_path / "diagnostics" / "ingestion" / "connectors_report.jsonl"
    _write_report(report_path)
    reachability_path = tmp_path / "diagnostics" / "ingestion" / "dtm" / "reachability.json"
    _write_reachability(reachability_path)
    out_path = tmp_path / "diagnostics" / "ingestion" / "summary.md"

    rc = summarize_connectors.main([
        "--report",
        str(report_path),
        "--out",
        str(out_path),
    ])
    assert rc == 0

    content = out_path.read_text(encoding="utf-8")
    assert "## Config used" in content
    assert "custom.yml" in content
    assert "Countries preview" in content
    assert "## Selector effectiveness" in content
    assert "Top selectors by rows" in content
    assert "## DTM Reachability" in content
    assert "dtmapi.iom.int:443" in content
    assert "ifconfig_me=203.0.113.10" in content
    assert "(deduplicated 1 duplicate entries for dtm_client)" in content
    dtm_rows = [line for line in content.splitlines() if line.startswith("| dtm_client |")]
    assert len(dtm_rows) == 1
