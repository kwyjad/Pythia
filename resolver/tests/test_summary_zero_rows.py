# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import json
from pathlib import Path

import pytest

from resolver.ingestion.diagnostics_emitter import append_jsonl, finalize_run, start_run

from scripts.ci import summarize_connectors


def _write_zero_row_report(path: Path) -> None:
    payload = {
        "connector_id": "dtm_client",
        "mode": "real",
        "status": "ok",
        "reason": "header-only; kept=0",
        "duration_ms": 90,
        "http": {"2xx": 0, "4xx": 0, "5xx": 0, "retries": 0, "rate_limit_remaining": None, "last_status": None},
        "counts": {"fetched": 15, "normalized": 10, "written": 0},
        "extras": {
            "status_raw": "ok-empty",
            "exit_code": 0,
            "rows_written": 0,
            "zero_rows_reason": "no_country_match",
            "config": {
                "config_path_used": "resolver/config/dtm.yml",
                "config_exists": True,
                "config_sha256": "abc123deadbeef",
                "admin_levels": ["admin0"],
                "countries_mode": "discovered",
                "countries_count": 2,
                "countries_preview": ["SSD", "ETH"],
                "no_date_filter": 0,
            },
            "normalize": {
                "rows_fetched": 15,
                "rows_normalized": 10,
                "rows_written": 0,
                "drop_reasons": {"no_country_match": 7, "filter_window_miss": 3},
            },
            "fetch": {"pages": 1, "total_received": 15},
            "per_country_counts": [
                {"country": "SSD", "rows": 0},
                {"country": "ETH", "rows": 0},
            ],
            "discovery": {
                "report": {
                    "used_stage": "static_iso3_minimal",
                    "reason": "fallback",
                    "stages": [],
                    "configured_labels": [],
                    "unresolved_labels": [],
                }
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload))
        handle.write("\n")


def test_zero_row_section_lists_root_causes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    report_path = tmp_path / "diagnostics" / "ingestion" / "connectors_report.jsonl"
    summary_path = tmp_path / "diagnostics" / "ingestion" / "summary.md"
    _write_zero_row_report(report_path)

    rc = summarize_connectors.main([
        "--report",
        str(report_path),
        "--out",
        str(summary_path),
    ])
    assert rc == 0

    content = summary_path.read_text(encoding="utf-8")
    assert "## Zero-row root cause" in content
    assert "Primary reason" in content
    assert "Top drop reasons" in content
    assert "Selectors with rows" in content
    assert "rows_received=15" in content
    assert "stage=static_iso3_minimal" in content


def test_summary_renders_per_country_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    report_path = tmp_path / "diagnostics" / "ingestion" / "connectors_report.jsonl"
    summary_path = tmp_path / "diagnostics" / "ingestion" / "summary.md"
    payload = {
        "connector_id": "dtm_client",
        "mode": "real",
        "status": "ok",
        "reason": "ok",
        "http": {},
        "counts": {"fetched": 10, "written": 8},
        "extras": {
            "per_country": [
                {
                    "country": "SSD",
                    "level": "admin0",
                    "param": "iso3",
                    "pages": 2,
                    "rows": 120,
                    "skipped_no_match": False,
                },
                {
                    "country": "NGA",
                    "level": "admin0",
                    "param": "name",
                    "pages": 0,
                    "rows": 0,
                    "skipped_no_match": True,
                    "reason": "no_country_match",
                },
            ],
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload))
        handle.write("\n")

    rc = summarize_connectors.main([
        "--report",
        str(report_path),
        "--out",
        str(summary_path),
    ])
    assert rc == 0

    content = summary_path.read_text(encoding="utf-8")
    assert "## DTM per-country results" in content
    assert "| SSD | admin0 | CountryISO3" in content
    assert "no_country_match" in content


def test_summary_includes_acled_http_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    diagnostics_dir = tmp_path / "diagnostics" / "ingestion"
    report_path = diagnostics_dir / "connectors_report.jsonl"
    summary_path = diagnostics_dir / "summary.md"

    ctx = start_run("acled_client", "real")
    result = finalize_run(ctx, status="ok", counts={"fetched": 0, "normalized": 0, "written": 0})
    append_jsonl(report_path, result)

    acled_client_dir = diagnostics_dir / "acled_client"
    acled_client_dir.mkdir(parents=True, exist_ok=True)
    run_info = {
        "rows_fetched": 0,
        "rows_normalized": 0,
        "rows_written": 0,
        "http_status": 200,
        "base_url": "https://api.acleddata.com/acled/read",
        "window": {"start": "2024-01-01", "end": "2024-03-31"},
        "params_keys": ["event_date", "event_date_where", "format", "limit", "page"],
    }
    (acled_client_dir / "acled_client_run.json").write_text(
        json.dumps(run_info, indent=2),
        encoding="utf-8",
    )

    zero_rows_src = Path(__file__).resolve().parent / "fixtures" / "ingestion" / "acled_zero_rows.json"
    zero_rows_dst = diagnostics_dir / "acled" / "zero_rows.json"
    zero_rows_dst.parent.mkdir(parents=True, exist_ok=True)
    zero_rows_dst.write_text(zero_rows_src.read_text(encoding="utf-8"), encoding="utf-8")

    mapping_dir = diagnostics_dir / "export_preview"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    mapping_record = {
        "file": "resolver/staging/acled.csv",
        "matched": False,
        "reasons": {"regex_miss": True},
    }
    (mapping_dir / "mapping_debug.jsonl").write_text(json.dumps(mapping_record) + "\n", encoding="utf-8")

    rc = summarize_connectors.main(
        ["--report", str(report_path), "--out", str(summary_path)]
    )
    assert rc == 0

    content = summary_path.read_text(encoding="utf-8")
    assert "## ACLED HTTP diagnostics" in content
    assert "Rows fetched:** 0" in content
    assert "Last HTTP status:** 200" in content
    assert "Base URL:** `https://api.acleddata.com/acled/read`" in content
    assert "Zero rows reason:" in content
    assert "ACLED export note" in content
