import json
from pathlib import Path

import pytest

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
