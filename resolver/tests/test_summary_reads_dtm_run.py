# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import json
from pathlib import Path

from scripts.ci import summarize_connectors


def test_summary_uses_dtm_run_totals(tmp_path: Path) -> None:
    diagnostics = tmp_path / "diagnostics" / "ingestion"
    diagnostics.mkdir(parents=True, exist_ok=True)
    run_path = diagnostics / "dtm_run.json"
    run_payload = {
        "generated_at": "2024-01-01T00:00:00Z",
        "window": {"start": "2023-01-01", "end": "2023-12-31", "disabled": False},
        "rows_written": 2,
        "missing_id_or_path": 0,
        "outputs": {"csv": "data/staging/dtm_displacement.csv", "meta": "meta"},
        "sources": {"valid": [], "invalid": []},
        "totals": {
            "rows_before": 3,
            "rows_after": 2,
            "rows_written": 2,
            "kept": 2,
            "dropped": 1,
            "parse_errors": 1,
            "invalid_sources": 0,
        },
    }
    run_path.write_text(json.dumps(run_payload), encoding="utf-8")

    report_path = diagnostics / "connectors_report.jsonl"
    payload = {
        "connector_id": "dtm_client",
        "mode": "real",
        "status": "ok",
        "reason": "kept=2, dropped=1, parse_errors=1",
        "duration_ms": 100,
        "http": {"2xx": 1, "4xx": 0, "5xx": 0, "retries": 0, "rate_limit_remaining": None, "last_status": None},
        "counts": {"fetched": 2, "normalized": 2, "written": 2},
        "extras": {
            "status_raw": "ok",
            "rows_written": 2,
            "run_details_path": str(run_path),
        },
    }
    report_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    entries = summarize_connectors.load_report(report_path)
    markdown = summarize_connectors.build_markdown(entries)
    dtm_rows = [line for line in markdown.splitlines() if line.startswith("| dtm_client |")]
    assert len(dtm_rows) == 1
    cells = [cell.strip() for cell in dtm_rows[0].strip("| ").split(" | ")]
    assert cells[7] == "2"
    assert cells[8] == "1"
    assert cells[9] == "1"


def test_summary_overrides_zero_counts_with_run_json(tmp_path: Path) -> None:
    diagnostics = tmp_path / "diagnostics" / "ingestion"
    diagnostics.mkdir(parents=True, exist_ok=True)
    run_path = diagnostics / "dtm_run.json"
    run_payload = {
        "rows": {"fetched": 5, "normalized": 5, "written": 5},
        "totals": {"rows_written": 5},
    }
    run_path.write_text(json.dumps(run_payload), encoding="utf-8")

    report_path = diagnostics / "connectors_report.jsonl"
    payload = {
        "connector_id": "dtm_client",
        "mode": "real",
        "status": "ok",
        "reason": None,
        "duration_ms": 0,
        "http": {"2xx": 1, "4xx": 0, "5xx": 0, "retries": 0, "rate_limit_remaining": None, "last_status": 200},
        "counts": {"fetched": 0, "normalized": 0, "written": 0},
        "extras": {"run_details_path": str(run_path)},
    }
    report_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    entries = summarize_connectors.load_report(report_path)
    assert entries[0]["counts"]["fetched"] == 5
    assert entries[0]["counts"]["normalized"] == 5
    assert entries[0]["counts"]["written"] == 5

    markdown = summarize_connectors.build_markdown(entries)
    assert "* **Rows fetched:** 5 (from run.json)" in markdown
    assert "* **Rows normalized:** 5 (from run.json)" in markdown
    assert "* **Rows written:** 5 (from run.json)" in markdown


def test_config_section_renders_parse_lines_and_warning(tmp_path: Path) -> None:
    diagnostics = tmp_path / "diagnostics" / "ingestion"
    diagnostics.mkdir(parents=True, exist_ok=True)
    report_path = diagnostics / "connectors_report.jsonl"
    config_payload = {
        "connector_id": "dtm_client",
        "mode": "real",
        "status": "ok",
        "reason": None,
        "duration_ms": 0,
        "http": {
            "2xx": 1,
            "4xx": 0,
            "5xx": 0,
            "retries": 0,
            "rate_limit_remaining": None,
            "last_status": 200,
        },
        "counts": {"fetched": 10, "normalized": 10, "written": 10},
        "extras": {
            "status_raw": "ok",
            "rows_written": 10,
            "config": {
                "config_path_used": "resolver/config/dtm.yml",
                "config_exists": True,
                "config_sha256": "abcdef123456",
                "countries_mode": "discovered",
                "countries_count": 7,
                "countries_preview": ["SSD", "ETH", "SOM", "NGA", "SUD"],
                "selected_iso3_preview": ["SSD", "ETH", "SOM"],
                "admin_levels": ["admin0", "admin1"],
                "no_date_filter": 0,
                "config_parse": {
                    "countries": [
                        "SSD",
                        "ETH",
                        "SOM",
                        "NGA",
                        "SUD",
                        "YEM",
                        "COD",
                    ],
                    "admin_levels": ["admin0", "admin1"],
                },
                "config_keys_found": {"countries": True, "admin_levels": True},
                "config_countries_count": 7,
            },
        },
    }
    report_path.write_text(json.dumps(config_payload) + "\n", encoding="utf-8")

    entries = summarize_connectors.load_report(report_path)
    markdown = summarize_connectors.build_markdown(entries)
    assert "- **Countries parse:** api.countries: found (7)" in markdown
    assert "- **Admin levels parse:** api.admin_levels: found ([admin0, admin1])" in markdown
    assert (
        "- ⚠ config had api.countries but selector list not applied (check loader/version)."
        in markdown
    )


def test_date_filter_section_renders(tmp_path: Path) -> None:
    diagnostics = tmp_path / "diagnostics" / "ingestion"
    diagnostics.mkdir(parents=True, exist_ok=True)
    report_path = diagnostics / "connectors_report.jsonl"
    payload = {
        "connector_id": "dtm_client",
        "mode": "real",
        "status": "ok",
        "reason": None,
        "duration_ms": 0,
        "http": {"2xx": 1, "4xx": 0, "5xx": 0, "retries": 0, "rate_limit_remaining": None, "last_status": 200},
        "counts": {"fetched": 10, "normalized": 10, "written": 10},
        "extras": {
            "status_raw": "ok",
            "rows_written": 10,
            "date_filter": {
                "date_column_used": "ReportingDate",
                "parsed_ok": 8,
                "parsed_total": 10,
                "inside": 6,
                "outside": 4,
                "parse_failed": 2,
                "window_start": "2024-01-01",
                "window_end": "2024-01-31",
                "inclusive": True,
                "skipped": False,
                "min_date": "2024-01-01",
                "max_date": "2024-02-15",
                "drop_counts": {"date_out_of_window": 2, "date_parse_failed": 2},
            },
            "artifacts": {
                "normalize_drops": "diagnostics/ingestion/dtm/normalize_drops.csv",
            },
        },
    }
    report_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    entries = summarize_connectors.load_report(report_path)
    markdown = summarize_connectors.build_markdown(entries)
    assert "## DTM Date Filter" in markdown
    assert "- **Date column:** `ReportingDate`" in markdown
    assert "- **Parsed:** 8/10" in markdown
    assert "Drop sample" in markdown
