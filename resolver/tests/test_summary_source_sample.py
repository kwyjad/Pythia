# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import csv
import json
from pathlib import Path

from scripts.ci import summarize_connectors


def _write_report(path: Path) -> None:
    payload = {
        "connector_id": "dtm_client",
        "mode": "real",
        "status": "ok",
        "reason": "done",
        "duration_ms": 120,
        "http": {"2xx": 1, "4xx": 0, "5xx": 0, "retries": 0, "rate_limit_remaining": None, "last_status": 200},
        "counts": {"fetched": 10, "normalized": 5, "written": 5},
        "extras": {
            "status_raw": "ok",
            "exit_code": 0,
            "rows_written": 5,
            "config": {
                "config_path_used": "resolver/config/dtm.yml",
                "config_exists": True,
                "config_sha256": "abc123deadbeef",
                "admin_levels": ["admin0", "admin1"],
                "countries_mode": "explicit_config",
                "countries_count": 3,
                "countries_preview": ["SSD", "ETH", "SOM"],
                "selected_iso3_preview": ["SSD", "ETH", "SOM"],
                "no_date_filter": 0,
            },
            "per_country_counts": [
                {"country": "SSD", "rows": 3, "level": "admin0"},
                {"country": "ETH", "rows": 2, "level": "admin0"},
            ],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload))
        handle.write("\n")


def _write_sample(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"CountryISO3": "SSD", "admin0Name": "South Sudan"},
        {"CountryISO3": "SSD", "admin0Name": "South Sudan"},
        {"CountryISO3": "ETH", "admin0Name": "Ethiopia"},
        {"CountryISO3": "SOM", "admin0Name": "Somalia"},
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["CountryISO3", "admin0Name"])
        writer.writeheader()
        writer.writerows(rows)


def test_summary_includes_sample_histogram(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    report_path = tmp_path / "diagnostics" / "ingestion" / "connectors_report.jsonl"
    summary_path = tmp_path / "diagnostics" / "ingestion" / "summary.md"
    sample_path = tmp_path / "diagnostics" / "ingestion" / "dtm" / "samples" / "admin0_head.csv"

    _write_report(report_path)
    _write_sample(sample_path)

    rc = summarize_connectors.main([
        "--report",
        str(report_path),
        "--out",
        str(summary_path),
    ])
    assert rc == 0

    content = summary_path.read_text(encoding="utf-8")
    assert "## Source sample: quick checks" in content
    assert "CountryISO3 top 5" in content
    assert "SSD (2)" in content
    assert "admin0Name top 5" in content
    assert "South Sudan (2)" in content
