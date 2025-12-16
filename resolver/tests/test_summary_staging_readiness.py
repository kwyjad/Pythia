# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import json
from pathlib import Path

import pytest

from scripts.ci import summarize_connectors


def _write_report(path: Path) -> None:
    payload = {
        "connector_id": "dtm_client",
        "mode": "real",
        "status": "ok",
        "reason": "done",
        "duration_ms": 120,
        "http": {"2xx": 1, "4xx": 0, "5xx": 0, "retries": 0, "rate_limit_remaining": None, "last_status": 200},
        "counts": {"fetched": 0, "normalized": 0, "written": 0},
        "extras": {
            "status_raw": "ok-empty",
            "exit_code": 0,
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
            "per_country_counts": [],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload))
        handle.write("\n")


@pytest.mark.parametrize(
    "resolver_files,data_files,expect_hint",
    [
        (["dtm_displacement.csv"], [], False),
        ([], ["dtm_displacement.csv"], True),
    ],
)
def test_summary_reports_staging_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resolver_files, data_files, expect_hint
) -> None:
    monkeypatch.chdir(tmp_path)
    resolver_dir = tmp_path / "resolver" / "staging"
    data_dir = tmp_path / "data" / "staging"

    for name in resolver_files:
        target = resolver_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("a,b\n1,2\n", encoding="utf-8")

    for name in data_files:
        target = data_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("a,b\n1,2\n", encoding="utf-8")

    report_path = tmp_path / "diagnostics" / "ingestion" / "connectors_report.jsonl"
    summary_path = tmp_path / "diagnostics" / "ingestion" / "summary.md"
    _write_report(report_path)

    rc = summarize_connectors.main([
        "--report",
        str(report_path),
        "--out",
        str(summary_path),
    ])
    assert rc == 0

    content = summary_path.read_text(encoding="utf-8")
    assert "## Staging readiness" in content
    assert "resolver/staging" in content
    assert "data/staging" in content
    if expect_hint:
        assert "RESOLVER_OUTPUT_DIR=resolver/staging" in content
    else:
        assert "RESOLVER_OUTPUT_DIR=resolver/staging" not in content
