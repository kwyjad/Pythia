from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.ci.summarize_connectors import render_summary_md


def test_summary_superreport_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    diagnostics_dir = tmp_path / "diagnostics" / "ingestion"
    logs_dir = diagnostics_dir / "logs"
    connector_dir = diagnostics_dir / "idmc"
    error_dir = diagnostics_dir / "acled_client"
    staging_dir = tmp_path / "resolver" / "staging"
    report_path = diagnostics_dir / "connectors_report.jsonl"

    monkeypatch.setenv("RESOLVER_GLOBAL_WINDOW_START", "2024-01-01")
    monkeypatch.setenv("RESOLVER_GLOBAL_WINDOW_END", "2024-03-31")
    monkeypatch.setenv("IDMC_API_TOKEN", "fake-token")

    logs_dir.mkdir(parents=True, exist_ok=True)
    connector_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    report_entries = [
        {
            "connector_id": "idmc",
            "status": "ok",
            "counts": {"fetched": 10, "normalized": 8, "written": 5},
            "http": {"requests": 3, "2xx": 3, "4xx": 0, "5xx": 0, "retries": 0, "timeouts": 0},
            "coverage": {"ym_min": "2024-01", "ym_max": "2024-03"},
            "samples": {"top_iso3": [["COL", 3], ["PER", 2]]},
            "extras": {"series": ["flow"], "flags": {"strict": False}},
        },
        {
            "connector_id": "acled_client",
            "status": "error",
            "counts": {"fetched": 0, "normalized": 0, "written": 0},
            "http": {"requests": 1, "2xx": 0, "4xx": 1, "5xx": 0, "retries": 2, "timeouts": 0},
            "extras": {},
        },
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        for entry in report_entries:
            handle.write(json.dumps(entry))
            handle.write("\n")

    (connector_dir / "reachability.json").write_text(
        json.dumps({"dns_ip": "203.0.113.10", "tcp_ms": 120, "tls_ok": True}),
        encoding="utf-8",
    )
    (connector_dir / "normalize.json").write_text(
        json.dumps({"drop_reasons": {"no_iso3": 2, "date_out_of_window": 1}}),
        encoding="utf-8",
    )
    (connector_dir / "why_zero.json").write_text(
        json.dumps(
            {
                "token_present": True,
                "countries_count": 2,
                "countries_sample": ["COL", "PER"],
                "window": {"start": "2024-01", "end": "2024-03"},
                "filters": {"date_out_of_window": 1, "no_iso3": 2},
                "config_source": "ingestion",
                "config_path_used": "resolver/config/idmc.yml",
                "loader_warnings": ["missing hazard map"],
            }
        ),
        encoding="utf-8",
    )
    (error_dir / "error.json").write_text(
        json.dumps({"exit_code": 2, "stderr_tail": "boom", "log_tail": "traceback"}),
        encoding="utf-8",
    )

    idmc_staging = staging_dir / "idmc"
    idmc_staging.mkdir(parents=True, exist_ok=True)
    (idmc_staging / "flow.csv").write_text("iso3,value\nCOL,1\nPER,2\n", encoding="utf-8")

    acled_staging = staging_dir / "acled_client"
    acled_staging.mkdir(parents=True, exist_ok=True)
    (acled_staging / "flow.csv").write_text("iso3,value\nKEN,4\n", encoding="utf-8")

    summary = render_summary_md(report_path, diagnostics_dir, staging_dir)

    assert "# Ingestion Superreport" in summary
    assert "## Run Overview" in summary
    assert "## Connector Diagnostics Matrix" in summary
    assert "## Connector Deep Dives" in summary
    assert "## Export & Snapshot" in summary
    assert "## Anomaly & Trend Checks" in summary
    assert "## Next Actions" in summary

    assert "idmc" in summary
    assert "acled_client" in summary
    assert "Why-zero payload" in summary
    assert "Investigate `acled_client` failure" in summary
    assert "Rows written (flow):" in summary
