from __future__ import annotations

import json
from pathlib import Path

from scripts.ci import summarize_connectors


def test_summary_table_includes_meta_column(tmp_path: Path) -> None:
    report_path = tmp_path / "diagnostics" / "ingestion" / "connectors_report.jsonl"
    meta_path = tmp_path / "data" / "staging" / "dtm_displacement.csv.meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({"row_count": 0}), encoding="utf-8")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "connector_id": "dtm_client",
        "mode": "real",
        "status": "ok",
        "reason": "header-only; kept=0",
        "duration_ms": 1000,
        "http": {"2xx": 0, "4xx": 0, "5xx": 0, "retries": 0, "rate_limit_remaining": None, "last_status": None},
        "counts": {"fetched": 0, "normalized": 0, "written": 0},
        "extras": {
            "status_raw": "ok-empty",
            "rows_written": 0,
            "meta_path": str(meta_path),
        },
    }
    report_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    entries = summarize_connectors.load_report(report_path)
    markdown = summarize_connectors.build_markdown(entries)

    dtm_rows = [line for line in markdown.splitlines() if line.startswith("| dtm_client |")]
    assert len(dtm_rows) == 1
    cells = [cell.strip() for cell in dtm_rows[0].strip("| ").split(" | ")]
    assert cells[2] == "ok-empty"
    assert cells[3] == "header-only; kept=0"
    assert cells[7] == "—"
    assert cells[8] == "—"
    assert cells[9] == "—"
    assert cells[-1] == str(meta_path)
