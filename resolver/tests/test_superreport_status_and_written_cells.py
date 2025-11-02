from __future__ import annotations

import re

from scripts.ci import summarize_connectors as sc


def _first_row(markdown: str) -> list[str]:
    lines = markdown.splitlines()
    header = next(line for line in lines if line.startswith("| Connector"))
    idx = lines.index(header)
    row = next(line for line in lines[idx + 2 :] if line.startswith("|"))
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def test_status_prefers_raw_value_and_kept_counts(tmp_path):
    entry = {
        "connector_id": "dtm-smoke",
        "mode": "smoke",
        "status": "ok",
        "reason": "empty window",
        "counts": {"fetched": 5, "normalized": 4, "written": 3},
        "http": {"2xx": 8, "4xx": 1, "5xx": 0, "retries": 2},
        "extras": {
            "status_raw": "ok-empty",
            "run_totals": {"rows_written": 7, "dropped": 2, "parse_errors": 1},
        },
    }
    markdown = sc.build_markdown([entry], diagnostics_root=tmp_path / "diagnostics" / "ingestion")
    cells = _first_row(markdown)
    assert cells[2] == "ok-empty"
    assert cells[3] == "empty window"
    assert re.fullmatch(r"8/1/0 \(2\)", cells[4])
    assert cells[5:8] == ["5", "4", "3"]
    assert cells[8:11] == ["7", "2", "1"]


def test_missing_counts_default_to_zero():
    entry = {
        "connector_id": "beta",
        "mode": "real",
        "status": "errored",
        "reason": None,
        "http": {},
        "extras": {},
    }
    markdown = sc.build_markdown([entry])
    cells = _first_row(markdown)
    assert cells[2] == "errored"
    assert cells[3] == "—"
    assert cells[4] == "—"
    assert cells[5:11] == ["0", "0", "0", "0", "0", "0"]
