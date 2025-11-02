from __future__ import annotations

import re
from typing import List

from resolver.diagnostics.ingestion import summarize_connectors as shim
from scripts.ci import summarize_connectors as sc

HEADER = (
    "| Connector | Mode | Status | Reason | HTTP 2xx/4xx/5xx (retries) | "
    "Fetched | Normalized | Written | Kept | Dropped | Parse errors | Logs | Meta rows | Meta |"
)
DIVIDER = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"


def _extract_matrix_rows(markdown: str) -> List[str]:
    lines = markdown.splitlines()
    try:
        header_index = lines.index(HEADER)
    except ValueError:
        return []
    matrix_lines: List[str] = []
    for line in lines[header_index + 2 :]:
        if not line.startswith("|"):
            break
        matrix_lines.append(line)
    return matrix_lines


def test_matrix_header_present_for_empty_entries():
    markdown = sc.build_markdown([])
    lines = markdown.splitlines()
    assert HEADER in lines
    assert DIVIDER in lines
    assert _extract_matrix_rows(markdown) == []


def test_counts_and_http_columns_have_expected_format():
    entries = [
        {
            "connector_id": "demo",
            "mode": "smoke",
            "status": "ok",
            "counts": {"fetched": 1, "normalized": 2, "written": 3},
            "http": {"2xx": 10, "4xx": 1, "5xx": 0, "retries": 4},
        }
    ]

    markdown = sc.build_markdown(entries)
    rows = _extract_matrix_rows(markdown)
    row_line = next(line for line in rows if line.startswith("| demo "))
    cells = [cell.strip() for cell in row_line.strip().strip("|").split("|")]
    assert len(cells) == 14
    assert cells[5] == "1"
    assert cells[6] == "2"
    assert cells[7] == "3"
    assert cells[8] == "3"
    assert cells[9] == "0"
    assert cells[10] == "0"
    assert re.fullmatch(r"\d+/\d+/\d+ \(\d+\)", cells[4])


def test_diagnostics_wrapper_matches_ci(tmp_path):
    diagnostics_root = tmp_path / "diagnostics" / "ingestion"
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "connector_id": "alpha",
            "status": "ok",
            "counts": {"fetched": 0, "normalized": 0, "written": 0},
        }
    ]

    markdown_ci = sc.build_markdown(entries, diagnostics_root=diagnostics_root)
    markdown_shim = shim.build_markdown(entries, diagnostics_root=diagnostics_root)
    assert markdown_ci == markdown_shim
