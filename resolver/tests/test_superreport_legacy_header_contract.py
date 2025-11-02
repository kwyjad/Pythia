from __future__ import annotations

from typing import List

from scripts.ci import summarize_connectors as sc

HEADER = (
    "| Connector | Mode | Status | Reason | HTTP 2xx/4xx/5xx (retries) | Fetched | Normalized | "
    "Written | Kept | Dropped | Parse errors | Logs | Meta rows | Meta |"
)
DIVIDER = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"


def _matrix_lines(markdown: str) -> List[str]:
    lines = markdown.splitlines()
    try:
        start = lines.index(HEADER)
    except ValueError:
        return []
    table_lines: List[str] = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        table_lines.append(line)
    return table_lines


def test_header_rendered_even_without_entries():
    markdown = sc.build_markdown([])
    lines = markdown.splitlines()
    assert HEADER in lines
    assert lines[lines.index(HEADER) + 1] == DIVIDER
    assert _matrix_lines(markdown) == []


def test_header_matches_expected_columns():
    entry = {
        "connector_id": "alpha",
        "mode": "smoke",
        "status": "ok",
        "counts": {"fetched": 1, "normalized": 2, "written": 3},
        "http": {"2xx": 1, "4xx": 0, "5xx": 0, "retries": 0},
    }
    markdown = sc.build_markdown([entry])
    lines = markdown.splitlines()
    assert HEADER in lines
    assert lines[lines.index(HEADER) + 1] == DIVIDER
    rows = _matrix_lines(markdown)
    assert len(rows) == 1
    cells = [cell.strip() for cell in rows[0].strip().strip("|").split("|")]
    assert len(cells) == 14
    assert cells[:5] == ["alpha", "smoke", "ok", "—", "1/0/0 (0)"]
    assert cells[5:11] == ["1", "2", "3", "3", "0", "0"]
    assert cells[11:] == ["—", "—", "—"]
