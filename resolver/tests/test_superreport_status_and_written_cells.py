from scripts.ci import summarize_connectors as sc


MATRIX_HEADER = (
    "| Connector | Mode | Status | Reason | HTTP 2xx/4xx/5xx (retries) | "
    "Fetched | Normalized | Written | Kept | Dropped | Parse errors | Logs | Meta rows | Meta |"
)


def _row_cells(markdown: str, connector: str) -> list[str]:
    lines = markdown.splitlines()
    try:
        header_index = lines.index(MATRIX_HEADER)
    except ValueError as exc:  # pragma: no cover - helper guard
        raise AssertionError("matrix header not found") from exc
    for line in lines[header_index + 2 :]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0] == connector:
            return cells
    raise AssertionError(f"Row for {connector!r} not found")


def test_status_prefers_raw_and_written_uses_totals():
    entries = [
        {
            "connector_id": "alpha",
            "mode": "smoke",
            "status": "ok",
            "reason": "no rows",
            "http": {"2xx": 9, "4xx": 1, "5xx": 0, "retries": 2},
            "extras": {
                "status_raw": "ok-empty",
                "run_totals": {"rows_written": 5, "dropped": 2, "parse_errors": 1},
            },
        }
    ]

    markdown = sc.build_markdown(entries)
    cells = _row_cells(markdown, "alpha")

    assert cells[2] == "ok-empty"  # Status column
    assert cells[4] == "9/1/0 (2)"  # HTTP column
    assert cells[5] == "0"
    assert cells[6] == "0"
    assert cells[7] == "5"  # Written uses run_totals fallback
    assert cells[8] == "5"  # Kept mirrors rows_written when no explicit value
    assert cells[9] == "2"
    assert cells[10] == "1"
