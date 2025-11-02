from scripts.ci import summarize_connectors as sc


MATRIX_HEADER = (
    "| Connector | Mode | Status | Reason | HTTP 2xx/4xx/5xx (retries) | "
    "Fetched | Normalized | Written | Kept | Dropped | Parse errors | Logs | Meta rows | Meta |"
)
MATRIX_DIVIDER = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"


def _extract_matrix_rows(markdown: str) -> list[str]:
    lines = markdown.splitlines()
    try:
        header_index = lines.index(MATRIX_HEADER)
    except ValueError:
        return []
    rows: list[str] = []
    for line in lines[header_index + 2 :]:
        if not line.startswith("|"):
            break
        rows.append(line)
    return rows


def test_matrix_header_matches_legacy_contract():
    markdown = sc.build_markdown([])
    lines = markdown.splitlines()
    assert MATRIX_HEADER in lines
    assert MATRIX_DIVIDER in lines
    assert lines[lines.index(MATRIX_HEADER) + 1] == MATRIX_DIVIDER
    assert "Country" not in MATRIX_HEADER
    assert "Counts f/n/w" not in MATRIX_HEADER
    assert _extract_matrix_rows(markdown) == []
