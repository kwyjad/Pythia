from __future__ import annotations
import json
from pathlib import Path
from resolver.diagnostics.ingestion import summarize_connectors as sc

def test_meta_rows_zero_renders_em_dash(tmp_path):
    # Arrange dirs to mimic diagnostics layout used by summarizer
    diag_root = tmp_path / "diagnostics" / "ingestion"
    dtm_dir = diag_root / "dtm_client"
    dtm_dir.mkdir(parents=True, exist_ok=True)

    # Create a *.meta.json file with zero rows; summarizer aggregates these
    # (gather_meta_json_files() commonly globs **/*.meta.json)
    meta_path = dtm_dir / "flow.meta.json"
    meta_path.write_text(json.dumps({"rows": 0}), encoding="utf-8")

    # Minimal entry ensures the connector name is recognized
    entries = [{
        "connector_id": "dtm_client",
        "status": "ok",
        "counts": {"fetched": 0, "normalized": 0, "written": 0},
        "http": {"2xx": 0, "4xx": 0, "5xx": 0, "retries": 0},
        "extras": {}
    }]

    # Act
    content = sc.build_markdown(
        entries,
        diagnostics_root=diag_root.as_posix(),
        staging_root=(tmp_path / "resolver" / "staging").as_posix(),
    )

    # Assert: find the row line and check "Meta rows" displays an em dash (—), not '0'
    # We look for the dtm row; the matrix has many columns, but presence of " | — | " near the tail
    # is sufficient to disambiguate Meta rows when rows==0.
    lines = [ln for ln in content.splitlines() if ln.strip().startswith("| dtm_client |")]
    assert lines, "Expected to find a dtm_client row in the diagnostics matrix"
    row = lines[0]
    assert " | — | " in row, f"Expected em dash for meta rows in line: {row}"
    assert " | 0 | " not in row, f"Zero must render as em dash in meta rows: {row}"
