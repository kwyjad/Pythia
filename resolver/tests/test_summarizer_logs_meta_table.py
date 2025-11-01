from __future__ import annotations

import json

from scripts.ci.summarize_connectors import build_markdown


def test_logs_meta_table(tmp_path):
    diagnostics_root = tmp_path / "diagnostics" / "ingestion"
    logs_dir = diagnostics_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "ingestion.log").write_text("hello", encoding="utf-8")

    meta_dir = diagnostics_root / "dtm"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "run.meta.json").write_text(json.dumps({"rows_written": 3}), encoding="utf-8")

    markdown = build_markdown(
        [],
        diagnostics_root=diagnostics_root,
        staging_root=tmp_path / "resolver" / "staging",
    )

    lines = markdown.splitlines()
    assert "| Logs | Meta rows |" in lines
    assert "| --- | ---: |" in lines
    assert "| 1 | 3 |" in lines
