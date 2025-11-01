"""Regression coverage for summarizer back-compat expectations."""
from __future__ import annotations

from pathlib import Path


from scripts.ci import summarize_connectors as sc


def test_display_reason_aliases_missing_report() -> None:
    assert sc._display_reason("missing or empty report") == "missing-report"


def test_build_markdown_has_dual_titles_and_table_header(tmp_path: Path) -> None:
    diagnostics = tmp_path / "diagnostics" / "ingestion"
    logs_dir = diagnostics / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "alpha_client.log").write_text("hello", encoding="utf-8")

    meta_dir = diagnostics / "alpha_client"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "run.meta.json").write_text("{""rows_written"": 2}", encoding="utf-8")

    staging_dir = tmp_path / "resolver" / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    entries = [
        {
            "connector_id": "alpha_client",
            "status": "ok",
            "reason": "",
            "counts": {"fetched": 1, "normalized": 1, "written": 1},
            "extras": {"status_raw": "ok"},
        }
    ]

    content = sc.build_markdown(entries, diagnostics_root=diagnostics, staging_root=staging_dir)
    assert "# Connector Diagnostics" in content
    assert "# Ingestion Superreport" in content
    assert "| Connector | Status | Reason | Logs | Meta rows | Meta |" in content
