"""Regression coverage for summarizer literals expected by fast tests."""
from __future__ import annotations

from scripts.ci import summarize_connectors as sc


def _mk_entry(**overrides):
    base = {"status": "ok", "extras": {}, "counts": {}}
    base.update(overrides)
    return base


def test_config_line_literal_present(tmp_path):
    diag_root = tmp_path / "diagnostics" / "ingestion"
    diag_root.mkdir(parents=True, exist_ok=True)
    entries = [
        _mk_entry(
            extras={
                "config": {
                    "config_source_label": "resolver",
                    "config_path_used": "resolver/config/dtm.yml",
                    "config_warnings": [],
                }
            }
        )
    ]
    markdown = sc.build_markdown(entries, diagnostics_root=diag_root)
    assert "Config source:" in markdown
    assert "Config: resolver/config/dtm.yml" in markdown


def test_selector_phrase_and_zero_row_primary_reason():
    entries = [
        _mk_entry(status="ok-empty", extras={"reason": "no-data"}, counts={"written": 0})
    ]
    markdown = sc.build_markdown(entries)
    assert "## Selector effectiveness" in markdown
    assert "Top selectors by rows" in markdown
    assert "## Zero-row root cause" in markdown
    assert "Primary reason" in markdown
    assert "Selectors with rows" in markdown


def test_logs_meta_three_column_headers(tmp_path):
    diag_root = tmp_path / "diagnostics" / "ingestion"
    logs_dir = diag_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "alpha.log").write_text("x", encoding="utf-8")
    markdown = sc.build_markdown([], diagnostics_root=diag_root)
    header = (
        "| Connector | Mode | Status | Reason | HTTP 2xx/4xx/5xx (retries) | Fetched | Normalized | Written | Kept | Dropped | Parse errors | Logs | Meta rows | Meta |"
    )
    assert header in markdown
