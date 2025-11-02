from __future__ import annotations

import json

from scripts.ci import summarize_connectors as sc


def _line_with_prefix(markdown: str, prefix: str) -> str:
    for line in markdown.splitlines():
        if line.startswith(prefix):
            return line
    raise AssertionError(f"Missing line starting with {prefix!r}")


def test_config_line_from_entry_metadata(tmp_path):
    diagnostics_root = tmp_path / "diagnostics" / "ingestion"
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "connector_id": "dtm",
            "status": "ok",
            "extras": {
                "config": {
                    "config_source_label": "resolver",
                    "config_path_used": "resolver/config/dtm.yml",
                }
            },
        }
    ]

    markdown = sc.build_markdown(entries, diagnostics_root=diagnostics_root)
    assert _line_with_prefix(markdown, "Config source:") == "Config source: resolver"
    assert _line_with_prefix(markdown, "Config:") == "Config: resolver/config/dtm.yml"


def test_config_line_from_why_zero_fallback(tmp_path):
    diagnostics_root = tmp_path / "diagnostics" / "ingestion"
    why_dir = diagnostics_root / "demo"
    why_dir.mkdir(parents=True, exist_ok=True)
    (why_dir / "why_zero.json").write_text(
        json.dumps({"config_path_used": "resolver/config/dtm.yml"}),
        encoding="utf-8",
    )

    entries = [{"connector_id": "demo", "status": "ok", "extras": {}}]

    markdown = sc.build_markdown(entries, diagnostics_root=diagnostics_root)
    assert _line_with_prefix(markdown, "Config source:") == "Config source: why_zero.json"
    assert _line_with_prefix(markdown, "Config:") == "Config: resolver/config/dtm.yml"


def test_config_line_uses_dtm_default_when_missing(tmp_path):
    diagnostics_root = tmp_path / "diagnostics" / "ingestion"
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    entries = [{"connector_id": "dtm-smoke", "status": "ok", "extras": {}}]

    markdown = sc.build_markdown(entries, diagnostics_root=diagnostics_root)
    assert _line_with_prefix(markdown, "Config:") == "Config: resolver/config/dtm.yml"


def test_config_line_unknown_for_non_dtm(tmp_path):
    diagnostics_root = tmp_path / "diagnostics" / "ingestion"
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    entries = [{"connector_id": "beta", "status": "ok", "extras": {}}]

    markdown = sc.build_markdown(entries, diagnostics_root=diagnostics_root)
    assert _line_with_prefix(markdown, "Config:") == "Config: unknown"
