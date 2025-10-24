from __future__ import annotations

import json
from pathlib import Path

from scripts.ci import summarize_connectors


def test_summarize_connectors_missing_report(tmp_path: Path, monkeypatch) -> None:
    report = tmp_path / "connectors_report.jsonl"
    summary = tmp_path / "summary.md"
    step_summary = tmp_path / "github_summary.md"

    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(step_summary))

    rc = summarize_connectors.main(
        [
            "--report",
            str(report),
            "--out",
            str(summary),
            "--github-step-summary",
        ]
    )
    assert rc == 0

    assert report.exists()
    lines = report.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    stub = json.loads(lines[0])
    assert stub["status"] == "error"
    assert stub["reason"] == "missing-report"

    content = summary.read_text(encoding="utf-8")
    assert "# Connector Diagnostics" in content
    assert "missing-report" in content
    assert "error" in content

    github_summary = step_summary.read_text(encoding="utf-8")
    assert "missing-report" in github_summary
