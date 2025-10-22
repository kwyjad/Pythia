from __future__ import annotations

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

    expected = summarize_connectors.MISSING_REPORT_SUMMARY
    assert summary.read_text() == expected
    assert step_summary.read_text().endswith(expected)
