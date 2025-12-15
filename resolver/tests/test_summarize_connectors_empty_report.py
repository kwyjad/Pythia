# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

from pathlib import Path

from scripts.ci import summarize_connectors


def test_summarize_connectors_empty_report(tmp_path: Path, monkeypatch) -> None:
    report = tmp_path / "connectors_report.jsonl"
    report.write_text("")
    summary = tmp_path / "summary.md"

    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    rc = summarize_connectors.main(
        [
            "--report",
            str(report),
            "--out",
            str(summary),
        ]
    )
    assert rc == 0

    content = summary.read_text()
    assert summarize_connectors.SUMMARY_TITLE in content
    assert "* **Connectors:** 0" in content
    assert "No connectors report was produced" not in content
