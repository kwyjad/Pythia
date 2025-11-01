from __future__ import annotations

import json
from scripts.ci.summarize_connectors import main


def test_missing_report_stub(tmp_path):
    report = tmp_path / "report.jsonl"
    out = tmp_path / "summary.md"
    diagnostics = tmp_path / "diagnostics"
    staging = tmp_path / "staging"

    exit_code = main(
        [
            "--report",
            str(report),
            "--diagnostics",
            str(diagnostics),
            "--staging",
            str(staging),
            "--out",
            str(out),
        ]
    )

    assert exit_code == 0
    lines = [line.strip() for line in report.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["status"] == "error"
    assert payload["reason"] == "missing or empty report"
    summary_text = out.read_text(encoding="utf-8")
    assert "# Ingestion Superreport" in summary_text
    assert "* **Connectors:** 0" in summary_text
