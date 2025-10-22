from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run_summary(tmp_path: Path) -> str:
    script = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "make_ai_summary.py"
    subprocess.run([sys.executable, str(script)], cwd=tmp_path, check=True)
    summary = tmp_path / ".ci" / "diagnostics" / "SUMMARY.md"
    return summary.read_text(encoding="utf-8")


def test_summary_reports_junit_and_exitcodes(tmp_path: Path) -> None:
    diag_dir = tmp_path / ".ci" / "diagnostics"
    diag_dir.mkdir(parents=True)
    exit_dir = tmp_path / ".ci" / "exitcodes"
    exit_dir.mkdir(parents=True)

    junit_content = """<?xml version='1.0' encoding='utf-8'?>\n<testsuite tests='1' failures='0' errors='0' skipped='0'>\n  <testcase classname='sample' name='test_ok'/>\n</testsuite>\n"""
    (diag_dir / "pytest-junit.xml").write_text(junit_content, encoding="utf-8")
    (exit_dir / "pytest-Linux").write_text("exit=0", encoding="utf-8")

    summary = _run_summary(tmp_path)

    assert "## Pytest Summary" in summary
    assert "Total" in summary
    assert "| `pytest-Linux` | 0" in summary


def test_summary_handles_missing_artifacts(tmp_path: Path) -> None:
    (tmp_path / ".ci" / "diagnostics").mkdir(parents=True)
    (tmp_path / ".ci" / "exitcodes").mkdir(parents=True)

    summary = _run_summary(tmp_path)

    assert "No pytest JUnit report found" in summary
    assert "No exit code files present." in summary
