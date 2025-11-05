from __future__ import annotations

import sys
from pathlib import Path


from resolver.tests.utils import run as run_proc


def _run_summary(tmp_path: Path) -> str:
    script = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "make_ai_summary.py"
    run_proc([sys.executable, str(script)], cwd=tmp_path, check=True)
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
    assert "- **JUnit report:** `.ci/diagnostics/pytest-junit.xml`" in summary
    assert "- **Totals:** tests=1 | passed=1 | failures=0 | errors=0 | skipped=0" in summary
    assert "| `pytest-Linux` | 0 | exit=0 |" in summary


def test_summary_infers_totals_from_tail_when_junit_missing(tmp_path: Path) -> None:
    diag_dir = tmp_path / ".ci" / "diagnostics"
    diag_dir.mkdir(parents=True)
    tail_path = diag_dir / "pytest-main.tail.txt"
    tail_lines = [
        "============================= test session starts =============================",
        "collected 2 items",
        "FAILED resolver/tests/test_example.py::test_failure - AssertionError: oh no",
        "PASSED resolver/tests/test_example.py::test_success",
        "=========================== short test summary info ===========================",
        "FAILED resolver/tests/test_example.py::test_failure - AssertionError",
        "========================= 1 failed, 1 passed in 0.12s =========================",
    ]
    tail_path.write_text("\n".join(tail_lines) + "\n", encoding="utf-8")

    summary = _run_summary(tmp_path)

    assert "- **JUnit report:** missing" in summary
    assert "- **Totals:** tests=2 | passed=1 | failures=1 | errors=0 | skipped=0" in summary
    assert "### Top failures" in summary
    assert "- resolver/tests/test_example.py::test_failure" in summary
