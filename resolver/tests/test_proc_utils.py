"""Regression tests for resolver.tests.utils.proc helpers."""

from __future__ import annotations

import sys

from resolver.tests.utils import run as run_proc


def test_proc_run_default_no_raise_nonzero() -> None:
    """Ensure helper returns CompletedProcess even on non-zero exit codes."""

    result = run_proc(
        [sys.executable, "-c", "import sys; sys.exit(3)"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 3
