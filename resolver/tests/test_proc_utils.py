"""Regression tests for resolver.tests.utils.proc helpers."""

from __future__ import annotations

import subprocess
import sys
from textwrap import dedent

import pytest

from resolver.tests.utils import run as run_proc


def test_proc_run_default_no_raise_nonzero() -> None:
    """Ensure helper returns CompletedProcess even on non-zero exit codes."""

    result = run_proc(
        [sys.executable, "-c", "import sys; sys.exit(3)"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 3


def test_proc_run_sets_offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Helper applies offline defaults when vars are absent."""

    monkeypatch.delenv("IDMC_NETWORK_MODE", raising=False)
    script = "import os; print(os.environ['IDMC_NETWORK_MODE'])"
    result = run_proc(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "fixture"


def test_proc_run_respects_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit env overrides should win over defaults."""

    monkeypatch.delenv("IDMC_NETWORK_MODE", raising=False)
    script = "import os; print(os.environ['IDMC_NETWORK_MODE'])"
    result = run_proc(
        [sys.executable, "-c", script],
        env={"IDMC_NETWORK_MODE": "live"},
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "live"


def test_proc_run_uses_env_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """The helper honours PYTEST_PROC_TIMEOUT when no explicit timeout is given."""

    monkeypatch.setenv("PYTEST_PROC_TIMEOUT", "0.1")
    with pytest.raises(subprocess.TimeoutExpired):
        run_proc(
            [
                sys.executable,
                "-c",
                dedent(
                    """
                    import time
                    time.sleep(5)
                    """
                ),
            ],
            capture_output=True,
            text=True,
        )
