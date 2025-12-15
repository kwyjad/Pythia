# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Smoke-level checks for resolver.tools.load_and_derive CLI usage."""

from __future__ import annotations

import shlex
import sys

from resolver.tests.utils import run as run_proc

def test_load_and_derive_accepts_period_before_subcommand() -> None:
    cmd = (
        f"{shlex.quote(sys.executable)} -m resolver.tools.load_and_derive"
        " --period ci-smoke"
        " load-canonical"
        " --in data/staging/ci-smoke/canonical"
    )
    result = run_proc(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )
    combined_output = (result.stdout or "") + (result.stderr or "")
    assert not (
        result.returncode == 2
        and "the following arguments are required: --period" in combined_output
    ), (
        "CLI rejected --period before the subcommand:\n"
        f"exit={result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "the following arguments are required: --period" not in combined_output
