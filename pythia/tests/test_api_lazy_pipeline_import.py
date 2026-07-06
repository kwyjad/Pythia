# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Importing the API must not pull in the pipeline module tree.

``pythia.pipeline.run`` imports the full horizon_scanner + calibration
stack at module scope (~100-300MB RSS). The API process on Render is
memory-constrained, so ``pythia.api.app`` must defer that import to the
``/v1/run`` handler body (which is itself gated off by default).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

REPO_ROOT = Path(__file__).resolve().parents[2]

_FORBIDDEN_PREFIXES = ("horizon_scanner", "forecaster", "pythia.pipeline")


def test_importing_api_app_does_not_load_pipeline_modules():
    code = (
        "import sys\n"
        "import pythia.api.app\n"
        f"bad = [m for m in sys.modules if m.startswith({_FORBIDDEN_PREFIXES!r})]\n"
        "assert not bad, f'pipeline modules loaded at API import time: {bad}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"API import pulled in pipeline modules:\n{result.stdout}\n{result.stderr}"
    )
