# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""CLI smoke tests for the IFRC GO stub."""

from __future__ import annotations

import sys
from pathlib import Path

from resolver.tests.utils import run as run_proc

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_stub(output: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "resolver.ingestion.ifrc_go_stub",
        "--out",
        str(output),
    ]
    run_proc(cmd, check=True, cwd=_repo_root(), capture_output=True, text=True)


def test_ifrc_stub_accepts_directory(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _run_stub(raw_dir)

    csv_path = raw_dir / "ifrc_go.csv"
    assert csv_path.exists(), "stub should write a CSV inside the provided directory"
    header = csv_path.read_text().splitlines()[0]
    assert header.startswith("event_id"), "CSV should include canonical header row"


def test_ifrc_stub_accepts_file_path(tmp_path: Path) -> None:
    csv_path = tmp_path / "custom.csv"
    _run_stub(csv_path)

    assert csv_path.exists(), "stub should respect explicit CSV file path"
    lines = csv_path.read_text().splitlines()
    assert lines, "CSV should not be empty"
    assert lines[0].startswith("event_id")
