# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Smoke test for the offline IDMC CLI."""
import json
import os
import sys


from resolver.tests.utils import run as run_proc


def read_last_connector_line() -> dict:
    path = os.path.join("diagnostics", "ingestion", "connectors.jsonl")
    assert os.path.exists(path), "connectors.jsonl missing"
    *_, last = open(path, "r", encoding="utf-8").read().strip().splitlines()
    return json.loads(last)


def test_idmc_offline_cli_smoke(tmp_path, monkeypatch):
    os.makedirs(os.path.join("diagnostics", "ingestion", "idmc"), exist_ok=True)
    cmd = [sys.executable, "-m", "resolver.ingestion.idmc.cli", "--skip-network"]
    result = run_proc(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    line = read_last_connector_line()
    assert line["connector"] == "idmc"
    assert line["status"] == "ok"
    assert line["rows_normalized"] >= 0
    assert "samples" in line and "normalized_preview" in line["samples"]
    assert os.path.exists(line["samples"]["normalized_preview"])
