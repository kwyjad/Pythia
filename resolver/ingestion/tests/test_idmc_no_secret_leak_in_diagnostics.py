# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Ensure IDMC diagnostics redact sensitive values."""
import json
from pathlib import Path

from resolver.ingestion.idmc import cli


def read_last_connector_line(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines, "connectors diagnostics must not be empty"
    return json.loads(lines[-1])


def test_idmc_connector_diagnostics_redact_tokens(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IDMC_API_TOKEN", "supersecret")

    exit_code = cli.main(["--skip-network"])
    assert exit_code == 0

    connectors_path = Path("diagnostics/ingestion/connectors.jsonl")
    assert connectors_path.exists(), "connectors.jsonl missing"

    text = connectors_path.read_text(encoding="utf-8")
    assert "supersecret" not in text
    assert "***REDACTED***" in text

    line = read_last_connector_line(connectors_path)
    env_block = line.get("run_env", {})
    assert env_block.get("IDMC_API_TOKEN") == "***REDACTED***"
