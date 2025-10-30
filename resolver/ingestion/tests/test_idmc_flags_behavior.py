"""Behavioural tests for IDMC CLI feature flags."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]


def run_cli(tmp_path, monkeypatch, *extra_args, env_extra=None):
    """Execute the IDMC CLI in an isolated directory and return the result."""

    monkeypatch.chdir(tmp_path)
    env = os.environ.copy()
    env.setdefault("IDMC_CACHE_DIR", str(tmp_path / "cache"))
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [str(REPO_ROOT)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    if env_extra:
        env.update(env_extra)

    cmd = [
        sys.executable,
        "-m",
        "resolver.ingestion.idmc.cli",
        "--skip-network",
        *extra_args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    connectors_path = tmp_path / "diagnostics" / "ingestion" / "connectors.jsonl"
    assert connectors_path.exists(), result.stderr
    last_line = connectors_path.read_text(encoding="utf-8").strip().splitlines()[-1]
    return result, json.loads(last_line)


def test_idmc_strict_empty_on_zero_rows(tmp_path, monkeypatch):
    result, line = run_cli(tmp_path, monkeypatch, "--strict-empty", "--window-days=0")

    assert result.returncode == 2
    assert line["status"] == "error"
    assert line["reason"] == "strict-empty-0-rows"
    assert line["rows_normalized"] == 0
    assert line["run_flags"]["strict_empty"] is True


def test_idmc_only_countries_filters(tmp_path, monkeypatch):
    result, line = run_cli(tmp_path, monkeypatch, "--only-countries=SDN")

    assert result.returncode == 0
    assert line["status"] == "ok"
    assert line["run_flags"]["only_countries"] == ["SDN"]
    assert line["debug"]["selected_countries_count"] == 1

    preview_path = Path(tmp_path, line["samples"]["normalized_preview"])
    preview = pd.read_csv(preview_path)
    assert (preview["iso3"] == "SDN").all()


def test_idmc_series_filter_zero_rows(tmp_path, monkeypatch):
    result, line = run_cli(tmp_path, monkeypatch, "--series=stock")

    assert result.returncode == 0
    assert line["rows_normalized"] == 0
    assert line["debug"]["selected_series"] == ["stock"]
    assert line["zero_rows"]["selectors"]["series"] == ["stock"]
