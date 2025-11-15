"""Tests for downstream error diagnostics instrumentation."""
from __future__ import annotations

import subprocess

import pandas as pd
import pytest

from scripts.ci import append_error_to_summary as append_module
from scripts.ci import append_stage_to_summary as stage_module


@pytest.fixture
def summary_path(tmp_path, monkeypatch):
    target = tmp_path / "diagnostics" / "ingestion" / "summary.md"
    monkeypatch.setattr(append_module, "SUMMARY_PATH", target)
    monkeypatch.setattr(stage_module, "SUMMARY_PATH", target)
    return target


def test_append_error_to_summary_writes_section(summary_path):
    append_module.append_error(
        "Example Section",
        "ExampleError",
        "Something went wrong",
        {"detail": "value"},
    )
    text = summary_path.read_text(encoding="utf-8")
    assert "## Example Section" in text
    assert "ExampleError" in text
    assert "Context" in text


def test_export_facts_db_error_appends(summary_path, monkeypatch):
    from resolver.tools import export_facts

    recorded: dict[str, str] = {}

    def fake_run(cmd, check=False):  # pragma: no cover - executed in test
        params: dict[str, str] = {}
        args_iter = iter(cmd[3:])
        for flag in args_iter:
            params[flag] = next(args_iter)
        append_module.append_error(
            params.get("--section", ""),
            params.get("--error-type", ""),
            params.get("--message", ""),
            append_module._load_context(params.get("--context", "{}")),
        )
        recorded["section"] = params.get("--section")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(export_facts.subprocess, "run", fake_run)

    export_facts._append_db_error_to_summary(
        section="Export Facts — DB write",
        exc=RuntimeError("boom"),
        db_url="duckdb:///tmp.db",
        facts_resolved=pd.DataFrame({"a": [1, 2]}),
        facts_deltas=pd.DataFrame({"a": [3]}),
    )

    assert recorded.get("section") == "Export Facts — DB write"
    text = summary_path.read_text(encoding="utf-8")
    assert "Export Facts — DB write" in text
    assert "\"facts_resolved_rows\": 2" in text


def test_precedence_engine_cli_error_appends(summary_path, monkeypatch):
    from resolver.tools import precedence_engine

    def fake_run(cmd, check=False):  # pragma: no cover - executed in test
        params: dict[str, str] = {}
        args_iter = iter(cmd[3:])
        for flag in args_iter:
            params[flag] = next(args_iter)
        append_module.append_error(
            params.get("--section", ""),
            params.get("--error-type", ""),
            params.get("--message", ""),
            append_module._load_context(params.get("--context", "{}")),
        )
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr(precedence_engine.subprocess, "run", fake_run)

    with pytest.raises(FileNotFoundError):
        precedence_engine.main(
            argv=[
                "--facts",
                "nonexistent.csv",
                "--cutoff",
                "2020-01-01",
            ]
        )

    text = summary_path.read_text(encoding="utf-8")
    assert "Precedence Engine — CLI error" in text
    assert "nonexistent.csv" in text


def test_append_stage_writes_section(summary_path):
    stage_module.append_stage(
        "Derive-freeze stage: Export canonical facts (start)",
        status="start",
        details="Begin export",
        context={"month_count": 3},
    )
    text = summary_path.read_text(encoding="utf-8")
    assert "Derive-freeze stage: Export canonical facts" in text
    assert "**Status:** start" in text
    assert "month_count" in text
