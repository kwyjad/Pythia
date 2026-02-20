# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Ensure CI workflows wire in the IDMC connector."""
from __future__ import annotations

import pathlib
import re

import yaml

WF_BACKFILL = pathlib.Path(".github/workflows/resolver-initial-backfill.yml")


def _load_yaml(path: pathlib.Path) -> object:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_backfill_mentions_idmc() -> None:
    content = WF_BACKFILL.read_text(encoding="utf-8")
    assert "Probe IDMC reachability" in content
    assert re.search(r"idmc", content, re.IGNORECASE)
    assert _load_yaml(WF_BACKFILL)


def test_backfill_runs_direct_idmc_step() -> None:
    data = _load_yaml(WF_BACKFILL)
    assert isinstance(data, dict)
    ingest = data.get("jobs", {}).get("ingest", {})
    steps = ingest.get("steps", [])
    assert isinstance(steps, list)
    names = [step.get("name") for step in steps if isinstance(step, dict)]
    assert "Run IDMC (HELIX single-shot)" in names
    direct_step = next(
        step for step in steps if isinstance(step, dict) and step.get("name") == "Run IDMC (HELIX single-shot)"
    )
    run_script = direct_step.get("run")
    assert isinstance(run_script, str)
    assert "--network-mode helix" in run_script
    assert "--start \"${{ steps.window.outputs.start_iso }}\"" in run_script
    assert "--end   \"${{ steps.window.outputs.end_iso }}\"" in run_script


def test_backfill_export_duckdb_uses_load_and_derive() -> None:
    """Verify the export-duckdb job uses load_and_derive pipeline."""
    data = _load_yaml(WF_BACKFILL)
    derive_job = data.get("jobs", {}).get("export-duckdb", {})
    steps = derive_job.get("steps", [])
    assert isinstance(steps, list)
    names = [step.get("name") for step in steps if isinstance(step, dict)]
    assert "Load, derive, and export to DuckDB" in names

    lda_step = next(
        step
        for step in steps
        if isinstance(step, dict) and step.get("name") == "Load, derive, and export to DuckDB"
    )
    run_script = lda_step.get("run")
    assert isinstance(run_script, str)
    assert "load_and_derive" in run_script

    # Ensure deleted modules are NOT referenced
    all_runs = " ".join(
        step.get("run", "") for step in steps if isinstance(step, dict)
    )
    assert "resolver.tools.export_facts" not in all_runs
    assert "resolver.cli.emdat_to_duckdb" not in all_runs
    assert "resolver.cli.idmc_to_duckdb" not in all_runs


def test_backfill_schedule_and_months_back_defaults() -> None:
    yaml_data = _load_yaml(WF_BACKFILL)
    on_block = yaml_data.get("on", {})
    assert on_block, "Expected an 'on' block in resolver-initial-backfill.yml"

    workflow_dispatch = on_block.get("workflow_dispatch", {})
    assert workflow_dispatch, "Expected 'workflow_dispatch' under 'on' in resolver-initial-backfill.yml"

    inputs = workflow_dispatch.get("inputs", {})
    assert inputs, "Expected 'inputs' under workflow_dispatch"

    months_back = inputs.get("months_back", {})
    assert months_back, "Expected 'months_back' input definition"
    assert months_back.get("default") == "3"

    schedule = on_block.get("schedule", [])
    assert schedule, "Expected 'schedule' under 'on'"
    assert any(item.get("cron") == "0 3 15 * *" for item in schedule if isinstance(item, dict))
