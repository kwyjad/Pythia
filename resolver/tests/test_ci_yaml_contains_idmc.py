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
    assert re.search(r"idmc", content, re.IGNORECASE)
    assert _load_yaml(WF_BACKFILL)


def test_backfill_runs_direct_idmc_step() -> None:
    data = _load_yaml(WF_BACKFILL)
    assert isinstance(data, dict)
    backfill = data.get("jobs", {}).get("backfill", {})
    steps = backfill.get("steps", [])
    assert isinstance(steps, list)
    names = [step.get("name") for step in steps if isinstance(step, dict)]
    assert "Phase 1: Run IDMC (HELIX)" in names
    direct_step = next(
        step for step in steps if isinstance(step, dict) and step.get("name") == "Phase 1: Run IDMC (HELIX)"
    )
    run_script = direct_step.get("run")
    assert isinstance(run_script, str)
    assert "--network-mode helix" in run_script
    assert "--start \"${{ steps.window.outputs.start_iso }}\"" in run_script
    assert "--end   \"${{ steps.window.outputs.end_iso }}\"" in run_script


def test_backfill_uses_load_and_derive() -> None:
    """Verify the backfill job uses load_and_derive pipeline."""
    data = _load_yaml(WF_BACKFILL)
    backfill_job = data.get("jobs", {}).get("backfill", {})
    steps = backfill_job.get("steps", [])
    assert isinstance(steps, list)
    names = [step.get("name") for step in steps if isinstance(step, dict)]
    assert "Phase 1: Normalize + load_and_derive → facts_resolved" in names

    lda_step = next(
        step
        for step in steps
        if isinstance(step, dict) and step.get("name") == "Phase 1: Normalize + load_and_derive → facts_resolved"
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


def test_backfill_has_context_source_phases() -> None:
    """Verify the consolidated backfill includes context source phases."""
    data = _load_yaml(WF_BACKFILL)
    backfill_job = data.get("jobs", {}).get("backfill", {})
    steps = backfill_job.get("steps", [])
    names = [step.get("name", "") for step in steps if isinstance(step, dict)]

    # Phase 2: Resolution sources
    assert any("FEWS NET" in n for n in names)
    assert any("GDACS" in n for n in names)

    # Phase 3: Structured data
    assert any("conflict" in n.lower() for n in names)

    # Phase 4: Context sources
    assert any("ENSO" in n for n in names)
    assert any("Seasonal TC" in n for n in names)
    assert any("HDX" in n for n in names)
    assert any("CrisisWatch" in n for n in names)
