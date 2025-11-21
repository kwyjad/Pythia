"""Ensure CI workflows wire in the IDMC connector."""
from __future__ import annotations

import pathlib
import re

import yaml

WF_MONTHLY = pathlib.Path(".github/workflows/resolver-monthly.yml")
WF_BACKFILL = pathlib.Path(".github/workflows/resolver-initial-backfill.yml")


def _load_yaml(path: pathlib.Path) -> object:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_monthly_mentions_idmc() -> None:
    content = WF_MONTHLY.read_text(encoding="utf-8")
    assert "Probe IDMC reachability" in content
    assert re.search(r"idmc", content, re.IGNORECASE)
    assert _load_yaml(WF_MONTHLY)  # basic YAML sanity


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


def test_backfill_uses_db_flag_for_idmc_writer() -> None:
    data = _load_yaml(WF_BACKFILL)
    ingest = data.get("jobs", {}).get("export-duckdb", {})
    steps = ingest.get("steps", [])
    assert isinstance(steps, list)
    writer = next(
        step
        for step in steps
        if isinstance(step, dict) and step.get("name") == "Load IDMC facts into DuckDB (auxiliary)"
    )
    run_script = writer.get("run")
    assert isinstance(run_script, str)
    assert "--db \"${RESOLVER_DB_URL}\"" in run_script
    assert "--db-url" not in run_script


def test_backfill_export_uses_db_flag_and_freeze_is_disabled() -> None:
    yaml_data = _load_yaml(WF_BACKFILL)
    derive_job = yaml_data.get("jobs", {}).get("export-duckdb", {})
    steps = derive_job.get("steps", [])
    assert isinstance(steps, list)
    export_step = next(
        step
        for step in steps
        if isinstance(step, dict) and step.get("name") == "Export canonical facts"
    )
    export_run = export_step.get("run")
    assert isinstance(export_run, str)
    assert "--db \"${{ env.RESOLVER_DB_URL }}\"" in export_run
    assert "--db-url" not in export_run

    names = [step.get("name") for step in steps if isinstance(step, dict)]
    assert "Derive and freeze monthly snapshots" not in names
    assert "Derive & Freeze (write DuckDB)" not in names


def test_backfill_schedule_and_months_back_defaults() -> None:
    yaml_data = _load_yaml(WF_BACKFILL)
    assert yaml_data.get("on", {}).get("workflow_dispatch", {})
    schedule = yaml_data.get("on", {}).get("schedule", [])
    assert any(item.get("cron") == "0 0 5 * *" for item in schedule if isinstance(item, dict))
    inputs = yaml_data.get("on", {}).get("workflow_dispatch", {}).get("inputs", {})
    months_back = inputs.get("months_back", {})
    assert months_back.get("default") == "1"
