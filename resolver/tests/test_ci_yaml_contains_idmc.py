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
    ingest = data.get("jobs", {}).get("derive-freeze", {})
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


def test_backfill_export_and_freeze_use_db_flag() -> None:
    yaml_data = _load_yaml(WF_BACKFILL)
    derive_job = yaml_data.get("jobs", {}).get("derive-freeze", {})
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

    freeze_script = next(
        step
        for step in steps
        if isinstance(step, dict) and step.get("name") == "Derive and freeze monthly snapshots"
    ).get("run")
    assert isinstance(freeze_script, str)
    assert "--db" in freeze_script
    assert "--db-url" not in freeze_script


def test_backfill_freeze_snapshot_duckdb_step_present() -> None:
    yaml_data = _load_yaml(WF_BACKFILL)
    derive_job = yaml_data.get("jobs", {}).get("derive-freeze", {})
    steps = derive_job.get("steps", [])
    assert isinstance(steps, list)
    freeze_step = next(
        step
        for step in steps
        if isinstance(step, dict) and step.get("name") == "Derive & Freeze (write DuckDB)"
    )
    run_script = freeze_step.get("run")
    assert isinstance(run_script, str)
    assert "--facts diagnostics/ingestion/export_preview/facts.csv" in run_script
    assert "--write-db=1" in run_script
    assert '--db-url "${{ env.RESOLVER_DUCKDB_URL }}"' in run_script
    env_block = freeze_step.get("env", {}) or {}
    assert env_block.get("RESOLVER_WRITE_DB") == "1"
    assert env_block.get("WRITE_DB") == "1"
