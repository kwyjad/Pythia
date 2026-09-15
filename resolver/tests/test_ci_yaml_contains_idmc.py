# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Ensure CI workflows wire in the IDMC connector."""
from __future__ import annotations

import pathlib
import re

import yaml

WF_BACKFILL = pathlib.Path(".github/workflows/resolver_update.yml")


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
    assert on_block, "Expected an 'on' block in resolver_update.yml"

    workflow_dispatch = on_block.get("workflow_dispatch", {})
    assert workflow_dispatch, "Expected 'workflow_dispatch' under 'on' in resolver_update.yml"

    inputs = workflow_dispatch.get("inputs", {})
    assert inputs, "Expected 'inputs' under workflow_dispatch"

    months_back = inputs.get("months_back", {})
    assert months_back, "Expected 'months_back' input definition"
    assert months_back.get("default") == "3"

    schedule = on_block.get("schedule", [])
    assert schedule, "Expected 'schedule' under 'on'"
    assert any(item.get("cron") == "0 3 28 * *" for item in schedule if isinstance(item, dict))


def test_backfill_ingests_before_the_monthly_forecast() -> None:
    """Resolver Update must run late in the month, ahead of the forecast cron.

    This workflow is the ONLY refresh for the tier-0 resolution sources (ACLED
    monthly fatalities, IFRC, IDMC, FEWS NET / IPC); the weekly
    ingest-structured-data.yml run covers only supplementary fast-changing
    sources. While this ran on the 15th and the pipeline on the 1st, every
    monthly forecast reasoned over core data ~17 days stale. Pin the ordering
    so a future cron edit cannot silently reintroduce that gap.
    """

    def _crons(path: pathlib.Path) -> list[str]:
        data = _load_yaml(path)
        assert isinstance(data, dict)
        # PyYAML parses a bare `on:` key as the boolean True.
        on_block = data.get("on") or data.get(True) or {}
        schedule = on_block.get("schedule", []) if isinstance(on_block, dict) else []
        return [
            str(item.get("cron"))
            for item in schedule
            if isinstance(item, dict) and item.get("cron")
        ]

    ingest_days = {c.split()[2] for c in _crons(WF_BACKFILL)}
    assert ingest_days == {"28"}, f"Resolver Update day-of-month drifted: {ingest_days}"

    pipeline = pathlib.Path(".github/workflows/pythia_pipeline_stage.yml")
    forecast_days = {c.split()[2] for c in _crons(pipeline) if c.split()[2] != "*"}
    assert forecast_days == {"1"}, f"Forecast pipeline day-of-month drifted: {forecast_days}"


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


def test_machine_inputs_land_before_the_drought_step() -> None:
    """The drought gate reads hdx_signals and seasonal_forecasts from the DB
    the job is holding, and the ladder's cap reads haz_raw_population. Until
    2026-09-03 the HDX store and NMME ingest ran AFTER Phase 2.5, so the gate
    always read last cycle's rows, and nothing loaded the population table.
    """

    data = _load_yaml(WF_BACKFILL)
    steps = data["jobs"]["backfill"]["steps"]
    names = [step.get("name", "") for step in steps if isinstance(step, dict)]

    def _index(fragment: str) -> int:
        matches = [i for i, n in enumerate(names) if fragment in n]
        assert matches, f"no step named like {fragment!r}: {names}"
        return matches[0]

    drought = _index("PA machine — drought")
    assert _index("HDX Signals") < drought
    assert _index("NMME") < drought
    assert _index("population") < drought
    # And the base rates read the drought rows, so they stay after it.
    assert _index("base rates") > drought


def test_machine_steps_are_wrapped_in_timeout_and_budgeted() -> None:
    """A step-level timeout SIGKILLs the shell and the ::warning branch never
    fires; coreutils `timeout` exits 124 which pipefail carries to it."""

    data = _load_yaml(WF_BACKFILL)
    steps = data["jobs"]["backfill"]["steps"]
    machine = [
        step for step in steps
        if isinstance(step, dict) and "PA machine —" in step.get("name", "")
    ]
    # flood, cyclone, drought, plus the ceiling reconsideration pass (Sept 2026)
    names = [step.get("name", "") for step in machine]
    assert len(machine) == 4, names
    assert any("reconsider ceiling rejections" in name for name in names), names
    for step in machine:
        run = step.get("run", "")
        assert "timeout " in run and "python -m resolver.hazard_resolution.cli" in run
        assert int(step.get("timeout-minutes", 0)) >= 30


def test_post_upload_diagnostics_never_red_the_ingest() -> None:
    data = _load_yaml(WF_BACKFILL)
    steps = data["jobs"]["backfill"]["steps"]
    names = [step.get("name", "") for step in steps if isinstance(step, dict)]
    upload = names.index("Upload canonical resolver DB")
    for step in steps[upload + 1:]:
        if not isinstance(step, dict) or "uses" not in step:
            continue
        if "upload-artifact" in step["uses"] and "reset" not in step.get("name", ""):
            assert step.get("continue-on-error") is True, step.get("name")


def test_resolver_update_is_gated_on_the_staged_pipeline() -> None:
    data = _load_yaml(WF_BACKFILL)
    jobs = data["jobs"]
    assert "gate" in jobs
    assert jobs["backfill"].get("needs") == "gate"
    assert "proceed" in str(jobs["backfill"].get("if", ""))


# ---------------------------------------------------------------------------
# Forecast-pipeline audit pins (2026-09-15)
# ---------------------------------------------------------------------------

WF_SIBYL = pathlib.Path(".github/workflows/run_sibyl.yml")
WF_PUBLISH = pathlib.Path(".github/workflows/publish_latest_data.yml")
WF_STAGE = pathlib.Path(".github/workflows/pythia_pipeline_stage.yml")
WF_POLLER = pathlib.Path(".github/workflows/poll_llm_batches.yml")
WF_INGEST = pathlib.Path(".github/workflows/ingest-structured-data.yml")
WF_BACKCAST = pathlib.Path(".github/workflows/haz_backcast.yml")


def _steps(path: pathlib.Path, job: str) -> list[dict]:
    data = _load_yaml(path)
    assert isinstance(data, dict)
    return list(data["jobs"][job]["steps"])


def _step(path: pathlib.Path, job: str, name: str) -> dict:
    for step in _steps(path, job):
        if step.get("name") == name:
            return step
    raise AssertionError(f"{path}: job {job} has no step named {name!r}")


def test_sibyl_publish_dispatch_is_the_chain_not_a_human_pick() -> None:
    """publish's regression guard hard-fails an explicit run_id with no chain
    flag — the mode reserved for a person's deliberate run pick. Sibyl is the
    forecast chain's SOLE publish trigger and must take the soft-skip branch,
    as the calibration chain has since 2026-08-26."""
    step = _step(WF_SIBYL, "sibyl", "Publish post-Sibyl DB to release")
    assert "-f dispatched_by_chain=true" in step["run"]
    assert "-f run_id=${{ github.run_id }}" in step["run"]


def test_sibyl_failure_cannot_withhold_the_forecast_release() -> None:
    """The gate and the Sibyl run sit between the DB download and the
    canonical upload; a red there left the month's forecasts unpublished."""
    for name in ("Gate on eligible questions with volatility scores", "Run Sibyl"):
        step = _step(WF_SIBYL, "sibyl", name)
        assert step.get("continue-on-error") is True, f"{name} must be non-fatal"
    run_sibyl = _step(WF_SIBYL, "sibyl", "Run Sibyl")
    # A gate that could not decide leaves the output empty: Sibyl is skipped, not run blind.
    assert "steps.gate.outputs.eligible != ''" in str(run_sibyl.get("if"))
    data = _load_yaml(WF_SIBYL)
    assert int(data["jobs"]["sibyl"]["timeout-minutes"]) >= 330


def test_publish_diagnostics_after_the_release_upload_are_non_fatal() -> None:
    """A crash in the inspection after `gh release upload` turned a completed
    publish red, skipped the API sync poke, and aged Publish in the watchdog."""
    data = _load_yaml(WF_PUBLISH)
    job = next(iter(data["jobs"].values()))
    names = [s.get("name") for s in job["steps"]]
    upload_idx = next(i for i, n in enumerate(names) if n and n.startswith("Upload resolver.duckdb"))
    for step in job["steps"][upload_idx + 1:]:
        if step.get("name") in ("Inspect published DB", "Upload resolver inspection artifact"):
            assert step.get("continue-on-error") is True, step.get("name")


def test_db_concurrency_group_is_on_the_writing_job_not_the_gate() -> None:
    """At workflow level the gate job (which usually SKIPS) took the group's
    single pending slot before deciding anything, and a newer queued run
    cancels whatever is pending — a poller-dispatched stage or the
    1st-of-month hs_submit cron itself."""
    for path, job in ((WF_INGEST, "ingest"), (WF_BACKCAST, "backcast")):
        data = _load_yaml(path)
        assert "concurrency" not in data, f"{path}: concurrency must not be workflow-level"
        assert data["jobs"][job]["concurrency"]["group"] == "pythia-resolver-db", path
        assert "concurrency" not in data["jobs"]["gate"], f"{path}: the gate must stay outside the group"


def test_collect_stages_have_room_for_the_in_process_wait_and_the_sync_fallback() -> None:
    data = _load_yaml(WF_STAGE)
    for job in ("hs-rc-collect", "hs-finalize-fc-submit", "fc-collect-finalize"):
        assert int(data["jobs"][job]["timeout-minutes"]) == 360, job
    # The re-batch wait must leave the sync fallback most of the stage.
    env = data["env"]
    assert float(env["PYTHIA_BATCH_RESUBMIT_WAIT_MIN"]) <= 120


def test_signature_regression_gate_only_fires_when_the_compare_ran() -> None:
    """With `!= 'true'` an earlier step failure (empty output) made every red
    stage end on a false 'DB signature regressed' annotation."""
    data = _load_yaml(WF_STAGE)
    seen = 0
    for job in data["jobs"].values():
        for step in job.get("steps", []):
            if str(step.get("name", "")).startswith(("Fail stage on DB signature regression", "Fail if canonical DB was not published")):
                seen += 1
                cond = str(step.get("if"))
                assert "steps.signature_after.outcome == 'success'" in cond
                assert "signature_ok == 'false'" in cond
    assert seen == 5


def test_fc_collect_resolves_the_epoch_from_stage_state_before_newest_hs_run() -> None:
    """A submit stage where every provider group failed is GREEN with no
    llm_batches row; the durable hs_stage_state mapping is this pipeline's
    epoch, and "newest hs_runs" would pin a manual run's question set."""
    step = _step(WF_STAGE, "fc-collect-finalize", "Read pipeline context from DB")
    run = step["run"]
    assert "hs_stage_state" in run and "pipeline_run_id" in run
    assert run.index("hs_stage_state") < run.index("FROM hs_runs ORDER BY generated_at DESC")


def test_poller_rearm_dispatch_retries_before_breaking_the_chain() -> None:
    step = _step(WF_POLLER, "poll", "Reschedule next poll")
    assert "for attempt in" in step["run"] and "gh workflow run poll_llm_batches.yml" in step["run"]
