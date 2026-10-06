# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""The canonical-DB guards from the 5 October 2026 incident.

A Resolver Update skipped by its pipeline gate concluded "success", uploaded
nothing, and started the resolutions chain; Compute SPD Scores then uploaded
a DB it had downloaded before an NMME refetch landed, and the refetch was
lost. These tests hold the two checks that stop both.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pytest

from scripts.ci import canonical_guard as cg

REPO_ID = 1234


def _artifact(run_id, created, *, name=cg.ARTIFACT, repo=REPO_ID, branch="main", expired=False):
    return {
        "name": name, "expired": expired, "created_at": created,
        "workflow_run": {"id": run_id, "head_repository_id": repo, "head_branch": branch},
    }


def _no_sleep(_s):
    return None


@pytest.fixture()
def env(monkeypatch, tmp_path):
    out = tmp_path / "out.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setenv("GITHUB_REPOSITORY", "kwyjad/pythia")
    monkeypatch.setenv("GITHUB_RUN_ID", "900")
    monkeypatch.setenv("PYTHIA_GATE_ATTEMPTS", "2")
    monkeypatch.setenv("PYTHIA_GATE_BACKOFF_SEC", "0")
    monkeypatch.delenv("DOWNLOADED_RUN_ID", raising=False)
    monkeypatch.delenv("DOWNLOAD_SOURCE", raising=False)
    return out


def _outputs(path: Path) -> dict:
    if not path.exists():
        return {}
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


# --- trigger-did-work --------------------------------------------------------


def test_a_chained_run_exits_early_when_its_trigger_uploaded_nothing(env, monkeypatch, capsys):
    """The 5 October shape: the gated Resolver Update uploaded only diagnostics."""
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_run")
    monkeypatch.setenv("TRIGGER_RUN_ID", "555")
    gh = lambda args: {"artifacts": [_artifact(555, "2026-10-05T13:33:00Z", name="backfill-diagnostics")]}
    assert cg.main(["trigger-did-work"], gh=gh, sleep=_no_sleep) == 0
    assert _outputs(env)["did_work"] == "false"
    assert "Trigger did no work" in capsys.readouterr().out


def test_a_chained_run_proceeds_when_its_trigger_uploaded_the_canonical_db(env, monkeypatch):
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_run")
    monkeypatch.setenv("TRIGGER_RUN_ID", "555")
    gh = lambda args: {"artifacts": [_artifact(555, "2026-10-05T13:33:00Z")]}
    cg.main(["trigger-did-work"], gh=gh, sleep=_no_sleep)
    assert _outputs(env)["did_work"] == "true"


def test_an_expired_canonical_artifact_is_not_work_done():
    assert not cg.run_uploaded_canonical([_artifact(1, "2026-10-01T00:00:00Z", expired=True)])


def test_a_manual_dispatch_is_never_held_back(env, monkeypatch):
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.delenv("TRIGGER_RUN_ID", raising=False)
    cg.main(["trigger-did-work"], gh=lambda a: pytest.fail("no API call expected"), sleep=_no_sleep)
    assert _outputs(env)["did_work"] == "true"


def test_an_unreadable_trigger_proceeds_with_a_warning(env, monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_run")
    monkeypatch.setenv("TRIGGER_RUN_ID", "555")

    def boom(_args):
        raise RuntimeError("HTTP 502")

    cg.main(["trigger-did-work"], gh=boom, sleep=_no_sleep)
    assert _outputs(env)["did_work"] == "true"
    assert "::warning" in capsys.readouterr().out


# --- upload-guard ---------------------------------------------------------------


def _listing(*artifacts):
    def gh(args):
        if args[1].endswith("kwyjad/pythia"):
            return {"id": REPO_ID}
        return {"artifacts": list(artifacts)}
    return gh


def test_upload_is_refused_when_a_newer_canonical_exists(env, capsys):
    """Compute SPD Scores downloaded run 100's DB; an NMME refetch (run 200)
    uploaded after it. Uploading now would discard run 200's rows."""
    gh = _listing(
        _artifact(200, "2026-10-05T15:00:00Z"),
        _artifact(100, "2026-10-05T12:00:00Z"),
    )
    rc = cg.main(["upload-guard", "--downloaded-run-id", "100", "--source", "discovered"],
                 gh=gh, sleep=_no_sleep)
    assert rc == 1
    assert "A newer canonical DB exists" in capsys.readouterr().out


def test_upload_is_allowed_when_the_downloaded_db_is_still_the_newest(env):
    gh = _listing(_artifact(100, "2026-10-05T12:00:00Z"), _artifact(50, "2026-10-04T12:00:00Z"))
    assert cg.main(["upload-guard", "--downloaded-run-id", "100"], gh=gh, sleep=_no_sleep) == 0


def test_this_runs_own_earlier_upload_and_foreign_artifacts_are_ignored(env):
    gh = _listing(
        _artifact(900, "2026-10-05T16:00:00Z"),                        # this run
        _artifact(300, "2026-10-05T17:00:00Z", repo=9999),             # a fork
        _artifact(301, "2026-10-05T17:00:00Z", branch="feature"),      # not main
        _artifact(100, "2026-10-05T12:00:00Z"),
    )
    assert cg.main(["upload-guard", "--downloaded-run-id", "100"], gh=gh, sleep=_no_sleep) == 0


def test_a_db_the_operator_forced_is_not_compared(env):
    rc = cg.main(["upload-guard", "--downloaded-run-id", "100", "--source", "forced"],
                 gh=lambda a: pytest.fail("no API call expected"), sleep=_no_sleep)
    assert rc == 0


def test_an_unreadable_listing_uploads_with_a_warning(env, capsys):
    def boom(_args):
        raise RuntimeError("HTTP 502")

    assert cg.main(["upload-guard", "--downloaded-run-id", "100"], gh=boom, sleep=_no_sleep) == 0
    assert "::warning" in capsys.readouterr().out


# --- lineage --------------------------------------------------------------------


def test_the_pipeline_records_the_db_it_forked_from_and_the_final_stage_reads_it(env, tmp_path):
    db = tmp_path / "staged.duckdb"
    assert cg.main(["record", "--db", str(db), "--downloaded-run-id", "100",
                    "--source", "discovered"]) == 0
    gh = _listing(_artifact(200, "2026-10-14T03:00:00Z"), _artifact(100, "2026-10-13T00:00:00Z"))
    assert cg.main(["upload-guard", "--db", str(db)], gh=gh, sleep=_no_sleep) == 1


def test_an_old_lineage_row_belongs_to_another_pipeline():
    con = duckdb.connect()
    cg.record_lineage(con, run_id="1", downloaded_run_id="100", source="discovered")
    later = datetime.now(timezone.utc) + timedelta(days=cg.LINEAGE_MAX_AGE_DAYS + 1)
    assert cg.read_lineage(con, now=later) is None
    assert cg.read_lineage(con) == ("100", "discovered")


def test_every_canonical_upload_is_preceded_by_the_guard():
    """Each workflow that uploads pythia-resolver-db runs the guard first."""
    import re

    wf_dir = Path(__file__).resolve().parents[1] / ".github" / "workflows"
    missing = []
    for path in sorted(wf_dir.glob("*.yml")):
        text = path.read_text()
        for m in re.finditer(r"name:\s*pythia-resolver-db\s*$", text, re.M):
            before = text[: m.start()]
            step_start = before.rfind("- name:")
            prior = before[:step_start]
            last_guard = prior.rfind("guard-canonical-upload")
            last_upload = prior.rfind("name: pythia-resolver-db")
            if last_guard == -1 or last_guard < last_upload:
                missing.append(path.name)
    assert not missing, f"canonical upload without a preceding guard in: {missing}"


@pytest.mark.parametrize("workflow, job", [
    ("compute_resolutions.yml", "compute-resolutions"),
    ("compute_scores.yml", "compute-scores"),
    ("compute_calibration_pythia.yml", "compute-calibration"),
    ("publish_latest_data.yml", "publish"),
    ("inspect_resolver_duckdb.yml", "inspect-resolver-db"),
])
def test_each_chained_workflow_waits_on_its_trigger_check(workflow, job):
    import yaml

    wf = yaml.safe_load((Path(__file__).resolve().parents[1] / ".github" / "workflows" / workflow).read_text())
    jobs = wf["jobs"]
    assert "trigger-did-work" in str(jobs["trigger"]["steps"])
    assert jobs[job]["needs"] == "trigger"
    assert jobs[job]["if"] == "needs.trigger.outputs.did_work == 'true'"
    # The trigger check must never take the DB group's one pending slot.
    assert "concurrency" not in wf or "pythia-resolver-db" not in str(wf.get("concurrency"))


@pytest.mark.parametrize("workflow, job, nxt", [
    ("compute_resolutions.yml", "compute-resolutions", "compute_scores.yml"),
    ("compute_scores.yml", "compute-scores", "compute_calibration_pythia.yml"),
])
def test_a_bot_dispatched_link_starts_the_next_one(workflow, job, nxt):
    """GitHub fires no workflow_run for a GITHUB_TOKEN-dispatched run: the
    6 October reset's Compute Resolutions finished and nothing followed."""
    import yaml

    wf = yaml.safe_load((Path(__file__).resolve().parents[1] / ".github" / "workflows" / workflow).read_text())
    steps = wf["jobs"][job]["steps"]
    cont = [s for s in steps if f"gh workflow run {nxt}" in str(s.get("run", ""))]
    assert cont, f"{workflow} never dispatches {nxt}"
    cond = cont[0]["if"]
    assert "github-actions[bot]" in cond and "workflow_dispatch" in cond
    assert wf["permissions"]["actions"] == "write"
