# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Chain diagnostics (per run) and the cycle zip built on the publish run.

No network: the Actions API and artifact downloads go through a fake ``gh``.
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from scripts.ci import build_cycle_diagnostics as bcd  # noqa: E402
from scripts.ci import chain_diagnostics as cd  # noqa: E402

REPO = "owner/pythia"
NAMES = {
    "resolver_update": "Resolver Update",
    "compute_resolutions": "Pythia — Compute Resolutions",
    "compute_scores": "Pythia — Compute SPD Scores",
    "compute_calibration": "Pythia — Compute Calibration Weights & Advice",
    "ingest_structured_data": "Ingest Structured Data",
    "refresh_crisiswatch": "Refresh CrisisWatch Data",
    "publish_latest_data": "Publish Latest Data (Release)",
}


# ---------------------------------------------------------------------------
# Fake gh
# ---------------------------------------------------------------------------


def _run(run_id, slug, created, *, event="workflow_run", repo=REPO, branch="main",
         conclusion="success", status="completed", title=None):
    return {
        "id": int(run_id), "name": NAMES.get(slug, slug), "event": event,
        "status": status, "conclusion": conclusion, "created_at": created,
        "updated_at": created, "run_started_at": created, "head_branch": branch,
        "head_repository": {"full_name": repo}, "display_title": title or NAMES.get(slug, slug),
        "html_url": f"https://github.com/{REPO}/actions/runs/{run_id}",
    }


class FakeGh:
    def __init__(self):
        self.runs: dict[str, dict] = {}
        self.artifacts: dict[str, list[dict]] = {}
        self.contents: dict[tuple[str, str], dict[str, str | bytes]] = {}
        self.downloads: list[tuple[str, str]] = []

    def add_run(self, info, artifacts=None):
        rid = str(info["id"])
        self.runs[rid] = info
        self.artifacts[rid] = []
        for name, files in (artifacts or {}).items():
            self.artifacts[rid].append({"name": name, "expired": False})
            self.contents[(rid, name)] = files

    def api(self, path):
        parts = path.split("?")[0].split("/")
        if parts[-1] == "artifacts":
            return {"artifacts": list(self.artifacts.get(parts[-2], []))}
        if "workflows" in parts:
            file = parts[parts.index("workflows") + 1]
            name = next(n for n, (_s, f) in cd.WORKFLOWS.items() if f == file)
            runs = [r for r in self.runs.values() if r["name"] == name]
            return {"workflow_runs": sorted(runs, key=lambda r: r["created_at"], reverse=True)}
        rid = parts[-1]
        if rid not in self.runs:
            raise RuntimeError("404")
        return self.runs[rid]

    def download(self, run_id, name, dest):
        self.downloads.append((str(run_id), name))
        files = self.contents.get((str(run_id), name))
        if files is None:
            return False, "download error: not found"
        for rel, body in files.items():
            p = Path(dest) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(body, bytes):
                p.write_bytes(body)
            else:
                p.write_text(body, encoding="utf-8")
        return True, "ok"


def _chain_artifact(slug, run_id, trigger, *, issues=(), downloaded=None, uploaded=True):
    lineage = {"slug": slug, "this_run": str(run_id), "trigger_run": trigger,
               "canonical_db_run_downloaded": downloaded or trigger,
               "canonical_db_run_uploaded": str(run_id) if uploaded else None,
               "upload_guard_verdict": "ok"}
    return {cd.artifact_name(slug, run_id): {
        "lineage.json": json.dumps(lineage),
        "run.json": json.dumps({"trigger_did_work": "true"}),
        "issues.json": json.dumps({"issues": list(issues)}),
        "summary.md": f"## {slug}\n",
    }}


def _scoring_chain(gh: FakeGh, *, with_lineage=True):
    gh.add_run(_run(100, "resolver_update", "2026-10-11T03:00:00Z", event="schedule"),
               {"resolver-debug-bundle-100": {"bundle.zip": b"PK\x03\x04fake"},
                "backfill-diagnostics": {"issues.json": json.dumps({"issues": [
                    {"id": "emdat_auth_rejected", "severity": "known", "title": "EM-DAT key"}]})},
                "pythia-resolver-db": {"resolver.duckdb": b"DB"}})
    specs = [(101, "compute_resolutions", "100", "2026-10-11T08:00:00Z"),
             (102, "compute_scores", "101", "2026-10-11T08:30:00Z"),
             (103, "compute_calibration", "102", "2026-10-11T09:00:00Z")]
    for rid, slug, trig, created in specs:
        arts = {"pythia-resolver-db": {"resolver.duckdb": b"DB"}}
        if with_lineage:
            issues = [{"id": "resolutions_mostly_zero_default:ACE/PA", "severity": "degraded",
                       "title": "ACE/PA mostly zero"}] if slug == "compute_resolutions" else []
            arts.update(_chain_artifact(slug, rid, trig, issues=issues))
        if slug == "compute_calibration":
            arts["pythia-ai-analysis-bundle"] = {"scored.zip": b"PK"}
        gh.add_run(_run(rid, slug, created), arts)
    gh.add_run(_run(200, "publish_latest_data", "2026-10-11T09:30:00Z", event="workflow_dispatch"))


def _build(gh, tmp_path, **kw):
    kw.setdefault("env", {})
    return bcd.build(gh=gh, repo=REPO, source_run_id="103", publish_run_id="200",
                     out_dir=tmp_path / "out", staging=tmp_path / "staging", wait_sec=0, **kw)


# ---------------------------------------------------------------------------
# The cycle zip
# ---------------------------------------------------------------------------


def test_lineage_walk_follows_trigger_run_across_four_runs(tmp_path):
    gh = FakeGh()
    _scoring_chain(gh)
    m = _build(gh, tmp_path)
    walked = [r["run_id"] for r in m["runs"]]
    assert walked == ["100", "101", "102", "103", "200"]
    reasons = {r["run_id"]: r["reason"] for r in m["runs"]}
    assert reasons["103"] == "source"
    assert reasons["101"].startswith("trigger")
    # Resolver Update is named by Compute Resolutions' lineage.json.
    assert reasons["100"].startswith("trigger")
    names = zipfile.ZipFile(m["zip"]).namelist()
    assert "README.md" in names and "CYCLE_SUMMARY.md" in names and "manifest.json" in names
    assert any(n.startswith("01_resolver_update__100/resolver-debug-bundle-100/") for n in names)
    assert any(n.startswith("03_compute_scores__102/chain-diagnostics-compute_scores-102/") for n in names)


def test_api_fallback_when_lineage_is_absent(tmp_path):
    gh = FakeGh()
    _scoring_chain(gh, with_lineage=False)
    m = _build(gh, tmp_path)
    assert [r["run_id"] for r in m["runs"]][:4] == ["100", "101", "102", "103"]
    assert all(r["reason"].startswith(("source", "upstream")) for r in m["runs"][:4])
    # An older chain-writing run with no chain diagnostics says so.
    assert any(x["artifact"] == cd.artifact_name("compute_scores", "102") for x in m["missing"])


def test_a_bot_dispatched_link_with_no_trigger_falls_back_to_the_api(tmp_path):
    gh = FakeGh()
    _scoring_chain(gh)
    # Scores was dispatched by Compute Resolutions' "Continue the chain" step:
    # its lineage.json names no trigger.
    gh.runs["102"]["event"] = "workflow_dispatch"
    gh.runs["102"]["actor"] = {"login": "github-actions[bot]"}
    name = cd.artifact_name("compute_scores", "102")
    lin = json.loads(gh.contents[("102", name)]["lineage.json"])
    lin["trigger_run"] = None
    gh.contents[("102", name)]["lineage.json"] = json.dumps(lin)
    m = _build(gh, tmp_path)
    reasons = {r["run_id"]: r["reason"] for r in m["runs"]}
    assert reasons["101"].startswith("upstream (Actions API: bot-dispatched")
    assert "100" in reasons


def test_fallback_ignores_an_upstream_too_old_or_after_the_run(tmp_path):
    gh = FakeGh()
    gh.add_run(_run(90, "resolver_update", "2026-09-01T03:00:00Z", event="schedule"))
    gh.add_run(_run(101, "compute_resolutions", "2026-10-11T08:00:00Z"))
    gh.add_run(_run(105, "resolver_update", "2026-10-12T03:00:00Z", event="schedule"))
    c = bcd.Collector(gh, REPO, tmp_path, wait_sec=0)
    assert c.fallback_upstream(gh.runs["101"]) is None


def test_window_keeps_only_our_own_runs(tmp_path):
    gh = FakeGh()
    _scoring_chain(gh)
    gh.add_run(_run(300, "ingest_structured_data", "2026-10-05T03:00:00Z", event="schedule"),
               {"ingest-diagnostics": {"x.md": "ok"}})
    gh.add_run(_run(301, "ingest_structured_data", "2026-10-06T03:00:00Z", event="pull_request"))
    gh.add_run(_run(302, "refresh_crisiswatch", "2026-10-07T06:17:00Z", event="schedule",
                    repo="fork/pythia"))
    gh.add_run(_run(303, "refresh_crisiswatch", "2026-10-08T06:17:00Z", event="schedule", branch="evil"))
    gh.add_run(_run(304, "refresh_crisiswatch", "2026-09-01T06:17:00Z", event="schedule"))
    gh.add_run(_run(305, "refresh_crisiswatch", "2026-10-09T06:17:00Z", event="schedule"),
               {"crisiswatch-data-305": {"crisiswatch_latest.json": "{}"}})
    m = _build(gh, tmp_path)
    ids = {r["run_id"] for r in m["runs"]}
    assert {"300", "305"} <= ids
    assert not ids & {"301", "302", "303", "304"}


def test_no_db_artifact_is_ever_included(tmp_path):
    gh = FakeGh()
    _scoring_chain(gh)
    m = _build(gh, tmp_path)
    assert not any(name.startswith("pythia-resolver-db") for _rid, name in gh.downloads)
    assert not any("resolver.duckdb" in f["path"] for f in m["files"])


def test_missing_and_expired_artifacts_land_in_the_manifest(tmp_path):
    gh = FakeGh()
    _scoring_chain(gh)
    gh.artifacts["103"].append({"name": "pythia-interpreter-scored", "expired": True})
    m = _build(gh, tmp_path)
    reasons = {(x["run_id"], x["artifact"]): x["reason"] for x in m["missing"]}
    assert reasons[("103", "pythia-interpreter-scored")] == "expired"
    assert reasons[("103", "calibration-summary")] == "absent"
    readme = zipfile.ZipFile(m["zip"]).read("README.md").decode()
    assert "pythia-interpreter-scored" in readme and "expired" in readme


def test_cycle_summary_merges_issues_worst_first(tmp_path):
    gh = FakeGh()
    _scoring_chain(gh)
    m = _build(gh, tmp_path)
    text = zipfile.ZipFile(m["zip"]).read("CYCLE_SUMMARY.md").decode()
    issues = text.split("## Every issue, worst first", 1)[1]
    assert issues.index("degraded") < issues.index("known")
    assert "| Resolver Update |" in text and "| Pythia — Compute SPD Scores |" in text


def test_a_planted_secret_fails_the_build(tmp_path):
    gh = FakeGh()
    _scoring_chain(gh)
    secret = "ghs_" + "a1B2c3D4e5F6g7H8i9J0" * 2
    gh.contents[("102", cd.artifact_name("compute_scores", "102"))]["summary.md"] = f"token {secret}"
    with pytest.raises(bcd.SecretLeak):
        _build(gh, tmp_path, env={"GH_TOKEN": secret})
    assert not list((tmp_path / "out").glob("*.zip"))


def test_size_ceiling_drops_in_the_stated_order(tmp_path):
    staging = tmp_path / "s"
    rnd = random.Random(1)

    def noise(n):
        return bytes(rnd.getrandbits(8) for _ in range(n))

    files = {
        "01_sibyl__1/pythia-debug-bundle-workflow-logs/run.txt": noise(40_000),
        "01_sibyl__1/pythia-debug-bundle/code_snapshot/x.py": noise(40_000),
        "01_sibyl__1/pythia-current-run-bundle/pack.zip": noise(40_000),
        "02_compute_scores__2/chain-diagnostics-compute_scores-2/big.csv":
            "\n".join(f"{i},{rnd.random()}" for i in range(6000)),
        "02_compute_scores__2/chain-diagnostics-compute_scores-2/summary.md": "keep",
        "README.md": "keep",
    }
    for rel, body in files.items():
        p = staging / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        (p.write_bytes if isinstance(body, bytes) else p.write_text)(body)
    size, dropped = bcd.enforce_size(staging, tmp_path / "z.zip", 60_000,
                                     {"01_sibyl__1": "sibyl", "02_compute_scores__2": "compute_scores"})
    stages = [d["stage"] for d in dropped]
    assert stages[:3] == ["workflow_logs", "code_snapshots", "sibyl_nested_zips"]
    assert (staging / "README.md").is_file()
    assert (staging / "02_compute_scores__2/chain-diagnostics-compute_scores-2/summary.md").is_file()
    assert all(d["original_bytes"] > 0 for d in dropped)


def test_size_ceiling_stops_at_the_first_stage_that_fits(tmp_path):
    staging = tmp_path / "s"
    p = staging / "01_sibyl__1/pythia-debug-bundle-workflow-logs/run.txt"
    p.parent.mkdir(parents=True)
    rnd = random.Random(2)
    p.write_bytes(bytes(rnd.getrandbits(8) for _ in range(50_000)))
    (staging / "README.md").write_text("keep")
    _size, dropped = bcd.enforce_size(staging, tmp_path / "z.zip", 10_000, {"01_sibyl__1": "sibyl"})
    assert [d["stage"] for d in dropped] == ["workflow_logs"]


def test_pipeline_stages_join_by_pipeline_id(tmp_path):
    gh = FakeGh()
    title = "Pythia Pipeline Stage: hs_20261013T000000 — {}"
    gh.add_run(_run(500, "Pythia Pipeline Stage", "2026-10-13T00:00:00Z", event="schedule",
                    title=title.format("hs_submit")), {"pythia-stage-diagnostics": {"a.json": "{}"}})
    gh.add_run(_run(501, "Pythia Pipeline Stage", "2026-10-13T05:00:00Z", event="workflow_dispatch",
                    title=title.format("fc_collect_finalize")))
    gh.add_run(_run(509, "Pythia Pipeline Stage", "2026-10-13T05:00:00Z", event="workflow_dispatch",
                    title="Pythia Pipeline Stage: other_pipe — hs_submit"))
    gh.add_run(_run(510, "Sibyl Deep Research", "2026-10-13T06:00:00Z", event="workflow_dispatch"),
               {"pythia-health-report": {"h.md": "x"}, "pythia-resolver-db": {"r.duckdb": b"x"}})
    m = bcd.build(gh=gh, repo=REPO, source_run_id="510", publish_run_id="", env={},
                  out_dir=tmp_path / "o", staging=tmp_path / "s", wait_sec=0)
    ids = [r["run_id"] for r in m["runs"]]
    assert ids == ["500", "501", "510"]


# ---------------------------------------------------------------------------
# The per-run writer
# ---------------------------------------------------------------------------


def _args(**kw):
    base = dict(workflow="compute_resolutions", db="", diagnostics_dir="", out_dir="",
                started_at="2000-01-01 00:00:00", include=[], no_api=True)
    base.update(kw)
    return argparse.Namespace(**base)


def test_writer_writes_every_file_and_a_stub_for_a_failed_section(tmp_path):
    db = tmp_path / "r.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE resolutions (question_id TEXT, horizon_m INT, value DOUBLE)")
    con.close()
    out = tmp_path / "d" / "compute_resolutions"
    cd.main(["before", "--workflow", "compute_resolutions", "--db", str(db),
             "--diagnostics-dir", str(tmp_path / "d")])
    con = duckdb.connect(str(db))
    con.execute("INSERT INTO resolutions VALUES ('q', 1, 3.0)")
    con.close()
    env = {"GITHUB_RUN_ID": "101", "CHAIN_TRIGGER_RUN_ID": "100", "CHAIN_DOWNLOADED_RUN_ID": "100",
           "CHAIN_UPLOAD_OUTCOME": "success", "CHAIN_GUARD_OUTCOME": "success",
           "CHAIN_DID_WORK": "true"}
    cd.write(_args(db=str(db), diagnostics_dir=str(tmp_path / "d")), env=env)
    for name in ("run.json", "lineage.json", "table_counts.csv", "issues.json", "issues.md",
                 "summary.md", "MISSING.json"):
        assert (out / name).is_file(), name
    lineage = json.loads((out / "lineage.json").read_text())
    assert lineage["trigger_run"] == "100" and lineage["canonical_db_run_uploaded"] == "101"
    assert "resolutions,0,1,1" in (out / "table_counts.csv").read_text()
    missing = json.loads((out / "MISSING.json").read_text())
    # No --report-json was written: the section is named, the step goes on.
    assert "compute_resolutions" in missing


def test_writer_reads_the_resolution_report(tmp_path):
    out = tmp_path / "d" / "compute_resolutions"
    (out / "reports").mkdir(parents=True)
    (out / "reports" / "compute_resolutions.json").write_text(json.dumps({
        "rows_written": 7, "resolved_from_source": 3, "resolved_as_zero": 4,
        "resolutions_before": 10, "resolutions_after": 17, "machine_retracted": 1,
        "outcome_counts": {"ACE/PA": {"sourced": 1, "zero_default": 4, "unresolved_for_lag": 2}},
        "mostly_zero_default_groups": ["ACE/PA"], "by_source_desc": {"zero_default": 4},
    }))
    cd.write(_args(diagnostics_dir=str(tmp_path / "d")), env={})
    assert "ACE/PA,1,4,2,0" in (out / "resolution_outcomes.csv").read_text()
    issues = json.loads((out / "issues.json").read_text())["issues"]
    assert any(i["id"].startswith("resolutions_mostly_zero_default") for i in issues)


def test_writer_marks_a_refused_upload_blocking(tmp_path):
    out = tmp_path / "d" / "compute_scores"
    cd.write(_args(workflow="compute_scores", diagnostics_dir=str(tmp_path / "d")),
             env={"GITHUB_RUN_ID": "7", "CHAIN_GUARD_OUTCOME": "failure"})
    lineage = json.loads((out / "lineage.json").read_text())
    assert lineage["upload_guard_verdict"] == "refused"
    assert lineage["canonical_db_run_uploaded"] is None
    issues = json.loads((out / "issues.json").read_text())["issues"]
    assert issues[0]["severity"] == "blocking"


def test_crisiswatch_state_reads_the_json_without_a_db(tmp_path):
    root = tmp_path / "repo"
    path = root / "horizon_scanner" / "data" / "crisiswatch_latest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"month": "September 2026", "year": 2026, "content_hash": "h",
                                "parse_accounting": {"unresolved_names": ["Atlantis"]},
                                "entries": [{"iso3": "SDN", "iso3_reason": "resolved"}, {"iso3": None}]}))
    state = cd.crisiswatch_state(root, tmp_path, env={}, today=cd.dt.date(2026, 10, 12))
    assert state["edition_after"] == "2026-09"
    assert state["expected_edition"] == "2026-09"
    assert state["gate_would_pass"] is True
    assert state["entries_by_iso3_reason"] == {"resolved": 1, "unresolved": 1}


def test_main_always_exits_zero(tmp_path):
    assert cd.main(["write", "--workflow", "compute_scores", "--db", str(tmp_path / "none.duckdb"),
                    "--diagnostics-dir", str(tmp_path), "--no-api"]) == 0
