# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The cycle zip's pipeline stages, cron hs_submit, poller ticks and stage
contexts, against the run listing recorded for the 13 Nov 2026 chain.

No network: a fake ``gh`` serves the recorded runs and honours the API's
``created`` range and ``page`` parameters, which the poller listing needs.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from urllib.parse import parse_qs

import pytest

pytest.importorskip("duckdb")

from scripts.ci import build_cycle_diagnostics as bcd  # noqa: E402
from scripts.ci import chain_diagnostics as cd  # noqa: E402

REPO = "owner/pythia"
FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "cycle_runs_2026_11_13.json")
                     .read_text(encoding="utf-8"))
CADENCE = dt.timedelta(minutes=5, seconds=42)


def _info(raw, name):
    return {**raw, "name": name, "status": "completed", "head_branch": "main",
            "head_repository": {"full_name": REPO}, "updated_at": raw["created_at"],
            "run_started_at": raw["created_at"],
            "html_url": f"https://github.com/{REPO}/actions/runs/{raw['id']}"}


def _parse(ts):
    return dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)


def _generated_pollers():
    """Ticks every 5.7 minutes from hs_submit to Sibyl, as on 13 Nov 2026."""
    start = _parse("2026-11-13T03:53:30Z")
    end = _parse("2026-11-14T00:00:00Z")
    out, t, rid = [], start, 43400000000
    while t < end:
        out.append({"id": rid, "display_title": "Poll LLM Batches", "event": "workflow_dispatch",
                    "conclusion": "success", "created_at": t.strftime("%Y-%m-%dT%H:%M:%SZ")})
        t += CADENCE
        rid += 1
    return out


class RecordedGh:
    def __init__(self, *, stage_contexts=True, did_work=None):
        self.runs: dict[str, dict] = {}
        self.artifacts: dict[str, list[dict]] = {}
        self.contents: dict[tuple[str, str], dict[str, str]] = {}
        self.listing_calls: list[str] = []
        for r in FIXTURE["stage_runs"]:
            self._add(_info(r, "Pythia Pipeline Stage"))
        self._add(_info(FIXTURE["sibyl_run"], "Sibyl Deep Research"))
        for r in FIXTURE["poller_runs"] + _generated_pollers():
            self._add(_info(r, "Poll LLM Batches"),
                      {"pythia-poller-decisions": {"poll_decision.json": json.dumps(
                          {"pipelines": [{"action": "waiting"}]})}})
        if stage_contexts:
            ctx = lambda stage, fc: {"stage_context.json": json.dumps(  # noqa: E731
                {"stage": stage, "hs_run_id": "hs_20261113T035247", "forecaster_run_id": fc,
                 "pipeline_id": FIXTURE["pipeline_id"]})}
            self._art("43362670678", "pythia-stage-context-hs_rc_collect", ctx("hs_rc_collect", ""))
            self._art("43363940081", "pythia-stage-context-hs_finalize_fc_submit",
                      ctx("hs_finalize_fc_submit", ""))
            self._art("43530492374", "pythia-stage-context", ctx("fc_collect_finalize", "fc_1794543371"))
        for rid, value in (did_work or {}).items():
            self._art(rid, cd.artifact_name("pipeline_stage", rid), {
                "run.json": json.dumps({"did_work": value, "stage": "x"}),
                "lineage.json": json.dumps({"trigger_run": None}),
                "issues.json": json.dumps({"issues": []})})

    def _add(self, info, artifacts=None):
        rid = str(info["id"])
        self.runs[rid] = info
        self.artifacts.setdefault(rid, [])
        for name, files in (artifacts or {}).items():
            self._art(rid, name, files)

    def _art(self, rid, name, files):
        self.artifacts.setdefault(rid, []).append({"name": name, "expired": False})
        self.contents[(rid, name)] = files

    def api(self, path):
        base, _, query = path.partition("?")
        parts = base.split("/")
        q = {k: v[0] for k, v in parse_qs(query).items()}
        if parts[-1] == "artifacts":
            return {"artifacts": list(self.artifacts.get(parts[-2], []))}
        if "workflows" in parts:
            self.listing_calls.append(path)
            file = parts[parts.index("workflows") + 1]
            name = next(n for n, (_s, f) in cd.WORKFLOWS.items() if f == file)
            runs = sorted((r for r in self.runs.values() if r["name"] == name),
                          key=lambda r: r["created_at"], reverse=True)
            if "created" in q:
                lo, hi = (_parse(x) for x in q["created"].split(".."))
                runs = [r for r in runs if lo <= _parse(r["created_at"]) <= hi]
            per = int(q.get("per_page", 30))
            page = int(q.get("page", 1))
            return {"workflow_runs": runs[(page - 1) * per: page * per]}
        rid = parts[-1]
        if rid not in self.runs:
            raise RuntimeError("404")
        return self.runs[rid]

    def download(self, run_id, name, dest):
        files = self.contents.get((str(run_id), name))
        if files is None:
            return False, "not found"
        for rel, body in files.items():
            p = Path(dest) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body, encoding="utf-8")
        return True, "ok"


def _build(tmp_path, gh, *, full):
    return bcd.build(gh=gh, repo=REPO, source_run_id=str(FIXTURE["sibyl_run"]["id"]), publish_run_id="",
                     env={}, out_dir=tmp_path / "o", staging=tmp_path / "s", wait_sec=0, full=full,
                     keep_staging=True)


def test_every_stage_of_the_pipeline_joins_and_the_cron_submit_too(tmp_path):
    m = _build(tmp_path, RecordedGh(), full=True)
    stages = {r["run_id"]: r["reason"] for r in m["runs"] if r["workflow"] == "Pythia Pipeline Stage"}
    assert set(stages) == {"43361558197", "43362670678", "43363940081", "43530492374"}
    assert "cron hs_submit" in stages["43361558197"]
    # The October cron run is a month older and never joins.
    assert "38345208458" not in stages


def test_full_mode_keeps_every_poller_tick_between_the_first_stage_and_sibyl(tmp_path):
    gh = RecordedGh()
    m = _build(tmp_path, gh, full=True)
    pollers = [r for r in m["runs"] if r["workflow"] == "Poll LLM Batches"]
    expected = [r for r in FIXTURE["poller_runs"] + _generated_pollers()
                if "2026-11-13T03:50:08Z" <= r["created_at"] <= "2026-11-14T00:04:56Z"]
    assert len(pollers) == len(expected) > 200
    assert "43816443524" not in {r["run_id"] for r in pollers}  # 16 Nov, outside the window
    # More than a page: the listing walked pages through the created filter.
    assert any("page=3" in c and "created=" in c for c in gh.listing_calls)
    assert all(r["reason"].startswith("poller tick") for r in pollers)
    folders = [p for p in (tmp_path / "s").iterdir() if "_poll_llm_batches__" in p.name]
    assert len(folders) == len(pollers)
    assert all((p / "pythia-poller-decisions" / "poll_decision.json").is_file() for p in folders[:3])


def test_the_ceiling_mode_keeps_forty_ticks_first_and_last_included(tmp_path):
    m = _build(tmp_path, RecordedGh(), full=False)
    pollers = sorted((r for r in m["runs"] if r["workflow"] == "Poll LLM Batches"),
                     key=lambda r: r["created_at"])
    assert len(pollers) == bcd.MAX_POLLER_RUNS
    every = sorted(r["created_at"] for r in FIXTURE["poller_runs"] + _generated_pollers()
                   if "2026-11-13T03:50:08Z" <= r["created_at"] <= "2026-11-14T00:04:56Z")
    assert pollers[0]["created_at"] == every[0] and pollers[-1]["created_at"] == every[-1]
    assert any("kept 40" in n for n in m["notes"])


def test_all_three_stage_contexts_are_collected_and_the_forecaster_one_names_the_run(tmp_path):
    m = _build(tmp_path, RecordedGh(), full=True)
    found = sorted(p.parent.name for p in (tmp_path / "s").rglob("stage_context.json"))
    assert found == ["pythia-stage-context", "pythia-stage-context-hs_finalize_fc_submit",
                     "pythia-stage-context-hs_rc_collect"]
    assert m["chain"]["fc_run_id"] == "fc_1794543371"


def test_the_cycle_summary_prints_a_stage_runs_own_did_work(tmp_path):
    gh = RecordedGh(did_work={"43362670678": "yes (3 table(s) changed)", "43363940081": "no"})
    _build(tmp_path, gh, full=False)
    text = (tmp_path / "s" / "CYCLE_SUMMARY.md").read_text(encoding="utf-8")
    row = next(line for line in text.splitlines() if "43362670678" in line)
    assert "yes (3 table(s) changed)" in row
    assert "| no |" in next(line for line in text.splitlines() if "43363940081" in line)
