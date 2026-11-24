# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Chain diagnostics for the pipeline stages and Sibyl (Nov 2026), the
experiment-arms writer and the earlier stages' context fetch.

The stage-health files are the 13 Nov 2026 chain's own, from the forecast
review fixtures."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from scripts.ci import chain_diagnostics as cd  # noqa: E402
from scripts.ci import experiment_arms as ea  # noqa: E402
from scripts.ci import sibyl_bundle_context as sbc  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "forecast_review" / "chain"
SIBYL_HEALTH = FIX / "11_sibyl__43531717339" / "pythia-stage-diagnostics-sibyl" / "diagnostics" / "stage_health__sibyl.json"
FC_HEALTH = FIX / "10_pipeline_stage__43530492374" / "pythia-stage-diagnostics" / "stage_health__fc_collect_finalize.json"
FC = "fc_1794543371"


def _args(workflow, diag, db):
    return argparse.Namespace(workflow=workflow, db=str(db), diagnostics_dir=str(diag), out_dir="",
                              started_at="2000-01-01 00:00:00", include=[], no_api=True)


def _db(tmp_path):
    db = tmp_path / "r.duckdb"
    con = duckdb.connect(str(db))
    con.execute("""CREATE TABLE forecasts_raw (run_id TEXT, question_id TEXT, model_name TEXT,
                   month_index INT, bucket_index INT, probability DOUBLE, status TEXT, spd_json TEXT)""")
    con.close()
    return db


def _fill(db):
    con = duckdb.connect(str(db))
    rows = []
    for q in ("Q1", "Q2", "Q3"):
        for m in ("gpt-6-sol", "claude-opus-5-5", "ensemble_mean_v2"):
            rows.append((FC, q, m, 1, 1, 0.5, "ok", json.dumps({"parse_mode": "ok"})))
    rows.append((FC, "Q4", "gpt-6-sol", 1, 1, 0.5, "ok", json.dumps({"parse_mode": "salvaged"})))
    rows.append((FC, "Q4", "claude-opus-5-5", None, None, None, "no_forecast", "{}"))
    con.executemany("INSERT INTO forecasts_raw VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    con.close()


def test_a_pipeline_stage_writes_did_work_its_stage_and_its_own_section(tmp_path):
    diag = tmp_path / "diagnostics"
    diag.mkdir()
    shutil.copy(FC_HEALTH, diag / FC_HEALTH.name)
    db = _db(tmp_path)
    cd.main(["before", "--workflow", "pipeline_stage", "--db", str(db), "--diagnostics-dir", str(diag)])
    _fill(db)
    env = {"GITHUB_RUN_ID": "43530492374", "CHAIN_STAGE": "fc_collect_finalize",
           "CHAIN_PIPELINE_ID": "hs_20261113T035247", "CHAIN_JOB_STATUS": "success",
           "FORECASTER_RUN_ID": FC}
    cd.write(_args("pipeline_stage", diag, db), env=env)
    out = diag / "pipeline_stage"
    run = json.loads((out / "run.json").read_text())
    assert run["stage"] == "fc_collect_finalize" and run["pipeline_id"] == "hs_20261113T035247"
    assert run["did_work"].startswith("yes") and "forecasts_raw +11" in run["tables_changed"]
    section = json.loads((out / "stage_section.json").read_text())
    assert section["rc_levels"]["n_assessed"] == 348
    assert section["stage_cost"]["n_calls"] == 729
    assert section["member_questions_ok"] == {"claude-opus-5-5": 3, "gpt-6-sol": 4}
    assert section["parse_salvaged_member_forecasts"] == 1
    assert section["no_forecast_by_member"] == {"claude-opus-5-5": 1}
    ids = {i["id"] for i in json.loads((out / "issues.json").read_text())["issues"]}
    assert {"member_forecasts_lost", "ensemble_members_short"} <= ids


def test_a_stage_that_changed_nothing_says_no_and_raises_an_issue(tmp_path):
    diag = tmp_path / "diagnostics"
    diag.mkdir()
    db = _db(tmp_path)
    cd.main(["before", "--workflow", "pipeline_stage", "--db", str(db), "--diagnostics-dir", str(diag)])
    cd.write(_args("pipeline_stage", diag, db),
             env={"GITHUB_RUN_ID": "1", "CHAIN_STAGE": "hs_rc_collect", "CHAIN_JOB_STATUS": "success"})
    run = json.loads((diag / "pipeline_stage" / "run.json").read_text())
    assert run["did_work"] == "no"
    issues = json.loads((diag / "pipeline_stage" / "issues.json").read_text())["issues"]
    assert any(i["id"] == "stage_wrote_nothing" for i in issues)


def test_sibyl_writes_its_counts_from_stage_health(tmp_path):
    diag = tmp_path / "diagnostics"
    diag.mkdir()
    payload = json.loads(SIBYL_HEALTH.read_text())
    payload["sibyl"]["budget_capped"] = True
    payload["sibyl"]["n_skipped"] = 4
    (diag / "stage_health__sibyl.json").write_text(json.dumps(payload))
    db = _db(tmp_path)
    cd.main(["before", "--workflow", "sibyl", "--db", str(db), "--diagnostics-dir", str(diag)])
    cd.write(_args("sibyl", diag, db), env={"GITHUB_RUN_ID": "43531717339", "CHAIN_STAGE": "sibyl"})
    section = json.loads((diag / "sibyl" / "sibyl_section.json").read_text())
    assert section["sibyl_run_id"] == "sibyl_1794614792427" and section["n_forecast"] == 15
    assert section["tools"]["n_search_calls"] == 351
    issues = json.loads((diag / "sibyl" / "issues.json").read_text())["issues"]
    assert any(i["id"] == "sibyl_budget_capped" and i["cost"] == 4 for i in issues)


def test_a_sibyl_run_that_was_gated_out_says_why(tmp_path):
    diag = tmp_path / "diagnostics"
    diag.mkdir()
    (diag / "stage_health__sibyl.json").write_text(json.dumps(
        {"sibyl": {"available": False, "reason": "no sibyl_runs row matching hs_run_id = ?"}}))
    cd.write(_args("sibyl", diag, ""), env={"GITHUB_RUN_ID": "45458289266", "CHAIN_STAGE": "sibyl"})
    section = json.loads((diag / "sibyl" / "sibyl_section.json").read_text())
    assert section == {"available": False, "reason": "no sibyl_runs row matching hs_run_id = ?"}


def test_a_missing_stage_health_is_a_named_gap_not_a_crash(tmp_path):
    diag = tmp_path / "diagnostics"
    diag.mkdir()
    cd.write(_args("sibyl", diag, ""), env={"GITHUB_RUN_ID": "9"})
    assert "sibyl" in json.loads((diag / "sibyl" / "MISSING.json").read_text())


def test_publish_post_records_the_cycle_zip(tmp_path):
    cyc = tmp_path / "cycle_out"
    cyc.mkdir()
    import zipfile

    zpath = cyc / "cycle_diagnostics__2026-11-14__1.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("CYCLE_SUMMARY.md", "# Cycle summary\n")
    out = tmp_path / "d" / "publish_post.json"
    cd.main(["publish-post", "--cycle-dir", str(cyc), "--out", str(out)])
    payload = json.loads(out.read_text())
    assert payload["cycle_zip"] == "cycle_diagnostics__2026-11-14__1.zip"
    assert payload["cycle_zip_bytes"] == zpath.stat().st_size
    # The summary is inside the zip, never beside it (24 Nov 2026 read false).
    assert payload["cycle_summary_written"] is True


# ---------------------------------------------------------------------------
# experiment arms
# ---------------------------------------------------------------------------


def _arms_db():
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE questions (question_id TEXT, track INT, hazard_code TEXT, metric TEXT)")
    con.execute("CREATE TABLE forecasts_ensemble (run_id TEXT, question_id TEXT, advice_arm TEXT, adversarial_arm TEXT)")
    con.execute("CREATE TABLE forecasts_raw (run_id TEXT, question_id TEXT, rc_shift_arm TEXT)")
    qs = [("A", 1, "ACE", "FATALITIES", "advice", "withheld", "shift"),
          ("B", 1, "FL", "PA", "no_advice", "shown", "control"),
          ("C", 2, "DR", "PHASE3PLUS_IN_NEED", "advice_empty", "n/a", None),
          ("D", 1, "TC", "EVENT_OCCURRENCE", None, "n/a", "shift"),
          ("E", 1, "ACE", "PA", "advice", "shown", None)]
    for q, t, h, m, adv, adx, rc in qs:
        con.execute("INSERT INTO questions VALUES (?, ?, ?, ?)", [q, t, h, m])
        con.execute("INSERT INTO forecasts_ensemble VALUES (?, ?, ?, ?)", [FC, q, adv, adx])
        con.execute("INSERT INTO forecasts_raw VALUES (?, ?, ?)", [FC, q, rc])
    return con


def test_experiment_arms_count_what_was_stamped():
    out = ea.collect(_arms_db(), FC, env={})
    assert out["n_questions"] == 5
    assert out["advice"]["counts"] == {"advice": 2, "no_advice": 1, "advice_empty": 1}
    assert out["advice"]["unstamped"] == 1 and out["advice"]["design_share"] == 0.5
    # RC shift: Track 1, not binary: A, B, E; E has no stamp.
    assert out["rc_shift"]["counts"] == {"shift": 1, "control": 1} and out["rc_shift"]["unstamped"] == 1
    adv = out["adversarial"]
    assert adv["counts"] == {"withheld": 1, "shown": 2} and adv["not_eligible"] == 2
    assert adv["withheld_question_ids"] == ["A"]
    assert adv["by_track_hazard_metric"]["t1|ACE/FATALITIES"] == {"withheld": 1}


def test_a_zero_share_turns_an_arm_off():
    out = ea.collect(_arms_db(), FC, env={"PYTHIA_ADVERSARIAL_WITHHELD_SHARE": "0"})
    assert out["adversarial"]["off"] is True


def test_the_arms_cli_never_fails(tmp_path):
    assert ea.main(["--db", str(tmp_path / "nope.duckdb"), "--fc-run-id", FC,
                    "--out", str(tmp_path / "a.json")]) == 0


# ---------------------------------------------------------------------------
# the HS-side stage contexts the Sibyl job fetches
# ---------------------------------------------------------------------------

RUNS = [
    {"id": 3, "display_title": "Pythia Pipeline Stage: hs_1 — hs_finalize_fc_submit",
     "conclusion": "success", "created_at": "2026-11-13T04:12:03Z"},
    {"id": 2, "display_title": "Pythia Pipeline Stage: hs_1 — hs_rc_collect",
     "conclusion": "success", "created_at": "2026-11-13T04:00:41Z"},
    {"id": 9, "display_title": "Pythia Pipeline Stage: hs_0 — hs_rc_collect",
     "conclusion": "success", "created_at": "2026-10-13T04:00:41Z"},
]


def test_the_two_hs_side_stage_runs_are_found_by_pipeline_id():
    assert sbc.stage_runs_of_pipeline("hs_1", RUNS) == {"hs_finalize_fc_submit": "3", "hs_rc_collect": "2"}


def test_each_earlier_context_lands_in_its_own_folder_or_a_stub(tmp_path):
    calls = []

    def runner(cmd):
        calls.append(cmd)
        if cmd[3] == "2":
            dest = Path(cmd[cmd.index("-D") + 1])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "stage_context.json").write_text("{}")
            return 0, "ok"
        return 1, "no artifact"

    res = sbc.fetch_earlier_stage_contexts(pipeline_id="hs_1", target=tmp_path, repo="o/r",
                                           list_runs=lambda: RUNS, runner=runner,
                                           sleep=lambda _s: None)
    assert res["hs_rc_collect"]["downloaded"] is True
    assert res["hs_finalize_fc_submit"]["downloaded"] is False
    assert (tmp_path / "hs_rc_collect" / "stage_context.json").is_file()
    stub = json.loads((tmp_path / "hs_finalize_fc_submit" / "MISSING.json").read_text())
    assert stub["artifact_name"] == "pythia-stage-context-hs_finalize_fc_submit"
    assert any("pythia-stage-context-hs_rc_collect" in c for c in calls)


def test_a_refused_download_is_tried_again(tmp_path):
    tries = []

    def runner(cmd):
        tries.append(cmd[3])
        if cmd[3] == "3" and tries.count("3") < 2:
            return 1, "HTTP 503: Egress is over the account limit"
        dest = Path(cmd[cmd.index("-D") + 1])
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "stage_context.json").write_text("{}")
        return 0, "ok"

    res = sbc.fetch_earlier_stage_contexts(pipeline_id="hs_1", target=tmp_path, repo="o/r",
                                           list_runs=lambda: RUNS, runner=runner,
                                           sleep=lambda _s: None)
    assert res["hs_finalize_fc_submit"]["downloaded"] is True
    assert tries.count("3") == 2


def test_no_pipeline_id_writes_two_stubs(tmp_path):
    sbc.fetch_earlier_stage_contexts(pipeline_id="", target=tmp_path, repo=None)
    assert (tmp_path / "hs_rc_collect" / "MISSING.json").is_file()
    assert (tmp_path / "hs_finalize_fc_submit" / "MISSING.json").is_file()
