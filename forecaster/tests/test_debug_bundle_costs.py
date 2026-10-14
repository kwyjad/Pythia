# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The debug bundle's costs (Oct 2026).

On 13 October 2026 Sibyl made 881 calls for $60.46, and the bundle's call
tables and cost total left every one of them out, because its filter read
the forecaster run and the HS triage phase only. The summary total and
run_comparison now read one clause, ``run_comparison.run_llm_where``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from scripts.debug_bundle import run_comparison  # noqa: E402

HS = "hs_20261013T041422"
FC = "fc_1791866322"
SIB = "sibyl_1791868847099"


def _db(tmp_path: Path):
    con = duckdb.connect(str(tmp_path / "c.duckdb"))
    con.execute("""CREATE TABLE llm_calls (call_id TEXT, phase TEXT, call_type TEXT, provider TEXT,
        model_id TEXT, model_name TEXT, run_id TEXT, hs_run_id TEXT, question_id TEXT, iso3 TEXT,
        hazard_code TEXT, cost_usd DOUBLE, usage_json TEXT, error_text TEXT, prompt_text TEXT,
        response_text TEXT, elapsed_ms INTEGER, timestamp TIMESTAMP)""")
    rows = [
        # forecaster, batched
        ("1", "spd_v2", "spd", "openai", "gpt-6-sol", FC, None, 2.0, {"service_tier": "batch"}, ""),
        ("2", "binary_v2", "binary", "google", "gemini-3.6-flash", FC, None, 1.0, {}, ""),
        # HS: one batchable RC pass, one grounding call (never batchable)
        ("3", "hs_triage", "rc_pass_1", "google", "gemini-3.6-flash", HS, HS, 0.5, {"service_tier": "batch"}, ""),
        ("4", "hs_triage", "rc_grounding", "brave", "brave-web-search", HS, HS, 0.25, {}, ""),
        # scenarios, never batchable
        ("5", "scenario_v2", "scenario", "google", "gemini-3.6-flash", FC, None, 0.75, {}, ""),
        # Sibyl: run_id is the Sibyl run, hs_run_id the HS run
        ("6", "sibyl", "sibyl_trial0_step1", "anthropic", "claude-opus-5-5", SIB, HS, 3.0, {}, ""),
        ("7", "sibyl", "sibyl_trial0_search", "brave", "brave-web-search", SIB, HS, 0.005, {}, "no_results"),
        # another run altogether
        ("8", "spd_v2", "spd", "openai", "gpt-6-sol", "fc_other", None, 99.0, {}, ""),
    ]
    for cid, phase, ct, prov, model, run, hs, cost, usage, err in rows:
        con.execute(
            "INSERT INTO llm_calls VALUES (?,?,?,?,?,?,?,?,NULL,NULL,NULL,?,?,?,'','',0,TIMESTAMP '2026-10-13 05:00:00')",
            [cid, phase, ct, prov, model, model, run, hs, cost, json.dumps(usage), err],
        )
    con.execute("CREATE TABLE sibyl_runs (sibyl_run_id TEXT, hs_run_id TEXT, created_at TIMESTAMP, "
                "run_cost_usd DOUBLE, n_selected INTEGER, n_forecast INTEGER)")
    con.execute(f"INSERT INTO sibyl_runs VALUES ('{SIB}', '{HS}', TIMESTAMP '2026-10-13 06:00:00', 3.005, 1, 1)")
    con.execute("CREATE TABLE hs_runs (hs_run_id TEXT, generated_at TIMESTAMP, is_test BOOLEAN)")
    con.execute(f"INSERT INTO hs_runs VALUES ('{HS}', TIMESTAMP '2026-10-13 04:14:22', FALSE)")
    con.execute("CREATE TABLE run_questions (hs_run_id TEXT, question_id TEXT, forecaster_run_id TEXT)")
    con.execute(f"INSERT INTO run_questions VALUES ('{HS}', 'Q1', '{FC}')")
    con.execute("CREATE TABLE questions (question_id TEXT, hs_run_id TEXT)")
    con.execute(f"INSERT INTO questions VALUES ('Q1', '{HS}')")
    con.execute("CREATE TABLE forecasts_ensemble (run_id TEXT, question_id TEXT)")
    con.execute(f"INSERT INTO forecasts_ensemble VALUES ('{FC}', 'Q1')")
    con.execute("CREATE TABLE hs_triage (run_id TEXT, iso3 TEXT)")
    return con


def _column(rows, metric):
    row = next(r for r in rows if r["metric"] == metric)
    return next(v for k, v in row.items() if k.startswith("run_0__"))


def test_run_llm_where_takes_the_three_runs_and_the_hs_stamp():
    where, params = run_comparison.run_llm_where(HS, FC, SIB)
    assert params == [FC, SIB, HS, HS]
    assert "hs_run_id = ?" in where


def test_run_comparison_carries_sibyl_and_per_phase_and_model_costs(tmp_path):
    con = _db(tmp_path)
    rows = run_comparison.collect(con, current_hs_run_id=HS)
    assert _column(rows, "sibyl_run_id") == SIB
    assert _column(rows, "total_cost_usd") == pytest.approx(7.505)
    assert _column(rows, "cost_sibyl_usd") == pytest.approx(3.005)
    assert _column(rows, "cost_phase:sibyl") == pytest.approx(3.005)
    assert _column(rows, "cost_model:claude-opus-5-5") == pytest.approx(3.0)
    # Batch tier over batchable work: spd, binary and the RC pass ($3.50),
    # of which $2.50 ran at batch tier.
    assert _column(rows, "batchable_cost_usd") == pytest.approx(3.5)
    assert _column(rows, "batch_tier_cost_pct") == pytest.approx(71.4)
    assert _column(rows, "n_llm_errors") == 1


def test_the_summary_total_equals_the_run_comparison_total(tmp_path):
    from scripts.dump_pythia_debug_bundle import _aggregate_usage_by_phase

    con = _db(tmp_path)
    by_phase = _aggregate_usage_by_phase(con, FC, HS, SIB)
    summary_total = round(sum(v["total_cost_usd"] for v in by_phase.values()), 4)
    rows = run_comparison.collect(con, current_hs_run_id=HS)
    assert summary_total == pytest.approx(_column(rows, "total_cost_usd"))
    assert "sibyl" in by_phase


def test_the_current_jobs_teed_logs_are_copied_into_the_bundle(tmp_path):
    from scripts.debug_bundle import workflow_logs

    logs = tmp_path / "job_logs"
    logs.mkdir()
    (logs / "sibyl_run.log").write_text("sibyl.run: budget cap ($60.00) reached\n")
    (tmp_path / "sibyl_stdout.txt").write_text("sibyl_run_id=sibyl_1\n")
    index = {"problems": []}
    workflow_logs.collect_local_logs(tmp_path / "out", index,
                                     paths=f"{tmp_path / 'sibyl_stdout.txt'},{logs}")
    files = sorted(p.name for p in (tmp_path / "out" / "current_job").iterdir())
    assert len(files) == 2 and len(index["local_logs"]) == 2
    assert any("budget cap" in p.read_text() for p in (tmp_path / "out" / "current_job").iterdir())


def test_a_shallow_checkout_reads_the_commit_range_from_the_compare_api(tmp_path, monkeypatch):
    from scripts.debug_bundle import code_snapshot

    class Api:
        repo = "kwyjad/Pythia"

        def get_json(self, path, params=None):
            assert path == "/repos/kwyjad/Pythia/compare/aaaaaaa...bbbbbbb"
            return {"total_commits": 2, "commits": [
                {"sha": "1111111aaaa", "commit": {"message": "First change\n\nbody"}},
                {"sha": "2222222bbbb", "commit": {"message": "Second change"}}]}

    monkeypatch.setattr(code_snapshot, "_run", lambda cmd, cwd: (1, "not a commit"))
    text = code_snapshot._commits_since(tmp_path, "aaaaaaa", "bbbbbbb", api=Api())
    assert "1111111 First change" in text and "2222222 Second change" in text
    assert "compare API" in text


def test_untracked_run_outputs_do_not_make_the_checkout_dirty(tmp_path):
    import subprocess

    from scripts.debug_bundle.env_config import _dirty

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "a.txt").write_text("x")
    subprocess.run(["git", "-C", str(tmp_path), "add", "a.txt"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "c"], check=True)
    (tmp_path / "diagnostics.json").write_text("{}")  # a run's own output
    out = _dirty(tmp_path)
    assert out["dirty"] is False and out["n_untracked"] == 1
    (tmp_path / "a.txt").write_text("changed")
    out = _dirty(tmp_path)
    assert out["dirty"] is True and out["tracked_changes"] == [" M a.txt"]
