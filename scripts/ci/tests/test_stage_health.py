# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for the per-stage health + cost snapshot.

As with batch_economics, the load-bearing property is that this can never fail
a pipeline stage — stages 2..N of a live pipeline run whatever is on main.
"""

from __future__ import annotations

import json

import duckdb
import pytest

from scripts.ci import stage_health

RUN = "hs_20260729T071137"


def _db(tmp_path, *, llm=True, triage=True):
    path = str(tmp_path / "t.duckdb")
    con = duckdb.connect(path)
    if llm:
        con.execute(
            """
            CREATE TABLE llm_calls (
                call_id TEXT, run_id TEXT, hs_run_id TEXT, call_type TEXT, phase TEXT,
                model_id TEXT, error_text TEXT, cost_usd DOUBLE, total_tokens BIGINT,
                timestamp TIMESTAMP, hazard_code TEXT, is_test BOOLEAN
            )
            """
        )
    if triage:
        con.execute(
            """
            CREATE TABLE hs_triage (
                run_id TEXT, regime_change_level INTEGER, track INTEGER, is_test BOOLEAN
            )
            """
        )
    con.close()
    return path


def _call(con, **kw):
    d = dict(call_id="c", run_id=None, hs_run_id=RUN, call_type="chat", phase="hs_triage",
             model_id="gemini-3.5-flash", error_text=None, cost_usd=0.1, total_tokens=100,
             timestamp="2026-07-29 07:11:40", hazard_code="RC_ACE_PASS_1", is_test=True)
    d.update(kw)
    con.execute(
        "INSERT INTO llm_calls VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [d["call_id"], d["run_id"], d["hs_run_id"], d["call_type"], d["phase"], d["model_id"],
         d["error_text"], d["cost_usd"], d["total_tokens"], d["timestamp"], d["hazard_code"],
         d["is_test"]],
    )


def _run(path, tmp_path, **kw):
    out = str(tmp_path / "h.json")
    argv = ["--db", path, "--stage", kw.pop("stage", "hs_submit"), "--hs-run-id",
            kw.pop("hs_run_id", RUN), "--out", out]
    if "since" in kw:
        argv += ["--since", kw.pop("since")]
    assert stage_health.main(argv) == 0
    return json.load(open(out))


# --- grounding -------------------------------------------------------------------


def test_separates_rc_and_triage_grounding(tmp_path):
    """RC's LIKE pattern would otherwise swallow TRIAGE_GROUNDING_* rows."""
    path = _db(tmp_path)
    con = duckdb.connect(path)
    _call(con, call_id="g1", hazard_code="GROUNDING_ACE")
    _call(con, call_id="g2", hazard_code="GROUNDING_DR")
    _call(con, call_id="t1", hazard_code="TRIAGE_GROUNDING_ACE")
    con.close()

    rep = _run(path, tmp_path)
    assert rep["grounding"]["rc"]["n_calls"] == 2
    assert rep["grounding"]["triage"]["n_calls"] == 1


def test_breaker_sentinel_is_counted_and_warned(tmp_path, capsys):
    path = _db(tmp_path)
    con = duckdb.connect(path)
    _call(con, call_id="g1", hazard_code="GROUNDING_ACE", model_id="grounding-breaker-tripped")
    con.close()

    rep = _run(path, tmp_path)
    rc = rep["grounding"]["rc"]
    assert rc["n_no_backend"] == 1
    assert "grounding-breaker-tripped" in rc["no_backend_by_reason"]
    out = capsys.readouterr().out
    assert "::warning title=Brave breaker tripped::" in out


def test_no_backend_majority_warns(tmp_path, capsys):
    path = _db(tmp_path)
    con = duckdb.connect(path)
    _call(con, call_id="g1", hazard_code="GROUNDING_ACE", model_id="grounding-failed")
    _call(con, call_id="g2", hazard_code="GROUNDING_DR", model_id="grounding-unavailable")
    con.close()
    _run(path, tmp_path)
    assert "::warning title=Grounding largely failed::" in capsys.readouterr().out


def test_healthy_grounding_produces_no_warning(tmp_path, capsys):
    path = _db(tmp_path)
    con = duckdb.connect(path)
    _call(con, call_id="g1", hazard_code="GROUNDING_ACE", model_id="brave-web-search")
    con.close()
    _run(path, tmp_path)
    assert "::warning" not in capsys.readouterr().out


def test_other_runs_are_excluded(tmp_path):
    path = _db(tmp_path)
    con = duckdb.connect(path)
    _call(con, call_id="g1", hazard_code="GROUNDING_ACE")
    _call(con, call_id="g2", hazard_code="GROUNDING_ACE", hs_run_id="hs_other")
    con.close()
    assert _run(path, tmp_path)["grounding"]["rc"]["n_calls"] == 1


# --- is_test ---------------------------------------------------------------------


def test_detects_mixed_is_test_across_tables(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("PYTHIA_TEST_MODE", raising=False)
    path = _db(tmp_path)
    con = duckdb.connect(path)
    _call(con, call_id="c1", is_test=True)
    con.execute("INSERT INTO hs_triage VALUES (?, 1, 1, FALSE)", [RUN])
    con.close()

    rep = _run(path, tmp_path)
    assert rep["is_test"]["consistent"] is False
    assert "::warning title=is_test inconsistent::" in capsys.readouterr().out


def test_flags_mismatch_against_expected_test_mode(tmp_path, capsys, monkeypatch):
    """The poller can deliver a boolean input as the string 'false'."""
    monkeypatch.setenv("PYTHIA_TEST_MODE", "1")
    path = _db(tmp_path)
    con = duckdb.connect(path)
    _call(con, call_id="c1", is_test=False)
    con.close()

    rep = _run(path, tmp_path)
    assert rep["is_test"]["matches_expected"] is False
    assert "::warning title=is_test does not match PYTHIA_TEST_MODE::" in capsys.readouterr().out


def test_consistent_is_test_matching_mode_is_silent(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("PYTHIA_TEST_MODE", "1")
    path = _db(tmp_path)
    con = duckdb.connect(path)
    _call(con, call_id="c1", is_test=True, model_id="brave-web-search", hazard_code="GROUNDING_ACE")
    con.execute("INSERT INTO hs_triage VALUES (?, 0, 2, TRUE)", [RUN])
    con.close()
    _run(path, tmp_path)
    assert "::warning" not in capsys.readouterr().out


# --- cost ------------------------------------------------------------------------


def test_stage_window_excludes_earlier_stage_spend(tmp_path):
    """llm_calls is cumulative in the travelling DB; the window is the whole point."""
    path = _db(tmp_path)
    con = duckdb.connect(path)
    _call(con, call_id="old", timestamp="2026-07-29 07:00:00", cost_usd=5.0)
    _call(con, call_id="new", timestamp="2026-07-29 07:30:00", cost_usd=2.0)
    con.close()

    rep = _run(path, tmp_path, since="2026-07-29 07:20:00")
    assert rep["cost"]["run_total"]["cost_usd"] == 7.0
    assert rep["cost"]["stage_total"]["cost_usd"] == 2.0


def test_cost_rolls_up_by_phase(tmp_path):
    path = _db(tmp_path)
    con = duckdb.connect(path)
    _call(con, call_id="a", phase="hs_triage", cost_usd=1.0)
    _call(con, call_id="b", phase="spd_v2", cost_usd=3.0)
    con.close()
    rep = _run(path, tmp_path, since="2026-07-29 00:00:00")
    by = {r["key"]: r["cost_usd"] for r in rep["cost"]["stage_by_phase"]}
    assert by["spd_v2"] == 3.0 and by["hs_triage"] == 1.0


def test_without_since_stage_totals_are_absent_not_wrong(tmp_path):
    path = _db(tmp_path)
    con = duckdb.connect(path)
    _call(con, call_id="a", cost_usd=1.0)
    con.close()
    rep = _run(path, tmp_path)
    assert rep["cost"]["stage_total"] is None


# --- rc levels -------------------------------------------------------------------


def test_rc_level_distribution(tmp_path):
    path = _db(tmp_path)
    con = duckdb.connect(path)
    for lvl, trk in ((0, 2), (1, 1), (1, 1)):
        con.execute("INSERT INTO hs_triage VALUES (?, ?, ?, TRUE)", [RUN, lvl, trk])
    con.close()
    rc = _run(path, tmp_path)["rc_levels"]
    assert rc["n_rows"] == 3 and rc["by_level"]["1"] == 2 and rc["by_track"]["1"] == 2


# --- never-fail-a-stage ----------------------------------------------------------


def test_missing_tables_exit_zero(tmp_path):
    path = _db(tmp_path, llm=False, triage=False)
    rep = _run(path, tmp_path)
    assert rep["grounding"]["available"] is False
    assert rep["rc_levels"]["available"] is False


def test_missing_db_exits_zero(tmp_path):
    assert stage_health.main(["--db", str(tmp_path / "nope.duckdb"), "--stage", "s"]) == 0


def test_unwritable_out_exits_zero(tmp_path):
    path = _db(tmp_path)
    assert stage_health.main(
        ["--db", path, "--stage", "s", "--out", "/proc/nope/x.json"]
    ) == 0


def test_writes_step_summary(tmp_path, monkeypatch):
    path = _db(tmp_path)
    con = duckdb.connect(path)
    _call(con, call_id="g1", hazard_code="GROUNDING_ACE", model_id="brave-web-search")
    con.close()
    s = tmp_path / "sum.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(s))
    _run(path, tmp_path)
    assert "Stage health" in s.read_text()
