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
    if "sibyl_run_id" in kw:
        argv += ["--sibyl-run-id", kw.pop("sibyl_run_id")]
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


def test_is_test_breakdown_is_rendered_in_markdown(tmp_path, monkeypatch):
    """The per-table counts must reach the LOG, not only the JSON artifact.

    When this check first fired in production it reported INCONSISTENT but the
    breakdown lived only in the artifact — unreadable from a restricted-network
    session, so the finding was visible and undiagnosable at the same time.
    """
    monkeypatch.delenv("PYTHIA_TEST_MODE", raising=False)
    path = _db(tmp_path)
    con = duckdb.connect(path)
    _call(con, call_id="c1", is_test=True)
    con.execute("INSERT INTO hs_triage VALUES (?, 1, 1, FALSE)", [RUN])
    con.close()

    rep = _run(path, tmp_path)
    md = stage_health._markdown(rep)
    assert "| llm_calls |" in md
    assert "| hs_triage |" in md


def test_absent_grounding_is_explained_not_blank(tmp_path):
    """A bare empty table can't distinguish 'failed' from 'happens elsewhere'."""
    path = _db(tmp_path)
    con = duckdb.connect(path)
    _call(con, call_id="c1", hazard_code="SPD_ACE")  # not a grounding row
    con.close()
    md = stage_health._markdown(_run(path, tmp_path))
    assert "No grounding calls in" in md


# --- sibyl -----------------------------------------------------------------------
#
# run_sibyl.yml emitted no diagnostics at all: Sibyl ran 40+ minutes on Claude
# Opus under a $40 hard cap and its spend was the one unmeasured number in the
# pipeline. On the reference DB it turned out to be $5.83 — larger than the
# $3.29 forecast pipeline it accompanies.

SIBYL_RUN = "sibyl_1785239977879"
OLD_SIBYL_RUN = "sibyl_previous_day"


def _sibyl_db(tmp_path, *, with_forecasts=True):
    path = _db(tmp_path)
    con = duckdb.connect(path)
    con.execute(
        """
        CREATE TABLE sibyl_runs (
            sibyl_run_id TEXT, hs_run_id TEXT, model TEXT, k INTEGER,
            aggregation TEXT, run_hard_cap_usd DOUBLE, budget_capped BOOLEAN,
            run_cost_usd DOUBLE, opus_cost_usd DOUBLE, brave_cost_usd DOUBLE,
            n_selected INTEGER, n_forecast INTEGER, n_skipped INTEGER,
            created_at TIMESTAMP
        )
        """
    )
    if with_forecasts:
        con.execute("CREATE TABLE sibyl_forecasts (sibyl_run_id TEXT, status TEXT)")
    con.close()
    return path


def _add_sibyl_run(con, sibyl_run_id, hs_run_id, created_at, **kw):
    d = dict(
        model="claude-opus-5", k=3, aggregation="linear_pool", cap=40.0,
        capped=False, cost=5.8338, opus=5.4838, brave=0.35,
        selected=4, forecast=4, skipped=0,
    )
    d.update(kw)
    con.execute(
        "INSERT INTO sibyl_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [sibyl_run_id, hs_run_id, d["model"], d["k"], d["aggregation"], d["cap"],
         d["capped"], d["cost"], d["opus"], d["brave"], d["selected"],
         d["forecast"], d["skipped"], created_at],
    )


def test_sibyl_section_reports_coverage_and_cost_split(tmp_path):
    path = _sibyl_db(tmp_path)
    con = duckdb.connect(path)
    _add_sibyl_run(con, SIBYL_RUN, RUN, "2026-07-29 10:00:00")
    for _ in range(4):
        con.execute("INSERT INTO sibyl_forecasts VALUES (?, 'ok')", [SIBYL_RUN])
    con.close()

    sb = _run(path, tmp_path, stage="sibyl", sibyl_run_id=SIBYL_RUN)["sibyl"]
    assert sb["available"] is True
    assert sb["run_cost_usd"] == pytest.approx(5.8338)
    assert sb["opus_cost_usd"] == pytest.approx(5.4838)
    assert sb["brave_cost_usd"] == pytest.approx(0.35)
    assert (sb["n_forecast"], sb["n_selected"]) == (4, 4)
    assert sb["by_status"] == {"ok": 4}


def test_sibyl_pins_on_run_id_rather_than_reporting_a_previous_run(tmp_path):
    """The hs_run_id fallback resolved the PREVIOUS day's run on the real DB.

    That is the stale-telemetry failure these diagnostics exist to catch, so a
    pinned sibyl_run_id must win over 'newest row'.
    """
    path = _sibyl_db(tmp_path)
    con = duckdb.connect(path)
    _add_sibyl_run(con, SIBYL_RUN, RUN, "2026-07-29 10:00:00", cost=5.83)
    # A later row for the same hs run — 'newest' would pick this one.
    _add_sibyl_run(con, OLD_SIBYL_RUN, RUN, "2026-07-29 23:00:00", cost=99.0)
    con.close()

    sb = _run(path, tmp_path, stage="sibyl", sibyl_run_id=SIBYL_RUN)["sibyl"]
    assert sb["sibyl_run_id"] == SIBYL_RUN
    assert sb["run_cost_usd"] == pytest.approx(5.83)


def test_sibyl_absent_rather_than_wrong_when_this_run_persisted_nothing(tmp_path):
    """Gated-out or crashed Sibyl must report nothing, not someone else's run."""
    path = _sibyl_db(tmp_path)
    con = duckdb.connect(path)
    _add_sibyl_run(con, OLD_SIBYL_RUN, "hs_yesterday", "2026-07-28 10:00:00")
    con.close()

    sb = _run(path, tmp_path, stage="sibyl", sibyl_run_id="sibyl_never_persisted")["sibyl"]
    assert sb["available"] is False
    assert OLD_SIBYL_RUN not in json.dumps(sb)


def test_sibyl_budget_cap_is_called_out(tmp_path):
    path = _sibyl_db(tmp_path)
    con = duckdb.connect(path)
    _add_sibyl_run(con, SIBYL_RUN, RUN, "2026-07-29 10:00:00",
                   capped=True, cost=40.7, selected=10, forecast=6, skipped=4)
    con.close()
    rep = _run(path, tmp_path, stage="sibyl", sibyl_run_id=SIBYL_RUN)
    md = stage_health._markdown(rep)
    assert "**CAPPED**" in md
    assert "6/10" in md


def test_sibyl_section_absent_on_a_pre_sibyl_db(tmp_path):
    """No sibyl tables at all must still exit 0 — never fail a stage."""
    path = _db(tmp_path)
    rep = _run(path, tmp_path, stage="sibyl", sibyl_run_id=SIBYL_RUN)
    assert rep["sibyl"]["available"] is False


def test_rc_levels_separate_seasonal_screen_outs_from_assessed_level_0(tmp_path):
    """hs_triage holds a row per country-hazard INCLUDING seasonal screen-outs.

    Those never reach the RC LLM and default to level 0, so folding them into
    by_level made stage_health report 435 level-0 rows for the 2026-08-01 run
    whose debug bundle correctly said 310 assessed at level 0 (+125 skipped).
    """
    path = str(tmp_path / "seasonal.duckdb")
    con = duckdb.connect(path)
    con.execute(
        """
        CREATE TABLE hs_triage (
            run_id TEXT, regime_change_level INTEGER, track INTEGER,
            is_test BOOLEAN, data_quality_json TEXT
        )
        """
    )
    for _ in range(3):  # assessed, level 0
        con.execute("INSERT INTO hs_triage VALUES (?, 0, 2, TRUE, ?)",
                    [RUN, '{"status": "ok"}'])
    for _ in range(5):  # seasonal screen-outs, never assessed
        con.execute("INSERT INTO hs_triage VALUES (?, 0, NULL, TRUE, ?)",
                    [RUN, '{"status": "seasonal_skip"}'])
    con.execute("INSERT INTO hs_triage VALUES (?, 2, 1, TRUE, ?)",
                [RUN, '{"status": "rc_promoted"}'])
    con.close()

    rc = _run(path, tmp_path)["rc_levels"]
    assert rc["n_rows"] == 9
    assert rc["n_not_assessed"] == 5
    assert rc["n_assessed"] == 4
    assert rc["by_level"]["0"] == 8            # every row, screen-outs included
    assert rc["by_level_assessed"]["0"] == 3   # only what RC actually scored
    assert rc["by_level_assessed"]["2"] == 1


def test_rc_levels_degrade_cleanly_without_data_quality_json(tmp_path):
    """A DB predating the column must not warn or mis-report."""
    path = _db(tmp_path)  # hs_triage here has no data_quality_json
    con = duckdb.connect(path)
    con.execute("INSERT INTO hs_triage VALUES (?, 0, 2, TRUE)", [RUN])
    con.close()
    rc = _run(path, tmp_path)["rc_levels"]
    assert rc["n_not_assessed"] == 0
    assert rc["by_level_assessed"] == rc["by_level"]
