# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""The scored bundle's error-attribution files (scripts/ai_bundle/error_attribution.py).

Each test builds a small DuckDB holding exactly the case it asks about: a
prior worse than the base rate shown and an update that helps; a flagged
question whose outcome moves and an unflagged one whose outcome does not; a
conflict spike nobody asked about; a binary event forecast at 3%; two advice
arms; a headline skill checked by hand. And every section must degrade to a
stub naming its reason when its tables are absent.
"""

from __future__ import annotations

import csv
import json
import zipfile
from datetime import date
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from scripts.ai_bundle import error_attribution as ea
from scripts.ai_bundle.build_scored_forecast_bundle import build_bundle, main

K = 7  # FATALITIES buckets: 0 | 1-4 | 5-24 | 25-99 | 100-499 | 500-999 | 1000+


def _schema(con) -> None:
    con.execute(
        "CREATE TABLE questions (question_id TEXT, hs_run_id TEXT, iso3 TEXT, hazard_code TEXT, "
        "metric TEXT, target_month TEXT, window_start_date DATE, window_end_date DATE, wording TEXT, "
        "status TEXT, track INTEGER, pythia_metadata_json TEXT, is_test BOOLEAN DEFAULT FALSE)"
    )
    con.execute(
        "CREATE TABLE scores (question_id TEXT, horizon_m INTEGER, metric TEXT, score_type TEXT, "
        "model_name TEXT, value DOUBLE, run_id TEXT, created_at TIMESTAMP DEFAULT now(), "
        "is_test BOOLEAN DEFAULT FALSE)"
    )
    con.execute(
        "CREATE TABLE resolutions (question_id TEXT, horizon_m INTEGER, observed_month TEXT, "
        "value DOUBLE, source_snapshot_ym TEXT, source_desc TEXT, is_test BOOLEAN DEFAULT FALSE)"
    )
    con.execute(
        "CREATE TABLE forecasts_raw (run_id TEXT, question_id TEXT, model_name TEXT, month_index INTEGER, "
        "bucket_index INTEGER, probability DOUBLE, ok BOOLEAN, elapsed_ms INTEGER, cost_usd DOUBLE, "
        "prompt_tokens INTEGER, completion_tokens INTEGER, total_tokens INTEGER, status TEXT, "
        "spd_json TEXT, human_explanation TEXT, reasoning_trace_json TEXT, rc_guidance TEXT, "
        "base_rate_block_version TEXT, recalibration_json TEXT, advice_arm TEXT)"
    )
    con.execute(
        "CREATE TABLE forecasts_ensemble (run_id TEXT, question_id TEXT, model_name TEXT, "
        "month_index INTEGER, bucket_index INTEGER, probability DOUBLE, ev_value DOUBLE, "
        "weights_profile TEXT, status TEXT, created_at TIMESTAMP, advice_arm TEXT, is_test BOOLEAN DEFAULT FALSE)"
    )
    con.execute(
        "CREATE TABLE hs_triage (run_id TEXT, iso3 TEXT, hazard_code TEXT, tier TEXT, triage_score DOUBLE, "
        "need_full_spd BOOLEAN, drivers_json TEXT, data_quality_json TEXT, scenario_stub TEXT, "
        "regime_change_likelihood DOUBLE, regime_change_magnitude DOUBLE, regime_change_score DOUBLE, "
        "regime_change_level INTEGER, regime_change_direction TEXT, regime_change_window TEXT, "
        "regime_change_json TEXT, track INTEGER)"
    )
    con.execute(
        "CREATE TABLE baseline_scored_forecasts (question_id TEXT, horizon_m INTEGER, model_name TEXT, "
        "metric TEXT, spd_json TEXT)"
    )
    con.execute(
        "CREATE TABLE acled_monthly_fatalities (iso3 TEXT, month DATE, fatalities DOUBLE, "
        "updated_at TIMESTAMP)"
    )


def _question(con, qid, iso3, hz, metric, ws="2026-08-01", track=1, hs="hs1"):
    con.execute(
        "INSERT INTO questions (question_id, hs_run_id, iso3, hazard_code, metric, target_month, "
        "window_start_date, window_end_date, wording, status, track) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [qid, hs, iso3, hz, metric, "2027-01", ws, "2027-01-31", "w", "active", track],
    )


def _triage(con, iso3, hz, level, direction, hs="hs1"):
    con.execute(
        "INSERT INTO hs_triage (run_id, iso3, hazard_code, tier, triage_score, regime_change_level, "
        "regime_change_direction, regime_change_score) VALUES (?,?,?,?,?,?,?,?)",
        [hs, iso3, hz, "priority", 0.7, level, direction, 0.3],
    )


def _spd(con, table, run, qid, model, h, probs, created="2026-08-01 05:00:00", **extra):
    for b, p in enumerate(probs, start=1):
        if table == "forecasts_raw":
            con.execute(
                "INSERT INTO forecasts_raw (run_id, question_id, model_name, month_index, bucket_index, "
                "probability, reasoning_trace_json, rc_guidance, base_rate_block_version, advice_arm, "
                "recalibration_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [run, qid, model, h, b, p, extra.get("trace"), extra.get("rc_guidance"),
                 extra.get("brv"), extra.get("arm"), extra.get("recal")],
            )
        else:
            con.execute(
                "INSERT INTO forecasts_ensemble (run_id, question_id, model_name, month_index, "
                "bucket_index, probability, created_at, advice_arm) VALUES (?,?,?,?,?,?,?,?)",
                [run, qid, model, h, b, p, created, extra.get("arm")],
            )


def _score(con, qid, model, h, st, v, run="r1"):
    con.execute(
        "INSERT INTO scores (question_id, horizon_m, metric, score_type, model_name, value, run_id) "
        "VALUES (?,?,?,?,?,?,?)",
        [qid, h, "x", st, model, v, None if model.startswith("__ext_") else run],
    )


def _resolve(con, qid, h, value, month="2026-08"):
    con.execute(
        "INSERT INTO resolutions (question_id, horizon_m, observed_month, value) VALUES (?,?,?,?)",
        [qid, h, month, value],
    )


def _baseline(con, qid, model, h, probs):
    con.execute(
        "INSERT INTO baseline_scored_forecasts VALUES (?,?,?,?,?)",
        [qid, h, model, "FATALITIES", json.dumps(probs)],
    )


PRIOR = [0.6, 0.2, 0.1, 0.05, 0.03, 0.01, 0.01]
DELTA = [-0.3, -0.1, 0.0, 0.1, 0.3, 0.0, 0.0]
POST = [round(p + d, 6) for p, d in zip(PRIOR, DELTA)]
CLIM = [0.05, 0.05, 0.1, 0.2, 0.4, 0.1, 0.1]
TRACE = {
    "prior": {"spd": PRIOR, "rationale": "base rate anchored on the 36-month history"},
    "updates": [
        {"signal": "ACLED shows escalation in the last three months", "direction": "UP",
         "magnitude": "LARGE", "months_affected": "1-2", "delta": DELTA, "post_update_spd": POST},
    ],
    "rc_assessment": "accepted",
}


@pytest.fixture
def db(tmp_path: Path) -> str:
    """Q1: ACE/FATALITIES, outcome 300 (bucket 5), a bad prior and a helpful update.
    Q2: the same country-hazard pattern, unflagged, outcome stays in its bucket."""
    path = tmp_path / "e.duckdb"
    con = duckdb.connect(str(path))
    _schema(con)
    _question(con, "Q1", "ETH", "ACE", "FATALITIES")
    _question(con, "Q2", "KEN", "ACE", "FATALITIES")
    _triage(con, "ETH", "ACE", 2, "up")
    _triage(con, "KEN", "ACE", 0, "unclear")
    for qid, outcome, last in (("Q1", 300.0, 10.0), ("Q2", 12.0, 10.0)):
        _resolve(con, qid, 1, outcome)
        _resolve(con, qid, 2, outcome, "2026-09")
        _spd(con, "forecasts_raw", "r1", qid, "model-a", 1, POST, trace=json.dumps(TRACE))
        _spd(con, "forecasts_raw", "r1", qid, "model-a", 2, POST, trace=json.dumps(TRACE))
        _spd(con, "forecasts_ensemble", "r1", qid, "ensemble_mean_v2", 1, POST)
        _spd(con, "forecasts_ensemble", "r1", qid, "ensemble_mean_v2", 2, POST)
        for h in (1, 2):
            _baseline(con, qid, "__ext_climatology", h, CLIM)
        _score(con, qid, "ensemble_mean_v2", 1, "brier", 0.4 if qid == "Q1" else 0.2)
        _score(con, qid, "__ext_climatology", 1, "brier", 0.5)
        _score(con, qid, "model-a", 1, "brier", 0.6)
        iso = "ETH" if qid == "Q1" else "KEN"
        con.execute(
            "INSERT INTO acled_monthly_fatalities VALUES (?, DATE '2026-07-01', ?, TIMESTAMP '2026-08-20')",
            [iso, last],
        )
    con.close()
    return str(path)


def _ctx(db: str, qids=("Q1", "Q2")) -> ea.Context:
    con = duckdb.connect(db)
    return ea.build_context(con, list(qids))


# ---------------------------------------------------------------------------
# input_partial_month
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hz, d, flag",
    [
        ("ACE", date(2026, 8, 1), True),   # July row written on 15 July
        ("ACE", date(2026, 9, 1), True),   # August row written on 28 August
        ("ACE", date(2026, 7, 15), False), # the 15 July ingest rewrote June complete
        ("ACE", date(2026, 10, 1), False), # complete-month rule in force
        ("FL", date(2026, 8, 1), False),   # no ACLED trajectory in a flood prompt
    ],
)
def test_input_partial_month_names_the_known_runs(hz, d, flag):
    got, basis = ea.input_partial_month(hz, d)
    assert got is flag
    assert basis


def test_input_partial_month_is_unknown_without_a_date():
    assert ea.input_partial_month("ACE", None)[0] is None


def test_months_affected_parses_the_trace_wordings():
    assert ea.months_affected("1-2") == {1, 2}
    assert ea.months_affected("all") == set(range(1, 7))
    assert ea.months_affected("months 3, 4") == {3, 4}
    assert ea.months_affected([5, 6]) == {5, 6}
    assert ea.months_affected("garbled") == set(range(1, 7))


# ---------------------------------------------------------------------------
# 1, 2. trace stages and update value
# ---------------------------------------------------------------------------


def test_a_prior_worse_than_the_base_rate_shows_as_a_bad_start(db):
    ctx = _ctx(db)
    rows = ea.build_trace_stages(ctx)
    q1 = [r for r in rows if r["question_id"] == "Q1" and r["horizon_m"] == 1]
    assert len(q1) == 1
    r = q1[0]
    assert r["shown_source"] == "climatology"
    assert r["realized_bucket"] == 5
    assert r["rps_prior"] > r["rps_shown"]           # the start was worse than the base rate
    assert r["rps_final"] < r["rps_prior"]           # and the adjustments helped
    assert r["expected_bucket_shift_prior_to_final"] > 0
    assert r["js_distance_prior_vs_shown"] > 0
    assert r["input_partial_month"] is True          # forecast on 1 August 2026
    summary = ea.summarise_trace_stages(rows)
    rps = [s for s in summary if s["score_type"] == "rps"][0]
    assert rps["n_questions"] == 2
    assert rps["prior_minus_shown"] > 0
    assert rps["start_verdict"] == "too few"         # two questions is not a finding


def test_an_update_that_moves_toward_the_outcome_is_credited(db):
    ctx = _ctx(db)
    rows = ea.build_update_value(ctx)
    q1 = [r for r in rows if r["question_id"] == "Q1"]
    assert {r["horizon_m"] for r in q1} == {1, 2}    # months_affected "1-2"
    for r in q1:
        assert r["moved_toward_outcome"] is True
        assert r["delta_rps"] < 0
        assert r["attribution_id"]
    summary = ea.summarise_update_value(rows)
    assert summary and summary[0]["n_updates"] == 2  # one update in each of two questions


# ---------------------------------------------------------------------------
# 3. rc_outcomes
# ---------------------------------------------------------------------------


def test_a_flagged_question_moves_and_an_unflagged_one_does_not(db):
    ctx = _ctx(db)
    rows = ea.build_rc_outcomes(ctx)
    q1 = [r for r in rows if r["question_id"] == "Q1"][0]
    q2 = [r for r in rows if r["question_id"] == "Q2"][0]
    assert q1["last_bucket"] == 3 and q1["outcome_bucket"] == 5
    assert q1["bucket_move"] == 2 and q1["direction_matched"] is True
    assert q2["bucket_move"] == 0 and q2["direction_matched"] is None   # RC 0 is the control
    summary = {(s["rc_level"], s["rc_direction"]): s for s in ea.summarise_rc_outcomes(rows)}
    assert summary[(2, "up")]["share_moved_2plus"] == 1.0
    assert summary[(0, "unclear")]["mean_abs_bucket_move"] == 0


# ---------------------------------------------------------------------------
# 4. unasked outcomes
# ---------------------------------------------------------------------------


def test_a_conflict_spike_nobody_asked_about_is_listed(db):
    con = duckdb.connect(db)
    con.execute(
        "INSERT INTO acled_monthly_fatalities VALUES ('SDN', DATE '2026-08-01', 650, TIMESTAMP '2026-09-20')"
    )
    con.execute(
        "INSERT INTO acled_monthly_fatalities VALUES ('SDN', DATE '2026-07-01', 3, TIMESTAMP '2026-08-20')"
    )
    ctx = ea.build_context(con, ["Q1", "Q2"])
    con.execute(
        "CREATE TABLE facts_resolved (iso3 TEXT, hazard_code TEXT, metric TEXT, ym TEXT, value DOUBLE, "
        "alertlevel TEXT)"
    )
    for _ in range(3):  # three rows for one country-month are one cell
        con.execute("INSERT INTO facts_resolved VALUES ('SDN', 'FL', 'event_occurrence', '2026-08', 1, 'Orange')")
    rows = ea.build_unasked_outcomes(ctx, countries=["ETH", "KEN", "SDN"])
    assert len([r for r in rows if r["hazard_code"] == "FL"]) == 1
    sdn = [r for r in rows if r["iso3"] == "SDN"]
    assert sdn and sdn[0]["trigger"] == "deaths_bucket_5_plus"
    assert sdn[0]["value"] == 650 and sdn[0]["month"] == "2026-08"
    assert sdn[0]["triage_tier"] == "not assessed"
    assert not [r for r in rows if r["iso3"] in ("ETH", "KEN")]   # asked, so not listed


def test_the_hs_country_list_resolves():
    iso3s = ea.hs_country_iso3s()
    assert len(iso3s) > 100 and "AFG" in iso3s


# ---------------------------------------------------------------------------
# 5. experiments and rollup split columns
# ---------------------------------------------------------------------------


def _arms_db(tmp_path: Path, n: int) -> str:
    path = tmp_path / f"arms{n}.duckdb"
    con = duckdb.connect(str(path))
    _schema(con)
    for i in range(2 * n):
        qid = f"A{i}"
        arm = "advice" if i < n else "no_advice"
        _question(con, qid, "ETH", "ACE", "FATALITIES")
        _resolve(con, qid, 1, 300.0)
        _spd(con, "forecasts_raw", "r1", qid, "model-a", 1, POST, arm=arm)
        _spd(con, "forecasts_ensemble", "r1", qid, "ensemble_mean_v2", 1, POST, arm=arm)
        # The no-advice arm sits 0.3 further from climatology, plus a little noise.
        _score(con, qid, "ensemble_mean_v2", 1, "brier", 0.40 + (0.3 if arm == "no_advice" else 0.0) + 0.01 * (i % 3))
        _score(con, qid, "__ext_climatology", 1, "brier", 0.5)
    con.close()
    return str(path)


def test_two_advice_arms_are_compared_and_too_few_says_so(tmp_path):
    ctx = _ctx(_arms_db(tmp_path, 3), [f"A{i}" for i in range(6)])
    rows = [r for r in ea.build_experiments(ctx) if r["flag"] == "advice_arm"]
    assert len(rows) == 1
    r = rows[0]
    assert {r["arm"], r["reference_arm"]} == {"advice", "no_advice"}
    assert r["verdict"] == "too few"


def test_a_clear_arm_difference_gets_a_verdict(tmp_path):
    ctx = _ctx(_arms_db(tmp_path, 12), [f"A{i}" for i in range(24)])
    r = [r for r in ea.build_experiments(ctx) if r["flag"] == "advice_arm"][0]
    assert r["n_arm"] == 12 and r["n_reference"] == 12
    assert r["verdict"] == "arm advice better"


def test_rollups_split_on_the_advice_arm(tmp_path):
    db = _arms_db(tmp_path, 3)
    out = tmp_path / "out"
    zp = build_bundle(db, out, months_back=0)
    with zipfile.ZipFile(zp) as zf:
        rows = list(csv.DictReader(zf.read("rollups.csv").decode().splitlines()))
    mean = [r for r in rows if r["model_name"] == "ensemble_mean_v2" and r["score_type"] == "brier"]
    assert {r["advice_arm"] for r in mean} == {"advice", "no_advice"}
    for r in mean:
        assert r["n_paired"] == "3"   # each arm paired with its own climatology rows only
        assert r["input_partial_month"] in ("True", "False")


# ---------------------------------------------------------------------------
# 6, 9. skill history and headline
# ---------------------------------------------------------------------------


def test_headline_matches_a_hand_computed_paired_skill(db):
    ctx = _ctx(db)
    head = ea.build_headline(ctx)
    (g,) = [g for g in head["groups"] if g["metric"] == "FATALITIES"]
    brier = g["scores"]["brier"]
    clim = brier["references"]["__ext_climatology"]
    # primary 0.4 + 0.2 against climatology 0.5 + 0.5: skill = 1 - 0.6 / 1.0
    assert clim["skill"] == pytest.approx(0.4)
    assert clim["wins"] == 2 and clim["n_paired_questions"] == 2
    assert clim["skill_ci90_low"] is not None and clim["skill_ci90_low"] <= 0.4 <= clim["skill_ci90_high"]
    assert brier["warning"] == "fewer than 10 paired questions"
    assert head["questions_resolved_per_horizon"] == {"1": 2, "2": 2}
    lines = "\n".join(ea.headline_digest_lines(head))
    assert "+0.400" in lines


def test_headline_is_also_split_by_horizon_and_the_digest_prints_h2_beside_h1(db):
    ctx = _ctx(db)
    head = ea.build_headline(ctx)
    (g,) = [g for g in head["groups"] if g["metric"] == "FATALITIES"]
    by_h = g["scores"]["brier"]["by_horizon"]
    # The fixture scores horizon 1 only, so h1 carries the whole pooled figure
    # and h2 is absent rather than invented.
    assert set(by_h) == {"1"}
    assert by_h["1"]["n_paired_questions"] == 2
    assert by_h["1"]["skill_vs_climatology"] == pytest.approx(0.4)
    lines = "\n".join(ea.headline_digest_lines(head))
    assert "### Horizon 1 beside horizon 2" in lines
    assert "| h1 n q | h1 skill vs clim [90%] | h2 n q | h2 skill vs clim [90%] |" in lines
    assert "| ACE | FATALITIES | T1 | Brier | 2 ⚠ | +0.400" in lines and "| 0 | — |" in lines


def test_skill_history_is_per_observed_month_and_split_on_partial_input(db):
    ctx = _ctx(db)
    rows = ea.build_skill_history(ctx)
    primary = [r for r in rows if r["forecaster"] == "primary" and r["reference"] == "__ext_climatology"]
    assert len(primary) == 1
    r = primary[0]
    assert r["observed_month"] == "2026-08" and r["n_questions"] == 2
    assert r["skill"] == pytest.approx(0.4)
    assert r["input_partial_month"] is True
    members = [r for r in rows if r["forecaster"] == "model-a"]
    assert members and members[0]["skill"] == pytest.approx(1 - 1.2 / 1.0)


# ---------------------------------------------------------------------------
# 7. tails and binary reliability
# ---------------------------------------------------------------------------


def test_a_binary_event_at_three_percent_is_a_tail_and_lands_in_the_lowest_bin(tmp_path):
    path = tmp_path / "b.duckdb"
    con = duckdb.connect(str(path))
    _schema(con)
    _question(con, "B1", "PHL", "TC", "EVENT_OCCURRENCE")
    _resolve(con, "B1", 1, 1.0)
    _spd(con, "forecasts_ensemble", "r1", "B1", "ensemble_mean_v2", 1, [0.03, 0.97])
    _spd(con, "forecasts_raw", "r1", "B1", "model-a", 1, [0.05, 0.95])
    _score(con, "B1", "ensemble_mean_v2", 1, "brier", 0.94)
    ctx = ea.build_context(con, ["B1"])
    tails, rel = ea.build_tails(ctx)
    mean = [t for t in tails if t["forecaster"] == "ensemble_mean_v2"][0]
    assert mean["p_realized"] == pytest.approx(0.03)
    assert mean["forecaster_kind"] == "aggregate"
    cell = [r for r in rel if r["forecaster"] == "ensemble_mean_v2"][0]
    assert cell["bin"] == "0-5%" and cell["observed_rate"] == 1.0 and cell["n"] == 1
    # 0.05 belongs to the 5-20% bin, not the lowest.
    assert [r for r in rel if r["forecaster"] == "model-a"][0]["bin"] == "5-20%"


# ---------------------------------------------------------------------------
# 8. base rate shown and inject health
# ---------------------------------------------------------------------------


def test_base_rate_shown_reads_the_months_before_the_forecast(db):
    con = duckdb.connect(db)
    con.execute(
        "INSERT INTO acled_monthly_fatalities VALUES ('ETH', DATE '2026-06-01', 40, TIMESTAMP '2026-07-15')"
    )
    con.execute(  # the month of the forecast is never shown
        "INSERT INTO acled_monthly_fatalities VALUES ('ETH', DATE '2026-08-01', 999, TIMESTAMP '2026-09-20')"
    )
    ctx = ea.build_context(con, ["Q1"])
    shown = ea.base_rate_shown(con, ctx.qmeta["Q1"])
    months = [m["month"] for m in shown["acled"]["months"]]
    assert months == ["2026-06", "2026-07"]
    july = shown["acled"]["months"][-1]
    assert july["updated_after_forecast"] is True     # rewritten 20 Aug, forecast 1 Aug
    assert shown["input_partial_month"] is True
    assert shown["level_volatility"] == {"shown": False}


def test_ace_inject_health_always_carries_crisiswatch(db):
    meta = {"iso3": "ETH", "hazard_code": "ACE", "metric": "FATALITIES", "track": 1,
            "input_partial_month": True, "input_partial_month_basis": "x"}
    inject = {
        "enso": {"available": False, "reason": "absent"},
        "crisiswatch": {"applicable": True, "available": True, "edition": "2026-05",
                        "edition_age_months": 3, "stale": True, "arrow": "deteriorated", "alert": None},
        "acled_cast": {"applicable": True, "available": True, "vintage": "2025-12-01",
                       "age_days": 243, "stale": True},
        "views": {"applicable": True, "available": False, "reason": "pruned"},
        "base_rate": {"available": True, "source": "acled"},
    }
    rows = {r["inject"]: r for r in ea.inject_health_rows("Q1", meta, inject)}
    assert rows["crisiswatch"]["stale"] is True and "deteriorated" in rows["crisiswatch"]["reason"]
    assert rows["acled_cast"]["stale"] is True and rows["views"]["present"] is False
    assert rows["acled_trajectory"]["stale"] is True
    text = "\n".join(ea.inject_digest_lines(list(rows.values())))
    assert "CrisisWatch for ACE questions" in text


def test_the_crisiswatch_status_carries_arrow_alert_and_age(db):
    from scripts.ai_bundle import provenance as prov

    con = duckdb.connect(db)
    con.execute(
        "CREATE TABLE crisiswatch_entries (iso3 TEXT, month INTEGER, year INTEGER, arrow TEXT, "
        "alert_type TEXT, fetched_at TIMESTAMP)"
    )
    con.execute("INSERT INTO crisiswatch_entries VALUES ('ETH', 6, 2026, 'deteriorated', 'conflict_risk', '2026-07-10')")
    got = prov._crisiswatch_status(con, "ETH", "ACE", date(2026, 9, 1))
    assert got["edition"] == "2026-06" and got["edition_age_months"] == 3 and got["stale"] is True
    assert got["arrow"] == "deteriorated" and got["alert"] == "conflict_risk"


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------


def test_every_section_stubs_without_a_context(tmp_path):
    result, digest = ea.emit_all(None, tmp_path, ctx_error="questions table absent")
    for name, info in result.files.items():
        assert info["status"] == "stub"
        assert (tmp_path / name).exists()
    with (tmp_path / "trace_stages.csv").open() as fh:
        assert list(csv.DictReader(fh)) == [{"stub_reason": "questions table absent"}]
    assert json.loads((tmp_path / "headline.json").read_text())["stub"] is True


def test_missing_tables_stub_their_sections_and_the_builder_exits_zero(tmp_path):
    path = tmp_path / "thin.duckdb"
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE questions (question_id TEXT, hs_run_id TEXT, iso3 TEXT, hazard_code TEXT, "
        "metric TEXT, target_month TEXT, window_start_date DATE, window_end_date DATE, wording TEXT, "
        "status TEXT, track INTEGER, pythia_metadata_json TEXT)"
    )
    con.execute("CREATE TABLE scores (question_id TEXT, horizon_m INTEGER, metric TEXT, "
                "score_type TEXT, model_name TEXT, value DOUBLE, run_id TEXT)")
    _question(con, "T1", "ETH", "ACE", "FATALITIES")
    con.execute("INSERT INTO scores VALUES ('T1', 1, 'x', 'brier', 'ensemble_mean_v2', 0.3, 'r1')")
    con.close()
    out = tmp_path / "out"
    assert main(["--db", str(path), "--out-dir", str(out), "--months-back", "0"]) == 0
    (zp,) = out.glob("*.zip")
    with zipfile.ZipFile(zp) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        files = manifest["error_attribution"]["files"]
        assert files["trace_stages.csv"]["status"] == "stub"
        assert "forecasts_raw" in files["trace_stages.csv"]["reason"]
        assert "stub_reason" in zf.read("trace_stages.csv").decode()
        for name in ("headline.json", "experiments.csv", "skill_history.csv", "inject_health.csv",
                     "unasked_outcomes.csv", "tail_outcomes.csv", "binary_reliability.csv"):
            assert name in zf.namelist()
            assert name in files


def test_the_digest_opens_with_the_headline_table(db, tmp_path):
    zp = build_bundle(db, tmp_path / "out", months_back=0)
    with zipfile.ZipFile(zp) as zf:
        digest = zf.read("digest.md").decode()
        head = json.loads(zf.read("headline.json"))
        index = list(csv.DictReader(zf.read("questions_index.csv").decode().splitlines()))
        record = json.loads(zf.read("questions/Q1.json"))
    first_section = next(line for line in digest.splitlines() if line.startswith("## "))
    assert first_section.startswith("## Headline")
    clim = head["groups"][0]["scores"]["brier"]["references"]["__ext_climatology"]
    assert f"{clim['skill']:+.3f}" in digest.split("## ", 2)[1]
    assert {r["input_partial_month"] for r in index} == {"True"}
    assert record["input_partial_month"] is True
    assert "acled" in record["base_rate_shown"]
    assert record["inject_status"]["crisiswatch"]["applicable"] is True


# ---------------------------------------------------------------------------
# prior_anchor_v1 / _v2: one recalibration group, reported apart and pooled
# ---------------------------------------------------------------------------

GROUP = "prior_anchor_v1|prior_anchor_v2"


def _versions_db(tmp_path: Path) -> str:
    path = tmp_path / "versions.duckdb"
    con = duckdb.connect(str(path))
    _schema(con)
    plan = [("prior_anchor_v1", 3, 0.30), ("prior_anchor_v2", 2, 0.20), (None, 2, 0.60)]
    i = 0
    for brv, n, brier in plan:
        for _ in range(n):
            qid = f"V{i}"
            i += 1
            _question(con, qid, "ETH", "ACE", "FATALITIES")
            _resolve(con, qid, 1, 300.0)
            _spd(con, "forecasts_raw", "r1", qid, "model-a", 1, POST, brv=brv)
            _spd(con, "forecasts_ensemble", "r1", qid, "ensemble_mean_v2", 1, POST)
            _score(con, qid, "ensemble_mean_v2", 1, "brier", brier)
            _score(con, qid, "__ext_climatology", 1, "brier", 0.5)
    con.close()
    return str(path)


def test_pooled_copies_only_for_the_equivalent_versions():
    rows = [{"base_rate_block_version": v} for v in ("prior_anchor_v1", "prior_anchor_v2", "none")]
    out = ea.with_pooled_block_versions(rows)
    assert [r["base_rate_block_version"] for r in out] == [
        "prior_anchor_v1", "prior_anchor_v2", "none", GROUP, GROUP,
    ]
    assert [r["block_version_pooled"] for r in out] == [False, False, False, True, True]


def test_rollups_report_v1_v2_and_the_pooled_group(tmp_path):
    zp = build_bundle(_versions_db(tmp_path), tmp_path / "out", months_back=0)
    with zipfile.ZipFile(zp) as zf:
        rows = list(csv.DictReader(zf.read("rollups.csv").decode().splitlines()))
    mean = {r["base_rate_block_version"]: r for r in rows
            if r["model_name"] == "ensemble_mean_v2" and r["score_type"] == "brier"}
    assert set(mean) == {"prior_anchor_v1", "prior_anchor_v2", GROUP, "none"}
    assert mean["prior_anchor_v1"]["n_samples"] == "3"
    assert mean["prior_anchor_v2"]["n_samples"] == "2"
    assert mean[GROUP]["n_samples"] == "5"
    assert float(mean[GROUP]["mean_value"]) == pytest.approx((3 * 0.3 + 2 * 0.2) / 5)
    assert mean[GROUP]["block_version_pooled"] == "True"
    assert mean["none"]["n_samples"] == "2" and mean["none"]["block_version_pooled"] == "False"
    # Paired against climatology inside the pooled group too.
    assert mean[GROUP]["n_paired"] == "5"


def test_experiments_compare_versions_apart_and_the_pooled_group(tmp_path):
    ctx = _ctx(_versions_db(tmp_path), [f"V{i}" for i in range(7)])
    rows = ea.build_experiments(ctx)
    per_version = {r["arm"] for r in rows if r["flag"] == "base_rate_block_version"} | {
        r["reference_arm"] for r in rows if r["flag"] == "base_rate_block_version"}
    assert per_version == {"prior_anchor_v1", "prior_anchor_v2", "none"}
    pooled = [r for r in rows if r["flag"] == "base_rate_block_group"]
    assert pooled and {pooled[0]["arm"], pooled[0]["reference_arm"]} == {GROUP, "none"}
    (p,) = [r for r in pooled if r["score_type"] == "brier"]
    assert {p["n_arm"], p["n_reference"]} == {5, 2}


def test_a_binary_question_stamped_with_an_rc_arm_is_in_neither_arm(tmp_path):
    """Until 2026-10-08 binary Track 1 questions were stamped with an RC arm
    although the shift guidance is SPD-only; the split must not read them."""
    path = tmp_path / "arm.duckdb"
    con = duckdb.connect(str(path))
    _schema(con)
    con.execute("ALTER TABLE forecasts_raw ADD COLUMN rc_shift_arm TEXT")
    _question(con, "S1", "ETH", "ACE", "FATALITIES")
    _question(con, "B1", "ETH", "FL", "EVENT_OCCURRENCE")
    _spd(con, "forecasts_raw", "r1", "S1", "model-a", 1, POST)
    _spd(con, "forecasts_raw", "r1", "B1", "model-a", 1, [0.2, 0.8])
    for qid, probs in (("S1", POST), ("B1", [0.2, 0.8])):
        _spd(con, "forecasts_ensemble", "r1", qid, "ensemble_mean_v2", 1, probs)
        _resolve(con, qid, 1, 1.0)
        _score(con, qid, "ensemble_mean_v2", 1, "brier", 0.3)
    con.execute("UPDATE forecasts_raw SET rc_shift_arm = 'shift'")
    ctx = ea.build_context(con, ["S1", "B1"])
    assert ctx.qmeta["S1"]["rc_shift_arm"] == "shift"
    assert ctx.qmeta["B1"]["rc_shift_arm"] == "none"
