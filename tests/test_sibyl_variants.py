# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Rearranged variants of trials already run (Oct 2026, review Part 3).

Each variant rebuilt from a stored forecast with production's own pooling,
w50 equal to what was published at a run weight of 0.5, the
disagreement-weighted rule at each threshold, single-trial lane series,
"not yet" below twenty questions, and stale rows removed.
"""

from __future__ import annotations

import json

import pytest

import sibyl.config as sibyl_config
from sibyl import score_variants as sv
from sibyl.aggregate import pool_months, publish_vectors
from sibyl.spd import apply_bucket_floor

pytestmark = pytest.mark.db

REF = {m: [0.2, 0.3, 0.2, 0.1, 0.1, 0.05, 0.05] for m in range(1, 7)}


def _trial(lane, role, m1_median, *, p_zero=0.1, outlier=False, evidence=True, error=None):
    q = {"0.05": m1_median / 5, "0.25": m1_median / 2, "0.5": m1_median,
         "0.75": m1_median * 2, "0.95": m1_median * 5}
    return {"lane": lane, "role": role, "evidence_ok": evidence, "error": error,
            "outlier_dropped": outlier,
            "month_1": {"p_zero": p_zero, "quantiles_positive": q},
            "month_6": {"p_zero": p_zero, "quantiles_positive": q}}


def _published(trials, ref, w, metric="FATALITIES"):
    pool = pool_months([sv.trial_months(t) for t in trials], metric)
    return {m: apply_bucket_floor(v) for m, v in publish_vectors(pool.vectors, ref, w).items()}


def _forecast(trials, weight=0.5, final=None):
    pooled = [t for t in sv.valid_trials(trials) if not t.get("outlier_dropped")]
    return {"question_id": "Q", "metric": "FATALITIES", "ref": REF, "ref_weight": weight,
            "trials": trials, "final": final or _published(pooled, REF, weight)}


# --- the variants ------------------------------------------------------------------

def test_w50_equals_what_was_published_at_a_run_weight_of_half():
    trials = [_trial("A", "production", 10), _trial("B", "production", 12),
              _trial("C", "production", 15)]
    f = _forecast(trials, weight=0.5)
    built = sv.build_variants(f)
    for m in range(1, 7):
        assert built[sv.W50_MODEL_NAME][m] == pytest.approx(f["final"][m], abs=1e-12)
    assert built[sv.W25_MODEL_NAME][1] != pytest.approx(f["final"][1])
    assert set(built) == set(sv.VARIANT_MODEL_NAMES)


def test_the_fixed_weights_move_toward_the_reference():
    trials = [_trial("A", "production", 400), _trial("B", "production", 500)]
    built = sv.build_variants(_forecast(trials))
    gap = [sum(abs(x - r) for x, r in zip(built[n][1], REF[1]))
           for n in (sv.W25_MODEL_NAME, sv.W50_MODEL_NAME, sv.W75_MODEL_NAME)]
    assert gap[0] > gap[1] > gap[2]


def test_abc_uses_the_production_trials_only():
    prod = [_trial("A", "production", 10), _trial("B", "production", 12),
            _trial("C", "production", 15)]
    extra = [_trial("D", "disagreement", 900), _trial("E", "disagreement", 1000)]
    f = _forecast(prod + extra)
    built = sv.build_variants(f)
    assert built[sv.ABC_MODEL_NAME][1] == pytest.approx(_published(prod, REF, 0.5)[1])
    assert built[sv.ABC_MODEL_NAME][1] != pytest.approx(f["final"][1])


def test_noguard_puts_back_the_trial_the_guard_left_out():
    trials = [_trial("A", "production", 10), _trial("B", "production", 12),
              _trial("C", "production", 20000, outlier=True)]
    f = _forecast(trials)
    built = sv.build_variants(f)
    assert built[sv.NOGUARD_MODEL_NAME][1] == pytest.approx(_published(trials, REF, 0.5)[1])
    assert built[sv.W50_MODEL_NAME][1] == pytest.approx(f["final"][1])


@pytest.mark.parametrize("jsd, w", [(None, 0.5), (0.11, 0.75), (0.10, 0.5), (0.05, 0.5),
                                    (0.03, 0.5), (0.0299, 0.25), (0.0, 0.25)])
def test_the_disagreement_weight_at_each_threshold(jsd, w):
    assert sv.dw_weight(jsd) == w


def test_the_disagreement_thresholds_are_fixed_config():
    assert (sibyl_config.DW_HIGH_JSD, sibyl_config.DW_LOW_JSD) == (0.10, 0.03)


def test_dw_uses_the_disagreement_of_the_pooled_trials():
    agree = [_trial("A", "production", 10), _trial("B", "production", 10)]
    built = sv.build_variants(_forecast(agree))
    assert built[sv.DW_MODEL_NAME][1] == pytest.approx(built[sv.W25_MODEL_NAME][1])
    differ = [_trial("A", "production", 2, p_zero=0.6), _trial("B", "production", 800)]
    built = sv.build_variants(_forecast(differ))
    assert built[sv.DW_MODEL_NAME][1] == pytest.approx(built[sv.W75_MODEL_NAME][1])


def test_invalid_trials_are_left_out_and_lanes_are_scored_one_by_one():
    trials = [_trial("A", "production", 10), _trial("B", "production", 12, evidence=False),
              _trial("C", "production", 15, error="model_step_failed"),
              _trial("R", "disagreement", 30)]
    lanes = sv.lane_vectors(_forecast(trials))
    assert set(lanes) == {"lane_A", "lane_R"}
    assert lanes["lane_A"][1] == pytest.approx(_published([trials[0]], None, 0.0)[1])
    # Months 2 to 5 are mixed as the pool mixes them.
    assert set(lanes["lane_A"]) == {1, 2, 3, 4, 5, 6}
    blind = [_trial("A", "production", 5, evidence=False)]
    assert sv.build_variants(_forecast(blind, final={1: [1.0]})) == {}


# --- in a database --------------------------------------------------------------

def _seed(tmp_path, monkeypatch, n):
    from tests.sibyl_test_utils import HS_RUN_ID, seed_db

    seed_db(tmp_path, monkeypatch)
    from pythia.db.schema import connect

    con = connect(read_only=False)
    con.execute("CREATE TABLE IF NOT EXISTS resolutions (question_id TEXT, horizon_m INTEGER, "
                "value DOUBLE, is_test BOOLEAN DEFAULT FALSE)")
    from pythia.tools.score_baselines import ensure_baseline_tables

    ensure_baseline_tables(con)
    trials = [_trial("A", "production", 10), _trial("B", "production", 12),
              _trial("C", "production", 15), _trial("R", "disagreement", 60)]
    pooled = sv.valid_trials(trials)
    final = _published(pooled, REF, 0.5)
    for i in range(n):
        qid = f"V{i:02d}_ACE_FATALITIES_2026-08"
        con.execute(
            "INSERT INTO questions (question_id, hs_run_id, iso3, hazard_code, metric, "
            "target_month, window_start_date, wording, status, track) VALUES "
            "(?, ?, 'ETH', 'ACE', 'FATALITIES', '2027-01', DATE '2026-08-01', 'w', 'active', 1)",
            [qid, HS_RUN_ID])
        con.execute(
            "INSERT INTO sibyl_forecasts (sibyl_run_id, run_id, question_id, hazard_code, "
            "metric, status, raw_by_month_json, reference_json, final_by_month_json, "
            "trials_json, evidence_ok, created_at) VALUES ('sr1', 'fc', ?, 'ACE', 'FATALITIES', "
            "'ok', ?, ?, ?, ?, TRUE, CURRENT_TIMESTAMP)",
            [qid, json.dumps({"vectors": {str(m): v for m, v in final.items()}}),
             json.dumps({"by_month": {str(m): v for m, v in REF.items()}, "weight": 0.5}),
             json.dumps({str(m): v for m, v in final.items()}), json.dumps(trials)])
        con.execute("INSERT INTO resolutions VALUES (?, 1, 12, FALSE)", [qid])
        # The published forecast's own score, as compute_scores writes it.
        j = sv._bucket(12, "FATALITIES")
        for st, v in sv.score_vector(final[1], j).items():
            con.execute("INSERT INTO scores (question_id, horizon_m, metric, score_type, "
                        "model_name, value, run_id) VALUES (?, 1, 'FATALITIES', ?, 'sibyl', ?, "
                        "'fc')", [qid, st, v])
    return con


def test_score_variants_writes_variants_and_lanes(tmp_path, monkeypatch):
    con = _seed(tmp_path, monkeypatch, 2)
    try:
        counters = sv.score_variants(con, as_of_month="2026-10")
        assert counters["scored_variants"] == 2 * 6 and counters["lane_rows"] == 2 * 4
        names = {r[0] for r in con.execute(
            "SELECT DISTINCT model_name FROM scores WHERE run_id IS NULL").fetchall()}
        assert set(sv.VARIANT_MODEL_NAMES) <= names
        series = {r[0] for r in con.execute(
            "SELECT DISTINCT series FROM sibyl_variant_scores").fetchall()}
        assert {"lane_A", "lane_B", "lane_C", "lane_R"} <= series
        # w50 scores exactly as the published sibyl row.
        a, b = con.execute(
            "SELECT a.value, b.value FROM scores a JOIN scores b ON a.question_id = b.question_id "
            "AND a.score_type = b.score_type WHERE a.model_name = '__ext_sibyl_w50' "
            "AND b.model_name = 'sibyl' AND a.score_type = 'brier' LIMIT 1").fetchone()
        assert a == pytest.approx(b, abs=1e-9)
    finally:
        con.close()


def test_stale_variant_rows_are_removed(tmp_path, monkeypatch):
    con = _seed(tmp_path, monkeypatch, 1)
    try:
        sv.score_variants(con, as_of_month="2026-10")
        con.execute("UPDATE sibyl_forecasts SET trials_json = '[]'")
        sv.score_variants(con, as_of_month="2026-10")
        n = con.execute(
            f"SELECT COUNT(*) FROM scores WHERE model_name IN "
            f"({', '.join(repr(m) for m in sv.VARIANT_MODEL_NAMES)})").fetchone()[0]
        lanes = con.execute(
            "SELECT COUNT(*) FROM sibyl_variant_scores WHERE series LIKE 'lane%'").fetchone()[0]
        assert (n, lanes) == (0, 0)
    finally:
        con.close()


def test_the_comparison_says_not_yet_below_twenty(tmp_path, monkeypatch):
    con = _seed(tmp_path, monkeypatch, 3)
    try:
        sv.score_variants(con, as_of_month="2026-10")
        out = sv.variant_comparison(con)
        w50 = out["variants"][sv.W50_MODEL_NAME]["all"]["brier"]
        assert w50["status"] == "not_yet" and w50["n_questions"] == 3
        assert "mean_diff" not in w50
        abc = out["variants"][sv.ABC_MODEL_NAME]
        assert abc["n_questions_different"] == 3  # lane R left out
    finally:
        con.close()


def test_the_comparison_reports_a_paired_interval_from_twenty(tmp_path, monkeypatch):
    con = _seed(tmp_path, monkeypatch, 20)
    try:
        sv.score_variants(con, as_of_month="2026-10")
        out = sv.variant_comparison(con)
        w50 = out["variants"][sv.W50_MODEL_NAME]["all"]["brier"]
        assert w50["status"] == "ok" and w50["mean_diff"] == pytest.approx(0.0, abs=1e-9)
        abc = out["variants"][sv.ABC_MODEL_NAME]["where_different"]["brier"]
        assert abc["status"] == "ok" and abc["n_questions"] == 20
    finally:
        con.close()


def test_nothing_adopts_a_variant():
    import inspect

    import sibyl.measure as measure
    import sibyl.run as run

    # The published weight and pool never read a variant's score.
    for mod in (measure, run):
        src = inspect.getsource(mod)
        assert "__ext_sibyl_w" not in src and "__ext_sibyl_dw" not in src
        assert "variant_comparison" not in src
