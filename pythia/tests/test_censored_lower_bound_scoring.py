# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""A lower-bound outcome is scored as "at least this bucket".

The IOM DTM rung of the ACE/PA ladder resolves a month to the rise in an IDP
stock between two rounds, a floor on new displacement. compute_scores
scored it as the exact answer, which rewards mass on the floor's bucket and
penalises mass above it, where the truth is. It is now scored under
brier_censored / log_censored / crps_censored, never pooled with the exact
types, and kept out of every reader that learns from outcomes.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("duckdb")

from pythia.tools import compute_scores as cs  # noqa: E402
from pythia.tools import scoring_class as sc  # noqa: E402

DTM = "dtm_conflict:idp_stock_increase:displaced_lower_bound"
P = [0.1, 0.2, 0.3, 0.2, 0.1, 0.1]  # PA: six buckets


# --- the formulas -------------------------------------------------------------


def test_censored_log_is_minus_log_of_the_tail():
    assert cs._log_score_censored(P, 2) == pytest.approx(-math.log(0.7))
    assert cs._log_score_censored(P, 2) <= cs._log_score(P, 2)


def test_censored_brier_scores_the_binary_event_at_or_above_k():
    assert cs._brier_censored(P, 2) == pytest.approx((0.7 - 1.0) ** 2)
    assert cs._brier_censored([0, 0, 1, 0, 0, 0], 2) == 0.0


def test_censored_rps_is_the_docstring_worked_example():
    # F_0 = 0.1, F_1 = 0.3: (0.01 + 0.09) / 5.
    assert cs._rps_censored(P, 2) == pytest.approx(0.02)
    # Never more than the exact RPS of any outcome at or above the bound.
    for j in range(2, 6):
        assert cs._rps_censored(P, 2) <= cs._rps(P, j) + 1e-12


def test_mass_above_the_bound_is_not_penalised():
    on_bound = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    above = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    for f in (cs._brier_censored, cs._log_score_censored, cs._rps_censored):
        assert f(above, 2) == pytest.approx(f(on_bound, 2))
    # The exact scores punished the forecast that put its mass above.
    assert cs._rps(above, 2) > cs._rps(on_bound, 2)


def test_lower_bound_markers():
    assert sc.is_lower_bound_source(DTM)
    assert sc.is_lower_bound_source("haz_resolutions:idmc_idu:displaced_lower_bound")
    assert not sc.is_lower_bound_source("idmc_conflict:new_displacements")
    assert not sc.is_lower_bound_source(None)
    assert sc.censored_applies("ACE", "PA", True)
    # FL/TC IDMC wins are scored as the answer (owner decision 2026-10-12).
    assert not sc.censored_applies("FL", "PA", True)
    assert not sc.censored_applies("ACE", "PA", False)


# --- end to end ------------------------------------------------------------------


def _seed(tmp_path, monkeypatch, resolutions):
    db_path = tmp_path / "scores.duckdb"
    url = f"duckdb:///{db_path}"
    monkeypatch.setenv("PYTHIA_DB_URL", url)
    monkeypatch.delenv("PYTHIA_TEST_MODE", raising=False)
    from pythia.db.schema import connect, ensure_schema

    ensure_schema()
    con = connect(read_only=False)
    try:
        con.execute("INSERT INTO hs_runs (hs_run_id, generated_at) VALUES ('hs_p', CURRENT_TIMESTAMP)")
        # A resolutions table from before is_lower_bound: the migration adds
        # and backfills it.
        con.execute(
            "CREATE TABLE IF NOT EXISTS scores (question_id TEXT, horizon_m INTEGER, metric TEXT, "
            "score_type TEXT, model_name TEXT, value DOUBLE, run_id TEXT, "
            "created_at TIMESTAMP DEFAULT now(), is_test BOOLEAN DEFAULT FALSE)"
        )
        con.execute("DROP TABLE IF EXISTS resolutions")
        con.execute(
            "CREATE TABLE resolutions (question_id TEXT, horizon_m INTEGER, "
            "observed_month TEXT, value DOUBLE, source_snapshot_ym TEXT, source_desc TEXT, "
            "created_at TIMESTAMP DEFAULT now(), is_test BOOLEAN DEFAULT FALSE)"
        )
        seen = set()
        for qid, hazard, metric, h, value, desc in resolutions:
            if qid not in seen:
                seen.add(qid)
                con.execute(
                "INSERT INTO questions (question_id, hs_run_id, iso3, hazard_code, metric, "
                "target_month, window_start_date, status, is_test) VALUES "
                "(?, 'hs_p', 'SDN', ?, ?, '2027-01', DATE '2026-08-01', 'active', FALSE)",
                [qid, hazard, metric],
                )
            con.execute(
                "INSERT INTO resolutions (question_id, horizon_m, observed_month, value, source_desc) "
                "VALUES (?, ?, '2026-08', ?, ?)", [qid, h, value, desc],
            )
            for b, p in enumerate(P, start=1):
                con.execute(
                    "INSERT INTO forecasts_raw (run_id, question_id, model_name, month_index, "
                    "bucket_index, probability, ok, status, is_test) "
                    "VALUES ('fc_1', ?, 'ensemble_mean_v2', ?, ?, ?, TRUE, 'ok', FALSE)",
                    [qid, h, b, p],
                )
            # An exact score an earlier run wrote against the bound.
            con.execute(
                "INSERT INTO scores (question_id, horizon_m, metric, score_type, model_name, "
                "value, run_id) VALUES (?, ?, ?, 'brier', 'ensemble_mean_v2', 9.9, 'fc_1')",
                [qid, h, metric],
            )
    finally:
        con.close()
    return url


def _scores(url):
    from pythia.db.schema import connect

    con = connect(read_only=False)
    try:
        return {
            (q, h, st): v for q, h, st, v in con.execute(
                "SELECT question_id, horizon_m, score_type, value FROM scores "
                "WHERE model_name = 'ensemble_mean_v2'"
            ).fetchall()
        }
    finally:
        con.close()


def test_a_dtm_lower_bound_is_scored_censored_and_never_exact(tmp_path, monkeypatch):
    url = _seed(tmp_path, monkeypatch, [
        ("SDN_ACE_PA_2026-08", "ACE", "PA", 1, 20000, DTM),          # bucket 2
        ("SDN_ACE_PA_2026-08", "ACE", "PA", 2, 20000, "idmc_conflict"),  # exact
        ("SDN_FL_PA_2026-08", "FL", "PA", 1, 20000,
         "haz_resolutions:idmc_idu:displaced_lower_bound"),           # FL: exact by decision
    ])
    report = cs.compute_scores(url)
    s = _scores(url)
    q = "SDN_ACE_PA_2026-08"
    assert {st for (qq, h, st) in s if qq == q and h == 1} == set(sc.CENSORED_SCORE_TYPES)
    assert s[(q, 1, "crps_censored")] == pytest.approx(0.02)
    assert s[(q, 1, "brier_censored")] == pytest.approx(0.09)
    assert {st for (qq, h, st) in s if qq == q and h == 2} == {"brier", "log", "crps"}
    assert {st for (qq, h, st) in s if qq == "SDN_FL_PA_2026-08"} == {"brier", "log", "crps"}
    assert report["lower_bound_resolutions"] == 2  # ACE DTM and FL IDMC both marked
    assert report["censored"]["exact_rows_replaced"] == 1
    assert report["censored"]["scored_rows"] == 3


def test_a_bound_in_bucket_0_scores_nothing_and_says_why(tmp_path, monkeypatch):
    url = _seed(tmp_path, monkeypatch, [("SDN_ACE_PA_2026-08", "ACE", "PA", 1, 0, DTM)])
    report = cs.compute_scores(url)
    assert _scores(url) == {}
    assert report["censored"]["skipped"] == {cs.CENSORED_SKIP_BUCKET_ZERO: 1}


def test_learning_readers_leave_lower_bounds_out(tmp_path, monkeypatch):
    url = _seed(tmp_path, monkeypatch, [
        ("SDN_ACE_PA_2026-08", "ACE", "PA", 1, 20000, DTM),
        ("SDN_ACE_PA_2026-08", "ACE", "PA", 2, 20000, "idmc_conflict"),
    ])
    cs.compute_scores(url)
    from pythia.db.schema import connect

    con = connect(read_only=False)
    try:
        kept = con.execute(
            f"SELECT horizon_m FROM resolutions r WHERE {sc.exact_outcome_sql('r')}"
        ).fetchall()
        assert kept == [(2,)]
        types = {t for (t,) in con.execute(
            f"SELECT DISTINCT score_type FROM scores s WHERE {sc.exact_score_types_sql('s')}"
        ).fetchall()}
        assert types == {"brier", "log", "crps"}
    finally:
        con.close()


def test_calibration_samples_exclude_censored_scores(tmp_path, monkeypatch):
    url = _seed(tmp_path, monkeypatch, [("SDN_ACE_PA_2026-08", "ACE", "PA", 1, 20000, DTM)])
    cs.compute_scores(url)
    from pythia.db.schema import connect
    from pythia.tools import compute_calibration_pythia as ccp

    con = connect(read_only=False)
    try:
        samples = ccp._load_samples(con, "2027-12")
    finally:
        con.close()
    assert not any(str(x.score_type).endswith("_censored") for x in samples)


def test_the_readers_that_learn_name_the_exclusion():
    """Each learning reader states which way it handles lower bounds."""
    import inspect

    from pythia.tools import compute_bucket_centroids, family_recalibration
    from pythia.tools import compute_calibration_pythia, generate_calibration_advice

    assert "exact_outcome_clause" in inspect.getsource(family_recalibration._member_rows)
    assert "exact_outcome_clause" in inspect.getsource(compute_bucket_centroids)
    assert "exact_outcome_clause" in inspect.getsource(generate_calibration_advice._scored)
    assert "exact_score_types_sql" in inspect.getsource(compute_calibration_pythia._load_samples)


def test_the_rescore_report_reads_before_and_after_read_only(tmp_path, monkeypatch):
    url = _seed(tmp_path, monkeypatch, [("SDN_ACE_PA_2026-08", "ACE", "PA", 1, 20000, DTM)])
    from pythia.db.schema import close_pooled_connections
    from scripts.ci import censored_rescore_report as rr

    close_pooled_connections()  # the report opens the file read-only
    db = url[len("duckdb:///"):]
    out = tmp_path / "out"
    assert rr.main(["--db", db, "--out-dir", str(out)]) == 0
    import json

    rep = json.loads((out / "censored_rescore.json").read_text())
    row = rep["rows"][0]
    assert row["before"]["brier"] == 9.9  # the exact score stored before
    assert row["after"]["crps_censored"] == pytest.approx(0.02)
    assert "brier_c" in (out / "censored_rescore.md").read_text()
