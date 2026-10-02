# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""A score is test data when the FORECAST it scores is test data.

Question ids are epoch-keyed, so a production question can also carry a
same-epoch test run's forecasts. compute_scores stamped is_test from the
question alone, so those scores read as production: on the 2026-10-02
release, 473 score rows on 6 questions across 12 models (Sibyl among them)
came from test-only forecasts and carried is_test = FALSE. The latest-run
rule in the advice generator then took the test run as "the forecast that
stands", because it was newer.
"""

from __future__ import annotations

import pytest

pytest.importorskip("duckdb")

QID = "SOM_ACE_FATALITIES_2026-08"
PROD_RUN = "fc_1784112479"
TEST_RUN = "fc_1785397217"  # later than the production run, as on the release


def _seed(tmp_path, monkeypatch) -> str:
    db_path = tmp_path / "scores.duckdb"
    url = f"duckdb:///{db_path}"
    monkeypatch.setenv("PYTHIA_DB_URL", url)
    monkeypatch.delenv("PYTHIA_TEST_MODE", raising=False)
    from pythia.db.schema import connect, ensure_schema

    ensure_schema()
    con = connect(read_only=False)
    try:
        con.execute("INSERT INTO hs_runs (hs_run_id, generated_at) VALUES ('hs_p', CURRENT_TIMESTAMP)")
        con.execute(
            "INSERT INTO questions (question_id, hs_run_id, iso3, hazard_code, metric, "
            "target_month, window_start_date, status, is_test) VALUES "
            "(?, 'hs_p', 'SOM', 'ACE', 'FATALITIES', '2027-01', DATE '2026-08-01', 'active', FALSE)",
            [QID],
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS resolutions (question_id TEXT, horizon_m INTEGER, "
            "observed_month TEXT, value DOUBLE, source_snapshot_ym TEXT, source_desc TEXT, "
            "created_at TIMESTAMP DEFAULT now(), is_test BOOLEAN DEFAULT FALSE)"
        )
        con.execute(
            "INSERT INTO resolutions (question_id, horizon_m, observed_month, value) "
            "VALUES (?, 1, '2026-08', 120)",
            [QID],
        )
        probs = [0.1, 0.1, 0.2, 0.3, 0.2, 0.05, 0.05]
        for run_id, model, is_test in (
            (PROD_RUN, "sibyl", False),
            (PROD_RUN, "ensemble_mean_v2", False),
            (TEST_RUN, "sibyl", True),
            (TEST_RUN, "ensemble_mean_v2", True),
        ):
            for b, p in enumerate(probs, start=1):
                con.execute(
                    "INSERT INTO forecasts_raw (run_id, question_id, model_name, month_index, "
                    "bucket_index, probability, ok, status, is_test) "
                    "VALUES (?, ?, ?, 1, ?, ?, TRUE, 'ok', ?)",
                    [run_id, QID, model, b, p, is_test],
                )
                con.execute(
                    "INSERT INTO forecasts_ensemble (run_id, question_id, iso3, hazard_code, "
                    "metric, model_name, month_index, bucket_index, probability, status, "
                    "created_at, is_test) VALUES (?, ?, 'SOM', 'ACE', 'FATALITIES', ?, 1, ?, ?, "
                    "'ok', CURRENT_TIMESTAMP, ?)",
                    [run_id, QID, model, b, p, is_test],
                )
    finally:
        con.close()
    return url


def test_scores_of_a_test_run_on_a_production_question_are_test(tmp_path, monkeypatch):
    url = _seed(tmp_path, monkeypatch)
    from pythia.db.schema import connect
    from pythia.tools.compute_scores import compute_scores

    compute_scores(url)
    con = connect(read_only=False)
    try:
        rows = con.execute(
            "SELECT run_id, model_name, bool_and(is_test), bool_or(is_test) "
            "FROM scores GROUP BY 1, 2 ORDER BY 1, 2"
        ).fetchall()
        eiv = dict(
            con.execute("SELECT run_id, bool_and(is_test) FROM eiv_scores GROUP BY 1").fetchall()
        )
    finally:
        con.close()
    by = {(r[0], r[1]): (r[2], r[3]) for r in rows}
    assert by[(PROD_RUN, "sibyl")] == (False, False)
    assert by[(PROD_RUN, "ensemble_mean_v2")] == (False, False)
    assert by[(TEST_RUN, "sibyl")] == (True, True)
    assert by[(TEST_RUN, "ensemble_mean_v2")] == (True, True)
    if eiv:
        assert eiv.get(TEST_RUN, True) is True
        assert eiv.get(PROD_RUN, False) is False


def test_latest_run_rule_skips_a_later_test_run(tmp_path, monkeypatch):
    url = _seed(tmp_path, monkeypatch)
    from pythia.db.schema import connect
    from pythia.tools.compute_scores import compute_scores
    from pythia.tools.generate_calibration_advice import _latest_run_clause

    compute_scores(url)
    con = connect(read_only=False)
    try:
        kept = con.execute(
            "SELECT DISTINCT s.run_id FROM scores s WHERE s.score_type = 'brier'"
            + _latest_run_clause(con, "scores", "s")
        ).fetchall()
    finally:
        con.close()
    assert kept == [(PROD_RUN,)]
