# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""A test run must be able to forecast questions that already exist.

The 6 October 2026 rehearsal (hs_20261006T085029) forecast 2 of its 21
questions: the other 19 belonged to the 1 October production epoch, a test
run changes nothing on a production question, and the forecaster selected a
run's questions by ``questions.hs_run_id``. ``run_questions`` records a run's
question set apart from a question's origin.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

import scripts.create_questions_from_triage as cqt  # noqa: E402
from pythia.db.schema import ensure_schema  # noqa: E402
from pythia.run_questions import (  # noqa: E402
    backfill_run_questions,
    hs_run_for_forecaster_run,
    latest_hs_run_with_questions,
    run_question_ids,
    stamp_forecaster_run,
)

PROD = "hs_20261001T040000"
TEST = "hs_20261006T085029"
PROD2 = "hs_20261002T040000"


class _Oct(date):
    @classmethod
    def today(cls):  # type: ignore[override]
        return date(2026, 10, 6)


def _seed_run(con, run_id: str, *, is_test: bool, generated_at: str, tracks: dict) -> None:
    con.execute(
        "INSERT INTO hs_runs (hs_run_id, generated_at, git_sha, config_profile, countries_json, is_test) "
        "VALUES (?, CAST(? AS TIMESTAMP), 'sha', 'default', '[]', ?)",
        [run_id, generated_at, is_test],
    )
    for (iso3, hz), (track, rc) in tracks.items():
        con.execute(
            """
            INSERT INTO hs_triage (run_id, iso3, hazard_code, tier, triage_score, need_full_spd,
                                   drivers_json, regime_shifts_json, data_quality_json,
                                   scenario_stub, track, regime_change_score, regime_change_level,
                                   is_test)
            VALUES (?, ?, ?, 'priority', 0.8, TRUE, '[]', '[]', '{}', '', ?, ?, ?, ?)
            """,
            [run_id, iso3, hz, track, rc, 1 if rc >= 0.15 else 0, is_test],
        )


@pytest.fixture()
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "pythia.duckdb"
    monkeypatch.setattr(cqt, "date", _Oct)
    monkeypatch.setattr(cqt, "_is_food_security_country", lambda iso3, con=None: True)
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db_path}")
    con = duckdb.connect(str(db_path))
    ensure_schema(con)
    # Production triage: SOM ACE on Track 1, SOM FL on Track 2.
    _seed_run(con, PROD, is_test=False, generated_at="2026-10-01 04:00:00",
              tracks={("SOM", "ACE"): (1, 0.6), ("SOM", "FL"): (2, 0.0)})
    # Test triage a week later sees them the other way round, plus ETH DR.
    _seed_run(con, TEST, is_test=True, generated_at="2026-10-06 08:50:29",
              tracks={("SOM", "ACE"): (2, 0.0), ("SOM", "FL"): (1, 0.4), ("ETH", "DR"): (1, 0.5)})
    con.close()
    return db_path


def _create(db_path: Path, run_id: str, monkeypatch, *, test: bool) -> None:
    if test:
        monkeypatch.setenv("PYTHIA_TEST_MODE", "1")
    else:
        monkeypatch.delenv("PYTHIA_TEST_MODE", raising=False)
    cqt.create_questions_from_triage(f"duckdb:///{db_path}", hs_run_id=run_id)
    monkeypatch.delenv("PYTHIA_TEST_MODE", raising=False)


def _questions(db_path: Path):
    con = duckdb.connect(str(db_path))
    try:
        return con.execute(
            "SELECT question_id, hs_run_id, track, pythia_metadata_json, COALESCE(is_test, FALSE) "
            "FROM questions ORDER BY 1"
        ).fetchall()
    finally:
        con.close()


def test_a_test_run_finds_every_question_for_its_countries(db, monkeypatch):
    _create(db, PROD, monkeypatch, test=False)
    before = _questions(db)
    _create(db, TEST, monkeypatch, test=True)
    after = {r[0]: r for r in _questions(db)}

    # The production rows are untouched: origin, track, metadata, flag.
    for row in before:
        assert after[row[0]] == row

    from forecaster.cli import _select_run_question_rows

    con = duckdb.connect(str(db))
    try:
        rows = _select_run_question_rows(con, TEST, {"SOM", "ETH"})
        prod_rows = _select_run_question_rows(con, PROD, {"SOM", "ETH"})
    finally:
        con.close()
    selected = {r[0]: r for r in rows}
    # Every question the test run asked for: the six production SOM questions
    # (ACE FATALITIES/PA, FL PA/EVENT_OCCURRENCE) and its own two ETH DR ones.
    assert set(selected) == {
        "SOM_ACE_FATALITIES_2026-11", "SOM_ACE_PA_2026-11",
        "SOM_FL_PA_2026-11", "SOM_FL_EVENT_OCCURRENCE_2026-11",
        "ETH_DR_PHASE3PLUS_IN_NEED_2026-11", "ETH_DR_EVENT_OCCURRENCE_2026-11",
    }
    # hs_run_id and track as the TEST run saw them.
    assert {r[1] for r in rows} == {TEST}
    assert selected["SOM_ACE_PA_2026-11"][9] == 2
    assert selected["SOM_FL_PA_2026-11"][9] == 1
    # The production run's view is unchanged.
    prod = {r[0]: r for r in prod_rows}
    assert set(prod) == {
        "SOM_ACE_FATALITIES_2026-11", "SOM_ACE_PA_2026-11",
        "SOM_FL_PA_2026-11", "SOM_FL_EVENT_OCCURRENCE_2026-11",
    }
    assert prod["SOM_ACE_PA_2026-11"][9] == 1 and {r[1] for r in prod_rows} == {PROD}


def test_the_test_runs_own_new_questions_are_test(db, monkeypatch):
    _create(db, PROD, monkeypatch, test=False)
    _create(db, TEST, monkeypatch, test=True)
    flags = {r[0]: r[4] for r in _questions(db)}
    assert flags["ETH_DR_EVENT_OCCURRENCE_2026-11"] is True
    assert flags["SOM_ACE_PA_2026-11"] is False
    con = duckdb.connect(str(db))
    try:
        links = con.execute(
            "SELECT hs_run_id, COUNT(*), BOOL_AND(is_test) FROM run_questions GROUP BY 1 ORDER BY 1"
        ).fetchall()
    finally:
        con.close()
    assert links == [(PROD, 4, False), (TEST, 6, True)]


def test_a_production_rerun_in_the_same_epoch_still_works(db, monkeypatch):
    _create(db, PROD, monkeypatch, test=False)
    _create(db, TEST, monkeypatch, test=True)
    con = duckdb.connect(str(db))
    _seed_run(con, PROD2, is_test=False, generated_at="2026-10-02 04:00:00",
              tracks={("SOM", "ACE"): (2, 0.0), ("ETH", "DR"): (1, 0.5)})
    con.close()
    _create(db, PROD2, monkeypatch, test=False)
    q = {r[0]: r for r in _questions(db)}
    # A production rerun still adopts: origin, track and the test flag move.
    assert q["SOM_ACE_PA_2026-11"][1] == PROD2 and q["SOM_ACE_PA_2026-11"][2] == 2
    assert q["ETH_DR_EVENT_OCCURRENCE_2026-11"][4] is False
    from forecaster.cli import _select_run_question_rows

    con = duckdb.connect(str(db))
    try:
        ids = {r[0] for r in _select_run_question_rows(con, PROD2, {"SOM", "ETH"})}
        # The first production run still sees the questions it asked for.
        first = {r[0] for r in _select_run_question_rows(con, PROD, {"SOM"})}
    finally:
        con.close()
    assert ids == {
        "SOM_ACE_FATALITIES_2026-11", "SOM_ACE_PA_2026-11",
        "ETH_DR_PHASE3PLUS_IN_NEED_2026-11", "ETH_DR_EVENT_OCCURRENCE_2026-11",
    }
    assert "SOM_FL_PA_2026-11" in first


def test_sibyl_selection_reads_the_runs_set_and_triage(db, monkeypatch):
    _create(db, PROD, monkeypatch, test=False)
    _create(db, TEST, monkeypatch, test=True)
    from sibyl.select_questions import latest_hs_run_id, load_candidates

    con = duckdb.connect(str(db))
    try:
        assert latest_hs_run_id(con) == TEST
        cands = {c.question_id: c for c in load_candidates(TEST, con=con)}
    finally:
        con.close()
    # ACE/FATALITIES and FL/PA are Sibyl-eligible; both are production rows.
    assert {"SOM_ACE_FATALITIES_2026-11", "SOM_FL_PA_2026-11"} <= set(cands)
    assert all(c.hs_run_id == TEST for c in cands.values())
    # Volatility from the TEST run's triage (FL 0.4), not production's (0.0).
    assert cands["SOM_FL_PA_2026-11"].volatility_score == pytest.approx(0.4)


def test_forecaster_run_maps_back_to_its_hs_run(db, monkeypatch):
    _create(db, PROD, monkeypatch, test=False)
    _create(db, TEST, monkeypatch, test=True)
    con = duckdb.connect(str(db))
    try:
        ids = run_question_ids(con, TEST)
        assert stamp_forecaster_run(con, hs_run_id=TEST, forecaster_run_id="fc_200", question_ids=ids) == 6
        assert hs_run_for_forecaster_run(con, "fc_200") == TEST
        from sibyl.spd import find_standard_run_id

        assert find_standard_run_id(con, "SOM_ACE_PA_2026-11", TEST) == "fc_200"
        # The production question row still names production.
        assert con.execute(
            "SELECT hs_run_id FROM questions WHERE question_id = 'SOM_ACE_PA_2026-11'"
        ).fetchone()[0] == PROD
    finally:
        con.close()


def test_a_test_runs_deviation_rows_on_production_questions_are_test(db, monkeypatch):
    _create(db, PROD, monkeypatch, test=False)
    _create(db, TEST, monkeypatch, test=True)
    import pythia.tools.compute_deviation as cd

    con = duckdb.connect(str(db))
    for run_id, is_test in (("fc_100", False), ("fc_200", True)):
        for m in range(1, 7):
            for b in range(1, 7):
                con.execute(
                    "INSERT INTO forecasts_raw (run_id, question_id, model_name, month_index, "
                    "bucket_index, probability, is_test) VALUES (?, 'SOM_ACE_PA_2026-11', "
                    "'ensemble_mean_v2', ?, ?, ?, ?)",
                    [run_id, m, b, 1 / 6, is_test],
                )
    con.close()
    monkeypatch.setattr(cd, "base_rate_spd", lambda *a, **k: ([1 / 6] * 6, "TEST", {}))
    cd.compute_deviation(f"duckdb:///{db}")
    con = duckdb.connect(str(db))
    try:
        rows = dict(con.execute(
            "SELECT run_id, is_test FROM forecast_deviation WHERE question_id = 'SOM_ACE_PA_2026-11'"
        ).fetchall())
    finally:
        con.close()
    assert rows == {"fc_100": False, "fc_200": True}


def test_backfill_reconstructs_past_runs_and_names_the_rest(tmp_path: Path):
    db_path = tmp_path / "old.duckdb"
    con = duckdb.connect(str(db_path))
    ensure_schema(con)
    _seed_run(con, PROD, is_test=False, generated_at="2026-10-01 04:00:00",
              tracks={("SOM", "ACE"): (1, 0.6)})
    _seed_run(con, TEST, is_test=True, generated_at="2026-10-06 08:50:29",
              tracks={("SOM", "ACE"): (2, 0.0)})
    con.execute(
        "INSERT INTO hs_runs (hs_run_id, generated_at, git_sha, config_profile, countries_json, is_test) "
        "VALUES ('hs_empty', CAST('2026-09-15' AS TIMESTAMP), 'sha', 'default', '[]', TRUE)"
    )
    for metric in ("FATALITIES", "PA"):
        con.execute(
            "INSERT INTO questions (question_id, hs_run_id, iso3, hazard_code, metric, "
            "window_start_date, status, track, pythia_metadata_json, is_test) "
            "VALUES (?, ?, 'SOM', 'ACE', ?, DATE '2026-11-01', 'active', 1, '{}', FALSE)",
            [f"SOM_ACE_{metric}_2026-11", PROD, metric],
        )
    con.execute("DELETE FROM run_questions")
    report = backfill_run_questions(con)
    links = con.execute(
        "SELECT hs_run_id, question_id, track, is_test FROM run_questions ORDER BY 1, 2"
    ).fetchall()
    # Idempotent: a second pass writes nothing.
    again = backfill_run_questions(con)
    latest = latest_hs_run_with_questions(con)
    con.close()
    assert report["runs_linked"] == 2 and report["links_written"] == 4
    assert report["unreconstructable"] == ["hs_empty"]
    assert (TEST, "SOM_ACE_PA_2026-11", 2, True) in links
    assert (PROD, "SOM_ACE_PA_2026-11", 1, False) in links
    assert again["links_written"] == 0
    assert latest == TEST
