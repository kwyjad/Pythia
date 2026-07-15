# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import json
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from pythia.db.schema import ensure_schema
from scripts.create_questions_from_triage import (
    SUPPORTED_HAZARD_METRICS,
    create_questions_from_triage,
)


def test_create_questions_from_triage_skips_aco(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path = tmp_path / "triage.duckdb"
    db_url = f"duckdb:///{db_path}"

    con = duckdb.connect(str(db_path))
    try:
        ensure_schema(con)
        con.execute(
            """
            INSERT INTO hs_triage (
                run_id, iso3, hazard_code, tier, triage_score, need_full_spd,
                drivers_json, regime_shifts_json, data_quality_json, scenario_stub
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "hs_run_1",
                "ETH",
                "ACO",
                "priority",
                0.8,
                True,
                "[]",
                "[]",
                "{}",
                "",
            ],
        )
    finally:
        con.close()

    created = create_questions_from_triage(db_url)
    assert created == 0

    con = duckdb.connect(str(db_path))
    try:
        count = con.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
    finally:
        con.close()

    assert count == 0


def test_create_questions_from_triage_creates_ace_questions(tmp_path: Path) -> None:
    db_path = tmp_path / "triage_ace.duckdb"
    db_url = f"duckdb:///{db_path}"

    con = duckdb.connect(str(db_path))
    try:
        ensure_schema(con)
        con.execute(
            """
            INSERT INTO hs_runs (hs_run_id, generated_at, git_sha, config_profile, countries_json)
            VALUES ('hs_run_ace', CURRENT_TIMESTAMP, 'abc123', 'default', '[]')
            """
        )
        con.execute(
            """
            INSERT INTO hs_triage (
                run_id, iso3, hazard_code, tier, triage_score, need_full_spd,
                drivers_json, regime_shifts_json, data_quality_json, scenario_stub
            ) VALUES ('hs_run_ace', 'SOM', 'ACE', 'priority', 0.85, TRUE, '[]', '[]', '{}', '')
            """
        )
    finally:
        con.close()

    created = create_questions_from_triage(db_url, hs_run_id="hs_run_ace")
    assert created == 2

    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute(
            """
            SELECT question_id, iso3, hazard_code, metric, hs_run_id, pythia_metadata_json
            FROM questions
            ORDER BY question_id
            """
        ).fetchall()
    finally:
        con.close()

    assert len(rows) == 2
    qids = {r[0] for r in rows}
    # question_ids are epoch-specific since Phase 2
    assert all("SOM_ACE_FATALITIES_" in q for q in qids if "FATALITIES" in q)
    assert all("SOM_ACE_PA_" in q for q in qids if "PA" in q)
    assert len(qids) == 2
    assert all(r[2] == "ACE" for r in rows)
    assert all(r[4] == "hs_run_ace" for r in rows)
    metas = [json.loads(r[5]) for r in rows]
    assert all(meta.get("source") == "hs_triage" for meta in metas)
    assert all(meta.get("hazard_family") == "conflict" for meta in metas)


def test_hw_not_in_supported_hazard_metrics() -> None:
    """HW (heatwave) should not be in SUPPORTED_HAZARD_METRICS."""
    assert "HW" not in SUPPORTED_HAZARD_METRICS


def test_create_questions_from_triage_skips_hw(tmp_path: Path) -> None:
    """HW triage rows should be skipped — no question created."""
    db_path = tmp_path / "triage_hw.duckdb"
    db_url = f"duckdb:///{db_path}"

    con = duckdb.connect(str(db_path))
    try:
        ensure_schema(con)
        con.execute(
            """
            INSERT INTO hs_triage (
                run_id, iso3, hazard_code, tier, triage_score, need_full_spd,
                drivers_json, regime_shifts_json, data_quality_json, scenario_stub
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "hs_run_hw",
                "IND",
                "HW",
                "priority",
                0.9,
                True,
                "[]",
                "[]",
                "{}",
                "",
            ],
        )
    finally:
        con.close()

    created = create_questions_from_triage(db_url)
    assert created == 0

    con = duckdb.connect(str(db_path))
    try:
        count = con.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
    finally:
        con.close()

    assert count == 0, f"HW questions should not be created, got {count}"


def test_create_questions_honors_blocked_need_full_spd(tmp_path: Path) -> None:
    """A priority-tier hazard whose need_full_spd was cleared (e.g. by the
    Brave circuit-breaker grounding gate) must NOT get questions; the stored
    need_full_spd column is the single source of truth."""
    db_path = tmp_path / "triage_gate.duckdb"
    db_url = f"duckdb:///{db_path}"

    con = duckdb.connect(str(db_path))
    try:
        ensure_schema(con)
        con.execute(
            """
            INSERT INTO hs_runs (hs_run_id, generated_at, git_sha, config_profile, countries_json)
            VALUES ('hs_run_gate', CURRENT_TIMESTAMP, 'abc123', 'default', '[]')
            """
        )
        # Priority tier but gated off — must be skipped.
        con.execute(
            """
            INSERT INTO hs_triage (
                run_id, iso3, hazard_code, tier, triage_score, need_full_spd,
                drivers_json, regime_shifts_json, data_quality_json, scenario_stub
            ) VALUES ('hs_run_gate', 'SOM', 'ACE', 'priority', 0.85, FALSE, '[]', '[]',
                      '{"brave_budget_gate": "blocked_no_grounding"}', '')
            """
        )
        # Quiet tier (RC-promoted defaults) with need_full_spd TRUE — must be kept.
        con.execute(
            """
            INSERT INTO hs_triage (
                run_id, iso3, hazard_code, tier, triage_score, need_full_spd,
                drivers_json, regime_shifts_json, data_quality_json, scenario_stub
            ) VALUES ('hs_run_gate', 'ETH', 'FL', 'quiet', 0.0, TRUE, '[]', '[]', '{}', '')
            """
        )
    finally:
        con.close()

    created = create_questions_from_triage(db_url, hs_run_id="hs_run_gate")

    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute("SELECT DISTINCT iso3, hazard_code FROM questions").fetchall()
    finally:
        con.close()

    assert ("SOM", "ACE") not in rows, "gated priority hazard must not produce questions"
    assert ("ETH", "FL") in rows, "need_full_spd=TRUE hazard must still produce questions"
    assert created > 0

def _seed_ace_triage(db_path: Path, hs_run_id: str) -> None:
    con = duckdb.connect(str(db_path))
    try:
        ensure_schema(con)
        con.execute(
            """
            INSERT INTO hs_runs (hs_run_id, generated_at, git_sha, config_profile, countries_json)
            VALUES (?, CURRENT_TIMESTAMP, 'abc123', 'default', '[]')
            """,
            [hs_run_id],
        )
        con.execute(
            """
            INSERT INTO hs_triage (
                run_id, iso3, hazard_code, tier, triage_score, need_full_spd,
                drivers_json, regime_shifts_json, data_quality_json, scenario_stub
            ) VALUES (?, 'SOM', 'ACE', 'priority', 0.85, TRUE, '[]', '[]', '{}', '')
            """,
            [hs_run_id],
        )
    finally:
        con.close()


def _question_is_test_flags(db_path: Path) -> set[bool]:
    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute("SELECT COALESCE(is_test, FALSE) FROM questions").fetchall()
    finally:
        con.close()
    return {bool(r[0]) for r in rows}


def test_same_epoch_production_rerun_clears_test_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A production run adopting same-epoch test questions must clear is_test,
    otherwise its forecasts stay hidden on the dashboard and excluded from
    resolution/scoring/calibration (which all inherit questions.is_test)."""
    db_path = tmp_path / "triage_istest.duckdb"
    db_url = f"duckdb:///{db_path}"

    _seed_ace_triage(db_path, "hs_run_test")
    monkeypatch.setenv("PYTHIA_TEST_MODE", "1")
    assert create_questions_from_triage(db_url, hs_run_id="hs_run_test") == 2
    assert _question_is_test_flags(db_path) == {True}

    # Same-epoch production re-run: UPDATE path must flip is_test to FALSE.
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            INSERT INTO hs_runs (hs_run_id, generated_at, git_sha, config_profile, countries_json)
            VALUES ('hs_run_prod', CURRENT_TIMESTAMP, 'abc123', 'default', '[]')
            """
        )
        con.execute(
            """
            INSERT INTO hs_triage (
                run_id, iso3, hazard_code, tier, triage_score, need_full_spd,
                drivers_json, regime_shifts_json, data_quality_json, scenario_stub
            ) VALUES ('hs_run_prod', 'SOM', 'ACE', 'priority', 0.85, TRUE, '[]', '[]', '{}', '')
            """
        )
    finally:
        con.close()
    monkeypatch.delenv("PYTHIA_TEST_MODE", raising=False)
    assert create_questions_from_triage(db_url, hs_run_id="hs_run_prod") == 0  # updated, not inserted
    assert _question_is_test_flags(db_path) == {False}


def test_same_epoch_test_rerun_never_taints_production_question(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The is_test transition is one-way (TRUE -> FALSE): a later test-mode
    re-run must not flip an existing production question back to test."""
    db_path = tmp_path / "triage_istest_rev.duckdb"
    db_url = f"duckdb:///{db_path}"

    _seed_ace_triage(db_path, "hs_run_prod")
    monkeypatch.delenv("PYTHIA_TEST_MODE", raising=False)
    assert create_questions_from_triage(db_url, hs_run_id="hs_run_prod") == 2
    assert _question_is_test_flags(db_path) == {False}

    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            INSERT INTO hs_runs (hs_run_id, generated_at, git_sha, config_profile, countries_json)
            VALUES ('hs_run_test2', CURRENT_TIMESTAMP, 'abc123', 'default', '[]')
            """
        )
        con.execute(
            """
            INSERT INTO hs_triage (
                run_id, iso3, hazard_code, tier, triage_score, need_full_spd,
                drivers_json, regime_shifts_json, data_quality_json, scenario_stub
            ) VALUES ('hs_run_test2', 'SOM', 'ACE', 'priority', 0.85, TRUE, '[]', '[]', '{}', '')
            """
        )
    finally:
        con.close()
    monkeypatch.setenv("PYTHIA_TEST_MODE", "1")
    assert create_questions_from_triage(db_url, hs_run_id="hs_run_test2") == 0
    assert _question_is_test_flags(db_path) == {False}


_FACTS_RESOLVED_DDL = """
CREATE TABLE IF NOT EXISTS facts_resolved (
    ym TEXT NOT NULL,
    iso3 TEXT NOT NULL,
    hazard_code TEXT NOT NULL,
    metric TEXT NOT NULL,
    series_semantics TEXT NOT NULL DEFAULT '',
    value DOUBLE
)
"""


def _write_country_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fewsnet: list[str]) -> None:
    """Point the generator at a real FEWS NET list and a MISSING IPC list."""
    import scripts.create_questions_from_triage as cq

    fewsnet_file = tmp_path / "fewsnet_countries.json"
    fewsnet_file.write_text(json.dumps(fewsnet))
    monkeypatch.setattr(cq, "_FEWSNET_COUNTRIES_FILE", fewsnet_file)
    monkeypatch.setattr(cq, "_IPC_COUNTRIES_FILE", tmp_path / "ipc_countries.json")


def test_food_security_db_fallback_covers_ipc_only_country(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A country absent from both JSON lists but with recent phase3plus_in_need
    rows in facts_resolved must count as food-security eligible (the IPC list
    is written at ingest time and never exists at question-generation time)."""
    import scripts.create_questions_from_triage as cq

    _write_country_files(monkeypatch, tmp_path, ["ETH"])
    cq._FOOD_SECURITY_DB_CACHE.clear()

    con = duckdb.connect(str(tmp_path / "fs.duckdb"))
    try:
        con.execute(_FACTS_RESOLVED_DDL)
        from datetime import date

        recent_ym = f"{date.today().year:04d}-{date.today().month:02d}"
        con.execute(
            "INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value) VALUES (?, 'COL', 'DR', 'phase3plus_in_need', 100000)",
            [recent_ym],
        )
        con.execute(
            "INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value) VALUES ('2019-01', 'XKX', 'DR', 'phase3plus_in_need', 5000)"
        )

        assert cq._is_food_security_country("ETH", con) is True  # FEWS NET file
        assert cq._is_food_security_country("COL", con) is True  # DB fallback
        # Older than the 36-month lookback: not resurrected
        assert cq._is_food_security_country("XKX", con) is False
        # No data anywhere
        assert cq._is_food_security_country("ZZZ", con) is False
    finally:
        cq._FOOD_SECURITY_DB_CACHE.clear()
        con.close()


def test_dr_phase3_question_generated_for_ipc_only_country(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import scripts.create_questions_from_triage as cq

    _write_country_files(monkeypatch, tmp_path, ["ETH"])
    cq._FOOD_SECURITY_DB_CACHE.clear()

    db_path = tmp_path / "triage.duckdb"
    db_url = f"duckdb:///{db_path}"
    con = duckdb.connect(str(db_path))
    try:
        ensure_schema(con)
        con.execute(_FACTS_RESOLVED_DDL)
        from datetime import date

        recent_ym = f"{date.today().year:04d}-{date.today().month:02d}"
        con.execute(
            "INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value) VALUES (?, 'COL', 'DR', 'phase3plus_in_need', 100000)",
            [recent_ym],
        )
        con.execute(
            """
            INSERT INTO hs_triage (
                run_id, iso3, hazard_code, tier, triage_score, need_full_spd,
                drivers_json, regime_shifts_json, data_quality_json, scenario_stub
            ) VALUES ('hs_run_fs', 'COL', 'DR', 'priority', 0.7, TRUE, '[]', '[]', '{}', '')
            """
        )
    finally:
        con.close()

    created = create_questions_from_triage(db_url, hs_run_id="hs_run_fs")
    assert created == 2  # PHASE3PLUS_IN_NEED + EVENT_OCCURRENCE

    con = duckdb.connect(str(db_path))
    try:
        metrics = {
            r[0]
            for r in con.execute(
                "SELECT metric FROM questions WHERE iso3 = 'COL'"
            ).fetchall()
        }
    finally:
        cq._FOOD_SECURITY_DB_CACHE.clear()
        con.close()
    assert metrics == {"PHASE3PLUS_IN_NEED", "EVENT_OCCURRENCE"}
