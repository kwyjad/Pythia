from __future__ import annotations

import json
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from pythia.db.schema import ensure_schema
from scripts.create_questions_from_triage import create_questions_from_triage


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
    assert qids == {"SOM_ACE_FATALITIES", "SOM_ACE_PA"}
    assert all(r[2] == "ACE" for r in rows)
    assert all(r[4] == "hs_run_ace" for r in rows)
    metas = [json.loads(r[5]) for r in rows]
    assert all(meta.get("source") == "hs_triage" for meta in metas)
    assert all(meta.get("hazard_family") == "conflict" for meta in metas)
