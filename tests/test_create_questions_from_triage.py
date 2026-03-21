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
