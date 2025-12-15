# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from pythia.db.schema import ensure_schema
from scripts import create_questions_from_triage as cqft


def test_create_questions_from_triage_creates_expected_questions(tmp_path: Path) -> None:
    db_path = tmp_path / "triage_questions.duckdb"
    db_url = f"duckdb:///{db_path}"

    con = duckdb.connect(str(db_path))
    try:
        ensure_schema(con)
        con.execute(
            """
            INSERT INTO hs_runs (hs_run_id, generated_at, git_sha, config_profile, countries_json)
            VALUES ('hs_test', CURRENT_TIMESTAMP, 'abc123', 'default', '[]')
            """
        )
        con.execute(
            """
            INSERT INTO hs_triage (
                run_id, iso3, hazard_code, tier, triage_score, need_full_spd,
                drivers_json, regime_shifts_json, data_quality_json, scenario_stub
            ) VALUES ('hs_test', 'ETH', 'ACE', 'priority', 0.8, TRUE, '[]', '[]', '{}', '')
            """
        )
    finally:
        con.close()

    created = cqft.create_questions_from_triage(db_url, hs_run_id="hs_test")
    assert created >= 2

    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute(
            """
            SELECT question_id, iso3, hazard_code, metric, status, hs_run_id, pythia_metadata_json
            FROM questions
            ORDER BY question_id
            """
        ).fetchall()
    finally:
        con.close()

    assert rows
    qids = {r[0] for r in rows}
    assert "ETH_ACE_FATALITIES" in qids
    assert "ETH_ACE_PA" in qids
    assert all(r[2] == "ACE" for r in rows)
    assert all(r[5] == "hs_test" for r in rows)
    assert all(r[4] == "active" for r in rows)
    assert all("hs_triage" in (r[6] or "") for r in rows)
