from __future__ import annotations

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
