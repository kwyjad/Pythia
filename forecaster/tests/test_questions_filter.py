from __future__ import annotations

import json
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

import forecaster.cli as cli  # type: ignore


def test_load_pythia_questions_filters_aco_and_demo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path = tmp_path / "questions.duckdb"
    con = duckdb.connect(str(db_path))

    con.execute(
        """
        CREATE TABLE questions (
            question_id TEXT,
            hs_run_id TEXT,
            scenario_ids_json TEXT,
            iso3 TEXT,
            hazard_code TEXT,
            metric TEXT,
            target_month TEXT,
            window_start_date DATE,
            window_end_date DATE,
            wording TEXT,
            status TEXT,
            pythia_metadata_json TEXT
        )
        """
    )

    rows = [
        (
            "ETH_ACE_FATALITIES",
            "hs_123",
            json.dumps(["S1"]),
            "ETH",
            "ACE",
            "FATALITIES",
            "2026-01",
            "2026-01-01",
            "2026-06-30",
            "HS question",
            "active",
            json.dumps({"source": "hs_triage"}),
        ),
        (
            "ETH_ACO_FATALITIES",
            None,
            json.dumps([]),
            "ETH",
            "ACO",
            "FATALITIES",
            "2026-01",
            "2026-01-01",
            "2026-06-30",
            "ACO question",
            "active",
            json.dumps({"source": "demo"}),
        ),
        (
            "ETH_ACE_FATALITIES_DEMO",
            None,
            json.dumps([]),
            "ETH",
            "ACE",
            "FATALITIES",
            "2026-01",
            "2026-01-01",
            "2026-06-30",
            "Demo question",
            "active",
            json.dumps({"source": "demo"}),
        ),
    ]

    con.executemany(
        """
        INSERT INTO questions (
            question_id, hs_run_id, scenario_ids_json, iso3, hazard_code, metric,
            target_month, window_start_date, window_end_date, wording, status, pythia_metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    con.close()

    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db_path}")

    questions = cli._load_pythia_questions(limit=10)
    ids = {q["id"] for q in questions}

    assert ids == {"ETH_ACE_FATALITIES"}

