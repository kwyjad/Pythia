# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for question metadata preservation in log_hs_questions_to_db.

The Horizon Scanner generates questions each run.  When a newer HS run
produces a question with the same ``question_id`` as an earlier run, the
original metadata (window_start_date, target_month, wording, etc.) MUST
be preserved.  The ``log_hs_questions_to_db`` function should skip
existing questions and only insert genuinely new ones.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from pythia.db.schema import connect, ensure_schema
from horizon_scanner.db_writer import log_hs_questions_to_db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_hs_run(monkeypatch, tmp_path: Path):
    """Create a DuckDB with schema and return the db_path."""
    db_path = tmp_path / "preservation.duckdb"
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db_path}")

    con = connect(read_only=False)
    try:
        ensure_schema(con)
        # Seed an HS run so the questions can reference it
        con.execute(
            "INSERT INTO hs_runs (hs_run_id, generated_at) "
            "VALUES ('run_dec', CURRENT_TIMESTAMP)"
        )
        con.execute(
            "INSERT INTO hs_runs (hs_run_id, generated_at) "
            "VALUES ('run_feb', CURRENT_TIMESTAMP)"
        )
    finally:
        con.close()

    return db_path


def _read_question(db_path: Path, question_id: str) -> dict | None:
    """Read a question row from the DB as a dict."""
    con = duckdb.connect(str(db_path))
    try:
        row = con.execute(
            "SELECT question_id, hs_run_id, target_month, window_start_date, wording "
            "FROM questions WHERE question_id = ?",
            [question_id],
        ).fetchone()
    finally:
        con.close()
    if not row:
        return None
    return {
        "question_id": row[0],
        "hs_run_id": row[1],
        "target_month": row[2],
        "window_start_date": row[3],
        "wording": row[4],
    }


def _count_questions(db_path: Path) -> int:
    con = duckdb.connect(str(db_path))
    try:
        return con.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestQuestionPreservation:
    """log_hs_questions_to_db must never overwrite existing questions."""

    def test_original_metadata_preserved(self, monkeypatch, tmp_path):
        """When a second HS run emits the same question_id, the original
        row (window_start_date, target_month, wording) must be kept."""
        db_path = _seed_hs_run(monkeypatch, tmp_path)

        # First HS run (December) creates the question
        dec_question = {
            "question_id": "MLI_FL_PA",
            "hs_run_id": "run_dec",
            "iso3": "MLI",
            "hazard_code": "FL",
            "metric": "PA",
            "target_month": "2026-06",
            "window_start_date": date(2026, 1, 1),
            "window_end_date": date(2026, 6, 30),
            "wording": "December forecast wording",
            "status": "active",
        }
        log_hs_questions_to_db("run_dec", [dec_question])

        # Verify December data was written
        q = _read_question(db_path, "MLI_FL_PA")
        assert q is not None
        assert q["hs_run_id"] == "run_dec"
        assert q["target_month"] == "2026-06"
        assert q["wording"] == "December forecast wording"

        # Second HS run (February) tries to write SAME question_id with different metadata
        feb_question = {
            "question_id": "MLI_FL_PA",
            "hs_run_id": "run_feb",
            "iso3": "MLI",
            "hazard_code": "FL",
            "metric": "PA",
            "target_month": "2026-08",
            "window_start_date": date(2026, 3, 1),
            "window_end_date": date(2026, 8, 31),
            "wording": "February forecast wording — THIS SHOULD NOT APPEAR",
            "status": "active",
        }
        log_hs_questions_to_db("run_feb", [feb_question])

        # Original December metadata MUST be preserved
        q = _read_question(db_path, "MLI_FL_PA")
        assert q is not None
        assert q["hs_run_id"] == "run_dec", (
            f"Expected original run_dec, got {q['hs_run_id']} — question was overwritten!"
        )
        assert q["target_month"] == "2026-06", (
            f"Expected original target 2026-06, got {q['target_month']} — overwritten!"
        )
        assert q["wording"] == "December forecast wording", (
            f"Expected original wording, got {q['wording']!r} — overwritten!"
        )

    def test_new_questions_still_inserted(self, monkeypatch, tmp_path):
        """Genuinely new questions (unseen question_id) should be inserted."""
        db_path = _seed_hs_run(monkeypatch, tmp_path)

        questions = [
            {
                "question_id": "ETH_DR_PA",
                "hs_run_id": "run_dec",
                "iso3": "ETH",
                "hazard_code": "DR",
                "metric": "PA",
                "target_month": "2026-06",
                "window_start_date": date(2026, 1, 1),
                "window_end_date": date(2026, 6, 30),
                "wording": "ETH drought question",
                "status": "active",
            },
            {
                "question_id": "SOM_ACE_FATALITIES",
                "hs_run_id": "run_dec",
                "iso3": "SOM",
                "hazard_code": "ACE",
                "metric": "FATALITIES",
                "target_month": "2026-06",
                "window_start_date": date(2026, 1, 1),
                "window_end_date": date(2026, 6, 30),
                "wording": "SOM conflict fatalities",
                "status": "active",
            },
        ]
        log_hs_questions_to_db("run_dec", questions)

        assert _count_questions(db_path) == 2
        assert _read_question(db_path, "ETH_DR_PA") is not None
        assert _read_question(db_path, "SOM_ACE_FATALITIES") is not None

    def test_mixed_new_and_existing(self, monkeypatch, tmp_path):
        """A batch with both existing and new questions: new ones inserted,
        existing ones preserved."""
        db_path = _seed_hs_run(monkeypatch, tmp_path)

        # Insert the first question
        log_hs_questions_to_db("run_dec", [
            {
                "question_id": "MLI_FL_PA",
                "hs_run_id": "run_dec",
                "iso3": "MLI",
                "hazard_code": "FL",
                "metric": "PA",
                "target_month": "2026-06",
                "window_start_date": date(2026, 1, 1),
                "window_end_date": date(2026, 6, 30),
                "wording": "Original wording",
                "status": "active",
            },
        ])
        assert _count_questions(db_path) == 1

        # Second run: one existing question + one new
        log_hs_questions_to_db("run_feb", [
            {
                "question_id": "MLI_FL_PA",  # existing
                "hs_run_id": "run_feb",
                "iso3": "MLI",
                "hazard_code": "FL",
                "metric": "PA",
                "target_month": "2026-08",
                "window_start_date": date(2026, 3, 1),
                "window_end_date": date(2026, 8, 31),
                "wording": "Overwritten wording",
                "status": "active",
            },
            {
                "question_id": "ETH_DR_PA",  # new
                "hs_run_id": "run_feb",
                "iso3": "ETH",
                "hazard_code": "DR",
                "metric": "PA",
                "target_month": "2026-08",
                "window_start_date": date(2026, 3, 1),
                "window_end_date": date(2026, 8, 31),
                "wording": "New ETH question",
                "status": "active",
            },
        ])

        assert _count_questions(db_path) == 2

        # Existing question is preserved
        mli = _read_question(db_path, "MLI_FL_PA")
        assert mli["hs_run_id"] == "run_dec"
        assert mli["wording"] == "Original wording"

        # New question was inserted
        eth = _read_question(db_path, "ETH_DR_PA")
        assert eth is not None
        assert eth["hs_run_id"] == "run_feb"
        assert eth["wording"] == "New ETH question"
