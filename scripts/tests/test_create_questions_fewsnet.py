# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for DR FEWS NET Phase 3+ question generation (Prompt 3.5)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest

from scripts.create_questions_from_triage import (
    _build_dr_fewsnet_question_wording,
    _is_fewsnet_country,
    _FEWSNET_COUNTRIES_FILE,
    create_questions_from_triage,
)
from pythia.db.schema import ensure_schema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_test_db(db_path: str, triage_rows: list[tuple]):
    """Set up a test DB with hs_runs, hs_triage, and questions tables."""
    con = duckdb.connect(db_path)
    ensure_schema(con)
    try:
        con.execute("INSERT INTO hs_runs (hs_run_id) VALUES ('test-run-1')")
    except Exception:
        pass
    for row in triage_rows:
        con.execute(
            "INSERT INTO hs_triage (run_id, iso3, hazard_code, tier, triage_score, need_full_spd, track) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            list(row),
        )
    con.close()


# ---------------------------------------------------------------------------
# Unit tests: _build_dr_fewsnet_question_wording
# ---------------------------------------------------------------------------

class TestBuildDrFewsnetQuestionWording:
    def test_mentions_ipc_phase3(self):
        wording = _build_dr_fewsnet_question_wording(
            "ETH", date(2026, 4, 1), date(2026, 9, 30)
        )
        assert "IPC Phase 3+" in wording

    def test_mentions_crisis_or_worse(self):
        wording = _build_dr_fewsnet_question_wording(
            "ETH", date(2026, 4, 1), date(2026, 9, 30)
        )
        assert "Crisis or worse" in wording

    def test_mentions_fewsnet(self):
        wording = _build_dr_fewsnet_question_wording(
            "ETH", date(2026, 4, 1), date(2026, 9, 30)
        )
        assert "FEWS NET" in wording

    def test_includes_country_name(self):
        wording = _build_dr_fewsnet_question_wording(
            "ETH", date(2026, 4, 1), date(2026, 9, 30)
        )
        # ETH is in COUNTRY_NAMES -> "Ethiopia"
        assert "Ethiopia" in wording

    def test_includes_dates(self):
        wording = _build_dr_fewsnet_question_wording(
            "SOM", date(2026, 4, 1), date(2026, 9, 30)
        )
        assert "2026-04-01" in wording
        assert "2026-09-30" in wording

    def test_unknown_country_uses_iso3(self):
        wording = _build_dr_fewsnet_question_wording(
            "KEN", date(2026, 4, 1), date(2026, 9, 30)
        )
        # KEN is not in COUNTRY_NAMES, so falls back to ISO3
        assert "KEN" in wording

    def test_mentions_current_situation(self):
        wording = _build_dr_fewsnet_question_wording(
            "ETH", date(2026, 4, 1), date(2026, 9, 30)
        )
        assert "Current Situation" in wording


# ---------------------------------------------------------------------------
# Unit tests: _is_fewsnet_country
# ---------------------------------------------------------------------------

class TestIsFewsnetCountry:
    def test_with_actual_json_file(self):
        """Test against the real fewsnet_countries.json if it exists."""
        if not _FEWSNET_COUNTRIES_FILE.exists():
            pytest.skip("fewsnet_countries.json not found")
        # ETH should be in the real file
        assert _is_fewsnet_country("ETH") is True
        assert _is_fewsnet_country("SOM") is True
        assert _is_fewsnet_country("KEN") is True
        # USA should not be a FEWS NET country
        assert _is_fewsnet_country("USA") is False

    def test_fail_open_when_file_missing(self):
        """When the JSON file is missing, should return True (fail-open)."""
        with patch(
            "scripts.create_questions_from_triage._FEWSNET_COUNTRIES_FILE",
            Path("/nonexistent/fewsnet_countries.json"),
        ):
            assert _is_fewsnet_country("USA") is True
            assert _is_fewsnet_country("XYZ") is True

    def test_fail_open_on_corrupt_json(self, tmp_path):
        """When the JSON file is corrupt, should return True (fail-open)."""
        bad_file = tmp_path / "fewsnet_countries.json"
        bad_file.write_text("NOT VALID JSON{{{")
        with patch(
            "scripts.create_questions_from_triage._FEWSNET_COUNTRIES_FILE",
            bad_file,
        ):
            assert _is_fewsnet_country("ETH") is True

    def test_case_insensitive(self, tmp_path):
        fewsnet_file = tmp_path / "fewsnet_countries.json"
        fewsnet_file.write_text(json.dumps(["ETH", "SOM"]))
        with patch(
            "scripts.create_questions_from_triage._FEWSNET_COUNTRIES_FILE",
            fewsnet_file,
        ):
            assert _is_fewsnet_country("eth") is True
            assert _is_fewsnet_country("Eth") is True
            assert _is_fewsnet_country("ETH") is True

    def test_not_in_list(self, tmp_path):
        fewsnet_file = tmp_path / "fewsnet_countries.json"
        fewsnet_file.write_text(json.dumps(["ETH", "SOM"]))
        with patch(
            "scripts.create_questions_from_triage._FEWSNET_COUNTRIES_FILE",
            fewsnet_file,
        ):
            assert _is_fewsnet_country("USA") is False
            assert _is_fewsnet_country("GBR") is False


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestDrPaBlocked:
    """DR/PA should be blocked for non-FEWS NET countries."""

    def test_non_fewsnet_country_no_pa(self, tmp_path):
        db_path = str(tmp_path / "test.duckdb")
        _setup_test_db(db_path, [
            ("test-run-1", "USA", "DR", "priority", 0.7, True, 1),
        ])
        fewsnet_file = tmp_path / "fewsnet_countries.json"
        fewsnet_file.write_text(json.dumps(["ETH", "SOM", "KEN"]))

        with patch(
            "scripts.create_questions_from_triage._FEWSNET_COUNTRIES_FILE",
            fewsnet_file,
        ):
            count = create_questions_from_triage(
                f"duckdb:///{db_path}", "test-run-1"
            )

        assert count == 1  # Only EVENT_OCCURRENCE

        con = duckdb.connect(db_path)
        rows = con.execute("SELECT metric FROM questions").fetchall()
        con.close()
        metrics = {r[0] for r in rows}
        assert "EVENT_OCCURRENCE" in metrics
        assert "PA" not in metrics
        assert "PHASE3PLUS_IN_NEED" not in metrics


class TestDrPhase3PlusGenerated:
    """DR/PA for FEWS NET countries should produce PHASE3PLUS_IN_NEED metric."""

    def test_fewsnet_country_gets_phase3plus(self, tmp_path):
        db_path = str(tmp_path / "test.duckdb")
        _setup_test_db(db_path, [
            ("test-run-1", "ETH", "DR", "priority", 0.7, True, 1),
        ])
        fewsnet_file = tmp_path / "fewsnet_countries.json"
        fewsnet_file.write_text(json.dumps(["ETH", "SOM", "KEN"]))

        with patch(
            "scripts.create_questions_from_triage._FEWSNET_COUNTRIES_FILE",
            fewsnet_file,
        ):
            count = create_questions_from_triage(
                f"duckdb:///{db_path}", "test-run-1"
            )

        assert count == 2  # PHASE3PLUS_IN_NEED + EVENT_OCCURRENCE

        con = duckdb.connect(db_path)
        rows = con.execute(
            "SELECT question_id, metric, wording FROM questions ORDER BY question_id"
        ).fetchall()
        con.close()

        metrics = {r[1] for r in rows}
        assert "PHASE3PLUS_IN_NEED" in metrics
        assert "EVENT_OCCURRENCE" in metrics
        assert "PA" not in metrics  # PA should be replaced by PHASE3PLUS_IN_NEED

    def test_question_id_includes_phase3plus(self, tmp_path):
        db_path = str(tmp_path / "test.duckdb")
        _setup_test_db(db_path, [
            ("test-run-1", "SOM", "DR", "priority", 0.7, True, 1),
        ])
        fewsnet_file = tmp_path / "fewsnet_countries.json"
        fewsnet_file.write_text(json.dumps(["ETH", "SOM", "KEN"]))

        with patch(
            "scripts.create_questions_from_triage._FEWSNET_COUNTRIES_FILE",
            fewsnet_file,
        ):
            create_questions_from_triage(f"duckdb:///{db_path}", "test-run-1")

        con = duckdb.connect(db_path)
        rows = con.execute("SELECT question_id FROM questions").fetchall()
        con.close()

        qids = [r[0] for r in rows]
        assert any("PHASE3PLUS_IN_NEED" in qid for qid in qids)

    def test_question_wording_mentions_ipc(self, tmp_path):
        db_path = str(tmp_path / "test.duckdb")
        _setup_test_db(db_path, [
            ("test-run-1", "ETH", "DR", "priority", 0.7, True, 1),
        ])
        fewsnet_file = tmp_path / "fewsnet_countries.json"
        fewsnet_file.write_text(json.dumps(["ETH", "SOM", "KEN"]))

        with patch(
            "scripts.create_questions_from_triage._FEWSNET_COUNTRIES_FILE",
            fewsnet_file,
        ):
            create_questions_from_triage(f"duckdb:///{db_path}", "test-run-1")

        con = duckdb.connect(db_path)
        rows = con.execute(
            "SELECT wording FROM questions WHERE metric = 'PHASE3PLUS_IN_NEED'"
        ).fetchall()
        con.close()

        assert len(rows) == 1
        wording = rows[0][0]
        assert "IPC Phase 3+" in wording
        assert "FEWS NET" in wording


class TestDrEventOccurrenceAllCountries:
    """DR/EVENT_OCCURRENCE should be generated for ALL countries."""

    def test_event_occurrence_for_non_fewsnet(self, tmp_path):
        db_path = str(tmp_path / "test.duckdb")
        _setup_test_db(db_path, [
            ("test-run-1", "USA", "DR", "priority", 0.7, True, 1),
        ])
        fewsnet_file = tmp_path / "fewsnet_countries.json"
        fewsnet_file.write_text(json.dumps(["ETH", "SOM"]))

        with patch(
            "scripts.create_questions_from_triage._FEWSNET_COUNTRIES_FILE",
            fewsnet_file,
        ):
            count = create_questions_from_triage(
                f"duckdb:///{db_path}", "test-run-1"
            )

        assert count == 1

        con = duckdb.connect(db_path)
        rows = con.execute("SELECT metric FROM questions").fetchall()
        con.close()
        assert rows[0][0] == "EVENT_OCCURRENCE"

    def test_event_occurrence_for_fewsnet(self, tmp_path):
        db_path = str(tmp_path / "test.duckdb")
        _setup_test_db(db_path, [
            ("test-run-1", "ETH", "DR", "priority", 0.7, True, 1),
        ])
        fewsnet_file = tmp_path / "fewsnet_countries.json"
        fewsnet_file.write_text(json.dumps(["ETH", "SOM"]))

        with patch(
            "scripts.create_questions_from_triage._FEWSNET_COUNTRIES_FILE",
            fewsnet_file,
        ):
            count = create_questions_from_triage(
                f"duckdb:///{db_path}", "test-run-1"
            )

        assert count == 2  # PHASE3PLUS_IN_NEED + EVENT_OCCURRENCE

        con = duckdb.connect(db_path)
        rows = con.execute("SELECT metric FROM questions").fetchall()
        con.close()
        metrics = {r[0] for r in rows}
        assert "EVENT_OCCURRENCE" in metrics


class TestNonDrUnchanged:
    """Non-DR hazards should be unaffected by FEWS NET logic."""

    def test_fl_still_generates_pa_and_event_occurrence(self, tmp_path):
        db_path = str(tmp_path / "test.duckdb")
        _setup_test_db(db_path, [
            ("test-run-1", "BGD", "FL", "priority", 0.8, True, 1),
        ])
        count = create_questions_from_triage(f"duckdb:///{db_path}", "test-run-1")
        assert count == 2  # PA + EVENT_OCCURRENCE

        con = duckdb.connect(db_path)
        rows = con.execute("SELECT metric FROM questions").fetchall()
        con.close()
        metrics = {r[0] for r in rows}
        assert "PA" in metrics
        assert "EVENT_OCCURRENCE" in metrics
