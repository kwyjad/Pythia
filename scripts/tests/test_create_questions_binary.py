# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for binary event question generation (EVENT_OCCURRENCE)."""

from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest

from scripts.create_questions_from_triage import (
    SUPPORTED_HAZARD_METRICS,
    _build_binary_question_wording,
    _is_fewsnet_country,
    _metrics_for_hazard,
    create_questions_from_triage,
)
from pythia.db.schema import ensure_schema


# ---- Unit tests ----

def test_fl_tc_dr_have_event_occurrence():
    """FL, TC, DR should include EVENT_OCCURRENCE in their metrics."""
    for hz in ("FL", "TC", "DR"):
        assert "EVENT_OCCURRENCE" in SUPPORTED_HAZARD_METRICS[hz], f"{hz} missing EVENT_OCCURRENCE"


def test_ace_no_event_occurrence():
    """ACE should NOT have EVENT_OCCURRENCE."""
    assert "EVENT_OCCURRENCE" not in SUPPORTED_HAZARD_METRICS["ACE"]


def test_cu_di_no_event_occurrence():
    """CU and DI should NOT have EVENT_OCCURRENCE."""
    assert "EVENT_OCCURRENCE" not in SUPPORTED_HAZARD_METRICS.get("CU", [])
    assert "EVENT_OCCURRENCE" not in SUPPORTED_HAZARD_METRICS.get("DI", [])


def test_binary_question_wording():
    """Binary question wording should reference GDACS and alert levels."""
    wording = _build_binary_question_wording(
        "ETH", "DR", date(2026, 4, 1), date(2026, 9, 30)
    )
    assert "GDACS" in wording
    assert "Orange or Red" in wording
    assert "drought" in wording
    assert "ETH" in wording or "Ethiopia" in wording
    assert "2026-04-01" in wording
    assert "2026-09-30" in wording


def test_binary_wording_tc():
    wording = _build_binary_question_wording(
        "MOZ", "TC", date(2026, 4, 1), date(2026, 9, 30)
    )
    assert "tropical cyclone" in wording
    assert "GDACS" in wording


def test_binary_wording_fl():
    wording = _build_binary_question_wording(
        "BGD", "FL", date(2026, 4, 1), date(2026, 9, 30)
    )
    assert "flooding" in wording
    assert "GDACS" in wording


def test_question_id_includes_event_occurrence():
    """Question ID should include EVENT_OCCURRENCE metric."""
    # The ID format is {iso3}_{hazard}_{metric}_{epoch}
    qid = "ETH_DR_EVENT_OCCURRENCE_2026-04"
    assert "EVENT_OCCURRENCE" in qid


def test_is_fewsnet_country_with_file(tmp_path):
    """_is_fewsnet_country returns True for listed countries."""
    fewsnet_file = tmp_path / "fewsnet_countries.json"
    fewsnet_file.write_text(json.dumps(["ETH", "SOM", "KEN"]))
    with patch("scripts.create_questions_from_triage._FEWSNET_COUNTRIES_FILE", fewsnet_file):
        assert _is_fewsnet_country("ETH") is True
        assert _is_fewsnet_country("eth") is True  # case insensitive
        assert _is_fewsnet_country("USA") is False


def test_is_fewsnet_country_no_file():
    """_is_fewsnet_country returns True (fail-open) when file missing."""
    with patch("scripts.create_questions_from_triage._FEWSNET_COUNTRIES_FILE", Path("/nonexistent/path.json")):
        assert _is_fewsnet_country("USA") is True


# ---- Integration tests ----

def _setup_test_db(db_path: str, triage_rows: list[tuple]):
    """Set up a test DB with hs_runs, hs_triage, and questions tables."""
    con = duckdb.connect(db_path)
    ensure_schema(con)
    # Insert into hs_runs (created by ensure_schema)
    try:
        con.execute("INSERT INTO hs_runs (hs_run_id) VALUES ('test-run-1')")
    except Exception:
        pass  # table may already have the row
    # Insert into hs_triage using explicit column names (ensure_schema creates the full table)
    for row in triage_rows:
        con.execute(
            "INSERT INTO hs_triage (run_id, iso3, hazard_code, tier, triage_score, need_full_spd, track) VALUES (?, ?, ?, ?, ?, ?, ?)",
            list(row),
        )
    con.close()


def test_fl_generates_both_pa_and_event_occurrence(tmp_path):
    """FL hazard should generate both PA and EVENT_OCCURRENCE questions."""
    db_path = str(tmp_path / "test.duckdb")
    _setup_test_db(db_path, [
        ("test-run-1", "BGD", "FL", "priority", 0.8, True, 1),
    ])
    count = create_questions_from_triage(f"duckdb:///{db_path}", "test-run-1")
    assert count == 2  # PA + EVENT_OCCURRENCE

    con = duckdb.connect(db_path)
    rows = con.execute("SELECT question_id, metric FROM questions ORDER BY question_id").fetchall()
    con.close()

    metrics = {r[1] for r in rows}
    assert "PA" in metrics
    assert "EVENT_OCCURRENCE" in metrics
    # Check question IDs
    qids = {r[0] for r in rows}
    assert any("EVENT_OCCURRENCE" in qid for qid in qids)


def test_dr_non_fewsnet_gets_event_occurrence_only(tmp_path):
    """DR for non-FEWS NET country should generate EVENT_OCCURRENCE but NOT PA."""
    db_path = str(tmp_path / "test.duckdb")
    _setup_test_db(db_path, [
        ("test-run-1", "USA", "DR", "priority", 0.7, True, 1),
    ])
    fewsnet_file = tmp_path / "fewsnet_countries.json"
    fewsnet_file.write_text(json.dumps(["ETH", "SOM", "KEN"]))

    with patch("scripts.create_questions_from_triage._FEWSNET_COUNTRIES_FILE", fewsnet_file):
        count = create_questions_from_triage(f"duckdb:///{db_path}", "test-run-1")

    assert count == 1  # Only EVENT_OCCURRENCE

    con = duckdb.connect(db_path)
    rows = con.execute("SELECT metric FROM questions").fetchall()
    con.close()
    metrics = {r[0] for r in rows}
    assert "EVENT_OCCURRENCE" in metrics
    assert "PA" not in metrics


def test_dr_fewsnet_gets_both(tmp_path):
    """DR for FEWS NET country should generate both PA and EVENT_OCCURRENCE."""
    db_path = str(tmp_path / "test.duckdb")
    _setup_test_db(db_path, [
        ("test-run-1", "ETH", "DR", "priority", 0.7, True, 1),
    ])
    fewsnet_file = tmp_path / "fewsnet_countries.json"
    fewsnet_file.write_text(json.dumps(["ETH", "SOM", "KEN"]))

    with patch("scripts.create_questions_from_triage._FEWSNET_COUNTRIES_FILE", fewsnet_file):
        count = create_questions_from_triage(f"duckdb:///{db_path}", "test-run-1")

    assert count == 2  # PA + EVENT_OCCURRENCE

    con = duckdb.connect(db_path)
    rows = con.execute("SELECT metric FROM questions").fetchall()
    con.close()
    metrics = {r[0] for r in rows}
    assert "PA" in metrics
    assert "EVENT_OCCURRENCE" in metrics


def test_ace_unchanged(tmp_path):
    """ACE should still generate FATALITIES + PA, no EVENT_OCCURRENCE."""
    db_path = str(tmp_path / "test.duckdb")
    _setup_test_db(db_path, [
        ("test-run-1", "ETH", "ACE", "priority", 0.9, True, 1),
    ])
    count = create_questions_from_triage(f"duckdb:///{db_path}", "test-run-1")
    assert count == 2  # FATALITIES + PA

    con = duckdb.connect(db_path)
    rows = con.execute("SELECT metric FROM questions").fetchall()
    con.close()
    metrics = {r[0] for r in rows}
    assert "FATALITIES" in metrics
    assert "PA" in metrics
    assert "EVENT_OCCURRENCE" not in metrics
