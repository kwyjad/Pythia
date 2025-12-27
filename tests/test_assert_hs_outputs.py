from __future__ import annotations

import os
from pathlib import Path

import pytest

try:  # pragma: no cover - skip guard for environments without DuckDB
    import duckdb  # noqa: F401
except ModuleNotFoundError:
    pytest.skip("duckdb not installed", allow_module_level=True)

from pythia.db.schema import ensure_schema
from resolver.db import duckdb_io
from scripts.ci import assert_hs_outputs


def _make_db(tmp_path: Path) -> str:
    db_path = tmp_path / "test.duckdb"
    db_url = f"duckdb:///{db_path}"
    con = duckdb_io.get_db(db_url)
    try:
        ensure_schema(con)
    finally:
        duckdb_io.close_db(con)
    return db_url


def test_assert_hs_outputs_triage_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_url = _make_db(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HS_RESOLVED_ISO3S", "AAA,BBB")

    exit_code = assert_hs_outputs.run_assertion(db_url, "hs_test", "triage")
    assert exit_code == 2

    diag_path = tmp_path / "diagnostics" / "hs_assertion.md"
    assert diag_path.exists()
    content = diag_path.read_text()
    assert "hs_triage count: 0" in content
    assert "AAA" in content and "BBB" in content


def test_assert_hs_outputs_questions_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_url = _make_db(tmp_path)
    con = duckdb_io.get_db(db_url)
    try:
        con.execute(
            """
            INSERT INTO hs_triage (
                run_id, iso3, hazard_code, tier, triage_score, need_full_spd,
                drivers_json, regime_shifts_json, data_quality_json, scenario_stub
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "hs_test",
                "USA",
                "FL",
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
        duckdb_io.close_db(con)

    monkeypatch.chdir(tmp_path)
    exit_code = assert_hs_outputs.run_assertion(db_url, "hs_test", "questions")
    assert exit_code == 3

    diag_path = tmp_path / "diagnostics" / "hs_assertion.md"
    assert diag_path.exists()
    content = diag_path.read_text()
    assert "questions count: 0" in content
    assert "eligible hs_triage rows: 1" in content
    assert "USA" in content


def test_assert_hs_outputs_questions_expected_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    db_url = _make_db(tmp_path)
    con = duckdb_io.get_db(db_url)
    try:
        con.execute(
            """
            INSERT INTO hs_triage (
                run_id, iso3, hazard_code, tier, triage_score, need_full_spd,
                drivers_json, regime_shifts_json, data_quality_json, scenario_stub
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "hs_test",
                "USA",
                "FL",
                "watchlist",
                0.1,
                False,
                "[]",
                "[]",
                "{}",
                "",
            ],
        )
    finally:
        duckdb_io.close_db(con)

    monkeypatch.chdir(tmp_path)
    exit_code = assert_hs_outputs.run_assertion(db_url, "hs_test", "questions")
    assert exit_code == 0

    out = capsys.readouterr().out
    assert (
        "::warning::HS produced 0 eligible hazards; 0 questions expected (skipping forecaster)."
        in out
    )

    diag_path = tmp_path / "diagnostics" / "hs_assertion.md"
    assert diag_path.exists()
    content = diag_path.read_text()
    assert "questions rows: 0" in content
    assert "eligible hs_triage rows: 0" in content
    assert "0 eligible hazards → 0 questions expected" in content
