from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")

from pythia.db import schema as db_schema
from pythia.db.schema import ensure_schema


def _columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    return {str(row[1]).lower() for row in con.execute(f"PRAGMA table_info('{table}')").fetchall()}


def test_ensure_schema_creates_tables(tmp_path, monkeypatch):
    db_path = tmp_path / "resolver_created.duckdb"
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db_path}")

    ensure_schema()

    con = duckdb.connect(str(db_path))
    try:
        hs_runs_cols = _columns(con, "hs_runs")
        forecasts_raw_cols = _columns(con, "forecasts_raw")

        assert {"hs_run_id", "generated_at", "countries_json"}.issubset(hs_runs_cols)
        assert {"run_id", "question_id", "model_name", "probability"}.issubset(
            forecasts_raw_cols
        )
    finally:
        con.close()


def test_ensure_schema_adds_missing_columns(tmp_path):
    db_path = tmp_path / "resolver_missing_cols.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE TABLE questions (question_id TEXT)")
        ensure_schema(con)
        ensure_schema(con)

        question_cols = _columns(con, "questions")
        assert "pythia_metadata_json" in question_cols
        assert "hs_run_id" in question_cols
    finally:
        con.close()


def test_get_db_url_prefers_env_over_config(monkeypatch):
    """Environment variable PYTHIA_DB_URL should override config app.db_url."""

    def fake_load_config():
        return {
            "app": {
                "db_url": "duckdb:///data/from-config.duckdb",
            }
        }

    monkeypatch.setattr(db_schema, "load_config", fake_load_config)

    env_url = "duckdb:///tmp/pythia-env-test.duckdb"
    monkeypatch.setenv("PYTHIA_DB_URL", env_url)

    url = db_schema.get_db_url()
    assert url == env_url
