# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")
if getattr(duckdb, "__pythia_stub__", False):
    pytest.skip("duckdb not installed", allow_module_level=True)

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
        assert {"requested_countries_json", "skipped_entries_json"}.issubset(hs_runs_cols)
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


def test_connect_normalises_read_only_for_file_backed_db(monkeypatch, tmp_path):
    """For file-backed DBs, connect() should always open read-write."""
    db_path = tmp_path / "test_duckdb_config.duckdb"
    url = f"duckdb:///{db_path}"

    def fake_get_db_url() -> str:
        return url

    monkeypatch.setattr(db_schema, "get_db_url", fake_get_db_url)

    calls: list[tuple[str, bool]] = []

    def fake_duckdb_connect(path: str, read_only: bool = False):
        calls.append((path, read_only))

        class DummyConn:
            def execute(self, *args, **kwargs):
                return self

            def fetchall(self):
                return []

            def close(self):
                pass

        return DummyConn()

    monkeypatch.setattr(db_schema.duckdb, "connect", fake_duckdb_connect)

    con1 = db_schema.connect(read_only=False)
    con2 = db_schema.connect(read_only=True)

    assert con1 is not None
    assert con2 is not None

    assert all(call[0] == str(db_path) for call in calls)
    assert all(call[1] is False for call in calls), calls
