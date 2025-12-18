# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ci import db_signature
from resolver.db import duckdb_io
from resolver.db._duckdb_available import DUCKDB_AVAILABLE

pytestmark = pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="duckdb not installed")


def _setup_db(db_path: Path) -> None:
    conn = duckdb_io.get_db(str(db_path))
    try:
        conn.execute("CREATE TABLE questions (id INTEGER)")
        conn.execute("INSERT INTO questions VALUES (1), (2)")
        conn.execute("CREATE TABLE llm_calls (id INTEGER)")
        conn.execute("INSERT INTO llm_calls VALUES (10)")
    finally:
        duckdb_io.close_db(conn)


def test_parse_table_list_handles_empty_and_spaces() -> None:
    assert db_signature.parse_table_list("") == []
    assert db_signature.parse_table_list(" questions , llm_calls , ") == ["questions", "llm_calls"]


def test_compare_signatures_detects_regression(tmp_path: Path) -> None:
    db_path = tmp_path / "resolver.duckdb"
    _setup_db(db_path)

    required = ["questions", "llm_calls"]
    optional = ["scenarios"]

    before = db_signature.compute_signature(db_path, required, optional)

    conn = duckdb_io.get_db(str(db_path))
    try:
        conn.execute("INSERT INTO questions VALUES (3)")
        conn.execute("INSERT INTO llm_calls VALUES (11)")
    finally:
        duckdb_io.close_db(conn)

    after = db_signature.compute_signature(db_path, required, optional)
    assert db_signature.compare_signatures(before, after, required) == []

    conn = duckdb_io.get_db(str(db_path))
    try:
        conn.execute("DELETE FROM questions WHERE id >= 2")
    finally:
        duckdb_io.close_db(conn)

    regressed = db_signature.compute_signature(db_path, required, optional)
    regressions = db_signature.compare_signatures(before, regressed, required)
    assert regressions == ["questions: 1 < 2"] or regressions == ["questions: 1.0 < 2"]

    out_path = tmp_path / "signature.json"
    db_signature.write_signature(before, out_path)
    assert out_path.exists()
