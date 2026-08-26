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


def test_compare_optional_signatures_warns_never_blindly(tmp_path: Path) -> None:
    """Optional regressions are reported; legitimate absence is not.

    Before this existed the optional half of the signature was computed,
    written and never read — the haz_* tables could drop to zero and the
    compare still exited 0.
    """

    db_path = tmp_path / "resolver.duckdb"
    _setup_db(db_path)

    conn = duckdb_io.get_db(str(db_path))
    try:
        conn.execute("CREATE TABLE haz_triggers (id INTEGER)")
        conn.execute("INSERT INTO haz_triggers VALUES (1), (2), (3)")
    finally:
        duckdb_io.close_db(conn)

    optional = ["haz_triggers", "haz_resolutions"]
    before = db_signature.compute_signature(db_path, ["questions"], optional)

    # A shrink in a present optional table is a warning...
    conn = duckdb_io.get_db(str(db_path))
    try:
        conn.execute("DELETE FROM haz_triggers WHERE id >= 2")
    finally:
        duckdb_io.close_db(conn)
    shrunk = db_signature.compute_signature(db_path, ["questions"], optional)
    warnings = db_signature.compare_optional_signatures(before, shrunk, optional)
    assert warnings == ["haz_triggers: 1 < 3"]

    # ...a table the baseline never had is NOT (haz_resolutions is absent in
    # both signatures — an older DB predating the feature must stay quiet).
    assert not any("haz_resolutions" in w for w in warnings)

    # A present-before, missing-after table IS a warning.
    conn = duckdb_io.get_db(str(db_path))
    try:
        conn.execute("DROP TABLE haz_triggers")
    finally:
        duckdb_io.close_db(conn)
    dropped = db_signature.compute_signature(db_path, ["questions"], optional)
    warnings = db_signature.compare_optional_signatures(before, dropped, optional)
    assert warnings == ["haz_triggers: present before (3 rows), now missing"]

    # And none of it is fatal: compare_signatures (the required check) still
    # returns no regressions for this DB.
    assert db_signature.compare_signatures(before, dropped, ["questions"]) == []
