from __future__ import annotations

import os

import pytest

from resolver.db import duckdb_io
from resolver.db.conn_shared import clear_all_cached_connections


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_all_cached_connections()
    yield
    clear_all_cached_connections()


def test_duckdb_path_canonicalization(tmp_path, monkeypatch):
    db_file = tmp_path / "canonical" / "state.duckdb"
    db_file.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb_io.get_db(str(db_file))
    conn.execute("CREATE TABLE IF NOT EXISTS t (value INTEGER)")
    conn.execute("INSERT INTO t VALUES (1), (2)")
    conn.close()

    assert db_file.exists(), "duckdb should materialise on first write"

    monkeypatch.setenv("RESOLVER_DB_URL", f"duckdb:///{db_file}")
    reopened = duckdb_io.get_db()
    count = reopened.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    assert count == 2

    resolved_path = getattr(reopened, "_path", None)
    assert resolved_path is not None
    assert os.path.samefile(resolved_path, db_file)

    reopened.close()
