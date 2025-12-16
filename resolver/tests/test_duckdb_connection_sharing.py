# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import os

import pytest

from concurrent.futures import ThreadPoolExecutor

from resolver.db import conn_shared, duckdb_io

pytest.importorskip(
    "duckdb",
    reason=(
        "duckdb not installed. Install via extras: `pip install .[db]` or use "
        "`scripts/install_db_extra_offline.(sh|ps1)`"
    ),
)


def test_same_url_returns_same_connection(tmp_path, monkeypatch):
    db_path = tmp_path / "shared.duckdb"
    url = f"duckdb:///{db_path}"
    monkeypatch.setenv("RESOLVER_DB_URL", url)
    monkeypatch.delenv("RESOLVER_DISABLE_CONN_CACHE", raising=False)

    conn_w = duckdb_io.get_db(url)
    conn_w.execute(
        """
        CREATE TABLE IF NOT EXISTS facts_deltas (
            ym VARCHAR,
            iso3 VARCHAR,
            hazard_code VARCHAR,
            metric VARCHAR,
            value_new DOUBLE,
            as_of DATE
        )
        """
    )
    conn_w.execute(
        "INSERT INTO facts_deltas VALUES (?, ?, ?, ?, ?, ?)",
        ["2024-02", "PHL", "TC", "in_need", 500.0, "2024-02-28"],
    )

    conn_r = duckdb_io.get_db(url)

    cache_disabled = os.getenv("RESOLVER_DISABLE_CONN_CACHE") == "1"
    if cache_disabled:
        assert conn_w is not conn_r
    else:
        assert conn_w is conn_r
    count = conn_r.execute(
        "SELECT COUNT(*) FROM facts_deltas WHERE ym = ? AND iso3 = ? AND hazard_code = ?",
        ["2024-02", "PHL", "TC"],
    ).fetchone()[0]
    assert count == 1


def test_eviction_on_close(tmp_path, monkeypatch):
    db_path = tmp_path / "evict.duckdb"
    url = f"duckdb:///{db_path}"
    monkeypatch.setenv("RESOLVER_DB_URL", url)

    conn1 = duckdb_io.get_db(url)
    conn1.execute("CREATE TABLE IF NOT EXISTS t (x INTEGER)")
    conn1.close()

    conn2 = duckdb_io.get_db(url)
    assert conn1 is not conn2
    conn2.execute("INSERT INTO t VALUES (1)")
    assert conn2.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1


def test_healthcheck_reopen(tmp_path, monkeypatch):
    db_path = tmp_path / "health.duckdb"
    url = f"duckdb:///{db_path}"
    monkeypatch.setenv("RESOLVER_DB_URL", url)

    conn = duckdb_io.get_db(url)
    conn.execute("CREATE TABLE IF NOT EXISTS h (x INTEGER)")
    wrapper, _ = conn_shared.get_shared_duckdb_conn(url)
    wrapper._raw.close()  # type: ignore[attr-defined]
    wrapper._closed = False  # ensure wrapper looks open for the cache

    conn2 = duckdb_io.get_db(url)
    conn2.execute("INSERT INTO h VALUES (1)")
    assert conn2.execute("SELECT COUNT(*) FROM h").fetchone()[0] == 1


def test_nocache_returns_fresh_handles(tmp_path, monkeypatch):
    db_path = tmp_path / "nocache.duckdb"
    url = f"duckdb:///{db_path}"
    monkeypatch.setenv("RESOLVER_DB_URL", url)

    monkeypatch.setenv("RESOLVER_DISABLE_CONN_CACHE", "1")
    try:
        c1 = duckdb_io.get_db(url)
        c2 = duckdb_io.get_db(url)
        assert c1 is not c2
        c1.execute("SELECT 1").fetchone()
        c2.execute("SELECT 1").fetchone()
    finally:
        monkeypatch.delenv("RESOLVER_DISABLE_CONN_CACHE", raising=False)


def test_thread_local_mode_isolated_connections(tmp_path, monkeypatch):
    db_path = tmp_path / "threadcache.duckdb"
    url = f"duckdb:///{db_path}"
    monkeypatch.setenv("RESOLVER_DB_URL", url)
    monkeypatch.setenv("RESOLVER_CONN_CACHE_MODE", "thread")

    def _connect_and_ident() -> int:
        conn = duckdb_io.get_db(url)
        conn.execute("SELECT 42")
        return id(conn)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(_connect_and_ident) for _ in range(2)]
            conn_ids = [future.result() for future in futures]
    finally:
        monkeypatch.delenv("RESOLVER_CONN_CACHE_MODE", raising=False)

    assert len(set(conn_ids)) == 2
