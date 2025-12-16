# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for canonical DuckDB URL and path handling."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("duckdb")

from resolver.db import duckdb_io


def test_duckdb_path_canonicalization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    db_path = workdir / "resolver.duckdb"

    monkeypatch.chdir(workdir)
    # Provide an env URL pointing at the same file using DuckDB scheme.
    monkeypatch.setenv("RESOLVER_DB_URL", f"duckdb:///{db_path}")

    # Open via a relative filesystem path first to ensure canonicalisation.
    conn_explicit = duckdb_io.get_db("resolver.duckdb")
    duckdb_io.init_schema(conn_explicit)
    conn_explicit.execute(
        "INSERT INTO manifests(path, sha256) VALUES ('example', 'abc123')"
    )

    # Opening via the environment URL should reuse the same canonical connection.
    conn_env = duckdb_io.get_db(None)
    try:
        assert conn_env is conn_explicit
        rows = conn_env.execute(
            "SELECT COUNT(*) FROM manifests WHERE path = 'example'"
        ).fetchone()[0]
        assert rows == 1
    finally:
        conn_env.close()
