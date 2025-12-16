# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


duckdb = pytest.importorskip("duckdb")


def _load_verify_duckdb_counts():
    """
    Import scripts/ci/verify_duckdb_counts.py directly from the repo tree,
    regardless of how the package is installed.
    """
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "scripts" / "ci" / "verify_duckdb_counts.py"
    assert path.exists(), f"verify_duckdb_counts.py not found at {path}"
    spec = importlib.util.spec_from_file_location("verify_duckdb_counts", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return mod


def test_fetch_breakdown_with_source_column(tmp_path: Path) -> None:
    """facts_resolved with a direct 'source' column should work and appear in breakdown."""
    vdc = _load_verify_duckdb_counts()
    con = duckdb.connect(database=":memory:")
    con.execute(
        """
        CREATE TABLE facts_resolved (
            iso3 TEXT,
            metric TEXT,
            semantics TEXT,
            source TEXT
        );
        """
    )
    con.execute(
        """
        INSERT INTO facts_resolved VALUES
            ('UKR', 'affected', 'stock', 'emdat'),
            ('UKR', 'affected', 'stock', 'emdat');
        """
    )

    rows = vdc._fetch_breakdown(con)  # type: ignore[attr-defined]
    assert any(
        r[0] == "facts_resolved" and r[1] == "emdat" and r[2] == "affected"
        for r in rows
    )


def test_fetch_breakdown_with_source_id_only(tmp_path: Path) -> None:
    """
    A table with only 'source_id' but no 'source' should still work and use
    source_id as a fallback for the source key.
    """
    vdc = _load_verify_duckdb_counts()
    con = duckdb.connect(database=":memory:")
    con.execute(
        """
        CREATE TABLE facts_deltas (
            iso3 TEXT,
            metric TEXT,
            semantics TEXT,
            source_id TEXT
        );
        """
    )
    con.execute(
        """
        INSERT INTO facts_deltas VALUES
            ('UKR', 'new_displacements', 'new', 'IDMC');
        """
    )

    rows = vdc._fetch_breakdown(con)  # type: ignore[attr-defined]
    assert any(
        r[0] == "facts_deltas" and r[1] == "IDMC" and r[2] == "new_displacements"
        for r in rows
    )


def test_fetch_breakdown_includes_acled_without_source_column(tmp_path: Path) -> None:
    """
    acled_monthly_fatalities has no 'source' column; breakdown should still
    work and label source as 'ACLED'.
    """
    vdc = _load_verify_duckdb_counts()
    con = duckdb.connect(database=":memory:")
    con.execute(
        """
        CREATE TABLE acled_monthly_fatalities (
            iso3 TEXT,
            month DATE,
            fatalities DOUBLE,
            source_id TEXT,
            series TEXT
        );
        """
    )
    con.execute(
        """
        INSERT INTO acled_monthly_fatalities VALUES
            ('AFG', DATE '2025-10-01', 123.0, 'acled_api', 'fatalities_battle_month');
        """
    )

    rows = vdc._fetch_breakdown(con)  # type: ignore[attr-defined]
    assert any(
        r[0] == "acled_monthly_fatalities" and r[1] == "ACLED"
        for r in rows
    )
