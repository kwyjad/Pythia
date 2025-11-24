from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")


def _load_verify_duckdb_counts():
    """
    Import scripts/ci/verify_duckdb_counts.py in a way that works
    whether or not 'scripts' is a Python package.
    """
    try:
        from scripts.ci import verify_duckdb_counts as mod  # type: ignore
        return mod
    except Exception:
        root = Path(__file__).resolve().parents[2]
        path = root / "scripts" / "ci" / "verify_duckdb_counts.py"
        spec = importlib.util.spec_from_file_location("verify_duckdb_counts", path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[arg-type]
        return mod


def test_fetch_breakdown_with_source_column():
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
            ('UKR', 'idp_displacement_stock_dtm', 'stock', 'IOM DTM');
        """
    )

    breakdown = vdc._fetch_breakdown(con)  # type: ignore[attr-defined]
    assert any(
        row[0] == "facts_resolved" and row[1] == "IOM DTM" for row in breakdown
    )


def test_fetch_breakdown_with_source_id_only():
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
            ('UKR', 'idp_displacement_stock_dtm', 'new', 'IOM DTM');
        """
    )

    breakdown = vdc._fetch_breakdown(con)  # type: ignore[attr-defined]
    assert any(
        row[0] == "facts_deltas" and row[1] == "IOM DTM" for row in breakdown
    )


def test_fetch_breakdown_includes_acled_constant_source():
    vdc = _load_verify_duckdb_counts()
    con = duckdb.connect(database=":memory:")
    con.execute(
        """
        CREATE TABLE acled_monthly_fatalities (
            iso3 TEXT,
            month DATE,
            fatalities DOUBLE
        );
        """
    )
    con.execute(
        """
        INSERT INTO acled_monthly_fatalities VALUES
            ('AFG', DATE '2025-10-01', 42.0);
        """
    )

    breakdown = vdc._fetch_breakdown(con)  # type: ignore[attr-defined]
    assert any(
        row[0] == "acled_monthly_fatalities" and row[1] == "ACLED"
        for row in breakdown
    )
