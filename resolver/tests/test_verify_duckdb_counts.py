from __future__ import annotations

import importlib.util
import subprocess
import sys
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


def test_resolver_db_url_used_when_no_cli_path(tmp_path, monkeypatch):
    """
    Mirrors the expectations of test_exporter_db_parity_smoke.test_verify_duckdb_counts_writes_markdown:

    - Create a DuckDB file at tmp_path / 'resolver.duckdb'.
    - Set RESOLVER_DB_URL to duckdb:///<that-path>.
    - Run 'python -m scripts.ci.verify_duckdb_counts' with no db_path arg.
    - Expect exit code 0 (no CalledProcessError).
    """

    db_path = tmp_path / "resolver.duckdb"
    con = duckdb.connect(db_path)
    con.execute(
        """
        CREATE TABLE facts_resolved (
            source VARCHAR,
            metric VARCHAR,
            series_semantics VARCHAR,
            value INTEGER
        );
        """
    )
    con.execute(
        "INSERT INTO facts_resolved VALUES (?, ?, ?, ?)",
        ("idmc", "affected", "stock", 10),
    )
    con.close()

    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo_root))
    monkeypatch.chdir(tmp_path)

    diagnostics = Path("diagnostics/ingestion")
    diagnostics.mkdir(parents=True, exist_ok=True)
    summary_file = diagnostics / "summary.md"
    summary_file.write_text("## Existing summary\n", encoding="utf-8")

    step_summary = tmp_path / "step-summary.md"

    monkeypatch.setenv("RESOLVER_DB_URL", f"duckdb:///{db_path}")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(step_summary))

    subprocess.run(
        [sys.executable, "-m", "scripts.ci.verify_duckdb_counts"],
        check=True,
    )

    text = summary_file.read_text(encoding="utf-8")
    assert "DuckDB" in text or "rows" in text
