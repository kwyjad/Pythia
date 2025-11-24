import argparse
import os
from typing import List, Tuple

import pytest

duckdb = pytest.importorskip("duckdb")

from scripts.ci.verify_duckdb_counts import _fetch_breakdown, _resolve_db_path


def test_fetch_breakdown_handles_null_source_and_group_by_alias_safely() -> None:
    con = duckdb.connect(":memory:")
    try:
        con.execute(
            """
            CREATE TABLE facts_resolved (
                source VARCHAR,
                metric VARCHAR,
                series_semantics VARCHAR
            );
            """
        )
        con.execute(
            "INSERT INTO facts_resolved VALUES "
            "(NULL, 'idp_displacement_stock_dtm', 'stock'),"
            "('IOM DTM', 'idp_displacement_stock_dtm', 'stock'),"
            "('emdat', 'affected', 'new');"
        )

        rows: List[Tuple[str, str, str, int]] = _fetch_breakdown(con)
        result = {(src, metric, semantics): count for src, metric, semantics, count in rows}

        assert result[("", "idp_displacement_stock_dtm", "stock")] == 1
        assert result[("IOM DTM", "idp_displacement_stock_dtm", "stock")] == 1
        assert result[("emdat", "affected", "new")] == 1

        assert rows == [
            ("", "idp_displacement_stock_dtm", "stock", 1),
            ("IOM DTM", "idp_displacement_stock_dtm", "stock", 1),
            ("emdat", "affected", "new", 1),
        ]
    finally:
        con.close()


def test_resolve_db_path_prefers_env_duckdb_url(monkeypatch, tmp_path) -> None:
    target = tmp_path / "resolver.duckdb"
    monkeypatch.setenv("RESOLVER_DB_URL", f"duckdb:///{target}")
    args = argparse.Namespace(db_path=None, tables=None, allow_missing=False)

    resolved = _resolve_db_path(args)

    assert resolved == str(target)
    monkeypatch.delenv("RESOLVER_DB_URL", raising=False)


def test_resolve_db_path_defaults_to_data_resolver(monkeypatch) -> None:
    monkeypatch.delenv("RESOLVER_DB_URL", raising=False)
    args = argparse.Namespace(db_path=None, tables=None, allow_missing=False)

    resolved = _resolve_db_path(args)

    assert resolved.endswith(os.path.join("data", "resolver.duckdb"))
