#!/usr/bin/env python3
"""Post-run DuckDB diagnostics for Horizon Scanner + Forecaster outputs."""
from __future__ import annotations

import argparse
import os
from typing import Iterable, Tuple

from resolver.db import duckdb_io


def _print_table_counts(conn, tables: Iterable[Tuple[str, str]]) -> None:
    for label, query in tables:
        try:
            count = conn.execute(query).fetchone()[0]
            print(f"- {label}: {int(count)} rows")
        except Exception as exc:  # pragma: no cover - diagnostics only
            print(f"- {label}: unavailable ({type(exc).__name__}: {exc})")


def _print_forecast_details(conn) -> None:
    try:
        n_ens = conn.execute("SELECT COUNT(*) FROM forecasts_ensemble").fetchone()[0]
        n_raw = conn.execute("SELECT COUNT(*) FROM forecasts_raw").fetchone()[0]
        print(f"- forecasts_ensemble: {int(n_ens)} rows")
        print(f"- forecasts_raw: {int(n_raw)} rows")

        rows = conn.execute(
            """
            SELECT question_id, COUNT(*) AS n_rows
            FROM forecasts_ensemble
            GROUP BY question_id
            ORDER BY question_id
            LIMIT 20
            """
        ).fetchall()
        print("### forecasts_ensemble per question_id (first 20)")
        if rows:
            for qid, n_rows in rows:
                print(f"- {qid}: {int(n_rows)} rows")
        else:
            print("- <no rows>")
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"[warn] Could not inspect forecasts tables: {type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize post-run DuckDB contents.")
    parser.add_argument(
        "--db-url",
        dest="db_url",
        default=None,
        help="DuckDB URL or path (defaults to RESOLVER_DB_URL or duckdb:///data/resolver.duckdb)",
    )
    args = parser.parse_args()

    fallback_db = os.environ.get("RESOLVER_DB_URL", "duckdb:///data/resolver.duckdb")
    db_url = args.db_url or fallback_db

    conn = duckdb_io.get_db(db_url)
    try:
        print("### DuckDB post-run diagnostics")
        print(f"- db_url: {db_url}")

        _print_table_counts(
            conn,
            (
                ("hs_runs", "SELECT COUNT(*) FROM hs_runs"),
                ("hs_scenarios", "SELECT COUNT(*) FROM hs_scenarios"),
                ("questions", "SELECT COUNT(*) FROM questions"),
            ),
        )
        _print_forecast_details(conn)
    finally:
        duckdb_io.close_db(conn)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
