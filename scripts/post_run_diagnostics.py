from __future__ import annotations

"""Quick, stable post-run diagnostics for Pythia pipeline outputs."""

import argparse
from typing import Iterable

from resolver.db import duckdb_io

DEFAULT_DB_URL = "duckdb:///data/resolver.duckdb"


def _print_count(conn, table: str) -> None:
    try:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"Table {table}: {int(count)} rows")
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"[warn] Could not count table {table}: {type(exc).__name__}: {exc}")


def _print_forecast_breakdown(conn) -> None:
    try:
        n_ens = conn.execute("SELECT COUNT(*) FROM forecasts_ensemble").fetchone()[0]
        n_raw = conn.execute("SELECT COUNT(*) FROM forecasts_raw").fetchone()[0]
        print(f"Table forecasts_ensemble: {int(n_ens)} rows")
        print(f"Table forecasts_raw: {int(n_raw)} rows")
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"[warn] Could not inspect forecasts tables: {exc}")
        return

    try:
        rows: Iterable[tuple[str, int]] = conn.execute(
            """
            SELECT question_id, COUNT(*) AS n_rows
            FROM forecasts_ensemble
            GROUP BY question_id
            ORDER BY question_id
            LIMIT 20
            """
        ).fetchall()
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"[warn] Could not summarize forecasts_ensemble: {exc}")
        return

    rows = list(rows)
    if not rows:
        print("No rows in forecasts_ensemble for this run.")
        return

    print("Forecasts per question_id (first 20):")
    for question_id, n_rows in rows:
        print(f"  {question_id}: {int(n_rows)} rows")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_URL,
        help="DuckDB path or duckdb:/// URL (default: duckdb:///data/resolver.duckdb)",
    )
    args = parser.parse_args(argv)

    conn = None
    try:
        conn = duckdb_io.get_db(args.db)
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"[warn] Could not open DuckDB at {args.db}: {type(exc).__name__}: {exc}")
        return 0

    try:
        _print_count(conn, "hs_runs")
        _print_count(conn, "hs_scenarios")
        _print_count(conn, "questions")
        _print_forecast_breakdown(conn)
    finally:
        duckdb_io.close_db(conn)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
