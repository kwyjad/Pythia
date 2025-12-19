# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import argparse
from typing import Iterable, List, Sequence, Tuple

import duckdb
from resolver.db import duckdb_io


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print LLM call latency p50/p95 grouped by (phase, provider, model)."
    )
    parser.add_argument(
        "--db",
        required=True,
        help='Database URL, e.g. "duckdb:///data/resolver.duckdb".',
    )
    return parser.parse_args()


def _table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    row = con.execute(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE lower(table_name) = lower(?)
        LIMIT 1
        """,
        [table_name],
    ).fetchone()
    return bool(row and row[0])


def _has_columns(
    con: duckdb.DuckDBPyConnection, table_name: str, required: Iterable[str]
) -> bool:
    cols = con.execute(
        """
        SELECT lower(column_name)
        FROM information_schema.columns
        WHERE lower(table_name) = lower(?)
        """,
        [table_name],
    ).fetchall()
    present = {c[0] for c in cols}
    return all(col.lower() in present for col in required)


def _query_latency(
    con: duckdb.DuckDBPyConnection,
) -> List[Tuple[str, str, str, int, float, float, float]]:
    return con.execute(
        """
        SELECT
          phase,
          CASE
            WHEN lower(model_name) LIKE 'gpt-%' THEN 'openai'
            WHEN lower(model_name) LIKE 'claude-%' THEN 'anthropic'
            WHEN lower(model_name) LIKE 'gemini-%' THEN 'google'
            WHEN lower(model_name) LIKE 'grok-%' THEN 'xai'
            ELSE 'unknown'
          END AS provider,
          model_name,
          COUNT(*) AS n_calls,
          quantile_cont(latency_ms, 0.5) AS p50_ms,
          quantile_cont(latency_ms, 0.95) AS p95_ms,
          avg(latency_ms) AS avg_ms
        FROM llm_calls
        WHERE success = TRUE
          AND latency_ms IS NOT NULL
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
        """
    ).fetchall()


def _print_markdown_table(rows: Sequence[Tuple[str, str, str, int, float, float, float]]) -> None:
    print("### LLM latency p50/p95 by phase/provider/model")
    print("| phase | provider | model_name | n_calls | p50_ms | p95_ms | avg_ms |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    if not rows:
        print("| (none) | (none) | (none) | 0 | 0 | 0 | 0 |")
        return
    for phase, provider, model_name, n_calls, p50_ms, p95_ms, avg_ms in rows:
        print(
            f"| {phase} | {provider} | {model_name} | "
            f"{int(n_calls)} | {p50_ms:.2f} | {p95_ms:.2f} | {avg_ms:.2f} |"
        )


def main() -> None:
    args = _parse_args()
    con = duckdb_io.get_db(args.db)
    try:
        if not _table_exists(con, "llm_calls"):
            raise SystemExit("llm_calls table not found.")
        required_cols = {"latency_ms", "phase", "model_name", "success"}
        if not _has_columns(con, "llm_calls", required_cols):
            missing = ", ".join(sorted(required_cols))
            raise SystemExit(f"llm_calls is missing required columns: {missing}")

        rows = _query_latency(con)
    finally:
        duckdb_io.close_db(con)

    _print_markdown_table(rows)


if __name__ == "__main__":
    main()
