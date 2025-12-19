# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import argparse
from typing import Iterable, List, Tuple

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
          COALESCE(phase, '') AS phase,
          COALESCE(provider, '') AS provider,
          COALESCE(model_id, '') AS model_id,
          COUNT(*) AS n_calls,
          quantile_cont(elapsed_ms, 0.5) AS p50_ms,
          quantile_cont(elapsed_ms, 0.95) AS p95_ms,
          avg(elapsed_ms) AS avg_ms
        FROM llm_calls
        WHERE elapsed_ms IS NOT NULL
          AND (error_text IS NULL OR error_text = '')
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
        """
    ).fetchall()


def _table_info_markdown(con: duckdb.DuckDBPyConnection) -> List[str]:
    try:
        info_rows = con.execute("PRAGMA table_info('llm_calls')").fetchall()
    except Exception as exc:  # pragma: no cover - defensive
        return [f"_Unable to read llm_calls schema: {exc}_"]

    lines = [
        "",
        "PRAGMA table_info('llm_calls'):",
        "",
        "| cid | name | type | notnull | dflt_value | pk |",
        "| --- | ---- | ---- | ------- | ---------- | -- |",
    ]
    if info_rows:
        for row in info_rows:
            cid, name, col_type, notnull, dflt_value, pk = row
            lines.append(f"| {cid} | {name} | {col_type} | {notnull} | {dflt_value} | {pk} |")
    else:
        lines.append("| (none) | (none) | (none) | (none) | (none) | (none) |")
    return lines


def render_latency_markdown(con: duckdb.DuckDBPyConnection) -> str:
    required_cols = {"elapsed_ms", "phase", "provider", "model_id", "error_text"}
    lines: List[str] = []

    lines.append("### LLM latency p50/p95 (phase/provider/model_id)")
    lines.append("")

    table_missing = not _table_exists(con, "llm_calls")
    if table_missing:
        lines.append("_llm_calls table not found; latency summary unavailable._")
        lines.extend(_table_info_markdown(con))
        return "\n".join(lines)

    if not _has_columns(con, "llm_calls", required_cols):
        cols = con.execute("PRAGMA table_info('llm_calls')").fetchall()
        present = {str(row[1]).lower() for row in cols}
        missing_cols = sorted(required_cols - present)
        lines.append("_llm_calls is missing required columns: " + ", ".join(missing_cols) + "_")
        lines.extend(_table_info_markdown(con))
        return "\n".join(lines)

    rows = _query_latency(con)
    lines.append("| phase | provider | model_id | n_calls | p50_ms | p95_ms | avg_ms |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    if rows:
        for phase, provider, model_id, n_calls, p50_ms, p95_ms, avg_ms in rows:
            lines.append(
                f"| {phase} | {provider} | {model_id} | "
                f"{int(n_calls)} | {p50_ms:.2f} | {p95_ms:.2f} | {avg_ms:.2f} |"
            )
    else:
        lines.append("| (none) | (none) | (none) | 0 | 0 | 0 | 0 |")
        lines.append("")
        lines.append(
            "_Note: No llm_calls rows with elapsed_ms and empty error_text; see schema below._"
        )
        lines.extend(_table_info_markdown(con))
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    con = duckdb_io.get_db(args.db)
    try:
        markdown = render_latency_markdown(con)
    finally:
        duckdb_io.close_db(con)

    print(markdown)


if __name__ == "__main__":
    main()
