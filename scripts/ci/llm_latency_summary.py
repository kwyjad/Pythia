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
    parser.add_argument(
        "--hs-run-id",
        help="If provided, scope HS triage latency to this hs_run_id (phase = hs_triage).",
    )
    parser.add_argument(
        "--forecaster-run-id",
        help="If provided, scope Research/Scenario/SPD latency to this forecaster run id.",
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
    predicate: str,
    params: Sequence,
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
          AND ({predicate})
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
        """.format(
            predicate=predicate or "1=1"
        ),
        params,
    ).fetchall()


def _top_slow_calls(
    con: duckdb.DuckDBPyConnection,
    predicate: str,
    params: Sequence,
    limit: int = 10,
) -> List[Tuple[str, str, str, str, int]]:
    return con.execute(
        f"""
        SELECT
          COALESCE(phase, '') AS phase,
          COALESCE(provider, '') AS provider,
          COALESCE(model_id, '') AS model_id,
          COALESCE(question_id, '') AS question_id,
          COALESCE(elapsed_ms, 0) AS elapsed_ms
        FROM llm_calls
        WHERE elapsed_ms IS NOT NULL
          AND ({predicate})
        ORDER BY elapsed_ms DESC
        LIMIT {int(limit)}
        """,
        params,
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


def render_latency_markdown(
    con: duckdb.DuckDBPyConnection,
    predicate: str,
    params: Sequence,
    strategy_label: str | None = None,
) -> str:
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

    params = list(params or [])
    predicate_sql = predicate or "1=1"
    total_rows = con.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0]
    filtered_rows = (
        con.execute(f"SELECT COUNT(*) FROM llm_calls WHERE {predicate_sql}", params).fetchone()[0]
        if predicate
        else total_rows
    )
    rows = _query_latency(con, predicate_sql, params)
    slow_rows = _top_slow_calls(con, predicate_sql, params)
    if strategy_label:
        lines.append(f"_Filter strategy: {strategy_label}_")
        lines.append("")

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
            "_Note: No llm_calls rows matched; see diagnostics below for context and schema._"
        )
        lines.append("")
        lines.append(f"- Total llm_calls rows: {total_rows}")
        lines.append(f"- Rows after filter: {filtered_rows}")
        if strategy_label:
            lines.append(f"- Filter strategy: {strategy_label}")
        lines.append("")
        lines.extend(_table_info_markdown(con))
        return "\n".join(lines)

    lines.append("")
    lines.append("Top slow LLM calls (elapsed_ms desc):")
    lines.append("")
    lines.append("| phase | provider | model_id | question_id | elapsed_ms |")
    lines.append("| --- | --- | --- | --- | --- |")
    if slow_rows:
        for phase, provider, model_id, question_id, elapsed_ms in slow_rows:
            lines.append(f"| {phase} | {provider} | {model_id} | {question_id} | {elapsed_ms} |")
    else:
        lines.append("| (none) | (none) | (none) | (none) | 0 |")

    return "\n".join(lines)


def _build_predicate(args: argparse.Namespace) -> tuple[str, list[str], str]:
    clauses: list[str] = []
    params: list[str] = []
    strategy_parts: list[str] = []

    if args.hs_run_id:
        clauses.append("(phase = 'hs_triage' AND hs_run_id = ?)")
        params.append(args.hs_run_id)
        strategy_parts.append("hs_run_id")

    if args.forecaster_run_id:
        clauses.append("(phase <> 'hs_triage' AND run_id = ?)")
        params.append(args.forecaster_run_id)
        strategy_parts.append("run_id")

    if not clauses:
        raise SystemExit("At least one of --hs-run-id or --forecaster-run-id is required.")

    return " OR ".join(clauses), params, ", ".join(strategy_parts)


def main() -> None:
    args = _parse_args()
    predicate, params, strategy_label = _build_predicate(args)
    con = duckdb_io.get_db(args.db)
    try:
        markdown = render_latency_markdown(con, predicate, params, strategy_label=strategy_label)
    finally:
        duckdb_io.close_db(con)

    print(markdown)


if __name__ == "__main__":
    main()
