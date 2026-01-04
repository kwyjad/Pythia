#!/usr/bin/env python3
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Backfill llm_calls telemetry columns in DuckDB."""

from __future__ import annotations

import argparse
import json
from typing import Any

from pythia.db.schema import ensure_schema
from pythia.db.util import (
    derive_response_format,
    ensure_llm_calls_columns,
)
from resolver.db import duckdb_io
from resolver.query.debug_ui import _extract_hazard_scores


def _build_where_clauses(run_id: str | None, hs_run_id: str | None) -> tuple[str, list[Any]]:
    clauses = []
    params: list[Any] = []
    if run_id:
        clauses.append("run_id = ?")
        params.append(run_id)
    if hs_run_id:
        clauses.append("hs_run_id = ?")
        params.append(hs_run_id)
    where_sql = ""
    if clauses:
        where_sql = " AND " + " AND ".join(clauses)
    return where_sql, params


def _count_rows(conn, sql: str, params: list[Any]) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0]) if row else 0


def backfill_llm_calls_telemetry(
    db_url: str,
    *,
    run_id: str | None = None,
    hs_run_id: str | None = None,
) -> dict[str, Any]:
    conn = duckdb_io.get_db(db_url)
    summary: dict[str, Any] = {
        "columns_ensured": [
            "status",
            "error_type",
            "error_message",
            "response_format",
            "hazard_scores_json",
            "hazard_scores_parse_ok",
        ],
        "status_updated": 0,
        "error_type_updated": 0,
        "error_message_updated": 0,
        "response_format_updated": 0,
        "hazard_scores_filled": 0,
    }
    try:
        ensure_schema(conn)
        ensure_llm_calls_columns(conn)

        where_sql, params = _build_where_clauses(run_id, hs_run_id)

        status_filter = f"WHERE (status IS NULL OR TRIM(status) = ''){where_sql}"
        summary["status_updated"] = _count_rows(
            conn,
            f"SELECT COUNT(*) FROM llm_calls {status_filter}",
            params,
        )
        conn.execute(
            f"""
            UPDATE llm_calls
            SET status = CASE
                WHEN error_text IS NULL OR TRIM(error_text) = '' THEN 'ok'
                ELSE 'error'
            END
            {status_filter}
            """,
            params,
        )

        error_type_filter = f"WHERE (error_type IS NULL OR TRIM(error_type) = ''){where_sql}"
        summary["error_type_updated"] = _count_rows(
            conn,
            f"SELECT COUNT(*) FROM llm_calls {error_type_filter}",
            params,
        )
        conn.execute(
            f"""
            UPDATE llm_calls
            SET error_type = CASE
                WHEN error_text IS NULL OR TRIM(error_text) = '' THEN NULL
                WHEN LOWER(error_text) LIKE '%timeout%' OR LOWER(error_text) LIKE '%timed out%'
                    THEN 'timeout'
                WHEN LOWER(error_text) LIKE '%429%'
                    OR (LOWER(error_text) LIKE '%rate%' AND LOWER(error_text) LIKE '%limit%')
                    THEN 'rate_limit'
                WHEN LOWER(error_text) LIKE '%disabled after%'
                    OR LOWER(error_text) LIKE '%cooldown active%'
                    THEN 'provider_disabled'
                WHEN LOWER(error_text) LIKE '%parse failed%'
                    OR LOWER(error_text) LIKE '%jsondecodeerror%'
                    THEN 'parse_error'
                ELSE 'provider_error'
            END
            {error_type_filter}
            """,
            params,
        )

        error_message_filter = f"WHERE (error_message IS NULL OR TRIM(error_message) = ''){where_sql}"
        summary["error_message_updated"] = _count_rows(
            conn,
            f"SELECT COUNT(*) FROM llm_calls {error_message_filter}",
            params,
        )
        conn.execute(
            f"""
            UPDATE llm_calls
            SET error_message = CASE
                WHEN error_text IS NULL OR TRIM(error_text) = '' THEN NULL
                ELSE SUBSTR(error_text, 1, 500)
            END
            {error_message_filter}
            """,
            params,
        )

        response_filter = (
            "WHERE (response_format IS NULL OR TRIM(response_format) = '')"
            f"{where_sql}"
        )
        rows = conn.execute(
            f"SELECT call_id, response_text FROM llm_calls {response_filter}",
            params,
        ).fetchall()
        updated_response = 0
        for call_id, response_text in rows:
            response_format = derive_response_format(response_text)
            conn.execute(
                "UPDATE llm_calls SET response_format = ? WHERE call_id = ?",
                [response_format, call_id],
            )
            updated_response += 1
        summary["response_format_updated"] = updated_response

        hazard_filter = (
            "WHERE phase = 'hs_triage'"
            " AND status = 'ok'"
            " AND hazard_scores_json IS NULL"
            " AND response_text IS NOT NULL"
            " AND TRIM(response_text) <> ''"
            f"{where_sql}"
        )
        rows = conn.execute(
            f"SELECT call_id, response_text FROM llm_calls {hazard_filter}",
            params,
        ).fetchall()
        filled_scores = 0
        for call_id, response_text in rows:
            hazard_scores = _extract_hazard_scores(response_text)
            if not hazard_scores:
                continue
            hazard_scores_json = json.dumps(hazard_scores, ensure_ascii=False)
            conn.execute(
                """
                UPDATE llm_calls
                SET hazard_scores_json = ?, hazard_scores_parse_ok = ?
                WHERE call_id = ?
                """,
                [hazard_scores_json, True, call_id],
            )
            filled_scores += 1
        summary["hazard_scores_filled"] = filled_scores
    finally:
        duckdb_io.close_db(conn)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="DuckDB path or duckdb:/// URL")
    parser.add_argument("--run-id", dest="run_id", default=None, help="Optional run_id filter")
    parser.add_argument(
        "--hs-run-id", dest="hs_run_id", default=None, help="Optional hs_run_id filter"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = backfill_llm_calls_telemetry(
        args.db, run_id=args.run_id, hs_run_id=args.hs_run_id
    )
    print("llm_calls telemetry migration summary")
    print(f"columns_ensured={','.join(summary['columns_ensured'])}")
    print(f"status_updated={summary['status_updated']}")
    print(f"error_type_updated={summary['error_type_updated']}")
    print(f"error_message_updated={summary['error_message_updated']}")
    print(f"response_format_updated={summary['response_format_updated']}")
    print(f"hazard_scores_filled={summary['hazard_scores_filled']}")


if __name__ == "__main__":
    main()
