# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Tuple

import duckdb  # type: ignore

from pythia.db.schema import connect as pythia_connect, ensure_schema


LOG = logging.getLogger(__name__)


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def _summarize_questions(con: duckdb.DuckDBPyConnection) -> None:
    """
    Print a summary of questions grouped by (iso3, hazard_code, metric), split into:
      - HS-driven (hs_run_id not null)
      - legacy (hs_run_id null/empty)
    Also highlight any ACO questions.
    """
    _print_header("QUESTIONS SUMMARY")

    rows = con.execute(
        """
        SELECT
            iso3,
            UPPER(COALESCE(hazard_code, '')) AS hazard_code,
            metric,
            COUNT(*) AS n_total,
            SUM(CASE WHEN hs_run_id IS NOT NULL AND hs_run_id <> '' THEN 1 ELSE 0 END) AS n_hs,
            SUM(CASE WHEN hs_run_id IS NULL OR hs_run_id = '' THEN 1 ELSE 0 END) AS n_legacy,
            SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS n_active
        FROM questions
        GROUP BY iso3, hazard_code, metric
        ORDER BY iso3, hazard_code, metric
        """
    ).fetchall()

    if not rows:
        print("No rows in questions table.\n")
        return

    print("iso3  | hazard  | metric       | total | hs_rows | legacy | active")
    print("------+---------+-------------+-------+---------+--------+--------")
    for r in rows:
        iso3 = r["iso3"]
        hz = r["hazard_code"]
        metric = r["metric"]
        print(
            f"{iso3:4} | {hz:7} | {metric:11} | "
            f"{r['n_total']:5} | {r['n_hs']:7} | {r['n_legacy']:6} | {r['n_active']:6}"
        )

    # Explicitly list any active ACO questions (the ones we want gone)
    aco_rows = con.execute(
        """
        SELECT question_id, iso3, hazard_code, metric, status, hs_run_id
        FROM questions
        WHERE UPPER(hazard_code) = 'ACO' AND status = 'active'
        ORDER BY iso3, question_id
        """
    ).fetchall()

    print("\nActive ACO questions:")
    if not aco_rows:
        print("  (none)\n")
    else:
        for r in aco_rows:
            print(
                f"  {r['question_id']}: iso3={r['iso3']} metric={r['metric']} "
                f"hs_run_id={r['hs_run_id']}"
            )
        print()


def _summarize_hs_triage(con: duckdb.DuckDBPyConnection) -> None:
    """
    Print a summary of hs_triage rows per iso3/hazard_code, including run_id and tier.
    """
    _print_header("HS_TRIAGE SUMMARY")

    rows = con.execute(
        """
        SELECT
            run_id,
            iso3,
            hazard_code,
            tier,
            triage_score
        FROM hs_triage
        ORDER BY run_id, iso3, hazard_code
        """
    ).fetchall()

    if not rows:
        print("No rows in hs_triage table.\n")
        return

    print("run_id              | iso3 | hazard | tier       | triage_score")
    print("--------------------+------+--------+-----------+-------------")
    for r in rows:
        print(
            f"{r['run_id']:20} | {r['iso3']:4} | {r['hazard_code']:6} | "
            f"{(r['tier'] or ''):9} | {float(r['triage_score'] or 0.0):11.3f}"
        )
    print()


def _summarize_hs_llm_calls(con: duckdb.DuckDBPyConnection) -> None:
    """
    Print HS LLM calls (phase='hs_triage') grouped by iso3/hazard_code/hs_run_id.
    """
    _print_header("HS LLM CALLS (llm_calls, phase='hs_triage')")

    rows = con.execute(
        """
        SELECT
            hs_run_id,
            iso3,
            hazard_code,
            phase,
            provider,
            model_name,
            model_id,
            COUNT(*) AS n_calls
        FROM llm_calls
        WHERE phase = 'hs_triage'
        GROUP BY hs_run_id, iso3, hazard_code, phase, provider, model_name, model_id
        ORDER BY hs_run_id, iso3, hazard_code
        """
    ).fetchall()

    if not rows:
        print("No HS triage calls found in llm_calls.\n")
        return

    print("hs_run_id           | iso3 | hazard | provider | model_name            | model_id              | n_calls")
    print("--------------------+------+------+----------+------------------------+-----------------------+--------")
    for r in rows:
        print(
            f"{(r['hs_run_id'] or ''):20} | {r['iso3']:4} | {(r['hazard_code'] or ''):6} | "
            f"{(r['provider'] or ''):8} | {(r['model_name'] or ''):22} | "
            f"{(r['model_id'] or ''):21} | {r['n_calls']:7}"
        )
    print()


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    db_url = os.environ.get("PYTHIA_DB_URL")
    if not db_url:
        raise SystemExit("PYTHIA_DB_URL is not set; cannot inspect Pythia state.")

    LOG.info("Connecting to Pythia DB at %s", db_url)
    con = pythia_connect(read_only=True)
    ensure_schema(con)

    try:
        _summarize_questions(con)
        _summarize_hs_triage(con)
        _summarize_hs_llm_calls(con)
    finally:
        con.close()


if __name__ == "__main__":
    main()
