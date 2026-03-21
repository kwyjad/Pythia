# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Mark specific pipeline runs as test data in the Pythia DuckDB."""

import argparse
import re
import duckdb

# Tables keyed by run_id (forecaster runs)
RUN_ID_TABLES = [
    "forecasts_ensemble",
    "forecasts_raw",
    "question_research",
    "question_run_metrics",
    "scenarios",
    "question_context",
    "run_provenance",
]

# Tables keyed by hs_run_id (horizon scanner runs)
HS_RUN_ID_TABLES = [
    "hs_runs",
    "hs_scenarios",
    "hs_triage",
    "hs_country_reports",
    "hs_hazard_tail_packs",
    "hs_adversarial_checks",
    "questions",
]

# Tables keyed by either run_id or hs_run_id
DUAL_KEY_TABLES = [
    "llm_calls",
]

# Tables that inherit from questions (mark via question_id)
INHERITED_TABLES = [
    "resolutions",
    "scores",
    "eiv_scores",
]


def _table_exists(con, table: str) -> bool:
    try:
        result = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table],
        ).fetchone()
        return result[0] > 0 if result else False
    except Exception:
        return False


def _has_column(con, table: str, column: str) -> bool:
    try:
        cols = {
            row[1].lower()
            for row in con.execute(f"PRAGMA table_info('{table}')").fetchall()
        }
        return column.lower() in cols
    except Exception:
        return False


def _update_table(con, table: str, column: str, run_ids: list[str]) -> int:
    if not _table_exists(con, table) or not _has_column(con, table, "is_test") or not _has_column(con, table, column):
        return 0
    placeholders = ", ".join(["?"] * len(run_ids))
    before = con.execute(
        f"SELECT COUNT(*) FROM {table} WHERE COALESCE(is_test, FALSE) = FALSE AND {column} IN ({placeholders})",
        run_ids,
    ).fetchone()[0]
    if before > 0:
        con.execute(
            f"UPDATE {table} SET is_test = TRUE WHERE {column} IN ({placeholders})",
            run_ids,
        )
    return before


def mark_runs(db_url: str, raw_ids: str) -> None:
    # Parse run IDs: split on commas, spaces, or semicolons
    run_ids = [rid.strip() for rid in re.split(r"[,;\s]+", raw_ids) if rid.strip()]
    if not run_ids:
        print("ERROR: No valid run IDs provided.")
        return

    print(f"Marking {len(run_ids)} run(s) as test: {run_ids}")

    db_path = db_url.replace("duckdb:///", "")
    con = duckdb.connect(db_path)

    total_updates = 0

    # Update tables keyed by run_id
    for table in RUN_ID_TABLES:
        count = _update_table(con, table, "run_id", run_ids)
        if count > 0:
            print(f"  {table}: marked {count} rows (by run_id)")
            total_updates += count

    # Update tables keyed by hs_run_id
    for table in HS_RUN_ID_TABLES:
        count = _update_table(con, table, "hs_run_id", run_ids)
        if count > 0:
            print(f"  {table}: marked {count} rows (by hs_run_id)")
            total_updates += count

    # Update dual-key tables
    for table in DUAL_KEY_TABLES:
        for col in ["run_id", "hs_run_id"]:
            count = _update_table(con, table, col, run_ids)
            if count > 0:
                print(f"  {table}: marked {count} rows (by {col})")
                total_updates += count

    # Update inherited tables via question_id
    test_question_ids = []
    if _table_exists(con, "questions") and _has_column(con, "questions", "is_test"):
        rows = con.execute(
            "SELECT DISTINCT question_id FROM questions WHERE is_test = TRUE"
        ).fetchall()
        test_question_ids = [r[0] for r in rows]

    if test_question_ids:
        for table in INHERITED_TABLES:
            count = _update_table(con, table, "question_id", test_question_ids)
            if count > 0:
                print(f"  {table}: marked {count} rows (by question_id inheritance)")
                total_updates += count

    con.close()
    print(f"\nDone. Total rows marked as test: {total_updates}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mark pipeline runs as test data")
    parser.add_argument("--db", required=True, help="DuckDB URL (e.g. duckdb:///data/resolver.duckdb)")
    parser.add_argument("--run-ids", required=True, help="Comma or space separated run IDs")
    args = parser.parse_args()
    mark_runs(args.db, args.run_ids)
