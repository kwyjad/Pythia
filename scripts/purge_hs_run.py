# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""
Purge all data associated with a specific Horizon Scanner run ID.

Cascade-deletes from leaf tables upward so the DB is left in a
consistent state, as if the HS run never happened.  Canonical resolver
data (facts_resolved, facts_deltas, etc.) is never touched.

Usage:
    # Dry-run (default) — prints row counts, no mutations:
    python -m scripts.purge_hs_run --hs-run-id hs_20260301T000000

    # Execute for real:
    python -m scripts.purge_hs_run --hs-run-id hs_20260301T000000 --execute
"""

from __future__ import annotations

import argparse
import os
import sys

from pythia.db.schema import connect, ensure_schema


def _count(con, table: str, col: str, value: str) -> int:
    """Count rows in *table* where *col* = *value*."""
    try:
        row = con.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {col} = ?", [value]
        ).fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def _count_in(con, table: str, col: str, values: list[str]) -> int:
    """Count rows in *table* where *col* is in *values*."""
    if not values:
        return 0
    try:
        row = con.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {col} IN (SELECT UNNEST(?::TEXT[]))",
            [values],
        ).fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def _delete(con, table: str, col: str, value: str) -> int:
    """Delete rows and return count deleted."""
    try:
        before = _count(con, table, col, value)
        if before:
            con.execute(f"DELETE FROM {table} WHERE {col} = ?", [value])
        return before
    except Exception:
        return 0


def _delete_in(con, table: str, col: str, values: list[str]) -> int:
    """Delete rows where *col* IN *values* and return count deleted."""
    if not values:
        return 0
    try:
        before = _count_in(con, table, col, values)
        if before:
            con.execute(
                f"DELETE FROM {table} WHERE {col} IN (SELECT UNNEST(?::TEXT[]))",
                [values],
            )
        return before
    except Exception:
        return 0


def _table_total(con, table: str) -> int:
    """Total row count for a table (returns 0 if table doesn't exist)."""
    try:
        row = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return 0


# Tables deleted by hs_run_id directly
HS_DIRECT_TABLES = [
    ("hs_adversarial_checks", "hs_run_id"),
    ("hs_hazard_tail_packs", "hs_run_id"),
    ("hs_country_reports", "hs_run_id"),
    ("hs_scenarios", "hs_run_id"),
    ("hs_triage", "run_id"),  # uses run_id, not hs_run_id
    ("llm_calls", "hs_run_id"),
    ("run_provenance", "hs_run_id"),
]

# Tables deleted by question_id cascade
QUESTION_CASCADE_TABLES = [
    "scores",
    "resolutions",
    "component_performance",
    "forecasts_raw",
    "forecasts_ensemble",
    "question_research",
    "question_run_metrics",
]


def purge(hs_run_id: str, *, execute: bool = False) -> bool:
    """Purge all data for *hs_run_id*.  Returns True on success."""

    con = connect(read_only=not execute)
    ensure_schema(con)

    # --- Validate ---
    exists = con.execute(
        "SELECT 1 FROM hs_runs WHERE hs_run_id = ?", [hs_run_id]
    ).fetchone()
    if not exists:
        print(f"ERROR: hs_run_id '{hs_run_id}' not found in hs_runs.", file=sys.stderr)
        con.close()
        return False

    # --- Phase 1: collect question_ids ---
    question_rows = con.execute(
        "SELECT question_id FROM questions WHERE hs_run_id = ?", [hs_run_id]
    ).fetchall()
    question_ids = [r[0] for r in question_rows if r and r[0]]

    # --- Dry-run summary ---
    print(f"\n{'=' * 60}")
    print(f"{'DRY RUN' if not execute else 'EXECUTING PURGE'}: hs_run_id = {hs_run_id}")
    print(f"{'=' * 60}")
    print(f"Questions linked to this run: {len(question_ids)}")
    print()

    summary: list[tuple[str, int]] = []

    # Question cascade tables
    for table in QUESTION_CASCADE_TABLES:
        count = _count_in(con, table, "question_id", question_ids)
        summary.append((table, count))

    # Questions themselves
    q_count = _count(con, "questions", "hs_run_id", hs_run_id)
    summary.append(("questions", q_count))

    # HS direct tables
    for table, col in HS_DIRECT_TABLES:
        count = _count(con, table, col, hs_run_id)
        summary.append((table, count))

    # hs_runs itself
    summary.append(("hs_runs", 1))

    # Print table
    print(f"{'Table':<30} {'Rows to delete':>15}")
    print(f"{'-' * 30} {'-' * 15}")
    total = 0
    for table, count in summary:
        print(f"{table:<30} {count:>15,}")
        total += count
    print(f"{'-' * 30} {'-' * 15}")
    print(f"{'TOTAL':<30} {total:>15,}")
    print()

    if not execute:
        print("Dry run complete. Pass --execute to purge.")
        con.close()
        return True

    # --- Execute purge in a transaction ---
    print("Purging...")
    con.execute("BEGIN TRANSACTION")

    try:
        deleted: list[tuple[str, int]] = []

        # Phase 2: cascade through question_id
        for table in QUESTION_CASCADE_TABLES:
            n = _delete_in(con, table, "question_id", question_ids)
            deleted.append((table, n))

        # Phase 3: questions
        n = _delete(con, "questions", "hs_run_id", hs_run_id)
        deleted.append(("questions", n))

        # Phase 4: HS direct tables
        for table, col in HS_DIRECT_TABLES:
            n = _delete(con, table, col, hs_run_id)
            deleted.append((table, n))

        # Phase 5: hs_runs
        n = _delete(con, "hs_runs", "hs_run_id", hs_run_id)
        deleted.append(("hs_runs", n))

        con.execute("COMMIT")
    except Exception as exc:
        con.execute("ROLLBACK")
        print(f"ERROR: purge failed, rolled back: {exc}", file=sys.stderr)
        con.close()
        return False

    # --- After summary ---
    print(f"\n{'=' * 60}")
    print("PURGE COMPLETE")
    print(f"{'=' * 60}")
    print(f"{'Table':<30} {'Rows deleted':>15}")
    print(f"{'-' * 30} {'-' * 15}")
    purged_total = 0
    for table, count in deleted:
        print(f"{table:<30} {count:>15,}")
        purged_total += count
    print(f"{'-' * 30} {'-' * 15}")
    print(f"{'TOTAL':<30} {purged_total:>15,}")
    print()

    con.close()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Purge all data for a specific HS run ID from the Pythia DB."
    )
    parser.add_argument(
        "--hs-run-id",
        required=True,
        help="The HS run ID to purge (e.g. hs_20260301T000000).",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="DuckDB URL (default: PYTHIA_DB_URL env var).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Actually execute the purge. Without this flag, only a dry-run is performed.",
    )
    args = parser.parse_args()

    if args.db:
        os.environ["PYTHIA_DB_URL"] = args.db

    ok = purge(args.hs_run_id, execute=args.execute)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
