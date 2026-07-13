# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Reconcile ``llm_calls.is_test`` with each run's actual test status.

Historically the forecaster's ``log_forecaster_llm_call`` defaulted
``is_test=False``, so ``spd_v2`` / ``binary_v2`` / ``scenario_v2`` llm_calls
made during a **test** run were stamped ``is_test=False`` even though the run's
forecast outputs (``forecasts_ensemble`` / ``forecasts_raw`` / ``questions``)
and the HS/Sibyl llm_calls were correctly ``is_test=True``. That mis-stamping
leaks test-run costs into non-test cost views (the Run Results "0 forecasts but
$X cost" symptom, and the production Costs page).

The forward fix lives in ``forecaster/llm_logging.py`` (is_test now inherits
``is_test_mode()``). This script repairs the already-written rows.

A row in ``llm_calls`` is considered test when EITHER:
  * its ``run_id`` belongs to a run that produced ``is_test=TRUE`` rows in
    ``forecasts_ensemble`` / ``forecasts_raw`` / ``questions``, OR
  * its ``hs_run_id`` is an ``is_test=TRUE`` run in ``hs_runs``.

It only flips ``is_test=FALSE`` → ``TRUE`` (never the reverse — this cannot
mislabel real production calls as test), so it is idempotent and safe to wire
into any workflow that runs every cycle.

Usage::

    PYTHIA_DB_URL=duckdb:///path/to/resolver.duckdb \
        python -m scripts.reconcile_llm_calls_is_test [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from pythia.db.schema import connect

LOGGER = logging.getLogger(__name__)

# (table, id_column) sources of authoritative test-run ids, matched against
# llm_calls.run_id.
_RUN_ID_SOURCES = [
    ("forecasts_ensemble", "run_id"),
    ("forecasts_raw", "run_id"),
    ("questions", "run_id"),
]
# hs-run sources matched against llm_calls.hs_run_id.
_HS_RUN_ID_SOURCES = [
    ("hs_runs", "hs_run_id"),
]


def _table_columns(conn, table: str) -> set[str]:
    try:
        return {
            str(r[1]).lower()
            for r in conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        }
    except Exception:
        return set()


def _collect_test_ids(conn, sources) -> set[str]:
    """Return the set of ids that are is_test=TRUE across the given sources."""
    ids: set[str] = set()
    for table, col in sources:
        cols = _table_columns(conn, table)
        if not cols or col not in cols or "is_test" not in cols:
            continue
        try:
            rows = conn.execute(
                f"SELECT DISTINCT {col} FROM {table} "
                f"WHERE COALESCE(is_test, FALSE) = TRUE AND {col} IS NOT NULL"
            ).fetchall()
            ids.update(str(r[0]) for r in rows if r and r[0])
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("reconcile: failed reading %s.%s: %s", table, col, exc)
    return ids


def reconcile_llm_calls_is_test(conn, apply: bool = True) -> dict[str, int]:
    """Flip llm_calls.is_test FALSE→TRUE for calls belonging to test runs.

    A single combined ``(run_id IN … OR hs_run_id IN …)`` predicate is used so
    rows matched by both keys are counted (and updated) exactly once.

    Returns {"n_test_run_ids": .., "n_test_hs_run_ids": .., "total": rows_flipped}.
    """
    cols = _table_columns(conn, "llm_calls")
    if not cols or "is_test" not in cols:
        LOGGER.warning("reconcile: llm_calls table/is_test column missing; skipping.")
        return {"n_test_run_ids": 0, "n_test_hs_run_ids": 0, "total": 0}

    run_ids = _collect_test_ids(conn, _RUN_ID_SOURCES) if "run_id" in cols else set()
    hs_run_ids = _collect_test_ids(conn, _HS_RUN_ID_SOURCES) if "hs_run_id" in cols else set()

    clauses: list[str] = []
    params: list[str] = []
    if run_ids:
        clauses.append(f"run_id IN ({','.join(['?'] * len(run_ids))})")
        params.extend(sorted(run_ids))
    if hs_run_ids:
        clauses.append(f"hs_run_id IN ({','.join(['?'] * len(hs_run_ids))})")
        params.extend(sorted(hs_run_ids))

    if not clauses:
        LOGGER.info("reconcile_llm_calls_is_test: no test runs found; nothing to do.")
        return {"n_test_run_ids": 0, "n_test_hs_run_ids": 0, "total": 0}

    where = f"COALESCE(is_test, FALSE) = FALSE AND ({' OR '.join(clauses)})"
    n = int(conn.execute(f"SELECT COUNT(*) FROM llm_calls WHERE {where}", params).fetchone()[0] or 0)
    if n and apply:
        conn.execute(f"UPDATE llm_calls SET is_test = TRUE WHERE {where}", params)

    LOGGER.info(
        "reconcile_llm_calls_is_test: %s %d rows across %d test run_ids / %d test hs_run_ids.",
        "would flip" if not apply else "flipped",
        n, len(run_ids), len(hs_run_ids),
    )
    return {"n_test_run_ids": len(run_ids), "n_test_hs_run_ids": len(hs_run_ids), "total": n}


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Reconcile llm_calls.is_test with run test status")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report how many rows would change without writing.",
    )
    args = parser.parse_args()

    db_url = os.getenv("PYTHIA_DB_URL")
    print(f"[reconcile_llm_calls_is_test] PYTHIA_DB_URL={db_url or '(default)'} "
          f"dry_run={args.dry_run}")
    conn = connect(read_only=False)
    try:
        result = reconcile_llm_calls_is_test(conn, apply=not args.dry_run)
    finally:
        conn.close()
    print(f"[reconcile_llm_calls_is_test] {'would flip' if args.dry_run else 'flipped'} "
          f"{result['total']} rows "
          f"(test run_ids={result['n_test_run_ids']}, "
          f"test hs_run_ids={result['n_test_hs_run_ids']}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
