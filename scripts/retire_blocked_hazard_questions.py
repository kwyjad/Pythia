# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Mark questions for blocked hazards (DI / HW / CU / ACO) as retired.

These hazards are blocked at the catalog level in
``horizon_scanner/db_writer.py::BLOCKED_HAZARDS`` and the question generator
no longer produces new ones. But legacy rows persist forever in the
``questions`` table and clutter dashboards (e.g. the Resolution Coverage
Summary on /performance) with permanent 0% tiles that can never resolve.

This script is idempotent: it only updates rows whose status is not
already ``'retired'``, so it's safe to wire into any workflow and run
every cycle.

Usage::

    PYTHIA_DB_URL=duckdb:///path/to/resolver.duckdb \
        python -m scripts.retire_blocked_hazard_questions

Outputs the number of rows updated and the per-hazard breakdown.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Iterable

from horizon_scanner.db_writer import BLOCKED_HAZARDS
from pythia.db.schema import connect

LOGGER = logging.getLogger(__name__)


def retire_blocked_hazard_questions(
    conn, blocked: Iterable[str] = BLOCKED_HAZARDS
) -> dict[str, int]:
    """Mark active questions for any hazard in ``blocked`` as retired.

    Returns a dict of {hazard_code: rows_updated}.
    """
    blocked_set = sorted({h.upper() for h in blocked if h})
    if not blocked_set:
        return {}

    # Confirm questions table + status column exist before attempting the update.
    try:
        cols = {
            str(r[1]).lower()
            for r in conn.execute("PRAGMA table_info('questions')").fetchall()
        }
    except Exception as exc:
        LOGGER.warning("questions table not present; skipping retirement: %s", exc)
        return {}
    if "status" not in cols or "hazard_code" not in cols:
        LOGGER.warning(
            "questions table missing required columns (status / hazard_code); "
            "skipping retirement."
        )
        return {}

    # Count what we're about to retire so the script's output is informative.
    placeholders = ",".join(["?"] * len(blocked_set))
    counts: dict[str, int] = {}
    for hz in blocked_set:
        row = conn.execute(
            "SELECT COUNT(*) FROM questions "
            "WHERE UPPER(hazard_code) = ? AND COALESCE(status, '') != 'retired'",
            [hz],
        ).fetchone()
        counts[hz] = int(row[0]) if row else 0

    total_to_retire = sum(counts.values())
    if total_to_retire == 0:
        LOGGER.info("retire_blocked_hazard_questions: nothing to do (already retired).")
        return counts

    conn.execute(
        f"""
        UPDATE questions
           SET status = 'retired'
         WHERE UPPER(hazard_code) IN ({placeholders})
           AND COALESCE(status, '') != 'retired'
        """,
        blocked_set,
    )
    LOGGER.info(
        "retire_blocked_hazard_questions: retired %d questions across hazards=%s.",
        total_to_retire,
        ",".join(f"{hz}={n}" for hz, n in counts.items() if n > 0),
    )
    return counts


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    db_url = os.getenv("PYTHIA_DB_URL")
    print(f"[retire_blocked_hazard_questions] PYTHIA_DB_URL={db_url or '(default)'}")
    print(f"[retire_blocked_hazard_questions] BLOCKED_HAZARDS={sorted(BLOCKED_HAZARDS)}")
    conn = connect(read_only=False)
    try:
        counts = retire_blocked_hazard_questions(conn)
    finally:
        conn.close()
    total = sum(counts.values())
    print(f"[retire_blocked_hazard_questions] Retired {total} rows. By hazard: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
