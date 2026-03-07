#!/usr/bin/env python3
"""One-time backfill: set track=1 for all existing questions and triage entries.

All forecasts created before the Track 1/2 system was introduced used the full
ensemble, so they are retroactively Track 1.

Usage:
    python -m scripts.backfill_track          # uses default DB path
    python -m scripts.backfill_track --db /path/to/pythia.duckdb
"""

from __future__ import annotations

import argparse
import logging

from pythia.db.schema import connect

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def backfill_track(db_path: str | None = None) -> None:
    con = connect(db_path) if db_path else connect()

    # Backfill questions: all existing questions are Track 1.
    q_before = con.execute("SELECT COUNT(*) FROM questions WHERE track IS NULL").fetchone()[0]
    con.execute("UPDATE questions SET track = 1 WHERE track IS NULL")
    q_after = con.execute("SELECT COUNT(*) FROM questions WHERE track IS NULL").fetchone()[0]
    logger.info("questions: %d rows backfilled (remaining NULL: %d)", q_before - q_after, q_after)

    # Backfill hs_triage: entries that had need_full_spd=TRUE are Track 1.
    t_before = con.execute(
        "SELECT COUNT(*) FROM hs_triage WHERE track IS NULL AND need_full_spd = TRUE"
    ).fetchone()[0]
    con.execute("UPDATE hs_triage SET track = 1 WHERE track IS NULL AND need_full_spd = TRUE")
    t_after = con.execute(
        "SELECT COUNT(*) FROM hs_triage WHERE track IS NULL AND need_full_spd = TRUE"
    ).fetchone()[0]
    logger.info("hs_triage: %d rows backfilled (remaining NULL w/ need_full_spd: %d)", t_before - t_after, t_after)

    con.close()
    logger.info("Backfill complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill track column for existing data")
    parser.add_argument("--db", type=str, default=None, help="Path to DuckDB file")
    args = parser.parse_args()
    backfill_track(args.db)
