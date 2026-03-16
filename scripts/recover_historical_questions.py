# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Recover historical questions destroyed by the DELETE+INSERT bug.

The old ``create_questions_from_triage.py`` used stable question_ids
(e.g. ``ETH_ACE_FATALITIES``) and a DELETE+INSERT pattern.  Each HS run
overwrote the previous question row, destroying its ``hs_run_id``,
``window_start_date``, and ``target_month``.  After the March 2026 run,
all questions point to the March HS run with ``window_start_date = 2026-04-01``.

Fortunately, old forecasts are preserved in ``forecasts_ensemble`` and
``forecasts_raw`` (each run's DELETE only removes its own ``run_id``).
This script:

1. Queries ``run_provenance`` for all distinct forecaster_run_ids.
2. For each, gets the HS run date from ``hs_runs.generated_at``.
3. Computes the correct ``window_start_date`` and ``target_month``.
4. Finds all distinct question_ids in ``forecasts_ensemble`` for that run_id.
5. Creates epoch-specific questions (e.g. ``ETH_ACE_FATALITIES_2026-01``)
   with the correct metadata.
6. Updates ``forecasts_ensemble`` and ``forecasts_raw`` to reference the
   new question_ids.
7. Purges all existing ``resolutions`` and ``scores`` (computed with wrong
   windows).
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from typing import Optional

import duckdb

from pythia.db.schema import ensure_schema


DEFAULT_DB_URL = "duckdb:///data/resolver.duckdb"
LOG = logging.getLogger(__name__)
NUM_HORIZONS = 6


def _resolve_db_path(db_url: str) -> str:
    if db_url.startswith("duckdb:///"):
        return db_url[len("duckdb:///"):]
    return db_url


def _opening_from_run_date(run_date: date) -> date:
    """First day of the month after the run date."""
    m = run_date.month + 1
    y = run_date.year + (m - 1) // 12
    m = ((m - 1) % 12) + 1
    return date(y, m, 1)


def _closing_from_opening(opening: date) -> date:
    """Last day of the 6th horizon month from opening."""
    end_month = opening.month + 5
    end_year = opening.year + (end_month - 1) // 12
    end_month = ((end_month - 1) % 12) + 1
    if end_month == 12:
        next_year, next_month = end_year + 1, 1
    else:
        next_year, next_month = end_year, end_month + 1
    return date(next_year, next_month, 1) - timedelta(days=1)


def _epoch_label(opening: date) -> str:
    return f"{opening.year:04d}-{opening.month:02d}"


def _target_month_label(opening: date) -> str:
    end_month = opening.month + 5
    end_year = opening.year + (end_month - 1) // 12
    end_month = ((end_month - 1) % 12) + 1
    return f"{end_year:04d}-{end_month:02d}"


def _make_epoch_qid(old_qid: str, epoch: str) -> str:
    """Append epoch suffix to an old-style question_id.

    If the question_id already ends with a YYYY-MM pattern, return it as-is.
    """
    parts = old_qid.rsplit("_", 1)
    if len(parts) == 2:
        last = parts[1]
        if len(last) == 7 and last[4] == "-":
            # Already has an epoch suffix
            return old_qid
    return f"{old_qid}_{epoch}"


def recover(db_url: str, dry_run: bool = False) -> None:
    db_path = _resolve_db_path(db_url)
    con = duckdb.connect(db_path)
    try:
        ensure_schema(con)

        # ── Step 1: Get all forecaster runs with their HS run dates ──
        provenance_rows = con.execute("""
            SELECT rp.forecaster_run_id, rp.hs_run_id, h.generated_at
            FROM run_provenance rp
            JOIN hs_runs h ON rp.hs_run_id = h.hs_run_id
            WHERE rp.forecaster_run_id IS NOT NULL
            ORDER BY h.generated_at
        """).fetchall()

        if not provenance_rows:
            print("No run_provenance entries found. Nothing to recover.")
            return

        print(f"Found {len(provenance_rows)} forecaster runs to process.")

        total_questions_created = 0
        total_forecasts_updated = 0

        for forecaster_run_id, hs_run_id, generated_at in provenance_rows:
            # Parse run date
            if isinstance(generated_at, str):
                from datetime import datetime
                run_dt = datetime.fromisoformat(generated_at)
                run_date = run_dt.date()
            elif hasattr(generated_at, "date"):
                run_date = generated_at.date()
            else:
                run_date = generated_at

            opening = _opening_from_run_date(run_date)
            closing = _closing_from_opening(opening)
            epoch = _epoch_label(opening)
            target_month = _target_month_label(opening)

            # ── Step 2: Find all question_ids in forecasts for this run ──
            qids = con.execute("""
                SELECT DISTINCT question_id
                FROM forecasts_ensemble
                WHERE run_id = ?
            """, [forecaster_run_id]).fetchall()

            if not qids:
                LOG.info(
                    "No forecasts found for run %s (hs=%s); skipping.",
                    forecaster_run_id, hs_run_id,
                )
                continue

            print(
                f"  Run {forecaster_run_id} (hs={hs_run_id}, "
                f"date={run_date}, epoch={epoch}): "
                f"{len(qids)} question(s)"
            )

            for (old_qid,) in qids:
                if old_qid is None:
                    continue
                new_qid = _make_epoch_qid(old_qid, epoch)

                # ── Step 3: Create epoch-specific question if needed ──
                existing = con.execute(
                    "SELECT 1 FROM questions WHERE question_id = ?",
                    [new_qid],
                ).fetchone()

                if not existing:
                    # Parse iso3, hazard_code, metric from old question_id
                    # Format: {ISO3}_{HAZARD}_{METRIC} or
                    #         {ISO3}_{HAZARD}_{METRIC}_{EPOCH}
                    parts = old_qid.split("_")
                    if len(parts) >= 3:
                        iso3 = parts[0]
                        hazard_code = parts[1]
                        metric = parts[2]
                    else:
                        iso3 = parts[0] if parts else ""
                        hazard_code = parts[1] if len(parts) > 1 else ""
                        metric = parts[2] if len(parts) > 2 else ""

                    # Try to get wording and track from an existing question
                    # with the old ID
                    old_q = con.execute("""
                        SELECT wording, track, pythia_metadata_json
                        FROM questions WHERE question_id = ?
                    """, [old_qid]).fetchone()

                    wording = old_q[0] if old_q else f"Forecast for {iso3} {hazard_code} {metric}"
                    track = old_q[1] if old_q else None
                    meta_json = old_q[2] if old_q else "{}"

                    if not dry_run:
                        con.execute("""
                            INSERT INTO questions (
                                question_id, hs_run_id, scenario_ids_json,
                                iso3, hazard_code, metric,
                                target_month, window_start_date, window_end_date,
                                wording, status, pythia_metadata_json, track
                            ) VALUES (?, ?, '[]', ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
                        """, [
                            new_qid, hs_run_id,
                            iso3, hazard_code, metric,
                            target_month, opening, closing,
                            wording, meta_json, track,
                        ])
                    total_questions_created += 1

                # ── Step 4: Update forecasts to reference new question_id ──
                if old_qid != new_qid and not dry_run:
                    for table in ("forecasts_ensemble", "forecasts_raw"):
                        updated = con.execute(f"""
                            UPDATE {table}
                            SET question_id = ?
                            WHERE question_id = ? AND run_id = ?
                        """, [new_qid, old_qid, forecaster_run_id]).fetchone()
                    total_forecasts_updated += 1

        # ── Step 5: Migrate current old-style questions to epoch-specific IDs ──
        # Any remaining old-style question_ids (without epoch suffix) should
        # also be updated. These are the March questions currently in the DB.
        old_style_qs = con.execute("""
            SELECT question_id, hs_run_id, iso3, hazard_code, metric,
                   target_month, window_start_date, window_end_date,
                   wording, status, pythia_metadata_json, track
            FROM questions
        """).fetchall()

        migrated_current = 0
        for row in old_style_qs:
            old_qid = row[0]
            hs_run_id_q = row[1]
            ws_date = row[6]

            # Check if this already has an epoch suffix
            parts = old_qid.rsplit("_", 1)
            if len(parts) == 2 and len(parts[1]) == 7 and parts[1][4:5] == "-":
                continue  # already epoch-specific

            # Derive epoch from window_start_date
            if ws_date is not None:
                if isinstance(ws_date, str):
                    ws_parts = ws_date.split("-")
                    epoch = f"{ws_parts[0]}-{ws_parts[1]}"
                elif isinstance(ws_date, date):
                    epoch = _epoch_label(ws_date)
                else:
                    continue
            else:
                continue

            new_qid = f"{old_qid}_{epoch}"

            # Check if new ID already exists (from recovery above)
            existing = con.execute(
                "SELECT 1 FROM questions WHERE question_id = ?",
                [new_qid],
            ).fetchone()

            if existing:
                # New-style already exists; just delete the old-style duplicate
                if not dry_run:
                    con.execute(
                        "DELETE FROM questions WHERE question_id = ?",
                        [old_qid],
                    )
                continue

            # Rename the question in-place
            if not dry_run:
                con.execute(
                    "UPDATE questions SET question_id = ? WHERE question_id = ?",
                    [new_qid, old_qid],
                )
                # Update any remaining forecasts that still reference old ID
                for table in ("forecasts_ensemble", "forecasts_raw"):
                    con.execute(f"""
                        UPDATE {table}
                        SET question_id = ?
                        WHERE question_id = ?
                    """, [new_qid, old_qid])

            migrated_current += 1

        print(f"\nRecovery complete:")
        print(f"  Questions created: {total_questions_created}")
        print(f"  Forecast runs updated: {total_forecasts_updated}")
        print(f"  Current questions migrated: {migrated_current}")

        # ── Step 6: Purge stale resolutions and scores ──
        if not dry_run:
            for table in ("scores", "resolutions"):
                count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                con.execute(f"DELETE FROM {table}")
                print(f"  Purged {count} rows from {table}")

            # Reset question statuses
            con.execute(
                "UPDATE questions SET status = 'active' WHERE status = 'resolved'"
            )
            print("  Reset all 'resolved' questions to 'active'")

        if dry_run:
            print("\n  (DRY RUN — no changes written)")

    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover historical questions destroyed by the DELETE+INSERT bug."
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_URL,
        help="DuckDB URL, e.g. duckdb:///data/resolver.duckdb",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    recover(args.db, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
