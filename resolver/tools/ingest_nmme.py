#!/usr/bin/env python3
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Ingest NMME seasonal forecasts into the Pythia DuckDB.

Downloads ENSMEAN anomaly data from CPC FTP, aggregates to country
level, and upserts into the ``seasonal_forecasts`` table.

Usage:
    python -m resolver.tools.ingest_nmme
    python -m resolver.tools.ingest_nmme --year-month 202603
    python -m resolver.tools.ingest_nmme --db duckdb:///data/resolver.duckdb
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

LOG = logging.getLogger(__name__)


def _default_db_url() -> str:
    """Resolve the Pythia DuckDB URL from config or environment."""
    url = os.getenv("PYTHIA_DB_URL", "").strip()
    if url:
        return url

    try:
        from pythia.config import load as load_config
        cfg = load_config()
        url = str((cfg.get("app") or {}).get("db_url", "")).strip()
        if url:
            return url
    except Exception:
        pass

    from resolver.db.duckdb_io import DEFAULT_DB_URL
    return DEFAULT_DB_URL


def main(argv: list[str] | None = None) -> dict | None:
    parser = argparse.ArgumentParser(
        description="Ingest NMME seasonal forecasts into Pythia DuckDB."
    )
    parser.add_argument(
        "--year-month",
        default=None,
        help="Issue month as YYYYMM (auto-detects latest if omitted).",
    )
    parser.add_argument(
        "--max-leads",
        type=int,
        default=7,
        help="Number of lead months to fetch (default: 7).",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="DuckDB URL or path (default: from config / PYTHIA_DB_URL).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and process but do not write to DuckDB.",
    )
    parser.add_argument(
        "--backfill-months",
        type=int,
        default=0,
        help=(
            "Also ingest the N issue months BEFORE the one fetched, oldest "
            "first. A vintage the FTP no longer keeps is skipped and named, "
            "never fatal: the CPC directory is 'realtime_anom', so how far "
            "back it reaches is a property of the archive rather than of "
            "this code, and the run reports what it found. Each recovered "
            "vintage is a month the drought gate's NMME indicator can speak "
            "for — the gate read 2 of 3 feeds in 2026-07 because the "
            "earliest vintage held was issued that July at lead 1, so it is "
            "about August and says nothing about July."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    )

    # 1. Fetch and process.
    from resolver.ingestion.nmme import fetch_and_process

    LOG.info("Fetching NMME seasonal forecasts from CPC FTP …")
    df = fetch_and_process(
        year_month=args.year_month,
        max_leads=args.max_leads,
    )

    if df.empty:
        LOG.warning("No data produced — nothing to write.")
        return

    LOG.info(
        "Produced %d rows: %d countries × %d variables × %d leads, "
        "issue date %s",
        len(df),
        df["iso3"].nunique(),
        df["variable"].nunique(),
        df["lead_months"].nunique(),
        df["forecast_issue_date"].iloc[0],
    )

    if args.dry_run:
        LOG.info("Dry run — skipping DuckDB write.")
        print(df.to_string(index=False, max_rows=20))
        return

    # 2. Write to DuckDB.
    db_url = args.db or _default_db_url()
    LOG.info("Writing to DuckDB: %s", db_url)

    from resolver.db.duckdb_io import get_db, upsert_dataframe
    from pythia.db.schema import ensure_schema

    # Set fetched_at so the Resolver page shows when data was last ingested
    # (created_at only fires on INSERT, not on upsert UPDATE).
    from datetime import datetime, timezone
    df["fetched_at"] = datetime.now(timezone.utc)

    con = get_db(db_url)
    try:
        ensure_schema(con)
        purged = purge_unitless_rows(con)
        result = upsert_dataframe(
            con,
            "seasonal_forecasts",
            df,
            keys=["iso3", "variable", "lead_months", "forecast_issue_date"],
        )
        LOG.info(
            "Upsert complete: %d rows written (was %d → now %d)",
            result.rows_written,
            result.rows_before,
            result.rows_after,
        )
        summary = {
            "rows_purged_unitless": int(purged),
            "rows_written": int(result.rows_written),
            "rows_before": int(result.rows_before),
            "rows_after": int(result.rows_after),
        }
        if args.backfill_months > 0:
            summary["backfill"] = _backfill_earlier_issues(
                con,
                months=args.backfill_months,
                newest_issue=str(df["forecast_issue_date"].iloc[0]),
                max_leads=args.max_leads,
            )
        return summary
    finally:
        from resolver.db.duckdb_io import close_db
        close_db(con)


def _earlier_issue_months(newest_issue: str, months: int) -> list[str]:
    """The ``YYYYMM`` issue months before ``newest_issue``, oldest first."""

    from datetime import date as _date

    text = str(newest_issue)[:7]
    year, month = int(text[:4]), int(text[5:7])
    out: list[str] = []
    for back in range(months, 0, -1):
        index = (year * 12 + month - 1) - back
        out.append(f"{index // 12:04d}{index % 12 + 1:02d}")
    _ = _date  # imported for the reader; arithmetic is done on the index
    return out


#: The variables one NMME vintage carries. A vintage holding fewer is
#: partial and is fetched again.
_NMME_VARIABLES = 2


def _vintage_is_held(con, year_month: str) -> bool:
    """Does ``seasonal_forecasts`` already hold this issue month in full?

    Never raises: a table that cannot be read answers "no", and the vintage
    is fetched as it always was.
    """

    try:
        row = con.execute(
            """
            SELECT COUNT(DISTINCT variable)
            FROM seasonal_forecasts
            WHERE strftime(CAST(forecast_issue_date AS DATE), '%Y%m') = ?
              AND units IS NOT NULL
            """,
            [year_month],
        ).fetchone()
    except Exception:  # noqa: BLE001 - a guard that cannot read must not stop the fetch
        return False
    return bool(row) and int(row[0] or 0) >= _NMME_VARIABLES


def purge_unitless_rows(con) -> int:
    """Delete ``seasonal_forecasts`` rows written before the unit fix.

    Until Oct 2026 the precipitation anomaly was stored in raw mm/s rounded
    to four decimals (all 21,294 prate rows of the 1 Oct release sat within
    0.0002 of zero) and both variables carried a sigma-based category. Rows
    with no ``units`` are that vintage; deleting them makes the backfill
    fetch each vintage again in real units (``_vintage_is_held`` counts only
    rows carrying units). Idempotent; never raises.
    """
    try:
        n = con.execute(
            "SELECT COUNT(*) FROM seasonal_forecasts WHERE units IS NULL"
        ).fetchone()[0]
        if n:
            con.execute("DELETE FROM seasonal_forecasts WHERE units IS NULL")
            LOG.info("[nmme] purged %d row(s) stored before the unit fix", n)
        return int(n or 0)
    except Exception as exc:  # noqa: BLE001 - a purge must not stop the ingest
        LOG.warning("[nmme] unitless-row purge skipped: %s", exc)
        return 0


def _backfill_earlier_issues(
    con, *, months: int, newest_issue: str, max_leads: int
) -> dict:
    """Ingest earlier NMME vintages, reporting what the archive actually held.

    CPC publishes under ``realtime_anom``, which is usually a rolling
    window rather than an archive, so a month that is simply not there is
    an ordinary outcome and is named rather than raised — the same rule the
    NOAA candidate-URL walk follows. A vintage that IS there is a month the
    drought gate's NMME indicator can speak for, which is the point.
    """

    from resolver.db.duckdb_io import upsert_dataframe
    from resolver.ingestion.nmme import fetch_and_process

    wanted = _earlier_issue_months(newest_issue, months)
    recovered: list[str] = []
    absent: list[str] = []
    failed: dict[str, str] = {}
    rows_written = 0

    LOG.info(
        "[nmme] backfill: asking CPC for %d earlier issue month(s): %s",
        len(wanted), ", ".join(wanted),
    )
    held: list[str] = []
    for year_month in wanted:
        # A vintage already in the table is not asked for again. CPC does not
        # revise a published issue, and every run re-downloaded all twelve
        # to merge them onto themselves with a delta of zero (run
        # 36401252026: 2m41s for no row).
        if _vintage_is_held(con, year_month):
            held.append(year_month)
            continue
        try:
            frame = fetch_and_process(year_month=year_month, max_leads=max_leads)
        except FileNotFoundError as exc:
            absent.append(year_month)
            LOG.info("[nmme] backfill %s: not held by the archive (%s)", year_month, exc)
            continue
        except Exception as exc:  # noqa: BLE001 - one bad vintage is not the run
            failed[year_month] = str(exc)[:200]
            LOG.warning("[nmme] backfill %s failed: %s", year_month, exc)
            continue
        if frame is None or frame.empty:
            absent.append(year_month)
            LOG.info("[nmme] backfill %s: no rows produced", year_month)
            continue
        from datetime import datetime, timezone

        frame["fetched_at"] = datetime.now(timezone.utc)
        result = upsert_dataframe(
            con,
            "seasonal_forecasts",
            frame,
            keys=["iso3", "variable", "lead_months", "forecast_issue_date"],
        )
        rows_written += int(result.rows_written)
        recovered.append(year_month)
        LOG.info(
            "[nmme] backfill %s: %d rows written", year_month, result.rows_written
        )

    LOG.info(
        "[nmme] backfill complete: %d of %d vintage(s) recovered (%d rows), "
        "%d already held; not held by the archive: %s",
        len(recovered), len(wanted), rows_written, len(held),
        ", ".join(absent) or "none",
    )
    return {
        "wanted": wanted,
        "already_held": held,
        "recovered": recovered,
        "absent": absent,
        "failed": failed,
        "rows_written": rows_written,
    }


if __name__ == "__main__":
    main()
