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


def main(argv: list[str] | None = None) -> None:
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

    # Set created_at explicitly so upsert refreshes it on conflict
    # (DEFAULT CURRENT_TIMESTAMP only fires on INSERT, not UPDATE).
    from datetime import datetime, timezone
    df["created_at"] = datetime.now(timezone.utc)

    con = get_db(db_url)
    try:
        ensure_schema(con)
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
    finally:
        from resolver.db.duckdb_io import close_db
        close_db(con)


if __name__ == "__main__":
    main()
