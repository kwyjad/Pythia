# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Fetch conflict forecasts from VIEWS, conflictforecast.org, and ACLED CAST.

Standalone script (separate from run_pipeline.py) because these
forecasts write to the dedicated ``conflict_forecasts`` table, not
through the Resolver precedence/delta pipeline.

Usage:
    python -m resolver.tools.fetch_conflict_forecasts
    python -m resolver.tools.fetch_conflict_forecasts --sources views
    python -m resolver.tools.fetch_conflict_forecasts --sources conflictforecast_org
    python -m resolver.tools.fetch_conflict_forecasts --sources acled_cast
    python -m resolver.tools.fetch_conflict_forecasts --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from typing import Sequence

import pandas as pd

LOG = logging.getLogger(__name__)

# Lazy-import connectors to keep the module importable without pandas at
# parse time (mirrors the pattern in run_pipeline.py).
_FORECAST_CONNECTORS = {
    "views": "resolver.connectors.views.ViewsConnector",
    "conflictforecast_org": "resolver.connectors.conflictforecast.ConflictForecastOrgConnector",
    "acled_cast": "resolver.connectors.acled_cast.AcledCastConnector",
}

_EXPECTED_COLUMNS = [
    "source",
    "iso3",
    "hazard_code",
    "metric",
    "lead_months",
    "value",
    "forecast_issue_date",
    "target_month",
    "model_version",
]


def _import_connector(dotted_path: str):
    """Import a connector class from a dotted path."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def fetch_and_store(
    *,
    db_url: str | None = None,
    sources: Sequence[str] | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Fetch conflict forecasts and write to DuckDB.

    Parameters
    ----------
    db_url : override the DuckDB URL (default: from pythia config)
    sources : list of connector names (default: all)
    dry_run : if True, fetch but don't write to DB

    Returns
    -------
    dict mapping source name → row count
    """
    if sources is None:
        sources = list(_FORECAST_CONNECTORS.keys())

    unknown = [s for s in sources if s not in _FORECAST_CONNECTORS]
    if unknown:
        raise ValueError(f"Unknown forecast source(s): {unknown}")

    all_dfs: list[pd.DataFrame] = []
    row_counts: dict[str, int] = {}

    for name in sources:
        LOG.info("[fetch_conflict_forecasts] fetching %s ...", name)
        cls = _import_connector(_FORECAST_CONNECTORS[name])
        connector = cls()

        try:
            df = connector.fetch_forecasts()
        except Exception as exc:
            LOG.error("[fetch_conflict_forecasts] %s failed: %s", name, exc)
            row_counts[name] = 0
            continue

        if df.empty:
            LOG.info("[fetch_conflict_forecasts] %s returned 0 rows", name)
            row_counts[name] = 0
            continue

        # Validate expected columns
        missing = [c for c in _EXPECTED_COLUMNS if c not in df.columns]
        if missing:
            LOG.error(
                "[fetch_conflict_forecasts] %s missing columns: %s",
                name, missing,
            )
            row_counts[name] = 0
            continue

        row_counts[name] = len(df)
        all_dfs.append(df[_EXPECTED_COLUMNS])
        LOG.info("[fetch_conflict_forecasts] %s: %d rows", name, len(df))

    if not all_dfs:
        LOG.info("[fetch_conflict_forecasts] no data to write")
        return row_counts

    combined = pd.concat(all_dfs, ignore_index=True)

    if dry_run:
        LOG.info(
            "[fetch_conflict_forecasts] DRY RUN — would write %d rows",
            len(combined),
        )
        print(f"\n--- DRY RUN: {len(combined)} total rows ---")
        print(f"Sources: {dict(row_counts)}")
        print(f"Countries: {combined['iso3'].nunique()}")
        print(f"Metrics: {combined['metric'].unique().tolist()}")
        print(combined.head(20).to_string(index=False))
        return row_counts

    _write_to_db(combined, db_url=db_url)
    LOG.info(
        "[fetch_conflict_forecasts] wrote %d rows to conflict_forecasts",
        len(combined),
    )
    return row_counts


def _write_to_db(df: pd.DataFrame, *, db_url: str | None = None) -> None:
    """Write forecast rows to the conflict_forecasts table.

    Uses upsert semantics: delete existing rows for the same
    (source, forecast_issue_date) then insert.
    """
    from pythia.db.schema import connect, ensure_schema

    con = connect()
    try:
        ensure_schema(con)

        # Delete stale rows for each (source, forecast_issue_date) combo
        combos = df[["source", "forecast_issue_date"]].drop_duplicates()
        for _, row in combos.iterrows():
            con.execute(
                "DELETE FROM conflict_forecasts "
                "WHERE source = ? AND forecast_issue_date = ?",
                [row["source"], row["forecast_issue_date"]],
            )

        # Deduplicate: keep last row per unique key to avoid constraint violations
        # (some upstream CSVs contain multiple rows per country, e.g. time-series)
        key_cols = ["source", "iso3", "hazard_code", "metric", "lead_months", "forecast_issue_date"]
        df = df.drop_duplicates(subset=key_cols, keep="last")

        # Insert new rows
        con.execute(
            "INSERT INTO conflict_forecasts "
            "(source, iso3, hazard_code, metric, lead_months, value, "
            " forecast_issue_date, target_month, model_version) "
            "SELECT source, iso3, hazard_code, metric, lead_months, value, "
            "       forecast_issue_date, target_month, model_version "
            "FROM df"
        )
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch conflict forecasts from VIEWS, conflictforecast.org, and ACLED CAST",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=list(_FORECAST_CONNECTORS.keys()),
        default=None,
        help="Which sources to fetch (default: all)",
    )
    parser.add_argument(
        "--db",
        dest="db_url",
        default=None,
        help="Override DuckDB URL",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch data but don't write to DB",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.db_url:
        import os
        os.environ["PYTHIA_DB_URL"] = args.db_url

    try:
        counts = fetch_and_store(
            db_url=args.db_url,
            sources=args.sources,
            dry_run=args.dry_run,
        )
        total = sum(counts.values())
        print(f"\nDone. {total} rows across {len(counts)} source(s): {counts}")
    except Exception as exc:
        LOG.error("Fatal: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
