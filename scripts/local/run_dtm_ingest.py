"""Local helper to run DTM ingestion and export into DuckDB.

This script wraps the existing DTM connector and export_facts pipeline to make
it easy to run a rolling backfill or monthly top-up from a local machine.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path


def _compute_backfill_window(months: int = 36) -> tuple[str, str]:
    """Compute a rolling backfill window ending with the last full month.

    Example:
        today = 2025-12-01
        end = 2025-11-30 (last day of previous month)
        start = 2022-12-01 for months=36 (36 months inclusive)
    """
    today = dt.date.today()
    first_of_this_month = today.replace(day=1)
    end = first_of_this_month - dt.timedelta(days=1)

    total_months = end.year * 12 + (end.month - 1)
    start_total = total_months - (months - 1)
    start_year, start_month_index = divmod(start_total, 12)
    start_month = start_month_index + 1
    start = dt.date(start_year, start_month, 1)

    return start.isoformat(), end.isoformat()


def _compute_topup_window() -> tuple[str, str]:
    """Compute a top-up window for the previous calendar month."""
    today = dt.date.today()
    first_of_this_month = today.replace(day=1)
    end = first_of_this_month - dt.timedelta(days=1)
    start = end.replace(day=1)
    return start.isoformat(), end.isoformat()


def _repo_root() -> Path:
    """Return the repository root (two levels above this file)."""
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local DTM ingestion + export into DuckDB (backfill or monthly top-up)."
    )
    parser.add_argument(
        "--mode",
        choices=["backfill", "topup"],
        default="backfill",
        help="Ingestion mode: 'backfill' (rolling window, default) or 'topup' (previous month only).",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=36,
        help="Backfill window in months for --mode backfill (default: 36).",
    )
    parser.add_argument(
        "--start",
        help="Override RESOLVER_START_ISO (YYYY-MM-DD). If set, overrides computed window start.",
    )
    parser.add_argument(
        "--end",
        help="Override RESOLVER_END_ISO (YYYY-MM-DD). If set, overrides computed window end.",
    )
    parser.add_argument(
        "--db",
        help=(
            "DuckDB URL to write to, e.g. 'duckdb:///data/resolver.duckdb'. "
            "If not set, uses RESOLVER_DB_URL or defaults to 'duckdb:///data/resolver.duckdb'."
        ),
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Log level for downstream commands (default: LOG_LEVEL env or 'INFO').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only run DTM connector (stage CSV); do NOT write to DuckDB.",
    )

    args = parser.parse_args()

    if args.start and args.end:
        start_iso = args.start
        end_iso = args.end
    else:
        if args.mode == "backfill":
            start_iso, end_iso = _compute_backfill_window(months=args.months)
        else:
            start_iso, end_iso = _compute_topup_window()

    repo_root = _repo_root()
    os.chdir(repo_root)

    os.environ["RESOLVER_START_ISO"] = start_iso
    os.environ["RESOLVER_END_ISO"] = end_iso
    os.environ["LOG_LEVEL"] = args.log_level

    db_url = args.db or os.environ.get("RESOLVER_DB_URL") or "duckdb:///data/resolver.duckdb"
    os.environ["RESOLVER_DB_URL"] = db_url

    print(f"[run_dtm_ingest] Mode={args.mode}, window={start_iso} → {end_iso}")
    print(f"[run_dtm_ingest] RESOLVER_DB_URL={db_url}")
    sys.stdout.flush()

    dtm_cmd = [sys.executable, "-m", "resolver.ingestion.dtm_client"]
    print(f"[run_dtm_ingest] Running DTM connector: {' '.join(dtm_cmd)}")
    sys.stdout.flush()
    subprocess.run(dtm_cmd, check=True)

    if args.dry_run:
        print("[run_dtm_ingest] Dry run enabled; skipping export_facts / DB write.")
        return

    out_dir = repo_root / "resolver" / "exports" / f"dtm_local_{args.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_str = str(out_dir)

    export_cmd = [
        sys.executable,
        "-m",
        "resolver.tools.export_facts",
        "--in",
        "resolver/staging",
        "--out",
        out_dir_str,
        "--config",
        "resolver/tools/export_config.yml",
        "--write-db",
        "1",
        "--db",
        db_url,
    ]
    print(f"[run_dtm_ingest] Running export_facts: {' '.join(export_cmd)}")
    sys.stdout.flush()
    subprocess.run(export_cmd, check=True)

    print("[run_dtm_ingest] Completed DTM ingestion + export.")


if __name__ == "__main__":
    main()
