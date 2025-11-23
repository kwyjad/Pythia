"""CLI wrapper for building Resolver snapshots from DuckDB."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

from resolver.db.conn_shared import canonicalize_duckdb_target
from resolver.db import duckdb_io
from resolver.snapshot.builder import build_monthly_snapshot


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build monthly Resolver snapshots directly from DuckDB."
    )
    parser.add_argument(
        "--db-url",
        required=True,
        help="DuckDB URL, e.g. duckdb:///data/resolver_backfill.duckdb",
    )
    parser.add_argument(
        "--month",
        dest="months",
        action="append",
        help="Month to snapshot (YYYY-MM). Can be passed multiple times.",
    )
    parser.add_argument(
        "--months-back",
        dest="months_back",
        type=int,
        default=36,
        help="If no explicit --month is given, snapshot the last N months (default: 36).",
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        default="snapshots",
        help="Directory to write snapshot parquet files (default: snapshots).",
    )
    return parser.parse_args(argv)


def _compute_last_months(n: int) -> List[str]:
    months: List[str] = []
    today = dt.date.today().replace(day=1)
    for _ in range(n):
        prev_last_day = today - dt.timedelta(days=1)
        ym = f"{prev_last_day.year:04d}-{prev_last_day.month:02d}"
        months.append(ym)
        today = prev_last_day.replace(day=1)
    return sorted(dict.fromkeys(months))


def _resolve_db_target(db_url: str) -> tuple[str, str]:
    if not db_url.startswith("duckdb://"):
        sys.stderr.write(
            f"[snapshot_from_db] Unsupported db-url format (expected duckdb:///...): {db_url}\n"
        )
        sys.exit(1)
    try:
        path, canonical_url = canonicalize_duckdb_target(db_url)
    except Exception as exc:  # pragma: no cover - defensive
        sys.stderr.write(f"[snapshot_from_db] Failed to canonicalize db-url: {exc}\n")
        sys.exit(1)
    if path != ":memory:" and not Path(path).exists():
        sys.stderr.write(
            f"[snapshot_from_db] DuckDB file does not exist: {Path(path)}\n"
        )
        sys.exit(1)
    return path, canonical_url


def _ensure_months(args: argparse.Namespace) -> Iterable[str]:
    if args.months:
        return sorted(dict.fromkeys(args.months))
    return _compute_last_months(args.months_back)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    _, canonical_db_url = _resolve_db_target(args.db_url.strip())
    months = list(_ensure_months(args))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = duckdb_io.get_db(canonical_db_url)
    try:
        for ym in months:
            sys.stderr.write(
                f"[snapshot_from_db] Building snapshot for ym={ym} -> {out_dir / ym / 'facts.parquet'}\n"
            )
            result = build_monthly_snapshot(
                conn,
                ym=ym,
                snapshot_root=out_dir,
                write_parquet=True,
            )
            sys.stderr.write(
                "[snapshot_from_db] ym={ym} rows: resolved={res} deltas={deltas} acled={acled} total={total}\n".format(
                    ym=result.ym,
                    res=result.resolved_rows,
                    deltas=result.delta_rows,
                    acled=result.acled_rows,
                    total=result.snapshot_rows,
                )
            )
    finally:
        duckdb_io.close_db(conn)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
