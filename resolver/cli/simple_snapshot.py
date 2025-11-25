from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Optional

from resolver.db import duckdb_io


def _table_exists(con, table: str) -> bool:
    try:
        rows = con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = ?",
            [table],
        ).fetchall()
    except Exception:
        return False
    return bool(rows)


def _extract_one_table_for_month(con, table: str, ym: str, out_path: Path) -> bool:
    """Extract rows for ``ym`` from ``table`` into ``out_path`` as Parquet."""

    if not _table_exists(con, table):
        print(f"[simple_snapshot] Table '{table}' does not exist; skipping.")
        return False

    query = f"SELECT * FROM {table} WHERE ym = ?"
    try:
        con.execute(query, [ym])
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[simple_snapshot] Failed to query {table} for ym={ym}: {exc}")
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        con.execute(
            "COPY (SELECT * FROM {table} WHERE ym = ?) TO ? (FORMAT PARQUET)".format(
                table=table
            ),
            [ym, str(out_path)],
        )
    except Exception as exc:  # pragma: no cover - defensive
        print(
            f"[simple_snapshot] Failed to write Parquet for {table}, ym={ym}, path={out_path}: {exc}"
        )
        return False

    print(f"[simple_snapshot] Wrote snapshot for table '{table}', ym={ym} to {out_path}")
    return True


def _default_ym_from_istanbul_today() -> str:
    """Return YYYY-MM for the last full month in Europe/Istanbul local time."""

    today = dt.datetime.utcnow().date()
    approx = today - dt.timedelta(days=15)
    return f"{approx.year:04d}-{approx.month:02d}"


def write_simple_snapshot(
    db_path: Path,
    ym: str,
    out_root: Path | str = Path("snapshots") / "simple",
    include_deltas: bool = True,
) -> None:
    """Extract a snapshot for ``ym`` from DuckDB into Parquet files."""

    db_path = Path(db_path)
    out_root = Path(out_root)
    out_dir = out_root / ym

    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB database not found: {db_path}")

    db_url = f"duckdb:///{db_path}"
    con = duckdb_io.get_db(db_url)
    try:
        any_written = False

        facts_resolved_path = out_dir / "facts_resolved.parquet"
        if _extract_one_table_for_month(con, "facts_resolved", ym, facts_resolved_path):
            any_written = True

        if include_deltas:
            facts_deltas_path = out_dir / "facts_deltas.parquet"
            if _extract_one_table_for_month(con, "facts_deltas", ym, facts_deltas_path):
                any_written = True

        if not any_written:
            print(
                f"[simple_snapshot] No rows written for ym={ym}; check that facts_resolved/facts_deltas contain that month."
            )
    finally:
        duckdb_io.close_db(con)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a simple, read-only snapshot from DuckDB into Parquet files.\n"
            "- facts_resolved.parquet (required table)\n"
            "- facts_deltas.parquet (optional)\n"
        )
    )
    parser.add_argument(
        "--db",
        dest="db_path",
        required=True,
        help="Path to DuckDB database (e.g., data/resolver_backfill.duckdb)",
    )
    parser.add_argument(
        "--ym",
        dest="ym",
        default=None,
        help="Target year-month (YYYY-MM). If omitted, use last full month (approx).",
    )
    parser.add_argument(
        "--out-root",
        dest="out_root",
        default=str(Path("snapshots") / "simple"),
        help="Root directory for snapshots (default: snapshots/simple)",
    )
    parser.add_argument(
        "--no-deltas",
        dest="include_deltas",
        action="store_false",
        help="Do not write facts_deltas snapshot (only facts_resolved).",
    )

    args = parser.parse_args(argv)

    ym = args.ym or _default_ym_from_istanbul_today()
    try:
        write_simple_snapshot(Path(args.db_path), ym, Path(args.out_root), args.include_deltas)
    except FileNotFoundError as exc:
        print(f"[simple_snapshot] {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[simple_snapshot] Unexpected error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
