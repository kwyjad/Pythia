# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Iterable

from resolver.db import duckdb_io

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_MISSING_REQUIRED = 2
EXIT_REGRESSION = 3


class MissingRequiredTablesError(RuntimeError):
    def __init__(self, missing: list[str]):
        super().__init__(f"Missing required tables: {missing}")
        self.missing = missing


def parse_table_list(csv_str: str | None) -> list[str]:
    if not csv_str:
        return []
    return [item.strip() for item in csv_str.split(",") if item.strip()]


def _remove_stale_wal(db_path: Path) -> None:
    """Delete any WAL file next to the DB before opening.

    Downloaded artifacts may include a stale ``.wal`` file from the process
    that originally wrote the DB.  Replaying that WAL on a different machine
    crashes DuckDB with ``InternalError: Failure while replaying WAL file …
    Calling DatabaseManager::GetDefaultDatabase with no default database set``.
    Since the DB file itself is the authoritative snapshot, the WAL is
    expendable.
    """
    wal = db_path.with_suffix(db_path.suffix + ".wal")
    if wal.exists():
        print(f"Removing stale WAL file: {wal}")
        wal.unlink()


def compute_signature(
    db_path: str | os.PathLike[str],
    required: Iterable[str],
    optional: Iterable[str],
    *,
    allow_missing_required: bool = False,
    all_tables: bool = False,
) -> dict:
    db_file = Path(db_path)
    if not db_file.is_file():
        raise FileNotFoundError(f"Database not found at {db_file}")

    required_list = list(required)
    optional_list = list(optional)

    _remove_stale_wal(db_file)
    conn = duckdb_io.get_db(str(db_file))
    try:
        tables = sorted(row[0] for row in conn.execute("SHOW TABLES").fetchall())
        counts: dict[str, int | None] = {}
        # ``all_tables`` widens the baseline to every table in the DB. The
        # signature's own guard still reads only the required/optional lists;
        # the extra counts exist so a debug bundle can report a before/after
        # delta for tables nobody thought to name in advance — which is
        # exactly where an unexplained ingest gap tends to sit.
        wanted = required_list + optional_list + (tables if all_tables else [])
        for table in dict.fromkeys(wanted):
            counts[table] = (
                int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]) if table in tables else None
            )
    finally:
        duckdb_io.close_db(conn)

    missing = [table for table in required_list if counts.get(table) is None]
    if missing and not allow_missing_required:
        raise MissingRequiredTablesError(missing)

    signature = {
        "db_path": str(db_file.resolve()),
        "size_bytes": db_file.stat().st_size,
        "tables": tables,
        "required_counts": {table: counts.get(table) for table in required_list},
        "optional_counts": {table: counts.get(table) for table in optional_list},
    }
    if missing:
        signature["missing_required"] = missing
    if all_tables:
        signature["all_counts"] = {table: counts.get(table) for table in tables}
    return signature


def write_signature(signature: dict, out_path: str | os.PathLike[str]) -> None:
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(signature, f, indent=2, sort_keys=True)


def compare_signatures(before: dict, after: dict, required: Iterable[str]) -> list[str]:
    regressions: list[str] = []
    before_counts = before.get("required_counts", {})
    after_counts = after.get("required_counts", {})
    for table in required:
        before_count = before_counts.get(table)
        after_count = after_counts.get(table)
        if before_count is not None and after_count is not None and after_count < before_count:
            regressions.append(f"{table}: {after_count} < {before_count}")
    return regressions


def compare_optional_signatures(
    before: dict, after: dict, optional: Iterable[str]
) -> list[str]:
    """Regressions in OPTIONAL tables — reported, never fatal.

    Optional tables may legitimately be absent (an older DB predating the
    feature), so absence is not a regression when the table never existed
    in the baseline. What IS worth a warning: a table the baseline HAD that
    is now missing, and a row count that shrank. Without this the optional
    half of the signature was computed, written to the JSON and never read
    — a decorative guard (the haz_* tables could drop to zero and both the
    resolver_update and haz_backcast compare steps would exit 0).
    """

    warnings: list[str] = []
    before_counts = before.get("optional_counts", {})
    after_counts = after.get("optional_counts", {})
    for table in optional:
        before_count = before_counts.get(table)
        after_count = after_counts.get(table)
        if before_count is None:
            continue  # the baseline never had it — nothing to regress from
        if after_count is None:
            warnings.append(f"{table}: present before ({before_count} rows), now missing")
        elif after_count < before_count:
            warnings.append(f"{table}: {after_count} < {before_count}")
    return warnings


def print_signature(signature: dict, required: Iterable[str], optional: Iterable[str]) -> None:
    required_list = list(required)
    optional_list = list(optional)
    print(f"DB size (bytes): {signature['size_bytes']}")
    print(f"Tables ({len(signature['tables'])}): {', '.join(signature['tables'])}")
    if signature.get("missing_required"):
        print(f"Missing required tables: {', '.join(signature['missing_required'])}")
    print("Required row counts:")
    for table in required_list:
        print(f"  - {table}: {signature['required_counts'].get(table)}")
    print("Optional row counts:")
    for table in optional_list:
        print(f"  - {table}: {signature['optional_counts'].get(table)}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute and compare DuckDB signatures.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    write_parser = subparsers.add_parser("write", help="Write a database signature")
    write_parser.add_argument("--db", required=True, help="Path to the DuckDB database file")
    write_parser.add_argument("--required", default="", help="Comma-separated required tables")
    write_parser.add_argument("--optional", default="", help="Comma-separated optional tables")
    write_parser.add_argument("--out", required=True, help="Where to write the signature JSON")
    write_parser.add_argument(
        "--all-tables",
        action="store_true",
        help="Also count every table in the DB (adds an all_counts block)",
    )

    compare_parser = subparsers.add_parser("compare", help="Compare two database signatures")
    compare_parser.add_argument("--before", required=True, help="Path to the baseline signature JSON")
    compare_parser.add_argument("--after-db", required=True, help="Path to the DuckDB database file to compare")
    compare_parser.add_argument("--required", default="", help="Comma-separated required tables")
    compare_parser.add_argument("--optional", default="", help="Comma-separated optional tables")
    compare_parser.add_argument("--out", required=True, help="Where to write the updated signature JSON")
    compare_parser.add_argument(
        "--all-tables",
        action="store_true",
        help="Also count every table in the DB (adds an all_counts block)",
    )

    return parser


def _load_signature(path: str | os.PathLike[str]) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    required = parse_table_list(args.required)
    optional = parse_table_list(args.optional)

    try:
        if args.command == "write":
            signature = compute_signature(
                args.db, required, optional, all_tables=bool(getattr(args, "all_tables", False))
            )
            write_signature(signature, args.out)
            print_signature(signature, required, optional)
            return EXIT_OK

        if args.command == "compare":
            before = _load_signature(args.before)
            after_signature = compute_signature(
                args.after_db,
                required,
                optional,
                allow_missing_required=True,
                all_tables=bool(getattr(args, "all_tables", False)),
            )
            write_signature(after_signature, args.out)
            regressions = compare_signatures(before, after_signature, required)
            optional_warnings = compare_optional_signatures(before, after_signature, optional)
            print_signature(after_signature, required, optional)
            # Optional-table regressions warn but never fail: optional tables
            # may legitimately be absent on older DBs, and a hard fail here
            # would block workflows that never touch them.
            for warning in optional_warnings:
                print(f"::warning::optional table regressed: {warning}")
            if after_signature.get("missing_required"):
                sys.stderr.write(
                    "Missing required tables: " + ", ".join(after_signature["missing_required"]) + "\n"
                )
                return EXIT_MISSING_REQUIRED
            if regressions:
                sys.stderr.write("Required tables regressed: " + ", ".join(regressions) + "\n")
                return EXIT_REGRESSION
            return EXIT_OK
    except MissingRequiredTablesError as exc:
        sys.stderr.write(str(exc) + "\n")
        return EXIT_MISSING_REQUIRED
    except FileNotFoundError as exc:
        sys.stderr.write(str(exc) + "\n")
        return EXIT_ERROR
    except Exception as exc:  # pragma: no cover - defensive catch for unexpected failures
        sys.stderr.write(f"Unexpected error: {exc}\n")
        return EXIT_ERROR

    return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
