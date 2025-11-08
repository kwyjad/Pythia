"""IDMC → DuckDB helper CLI for one-command exports."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import duckdb

from resolver.db import duckdb_io
from resolver.tools import export_facts

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)

DEFAULT_STAGING = Path("resolver/staging/idmc")
DEFAULT_OUT = Path("diagnostics/ingestion/export_preview")
DEFAULT_DB_PATH = Path("./resolver_data/resolver.duckdb")


def _normalize_db_url_arg(raw: str | None) -> tuple[str, str]:
    """Return (DuckDB URL, filesystem path) for ``raw`` input."""

    candidate = (raw or "").strip() or str(DEFAULT_DB_PATH)
    if "://" in candidate and not candidate.lower().startswith("duckdb://"):
        return candidate, candidate
    if candidate.lower().startswith("duckdb://"):
        if candidate.lower().startswith("duckdb:///"):
            fs_part = candidate[len("duckdb:///") :]
            fs_path = Path(fs_part).expanduser().resolve()
            return f"duckdb:///{fs_path.as_posix()}", str(fs_path)
        return candidate, candidate
    fs_path = Path(candidate).expanduser().resolve()
    return f"duckdb:///{fs_path.as_posix()}", str(fs_path)


def _safe_count(conn: duckdb.DuckDBPyConnection, table: str) -> int:
    """Return the row count for ``table`` treating missing tables as zero."""

    try:
        value = conn.execute(
            f"SELECT COALESCE(COUNT(*), 0) FROM {table}"
        ).fetchone()[0]
    except duckdb.Error as exc:
        message = str(exc).lower()
        if "no such table" in message or "does not exist" in message:
            LOGGER.info("verification.table_missing | table=%s", table)
            return 0
        raise
    return int(value or 0)


def _gather_warnings(*sources: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for collection in sources:
        for entry in collection or []:
            if not entry:
                continue
            message = str(entry)
            if message not in seen:
                seen.add(message)
                ordered.append(message)
    return ordered


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staging-dir",
        default=str(DEFAULT_STAGING),
        help="Directory containing IDMC staging exports (default: %(default)s)",
    )
    parser.add_argument(
        "--db-url",
        default=str(DEFAULT_DB_PATH),
        help="DuckDB URL or path override (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Directory for exporter diagnostics (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("RESOLVER_LOG_LEVEL", "INFO"),
        help="Set logging level (default: %(default)s)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if the exporter reports warnings",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    LOGGER.setLevel(level)

    staging_dir = Path(args.staging_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    duckdb_url, duckdb_path = _normalize_db_url_arg(args.db_url)
    LOGGER.info(
        "idmc_to_duckdb.start | staging=%s out=%s duckdb_url=%s duckdb_path=%s",
        staging_dir,
        out_dir,
        duckdb_url,
        duckdb_path,
    )

    exporter_args = dict(
        inp=staging_dir,
        out_dir=out_dir,
        write_db="1",
        db_url=duckdb_url,
        only_strategy="idmc-staging",
    )
    LOGGER.info(
        "exporter.invoke | input=%s out=%s db_url=%s write_db=1",
        staging_dir,
        out_dir,
        duckdb_url,
    )

    try:
        result = export_facts.export_facts(**exporter_args)
    except export_facts.DuckDBWriteError as exc:
        LOGGER.error("Exporter DuckDB write failed", exc_info=True)
        print(f"DuckDB write failed: {exc}", file=sys.stderr)
        return 3
    except export_facts.ExportError as exc:
        LOGGER.error("Exporter failed", exc_info=True)
        print(f"Export failed: {exc}", file=sys.stderr)
        return 3

    db_stats = result.db_stats or {}

    def _stats(values: dict[str, object] | None) -> tuple[int, int]:
        if not values:
            return 0, 0
        delta = int(values.get("rows_delta", 0) or 0)
        total = int(values.get("rows_after", values.get("rows_before", 0) or 0) or 0)
        return delta, total

    resolved_delta, resolved_total = _stats(db_stats.get("facts_resolved"))
    deltas_delta, deltas_total = _stats(db_stats.get("facts_deltas"))
    total_delta = resolved_delta + deltas_delta

    print(f"✅ Wrote {total_delta} rows to DuckDB")
    print(f" - facts_resolved Δ={resolved_delta} total={resolved_total}")
    print(f" - facts_deltas  Δ={deltas_delta} total={deltas_total}")

    try:
        conn = duckdb_io.get_db(duckdb_url)
    except duckdb.Error as exc:
        LOGGER.error("Verification connection failed", exc_info=True)
        print(f"Verification failed: {exc}", file=sys.stderr)
        return 3

    try:
        deltas_count = _safe_count(conn, "facts_deltas")
        resolved_count = _safe_count(conn, "facts_resolved")
    except duckdb.Error as exc:
        LOGGER.error("Verification query failed", exc_info=True)
        print(f"Verification failed: {exc}", file=sys.stderr)
        return 3

    total_rows = deltas_count + resolved_count
    print(
        "Verification: facts_deltas={deltas} facts_resolved={resolved} total={total}".format(
            deltas=deltas_count,
            resolved=resolved_count,
            total=total_rows,
        )
    )

    if total_rows <= 0:
        LOGGER.error("Verification failed: DuckDB contains no rows after export")
        return 3

    staging_warnings = []
    if not (staging_dir / "stock.csv").is_file():
        staging_warnings.append("stock.csv: not present")

    collected_warnings = _gather_warnings(
        staging_warnings,
        result.warnings or [],
        (result.report or {}).get("warnings") or [],
    )

    if collected_warnings:
        print("Warnings:")
        for message in collected_warnings:
            print(f" - {message}")
    else:
        print("Warnings: none")

    exit_code = 0
    if collected_warnings and args.strict:
        exit_code = 2

    print(
        (
            "Summary: resolved_total={resolved} deltas_total={deltas} total_rows={total} "
            "| delta_resolved={resolved_delta} delta_deltas={deltas_delta} exit={code}"
        ).format(
            resolved=resolved_count,
            deltas=deltas_count,
            total=total_rows,
            resolved_delta=resolved_delta,
            deltas_delta=deltas_delta,
            code=exit_code,
        )
    )

    return exit_code


def main(argv: Sequence[str] | None = None) -> None:
    sys.exit(run(argv))


if __name__ == "__main__":
    main()
