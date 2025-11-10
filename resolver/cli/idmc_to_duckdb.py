"""IDMC → DuckDB helper CLI for one-command exports."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

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
        "--db",
        default=None,
        help="Alias for --db-url; DuckDB URL or path",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Directory for exporter diagnostics (default: %(default)s)",
    )
    parser.add_argument(
        "--facts-csv",
        default=None,
        help="Optional canonical facts.csv to load instead of staging",
    )
    parser.add_argument(
        "--write-db",
        action="store_true",
        help="Enable DuckDB writes (requires --db or --db-url)",
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
    parser.add_argument(
        "--append-summary",
        default=None,
        help="Optional file to append a markdown summary of DuckDB verification",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    LOGGER.setLevel(level)

    staging_dir = Path(args.staging_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    db_arg = args.db if getattr(args, "db", None) else args.db_url
    duckdb_url, duckdb_path = _normalize_db_url_arg(db_arg)
    writing_requested = bool(args.write_db)
    LOGGER.info(
        "idmc_to_duckdb.start | staging=%s out=%s duckdb_url=%s duckdb_path=%s",
        staging_dir,
        out_dir,
        duckdb_url,
        duckdb_path,
    )

    facts_dataframe: pd.DataFrame | None = None
    exporter_result: export_facts.ExportResult | None = None
    db_stats: dict[str, dict[str, object]] = {}

    prepared_resolved: pd.DataFrame | None = None
    prepared_deltas: pd.DataFrame | None = None

    if args.facts_csv:
        facts_path = Path(args.facts_csv)
        if not facts_path.is_file():
            LOGGER.error("facts.csv not found at %s", facts_path)
            print(f"facts.csv not found: {facts_path}", file=sys.stderr)
            return 2
        LOGGER.info(
            "facts_csv.selected | path=%s write_db=%s",
            facts_path,
            writing_requested,
        )
        try:
            facts_dataframe = pd.read_csv(facts_path)
        except Exception as exc:  # pragma: no cover - pandas-level parsing issues
            LOGGER.error("Failed to read facts.csv", exc_info=True)
            print(f"Failed to read facts.csv: {exc}", file=sys.stderr)
            return 2
        prepared_resolved, prepared_deltas = export_facts.prepare_duckdb_tables(
            facts_dataframe
        )
        db_stats = {}
    else:
        exporter_args = dict(
            inp=staging_dir,
            out_dir=out_dir,
            write_db="0",
            db_url=duckdb_url,
            only_strategy="idmc-staging",
        )
        LOGGER.info(
            "exporter.invoke | input=%s out=%s db_url=%s write_db=%s",
            staging_dir,
            out_dir,
            duckdb_url,
            "1" if writing_requested else "0",
        )

        try:
            exporter_result = export_facts.export_facts(**exporter_args)
        except export_facts.DuckDBWriteError as exc:
            LOGGER.error("Exporter DuckDB write failed", exc_info=True)
            print(f"DuckDB write failed: {exc}", file=sys.stderr)
            return 3
        except export_facts.ExportError as exc:
            LOGGER.error("Exporter failed", exc_info=True)
            print(f"Export failed: {exc}", file=sys.stderr)
            return 3

        facts_dataframe = exporter_result.dataframe
        if facts_dataframe is None and exporter_result.csv_path:
            csv_path = Path(exporter_result.csv_path)
            if csv_path.is_file():
                try:
                    facts_dataframe = pd.read_csv(csv_path)
                except Exception:  # pragma: no cover - diagnostics only
                    LOGGER.debug("Failed to backfill dataframe from %s", csv_path, exc_info=True)
        db_stats = exporter_result.db_stats or {}
        prepared_resolved = exporter_result.resolved_df or prepared_resolved
        prepared_deltas = exporter_result.deltas_df or prepared_deltas

    total_rows = 0
    resolved_count = 0
    deltas_count = 0
    zero_facts = facts_dataframe is None or facts_dataframe.empty
    write_results: dict[str, duckdb_io.UpsertResult] = {}
    if writing_requested:
        if zero_facts:
            LOGGER.error("write_db requested but no canonical facts rows found")
            print("No canonical facts rows available for DuckDB write", file=sys.stderr)
        else:
            conn = None
            try:
                conn = duckdb_io.get_db(duckdb_url)
            except duckdb.Error as exc:
                LOGGER.error("Writer connection failed", exc_info=True)
                print(f"DuckDB write failed: {exc}", file=sys.stderr)
                return 3
            try:
                write_results = duckdb_io.write_facts_tables(
                    conn,
                    facts_resolved=prepared_resolved,
                    facts_deltas=prepared_deltas,
                )
            except duckdb.Error as exc:
                LOGGER.error("DuckDB upsert failed", exc_info=True)
                print(f"DuckDB write failed: {exc}", file=sys.stderr)
                duckdb_io.close_db(conn)
                return 3
            except Exception as exc:
                LOGGER.error("DuckDB write encountered an unexpected error", exc_info=True)
                print(f"DuckDB write failed: {exc}", file=sys.stderr)
                duckdb_io.close_db(conn)
                return 3
            try:
                deltas_count = _safe_count(conn, "facts_deltas")
                resolved_count = _safe_count(conn, "facts_resolved")
            except duckdb.Error as exc:
                LOGGER.error("Verification query failed", exc_info=True)
                print(f"Verification failed: {exc}", file=sys.stderr)
                duckdb_io.close_db(conn)
                return 3
            finally:
                duckdb_io.close_db(conn)

            total_rows = deltas_count + resolved_count

            if total_rows <= 0:
                LOGGER.warning(
                    "Verification: DuckDB tables empty after export (flows may be empty)"
                )

    if write_results:
        merged_stats = dict(db_stats)
        merged_stats.update({name: result.to_dict() for name, result in write_results.items()})
        db_stats = merged_stats
    def _stats(values: dict[str, object] | None) -> tuple[int, int]:
        if not values:
            return 0, 0
        delta = int(values.get("rows_delta", 0) or 0)
        total = int(values.get("rows_after", values.get("rows_before", 0) or 0) or 0)
        return delta, total

    resolved_delta, resolved_total = _stats(db_stats.get("facts_resolved"))
    deltas_delta, deltas_total = _stats(db_stats.get("facts_deltas"))
    total_delta = resolved_delta + deltas_delta

    def emit(message: str) -> None:
        print(message)

    rows_message = f"✅ Wrote {total_delta} rows to DuckDB"
    if not writing_requested:
        rows_message += " (dry-run)"
    emit(rows_message)
    emit(f" - facts_resolved Δ={resolved_delta} total={resolved_total}")
    emit(f" - facts_deltas  Δ={deltas_delta} total={deltas_total}")

    if writing_requested:
        emit(
            "Verification: facts_deltas={deltas} facts_resolved={resolved} total={total}".format(
                deltas=deltas_count,
                resolved=resolved_count,
                total=total_rows,
            )
        )

    staging_warnings = []
    if not args.facts_csv and not (staging_dir / "stock.csv").is_file():
        staging_warnings.append("stock.csv: not present")

    collected_warnings = _gather_warnings(
        staging_warnings,
        (exporter_result.warnings if exporter_result else []) or [],
        ((exporter_result.report or {}) if exporter_result else {}).get("warnings") or [],
    )

    if collected_warnings:
        emit("Warnings:")
        for message in collected_warnings:
            emit(f" - {message}")
    else:
        emit("Warnings: none")

    exit_code = 0
    if writing_requested and zero_facts:
        exit_code = 4
    if collected_warnings and args.strict and exit_code == 0:
        exit_code = 2

    emit(
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

    summary_path = args.append_summary
    if summary_path:
        summary_lines = [
            "## IDMC — DuckDB verification",
            "",
        ]
        status_line = "executed" if writing_requested else "dry-run"
        summary_lines.append(f"- **Write mode:** {status_line}")
        summary_lines.append(
            f"- **Rows delta:** {total_delta} (facts_resolved Δ={resolved_delta}, facts_deltas Δ={deltas_delta})"
        )
        if writing_requested:
            summary_lines.append(
                f"- **Verification totals:** facts_resolved={resolved_count} facts_deltas={deltas_count} total={total_rows}"
            )
        if collected_warnings:
            joined = "; ".join(collected_warnings)
            summary_lines.append(f"- **Warnings:** {joined}")
        else:
            summary_lines.append("- **Warnings:** none")
        summary_lines.append("")
        try:
            summary_target = Path(summary_path)
            summary_target.parent.mkdir(parents=True, exist_ok=True)
            with summary_target.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(summary_lines))
        except OSError as exc:
            LOGGER.warning("Failed to append DuckDB summary to %s: %s", summary_path, exc)

    return exit_code


def main(argv: Sequence[str] | None = None) -> None:
    sys.exit(run(argv))


if __name__ == "__main__":
    main()
