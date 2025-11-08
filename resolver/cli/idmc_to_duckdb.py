"""IDMC → DuckDB helper CLI for one-command exports."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Sequence

from resolver.db import duckdb_io
from resolver.db.conn_shared import canonicalize_duckdb_target
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
DEFAULT_DB_URL = "./resolver_data/resolver.duckdb"


def _normalize_db_url(raw: str | None) -> tuple[str, str, str]:
    """Return printable path, filesystem path, and DuckDB URL for ``raw``."""

    candidate = (raw or "").strip()
    env_default = os.getenv("RESOLVER_DB_URL", "").strip()
    if not candidate:
        candidate = env_default or DEFAULT_DB_URL
    printable = (raw or "").strip() or (env_default or DEFAULT_DB_URL)

    path: str | None = None
    url = candidate
    try:
        path, url = canonicalize_duckdb_target(candidate)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.error(
            "Failed to canonicalise DuckDB target | raw=%s error=%s",
            candidate,
            exc,
        )
        path = None
        url = candidate
    if not str(url).startswith("duckdb://"):
        try:
            if str(candidate).lower().endswith(".duckdb"):
                resolved = Path(candidate).expanduser().resolve()
            else:
                resolved = Path(url).expanduser().resolve()
        except Exception:
            resolved = None
        if resolved is not None:
            path = resolved.as_posix()
            url = f"duckdb:///{resolved.as_posix()}"
    if path is None:
        path = url.replace("duckdb:///", "") if url.startswith("duckdb:///") else candidate
    if not printable:
        printable = candidate if isinstance(candidate, str) and candidate else path or url
    return printable, path, url


def _ensure_directory(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover - defensive
        LOGGER.debug("mkdir_failed | path=%s error=%s", path, exc)


def _ym_candidates(frame) -> list[str]:
    if frame is None:
        return []
    candidates: list[str] = []
    if "ym" in frame.columns:
        raw = frame["ym"].astype(str)
        candidates.extend(v.strip() for v in raw if v and v.strip())
    elif "as_of_date" in frame.columns:
        raw = frame["as_of_date"].astype(str)
        for value in raw:
            value = value.strip()
            if not value:
                continue
            candidates.append(value[:7])
    seen = []
    for ym in candidates:
        if ym not in seen:
            seen.append(ym)
    return seen


def _log_staging_inventory(staging_dir: Path) -> None:
    LOGGER.info("staging.directory | %s", staging_dir)
    expected = [
        ("flow.csv", staging_dir / "flow.csv"),
        ("idmc_facts_flow.parquet", staging_dir / "idmc_facts_flow.parquet"),
        ("stock.csv", staging_dir / "stock.csv"),
    ]
    expected_labels = {name for name, _ in expected}
    for label, path in expected:
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        LOGGER.debug("staging.file | name=%s exists=%s size=%s", label, exists, size)
    if staging_dir.exists():
        for path in staging_dir.iterdir():
            if path.name not in expected_labels:
                LOGGER.debug("staging.extra | %s", path.name)


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staging-dir",
        default=str(DEFAULT_STAGING),
        help="Directory containing IDMC staging exports (default: %(default)s)",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="DuckDB URL or path override (default: RESOLVER_DB_URL or ./resolver_data/resolver.duckdb)",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Directory for exporter diagnostics (default: %(default)s)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if the exporter reports warnings",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("RESOLVER_LOG_LEVEL", "INFO"),
        help="Set logging level (default: %(default)s)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    LOGGER.setLevel(level)

    staging_dir = Path(args.staging_dir)
    out_dir = Path(args.out)
    db_display, db_path, canonical_url = _normalize_db_url(args.db_url)

    LOGGER.info(
        "idmc_to_duckdb.start | staging=%s out=%s db_url=%s (display=%s path=%s)",
        staging_dir,
        out_dir,
        canonical_url,
        db_display,
        db_path,
    )
    LOGGER.debug(
        "environment | RESOLVER_DB_URL=%s RESOLVER_WRITE_DB=%s",
        os.getenv("RESOLVER_DB_URL", ""),
        os.getenv("RESOLVER_WRITE_DB", ""),
    )

    _ensure_directory(out_dir)
    if db_path and db_path != ":memory:" and not str(db_path).startswith("duckdb://"):
        try:
            parent = Path(db_path).expanduser().resolve().parent
        except Exception:
            parent = None
        if parent is not None:
            _ensure_directory(parent)
    _log_staging_inventory(staging_dir)

    flow_path = staging_dir / "flow.csv"
    parquet_path = staging_dir / "idmc_facts_flow.parquet"
    stock_path = staging_dir / "stock.csv"
    has_flow = flow_path.is_file()
    has_parquet = parquet_path.is_file()
    has_stock = stock_path.is_file()

    inferred_warnings: list[str] = []
    if not has_stock:
        inferred_warnings.append("stock.csv: not present")

    with ExitStack() as stack:
        input_dir = staging_dir
        if has_flow:
            temp_dir = stack.enter_context(
                tempfile.TemporaryDirectory(prefix="idmc_single_source_")
            )
            work_dir = Path(temp_dir)
            shutil.copy2(flow_path, work_dir / flow_path.name)
            LOGGER.info(
                "single_source.flow_only | work_dir=%s copy_stock=%s skip_parquet=%s",
                work_dir,
                has_stock,
                has_parquet,
            )
            if has_stock:
                shutil.copy2(stock_path, work_dir / stock_path.name)
            if has_parquet:
                inferred_warnings.append(
                    "idmc_facts_flow.parquet: skipped to avoid double-counting"
                )
            input_dir = work_dir
        else:
            LOGGER.info(
                "single_source.flow_missing | staging=%s parquet_present=%s",
                staging_dir,
                has_parquet,
            )

        try:
            LOGGER.info(
                "exporter.invoke | input=%s out=%s db_url=%s write_db=1 strategy=idmc-staging",
                input_dir,
                out_dir,
                canonical_url,
            )
            export_result = export_facts.export_facts(
                inp=input_dir,
                out_dir=out_dir,
                write_db="1",
                db_url=canonical_url,
                only_strategy="idmc-staging",
            )
        except export_facts.DuckDBWriteError as exc:
            LOGGER.error("Exporter DuckDB write failed", exc_info=True)
            print(f"❌ DuckDB write failed: {exc}", file=sys.stderr)
            return 3
        except export_facts.ExportError as exc:
            LOGGER.error("Exporter failed", exc_info=True)
            print(f"❌ Export failed: {exc}", file=sys.stderr)
            return 3

    dataframe = export_result.dataframe
    if dataframe is None or dataframe.empty:
        LOGGER.error("Exporter produced no rows; aborting")
        print("❌ Export produced no rows", file=sys.stderr)
        return 3

    exported_rows = int(export_result.rows)
    print(f"📄 Exported {export_result.rows} rows to {export_result.csv_path}")

    db_stats = export_result.db_stats or {}

    def _extract_delta_and_total(stats: dict[str, object] | None) -> tuple[int, int]:
        if not stats:
            return 0, 0
        delta = int(stats.get("rows_delta", 0) or 0)
        total_after = int(stats.get("rows_after", stats.get("rows_before", 0) or 0))
        return delta, total_after

    resolved_stats = db_stats.get("facts_resolved") or {}
    deltas_stats = db_stats.get("facts_deltas") or {}
    resolved_delta, resolved_total_reported = _extract_delta_and_total(resolved_stats)
    deltas_delta, deltas_total_reported = _extract_delta_and_total(deltas_stats)

    print(
        f"✅ Wrote {resolved_delta} rows to DuckDB (facts_resolved) — total {resolved_total_reported}"
    )
    print(
        f"✅ Wrote {deltas_delta} rows to DuckDB (facts_deltas) — total {deltas_total_reported}"
    )

    conn = None
    try:
        conn = duckdb_io.get_db(canonical_url)
        duckdb_io.init_schema(conn)

        def _table_count(name: str) -> int:
            try:
                return int(
                    conn.execute(
                        f"SELECT COALESCE(COUNT(*), 0) FROM {name}"
                    ).fetchone()[0]
                )
            except Exception as exc:
                message = str(exc).lower()
                if "no such table" in message or "does not exist" in message:
                    LOGGER.warning(
                        "Verification table missing; treating row count as zero | table=%s",
                        name,
                    )
                    return 0
                raise

        resolved_count = _table_count("facts_resolved")
        deltas_count = _table_count("facts_deltas")
    except Exception as exc:
        LOGGER.error("Verification query failed", exc_info=True)
        print(f"❌ Verification failed: {exc}", file=sys.stderr)
        return 3
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # pragma: no cover - cached connection may persist
                pass

    resolved_count = int(resolved_count)
    deltas_count = int(deltas_count)
    total_rows = resolved_count + deltas_count
    if total_rows <= 0:
        LOGGER.error("Verification failed: DuckDB contains no rows after export")
        return 3

    LOGGER.info(
        "verification.ok | resolved=%s deltas=%s total=%s",
        resolved_count,
        deltas_count,
        total_rows,
    )
    print(
        "✅ Verified DuckDB row counts: resolved={resolved} deltas={deltas} total={total}".format(
            resolved=resolved_count,
            deltas=deltas_count,
            total=total_rows,
        )
    )

    warnings: list[str] = []
    seen: set[str] = set()

    for warning in inferred_warnings:
        if warning and warning not in seen:
            warnings.append(warning)
            seen.add(warning)

    export_warnings = list(export_result.warnings or [])
    report_warnings = export_result.report.get("warnings") if export_result.report else []
    for warning in (export_warnings or []) + (report_warnings or []):
        if not warning:
            continue
        message = str(warning)
        if message not in seen:
            warnings.append(message)
            seen.add(message)

    if warnings:
        print("Warnings:")
        for message in warnings:
            print(f"- {message}")
    else:
        print("Warnings: none")

    total_delta = resolved_delta + deltas_delta
    resolved_total = resolved_count
    deltas_total = deltas_count

    exit_code = 0
    if warnings and args.strict:
        LOGGER.error("Strict mode enabled and warnings present; exiting with warnings")
        exit_code = 2

    print(
        (
            "Summary: Exported {exported} rows; wrote {delta} rows "
            "(resolved Δ={resolved_delta}, deltas Δ={deltas_delta}) "
            "→ totals resolved={resolved_total}, deltas={deltas_total}, total={total} "
            "| semantics stock→facts_resolved, new→facts_deltas "
            "to DuckDB @ {path}; warnings: {warns}; exit={code}"
        ).format(
            exported=exported_rows,
            delta=total_delta,
            resolved_delta=resolved_delta,
            deltas_delta=deltas_delta,
            resolved_total=resolved_total,
            deltas_total=deltas_total,
            total=total_rows,
            path=db_display,
            warns=len(warnings),
            code=exit_code,
        )
    )

    matched_files = export_result.report.get("matched_files") if export_result.report else []
    matched_sources: set[str] = set()
    for entry in matched_files or []:
        source = entry.get("source")
        if source:
            matched_sources.add(str(source))
    if matched_sources:
        print("Sources:")
        for source in sorted(matched_sources):
            print(f" - {source}")

    return exit_code


def main(argv: Sequence[str] | None = None) -> None:
    sys.exit(run(argv))


if __name__ == "__main__":
    main()
