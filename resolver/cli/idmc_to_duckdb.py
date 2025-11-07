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


def _normalize_db_url(raw: str | None) -> tuple[str, str]:
    """Return canonical filesystem path and DuckDB URL for ``raw``."""

    candidate = (raw or "").strip()
    if not candidate:
        env_default = os.getenv("RESOLVER_DB_URL", "").strip()
        candidate = env_default or DEFAULT_DB_URL
    try:
        path, url = canonicalize_duckdb_target(candidate)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.error(
            "Failed to canonicalise DuckDB target | raw=%s error=%s",
            candidate,
            exc,
        )
        raise
    return path, url


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
    db_path, canonical_url = _normalize_db_url(args.db_url)

    LOGGER.info("idmc_to_duckdb.start | staging=%s out=%s db_url=%s", staging_dir, out_dir, canonical_url)
    LOGGER.debug(
        "environment | RESOLVER_DB_URL=%s RESOLVER_WRITE_DB=%s",
        os.getenv("RESOLVER_DB_URL", ""),
        os.getenv("RESOLVER_WRITE_DB", ""),
    )

    _ensure_directory(out_dir)
    if db_path != ":memory":
        _ensure_directory(Path(db_path).expanduser().resolve().parent)
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
            export_result = export_facts.export_facts(
                inp=input_dir,
                out_dir=out_dir,
                write_db=False,
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

    filtered_df = dataframe
    removed_rows = 0
    if "source" in dataframe.columns:
        normalized_sources = dataframe["source"].astype(str).str.strip().str.lower()
        idmc_mask = normalized_sources == "idmc_flow"
        removed_rows = int((~idmc_mask).sum())
        if removed_rows:
            filtered_df = dataframe.loc[idmc_mask].reset_index(drop=True)
            LOGGER.info(
                "idmc_filter.source | removed=%s kept=%s", removed_rows, len(filtered_df)
            )
            if filtered_df.empty:
                LOGGER.error("Filtering removed all rows; aborting")
                print("❌ No idmc_flow rows remain after filtering", file=sys.stderr)
                return 3
            try:
                filtered_df.to_csv(export_result.csv_path, index=False)
                if export_result.parquet_path:
                    filtered_df.to_parquet(export_result.parquet_path, index=False)
            except Exception as exc:  # pragma: no cover - diagnostics only
                LOGGER.warning("Failed to rewrite filtered artifacts: %s", exc, exc_info=True)
            inferred_warnings.append(
                f"Filtered {removed_rows} non-idmc_flow rows before persistence"
            )

    export_result.dataframe = filtered_df
    export_result.rows = len(filtered_df)
    if isinstance(export_result.report, dict):
        export_result.report["rows_exported"] = len(filtered_df)

    try:
        db_stats = export_facts._maybe_write_to_db(
            facts_resolved=filtered_df,
            db_url=canonical_url,
            write_db=True,
            fail_on_error=True,
        )
    except export_facts.DuckDBWriteError as exc:
        LOGGER.error("Filtered DuckDB write failed", exc_info=True)
        print(f"❌ DuckDB write failed: {exc}", file=sys.stderr)
        return 3

    export_result.db_stats = db_stats

    print(f"📄 Exported {export_result.rows} rows to {export_result.csv_path}")

    db_stats = export_result.db_stats.get("facts_resolved") if export_result.db_stats else None
    if db_stats:
        delta = int(db_stats.get("rows_delta", 0))
        total_after = int(db_stats.get("rows_after", 0))
        print(f"✅ Wrote {delta} rows to DuckDB (facts_resolved) — total {total_after}")
    else:
        print("⚠️ DuckDB write stats unavailable; verifying manually…")

    conn = None
    try:
        conn = duckdb_io.get_db(canonical_url)
        duckdb_io.init_schema(conn)
        ym_values = _ym_candidates(export_result.dataframe)
        if ym_values:
            placeholders = ",".join(["?"] * len(ym_values))
            query = f"SELECT COUNT(*) FROM facts_resolved WHERE ym IN ({placeholders})"
            total_rows = conn.execute(query, ym_values).fetchone()[0]
        else:
            total_rows = conn.execute("SELECT COUNT(*) FROM facts_resolved").fetchone()[0]
    except Exception as exc:
        message = str(exc).lower()
        if "no such table" in message or "does not exist" in message:
            LOGGER.warning("Verification table missing; treating row count as zero")
            total_rows = 0
        else:
            LOGGER.error("Verification query failed", exc_info=True)
            print(f"❌ Verification failed: {exc}", file=sys.stderr)
            return 3
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # pragma: no cover - cached connection may persist
                pass

    total_rows = int(total_rows)
    exported_rows = int(export_result.rows)
    if total_rows <= 0:
        LOGGER.error("Verification failed: DuckDB contains no rows after export")
        return 3

    LOGGER.info("verification.ok | total_rows=%s", total_rows)
    if not db_stats:
        print(f"✅ Verified DuckDB row count: total {total_rows}")

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
            print(f" - ⚠️ {message}")
    else:
        print("Warnings: none")

    rows_delta = 0
    rows_total = total_rows
    if db_stats:
        rows_delta = int(db_stats.get("rows_delta", rows_delta))
        rows_total = int(db_stats.get("rows_after", rows_total))

    exit_code = 0
    if warnings and args.strict:
        LOGGER.error("Strict mode enabled and warnings present; exiting with warnings")
        exit_code = 2

    print(
        "Summary: Exported {exported} rows; wrote {delta} rows (total {total}) to DuckDB @ {path}; "
        "warnings: {warns}; exit={code}".format(
            exported=exported_rows,
            delta=rows_delta,
            total=rows_total,
            path=db_path,
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
