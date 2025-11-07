"""IDMC → DuckDB helper CLI for one-command exports."""

from __future__ import annotations

import argparse
import logging
import os
import sys
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


def _resolve_db_url(raw: str | None) -> str:
    env_default = os.getenv("RESOLVER_DB_URL", DEFAULT_DB_URL).strip()
    candidate = (raw or "").strip() or env_default
    if not candidate:
        candidate = DEFAULT_DB_URL
    return candidate


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
    db_url_raw = _resolve_db_url(args.db_url)
    db_path, canonical_url = canonicalize_duckdb_target(db_url_raw)

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

    try:
        export_result = export_facts.export_facts(
            inp=staging_dir,
            out_dir=out_dir,
            write_db=True,
            db_url=canonical_url,
        )
    except export_facts.DuckDBWriteError as exc:
        LOGGER.error("Exporter DuckDB write failed", exc_info=True)
        print(f"❌ DuckDB write failed: {exc}", file=sys.stderr)
        return 3
    except export_facts.ExportError as exc:
        LOGGER.error("Exporter failed", exc_info=True)
        print(f"❌ Export failed: {exc}", file=sys.stderr)
        return 3

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
    if total_rows <= 0 or total_rows < exported_rows:
        LOGGER.error(
            "Verification failed: DB rows (%s) lower than exported rows (%s)",
            total_rows,
            exported_rows,
        )
        return 3

    LOGGER.info("verification.ok | total_rows=%s", total_rows)
    if not db_stats:
        print(f"✅ Verified DuckDB row count: total {total_rows}")

    warnings = list(export_result.warnings)
    report_warnings = export_result.report.get("warnings") if export_result.report else []
    for warning in report_warnings or []:
        if warning not in warnings:
            warnings.append(str(warning))

    if warnings:
        print("Warnings:")
        for message in warnings:
            print(f" - ⚠️ {message}")
        if args.strict:
            LOGGER.error("Strict mode enabled and warnings present; exiting with failure")
            return 2

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

    return 0


def main(argv: Sequence[str] | None = None) -> None:
    sys.exit(run(argv))


if __name__ == "__main__":
    main()
