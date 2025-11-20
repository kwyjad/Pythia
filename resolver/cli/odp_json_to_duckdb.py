"""CLI for running the UNHCR ODP JSON → DuckDB pipeline."""

from __future__ import annotations

import argparse
import traceback
from datetime import date
from pathlib import Path
from typing import Sequence

from resolver.common.logs import configure_root_logger, get_logger
from resolver.diagnostics import odp_smoke
from resolver.ingestion import odp_duckdb, odp_series

logger = get_logger(__name__)


def _normalize_db_url_arg(raw: str | None) -> tuple[str, str]:
    """Return (DuckDB URL, filesystem path) for ``raw`` input."""

    candidate = (raw or "").strip()
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m resolver.cli.odp_json_to_duckdb",
        description=(
            "Fetch UNHCR ODP JSON widgets, normalize monthly series, "
            "and write them into DuckDB (odp_timeseries_raw)."
        ),
    )
    parser.add_argument(
        "--db",
        required=True,
        help="DuckDB path or duckdb:/// URL (e.g., data/resolver_backfill.duckdb).",
    )
    parser.add_argument(
        "--config",
        default="resolver/ingestion/config/unhcr_odp.yml",
        help="ODP HTML/JSON discovery config (YAML).",
    )
    parser.add_argument(
        "--normalizers",
        default="resolver/ingestion/config/odp_normalizers.yml",
        help="ODP JSON normalization registry (YAML).",
    )
    parser.add_argument(
        "--today",
        default=None,
        help=(
            "Optional override for 'today' when computing ym/as_of_date; "
            "format YYYY-MM-DD (used mainly for tests)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Optional log level override (e.g. DEBUG, INFO).",
    )
    parser.add_argument(
        "--summary",
        default="diagnostics/odp_json/odp_smoke_summary.md",
        help="Destination path for the smoke summary markdown.",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.log_level:
        configure_root_logger(level=args.log_level)

    db_url, db_path = _normalize_db_url_arg(args.db)
    logger.info("ODP DuckDB CLI: using DB %s (url=%s)", db_path, db_url)

    stats = odp_series.OdpPipelineStats()

    if args.today:
        try:
            parts = [int(x) for x in args.today.split("-")]
            today = date(*parts)
        except Exception as exc:  # noqa: BLE001
            logger.error("Invalid --today value %r: %s", args.today, exc)
            exit_code = 2
            pipeline_error = exc
            summary_text = odp_smoke.build_smoke_summary(
                config_path=args.config,
                normalizers_path=args.normalizers,
                db_path=db_path,
                diagnostics_dir=Path(args.summary).parent,
                rows=None,
                error=pipeline_error,
                traceback_text=traceback.format_exc(),
                stats=stats,
            )
            odp_smoke.write_smoke_summary(summary_text, args.summary)
            return exit_code
    else:
        today = None

    exit_code = 0
    pipeline_error: Exception | None = None
    rows: int | None = None
    try:
        rows = odp_duckdb.build_and_write_odp_series(
            config_path=args.config,
            normalizers_path=args.normalizers,
            db_url=db_url,
            fetch_html=None,
            fetch_json=None,
            today=today,
            stats=stats,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("ODP DuckDB CLI: pipeline failed: %s", exc)
        pipeline_error = exc
        exit_code = 1
    else:
        logger.info("ODP DuckDB CLI: completed successfully (rows=%s)", rows)
    try:
        summary_text = odp_smoke.build_smoke_summary(
            config_path=args.config,
            normalizers_path=args.normalizers,
            db_path=db_path,
            diagnostics_dir=Path(args.summary).parent,
            rows=rows,
            error=pipeline_error,
            traceback_text=traceback.format_exc() if pipeline_error else None,
            stats=stats,
        )
        odp_smoke.write_smoke_summary(summary_text, args.summary)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to write ODP smoke summary: %s", exc)

    if exit_code == 0:
        return 0
    return exit_code


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run(argv))


if __name__ == "__main__":
    main()
