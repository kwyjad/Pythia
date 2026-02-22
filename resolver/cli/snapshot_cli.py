# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Resolver snapshot orchestration CLI.

Uses the new connector-based pipeline (``resolver.tools.run_pipeline``)
to fetch, validate, enrich, resolve precedence, compute deltas, and
optionally write to DuckDB — then freezes a monthly snapshot.
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Callable, Optional

from resolver.tools.build_forecaster_features import (
    FeatureBuildError,
    build_features as build_forecaster_features,
)
from resolver.tools.freeze_snapshot import SnapshotError, freeze_snapshot
from resolver.tools.run_pipeline import PipelineResult, run_pipeline

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOTS = ROOT / "snapshots"


class PipelineError(Exception):
    """Raised when the pipeline fails."""


def _parse_bool_flag(value: str | bool | None) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _month_cutoff(ym: str) -> str:
    try:
        year, month = map(int, ym.split("-"))
        dt.date(year, month, 1)
    except Exception as exc:  # pragma: no cover - defensive parsing
        raise SnapshotError(f"--ym must be YYYY-MM (received '{ym}')") from exc
    last_day = calendar.monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-{last_day:02d}"


def make_monthly(args: argparse.Namespace) -> int:
    ym = args.ym
    _month_cutoff(ym)  # validate format

    snapshots_dir = Path(args.outdir)
    write_db = _parse_bool_flag(args.write_db)
    db_url = args.db_url

    connector_names = args.connectors or None
    dry_run = write_db is False or (write_db is None and not db_url)

    snapshots_dir.mkdir(parents=True, exist_ok=True)

    # --- Run the new connector-based pipeline ---
    pipeline_result: PipelineResult = run_pipeline(
        db_url=db_url if not dry_run else None,
        connectors=connector_names,
        dry_run=dry_run,
    )

    if pipeline_result.total_facts == 0:
        raise PipelineError("Pipeline produced 0 facts — nothing to snapshot")

    errors = [cr for cr in pipeline_result.connector_results if cr.status == "error"]
    if errors:
        names = ", ".join(cr.name for cr in errors)
        print(f"⚠️  Connectors with errors: {names}", file=sys.stderr)

    # --- Build forecaster features ---
    try:
        features_frame = build_forecaster_features(
            snapshots_dir=snapshots_dir,
            db_url=db_url,
        )
    except FeatureBuildError as exc:
        raise SnapshotError(f"Failed to build resolver features: {exc}") from exc

    print("✅ Monthly pipeline complete")
    print(f"Month: {ym}")
    print(f"Total facts: {pipeline_result.total_facts}")
    print(f"Resolved rows: {pipeline_result.resolved_rows}")
    print(f"Delta rows: {pipeline_result.delta_rows}")
    print(f"DB written: {pipeline_result.db_written}")
    print(f"Resolver features rows: {len(features_frame)}")
    return 0


def list_snapshots(args: argparse.Namespace) -> int:
    base = Path(args.outdir)
    if not base.exists():
        print(f"No snapshots directory at {base}")
        return 0

    entries = sorted(p for p in base.iterdir() if p.is_dir())
    if not entries:
        print(f"No snapshot folders under {base}")
        return 0

    for entry in entries:
        manifest_path = entry / "manifest.json"
        if manifest_path.exists():
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            created = data.get("created_at_utc", "?")
            rows = data.get("resolved_rows", data.get("rows", "?"))
            print(f"{entry.name}: {rows} rows (created {created})")
        else:
            print(f"{entry.name}: manifest.json missing")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolver monthly snapshot orchestrator",
    )
    subparsers = parser.add_subparsers(dest="command")

    make = subparsers.add_parser(
        "make-monthly",
        help="Run pipeline connectors, resolve, and snapshot",
    )
    make.add_argument("--ym", required=True, help="Target year-month (YYYY-MM)")
    make.add_argument(
        "--connectors",
        nargs="*",
        default=None,
        help="Run only these connectors (default: all registered)",
    )
    make.add_argument(
        "--outdir",
        default=str(DEFAULT_SNAPSHOTS),
        help="Snapshot base directory",
    )
    make.add_argument(
        "--write-db",
        default=None,
        choices=["0", "1"],
        help="Set to 1 or 0 to force-enable or disable DuckDB writes",
    )
    make.add_argument(
        "--db-url",
        default=None,
        help="Optional DuckDB URL override",
    )
    make.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing snapshot directory if present",
    )
    make.set_defaults(func=make_monthly)

    list_cmd = subparsers.add_parser(
        "list-snapshots",
        help="List available snapshot folders",
    )
    list_cmd.add_argument(
        "--outdir",
        default=str(DEFAULT_SNAPSHOTS),
        help="Snapshot base directory",
    )
    list_cmd.set_defaults(func=list_snapshots)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    func: Optional[Callable[[argparse.Namespace], int]] = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 1
    try:
        return func(args)
    except (SnapshotError, PipelineError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
