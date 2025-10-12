"""Resolver snapshot orchestration CLI."""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from resolver.tools.build_forecaster_features import (
    FeatureBuildError,
    build_features as build_forecaster_features,
)
from resolver.tools.export_facts import (
    DEFAULT_CONFIG as DEFAULT_EXPORT_CONFIG,
    ExportError,
    export_facts,
)
from resolver.tools.freeze_snapshot import SnapshotError, freeze_snapshot

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGING = ROOT / "staging"
DEFAULT_EXPORTS = ROOT / "exports"
DEFAULT_SNAPSHOTS = ROOT / "snapshots"


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


def _run_subprocess(args: list[str]) -> None:
    completed = subprocess.run(args)
    if completed.returncode != 0:
        raise SnapshotError(
            f"Command failed ({completed.returncode}): {' '.join(args)}"
        )


def _validate_facts(facts_path: Path) -> None:
    _run_subprocess([
        sys.executable,
        "-m",
        "resolver.tools.validate_facts",
        "--facts",
        str(facts_path),
    ])


def make_monthly(args: argparse.Namespace) -> int:
    ym = args.ym
    cutoff = _month_cutoff(ym)

    staging = Path(args.staging)
    exports_dir = Path(args.exports_dir)
    snapshots_dir = Path(args.outdir)
    export_cfg = Path(args.export_config)
    write_db = _parse_bool_flag(args.write_db)
    db_url = args.db_url

    if not staging.exists():
        raise SnapshotError(f"Staging path not found: {staging}")
    exports_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    export_result = export_facts(
        inp=staging,
        config_path=export_cfg,
        out_dir=exports_dir,
        write_db=write_db,
        db_url=db_url,
    )

    _validate_facts(export_result.csv_path)

    _run_subprocess(
        [
            sys.executable,
            "-m",
            "resolver.tools.precedence_engine",
            "--facts",
            str(export_result.csv_path),
            "--cutoff",
            cutoff,
            "--outdir",
            str(exports_dir),
        ]
    )

    resolved_csv = exports_dir / "resolved.csv"
    if not resolved_csv.exists():
        raise SnapshotError(f"precedence_engine did not produce {resolved_csv}")

    deltas_csv = exports_dir / "deltas.csv"
    _run_subprocess(
        [
            sys.executable,
            "-m",
            "resolver.tools.make_deltas",
            "--resolved",
            str(resolved_csv),
            "--out",
            str(deltas_csv),
        ]
    )

    if not deltas_csv.exists():
        raise SnapshotError("make_deltas did not produce deltas.csv")

    snapshot = freeze_snapshot(
        facts=export_result.csv_path,
        month=ym,
        outdir=snapshots_dir,
        overwrite=args.overwrite,
        deltas=deltas_csv,
        resolved_csv=resolved_csv,
        write_db=write_db,
        db_url=db_url,
    )

    try:
        features_frame = build_forecaster_features(
            snapshots_dir=snapshots_dir,
            db_url=db_url,
        )
    except FeatureBuildError as exc:
        raise SnapshotError(f"Failed to build resolver features: {exc}") from exc

    manifest_data = json.loads(snapshot.manifest.read_text(encoding="utf-8"))
    resolved_rows = manifest_data.get("resolved_rows", manifest_data.get("rows", 0))
    print("✅ Monthly snapshot complete")
    print(f"Month: {snapshot.ym}")
    print(f"Exports dir: {exports_dir}")
    print(f"Snapshot dir: {snapshot.out_dir}")
    print(f"Resolved rows: {resolved_rows}")
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
        help="Export, validate, and freeze the monthly snapshot",
    )
    make.add_argument("--ym", required=True, help="Target year-month (YYYY-MM)")
    make.add_argument(
        "--staging",
        default=str(DEFAULT_STAGING),
        help="Path to staging inputs for export_facts",
    )
    make.add_argument(
        "--exports-dir",
        default=str(DEFAULT_EXPORTS),
        help="Directory for intermediate exports",
    )
    make.add_argument(
        "--outdir",
        default=str(DEFAULT_SNAPSHOTS),
        help="Snapshot base directory",
    )
    make.add_argument(
        "--export-config",
        default=str(DEFAULT_EXPORT_CONFIG),
        help="Path to export_config.yml",
    )
    make.add_argument(
        "--write-db",
        default=None,
        choices=["0", "1"],
        help="Set to 1 or 0 to force-enable or disable DuckDB writes (defaults to auto via RESOLVER_DB_URL)",
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
    except (SnapshotError, ExportError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
