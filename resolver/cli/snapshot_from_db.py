"""CLI wrapper for DB-first snapshot builder (skeleton)."""

from __future__ import annotations

import argparse
from typing import Sequence

from resolver.snapshot import builder


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build Resolver monthly snapshots directly from DuckDB canonical tables."
            " Currently a stub that outlines the intended plan."
        ),
    )
    parser.add_argument(
        "--month",
        help="Target month (YYYY-MM). Optional while the builder is a stub.",
    )
    parser.add_argument(
        "--db-url",
        default="duckdb:///resolver/db/resolver.duckdb",
        help="DuckDB URL containing canonical facts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the intended plan without performing work (stub default).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    return builder.run_stub_snapshot(
        month=args.month,
        db_url=args.db_url,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())

