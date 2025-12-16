#!/usr/bin/env python3
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""CLI entry point for building the Forecaster LLM context bundle."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

from resolver.tools import llm_context

try:  # pragma: no cover - Python <3.9 fallback
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore

ISTANBUL_TZ = ZoneInfo("Europe/Istanbul")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Resolver LLM context bundle")
    parser.add_argument(
        "--months",
        type=int,
        default=12,
        help="Number of trailing finalized months to include (default: 12)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("context"),
        help="Directory where bundle files will be written",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="db",
        help="Selector backend to use when loading monthly data (default: db)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the context frame and print a summary without writing files",
    )
    return parser.parse_args(argv)


def _last_finalized_month(reference: dt.datetime | None = None) -> str:
    if reference is None:
        now = dt.datetime.now(ISTANBUL_TZ)
    else:
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=ISTANBUL_TZ)
        else:
            reference = reference.astimezone(ISTANBUL_TZ)
        now = reference

    year, month = now.year, now.month - 1
    if month == 0:
        month = 12
        year -= 1
    return f"{year:04d}-{month:02d}"


def _target_months(count: int, reference: dt.datetime | None = None) -> list[str]:
    if count <= 0:
        return []

    end = _last_finalized_month(reference)
    year, month = map(int, end.split("-"))

    months: list[str] = []
    for _ in range(count):
        months.append(f"{year:04d}-{month:02d}")
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    months.reverse()
    return months


def _month_counts(frame, months: Sequence[str]) -> dict[str, int]:
    if not months:
        return {}
    counts = Counter(frame["ym"]) if not frame.empty else Counter()
    return {ym: int(counts.get(ym, 0)) for ym in months}


def _print_summary(frame, month_counts: dict[str, int]) -> None:
    print("Context month counts:")
    if not month_counts:
        print("  (no months requested)")
    else:
        for ym in sorted(month_counts):
            print(f"  {ym}: {month_counts[ym]}")

    if frame.empty:
        print("[llm-context] context frame is empty.")
        return

    print("Context rows by country/hazard:")
    pair_counts = (
        frame.groupby(["iso3", "hazard_code"])["value"].count().sort_values(ascending=False)
    )
    for (iso3, hazard_code), count in pair_counts.items():
        print(f"  {iso3}/{hazard_code}: {int(count)}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    months = _target_months(args.months)
    frame = llm_context.build_context_frame(months=months, backend=args.backend)

    month_counts = _month_counts(frame, months)
    print(
        f"[llm-context] rows={len(frame)} months={len(months)} backend={args.backend}"
    )
    _print_summary(frame, month_counts)

    if args.dry_run:
        return 0

    if frame.empty:
        print("[llm-context] no context rows generated; aborting with non-zero exit code.")
        return 1

    bundle = llm_context.write_context_bundle(
        months=months,
        outdir=args.outdir,
        backend=args.backend,
        frame=frame,
    )
    print(f"[llm-context] wrote: {bundle}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
