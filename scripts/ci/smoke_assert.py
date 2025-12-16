# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Validate smoke canonical outputs contain the expected number of data rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


def count_rows(csv_path: Path) -> int:
    """Return the number of non-empty data rows in ``csv_path``."""

    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)  # header
            rows = 0
            for record in reader:
                if not any(cell.strip() for cell in record):
                    continue
                rows += 1
            return rows
    except FileNotFoundError:
        return 0
    except Exception:
        return 0


def build_report(canonical_dir: Path) -> dict[str, object]:
    """Build the smoke assertion payload for ``canonical_dir``."""

    files: list[dict[str, object]] = []
    total_rows = 0

    if canonical_dir.exists():
        for csv_path in sorted(canonical_dir.glob("*.csv")):
            rows = count_rows(csv_path)
            total_rows += rows
            files.append({
                "path": str(csv_path.resolve()),
                "rows": rows,
            })

    resolved_dir = canonical_dir.resolve() if canonical_dir.exists() else canonical_dir
    return {
        "canonical_dir": str(resolved_dir),
        "files": files,
        "total_rows": total_rows,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assert smoke canonical outputs contain data rows",
    )
    parser.add_argument(
        "--canonical-dir",
        type=Path,
        required=True,
        help="Canonical directory containing stubbed CSVs",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=1,
        help="Minimum total canonical rows required for success",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(".ci/diagnostics/smoke-assert.json"),
        help="Path to the JSON report written for diagnostics",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    report = build_report(args.canonical_dir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        "SMOKE: canonical rows total={total} (min={minimum}) in {count} file(s)".format(
            total=report["total_rows"],
            minimum=args.min_rows,
            count=len(report["files"]),
        )
    )
    print(f"Report written to {args.out}")

    return 0 if report["total_rows"] >= args.min_rows else 2


if __name__ == "__main__":  # pragma: no cover - exercised in CI
    raise SystemExit(main())
