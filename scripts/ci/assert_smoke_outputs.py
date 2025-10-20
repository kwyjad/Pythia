"""CI helper to verify stub-based smoke outputs contain data rows."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


def count_rows_excluding_header(csv_path: Path) -> int:
    """Return the number of non-empty records in ``csv_path``.

    The header row is ignored and trailing blank lines are skipped so the
    resulting total matches the markdown quick stats produced by diagnostics.
    Missing or unreadable files yield ``0``.
    """

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
    """Collect row counts for CSVs below ``canonical_dir``."""

    files: list[dict[str, object]] = []
    total_rows = 0
    if canonical_dir.exists():
        for csv_path in sorted(canonical_dir.glob("*.csv")):
            rows = count_rows_excluding_header(csv_path)
            total_rows += rows
            files.append({
                "path": str(csv_path.resolve()),
                "rows": rows,
            })

    return {
        "canonical_dir": str(canonical_dir.resolve()) if canonical_dir.exists() else str(canonical_dir),
        "files": files,
        "total_rows": total_rows,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assert smoke canonical outputs contain data rows",
    )
    parser.add_argument("--staging", type=Path, required=True, help="Base staging directory")
    parser.add_argument("--period", required=True, help="Resolver period label")
    parser.add_argument("--min-rows", type=int, default=1, help="Minimum total canonical rows required")

    args = parser.parse_args(list(argv) if argv is not None else None)

    canonical_dir = (args.staging / args.period / "canonical").resolve()
    report = build_report(canonical_dir)

    diagnostics_dir = Path(".ci/diagnostics")
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    report_path = diagnostics_dir / "smoke-assert.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        "SMOKE: canonical rows total={total} (min={minimum}) in {count} file(s)".format(
            total=report["total_rows"],
            minimum=args.min_rows,
            count=len(report["files"]),
        )
    )
    print(f"Report written to {report_path}")

    return 0 if report["total_rows"] >= args.min_rows else 1


if __name__ == "__main__":  # pragma: no cover - exercised in CI
    raise SystemExit(main())
