"""CI helper to verify stub-based smoke outputs contain data rows."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _count_rows(csv_path: Path) -> int:
    """Return number of data rows (excluding header) in ``csv_path``.

    Missing files or unreadable files yield ``0``.
    """
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            # Consume header
            next(reader, None)
            return sum(1 for _ in reader)
    except Exception:
        return 0


def build_report(base: Path) -> Dict[str, Any]:
    """Build a JSON-serialisable report for all CSVs under ``base``."""
    files: List[Dict[str, Any]] = []
    total_rows = 0
    for csv_path in sorted(base.glob("*.csv")):
        rows = _count_rows(csv_path)
        total_rows += rows
        files.append({
            "path": str(csv_path),
            "rows": rows,
        })

    return {
        "canonical_dir": str(base),
        "files": files,
        "total_rows": total_rows,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assert smoke canonical outputs contain data rows")
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
        f"Smoke canonical rows: {report['total_rows']} (min required {args.min_rows})\n"
        f"Report written to {report_path}"
    )

    return 0 if report["total_rows"] >= args.min_rows else 1


if __name__ == "__main__":  # pragma: no cover - exercised in CI
    raise SystemExit(main())
