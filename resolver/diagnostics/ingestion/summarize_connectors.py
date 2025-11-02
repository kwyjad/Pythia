"""Lightweight diagnostics summarizer for connectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def build_summary(staging_dir: str) -> Dict[str, object]:
    staging_path = Path(staging_dir)
    report: Dict[str, object] = {
        "staging_exists": staging_path.exists(),
        "files": [],
        "rows_total": 0,
    }

    if not staging_path.exists():
        return report

    files: List[Dict[str, object]] = []
    rows_total = 0

    for csv_path in staging_path.rglob("*.csv"):
        try:
            with csv_path.open("r", encoding="utf-8", errors="ignore") as handle:
                rows = sum(1 for _ in handle) - 1
            rows = max(rows, 0)
            files.append({"path": str(csv_path), "rows": rows})
            rows_total += rows
        except Exception:
            files.append({"path": str(csv_path), "rows": "unreadable"})

    report["files"] = files
    report["rows_total"] = rows_total
    return report


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize connector staging CSV outputs")
    parser.add_argument("--staging", default="data/staging", help="Path to the staging root")
    parser.add_argument(
        "--out",
        default=".ci/diagnostics/connectors-summary.json",
        help="Destination path for the JSON summary",
    )
    return parser.parse_args()


def main() -> None:
    args = _cli()
    summary = build_summary(args.staging)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
