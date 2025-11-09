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

    if not staging_path.exists() or not staging_path.is_dir():
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
    probe_path = staging_path.parent / "idmc" / "probe.json"
    try:
        if probe_path.exists():
            probe_data = json.loads(probe_path.read_text(encoding="utf-8"))
            report["idmc_probe"] = probe_data
    except Exception:
        report["idmc_probe"] = "unreadable"
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
