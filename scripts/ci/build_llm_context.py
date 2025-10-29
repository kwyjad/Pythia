#!/usr/bin/env python3
"""Build a minimal LLM context bundle for CI diagnostics."""

from __future__ import annotations

import csv
import datetime as dt
import json
import pathlib
from typing import Any, Dict, List


def _load_rows(csv_path: pathlib.Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not csv_path.exists():
        return rows
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(row)
    except Exception:
        # Fall back to an empty payload if the preview file is unreadable.
        rows = []
    return rows


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    facts_csv = repo_root / "diagnostics" / "ingestion" / "export_preview" / "facts.csv"

    diagnostics_dir = repo_root / "diagnostics" / "context"
    legacy_dir = repo_root / "context"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    legacy_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(facts_csv)
    payload = {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "source": str(facts_csv),
        "row_count": len(rows),
        "sample": rows[:10],
    }

    bundle_path = diagnostics_dir / "llm_context.json"
    legacy_bundle_path = legacy_dir / "llm_context.json"

    for destination in (bundle_path, legacy_bundle_path):
        try:
            with destination.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        except Exception:
            # Ignore write errors so that the workflow can continue.
            pass

    summary_path = repo_root / "summary.md"
    block = [
        "## LLM Context",
        f"- **context path:** `{bundle_path}`",
        f"- **facts preview path:** `{facts_csv}`",
        f"- **rows included:** {len(rows)}",
        "",
    ]
    try:
        with summary_path.open("a", encoding="utf-8") as summary_handle:
            summary_handle.write("\n".join(block) + "\n")
    except Exception:
        # The summary file is optional for CI runs.
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
