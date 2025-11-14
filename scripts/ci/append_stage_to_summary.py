"""Append derive-freeze stage markers to the ingestion diagnostics summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SUMMARY_PATH = Path("diagnostics/ingestion/summary.md")


def _coerce_context(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except Exception:
        return {"raw_context": raw}
    if isinstance(loaded, dict):
        return loaded
    return {"raw_context": loaded}


def append_stage(section: str, *, status: str | None = None, details: str | None = None, context: dict[str, Any] | None = None) -> None:
    """Append a stage marker block to the ingestion summary."""

    lines = [f"## {section}", ""]
    if status:
        lines.append(f"- **Status:** {status}")
    if details:
        lines.append(f"- **Details:** {details}")
    if context:
        serialized = json.dumps(context, sort_keys=True)
        lines.append(f"- **Context:** `{serialized}`")
    lines.append("")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with SUMMARY_PATH.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    except Exception:
        # Stage markers are best-effort diagnostics.
        pass


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Append stage markers to diagnostics/ingestion/summary.md",
    )
    parser.add_argument("--section", required=True, help="Section title (Markdown H2)")
    parser.add_argument("--status", help="Optional status string to include")
    parser.add_argument("--details", help="Optional free-form details line")
    parser.add_argument(
        "--context",
        help="Optional JSON context blob for troubleshooting",
    )
    args = parser.parse_args(argv)

    context = _coerce_context(args.context)
    append_stage(
        args.section,
        status=args.status,
        details=args.details,
        context=context if context else None,
    )


if __name__ == "__main__":
    main()
