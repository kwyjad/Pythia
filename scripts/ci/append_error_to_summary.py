# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Append structured error diagnostics to ingestion summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SUMMARY_PATH = Path("diagnostics/ingestion/summary.md")


def _load_context(raw_context: str) -> dict[str, Any]:
    """Parse a JSON string, falling back to raw representation on failure."""
    try:
        loaded = json.loads(raw_context)
        if isinstance(loaded, dict):
            return loaded
        return {"raw_context": loaded}
    except Exception:
        return {"raw_context": raw_context}


def append_error(section: str, error_type: str, message: str, context: dict[str, Any]) -> None:
    """Append an error block to the ingestion summary file."""
    lines = [
        f"## {section}",
        "",
        f"- **Error type:** `{error_type}`",
        f"- **Error message:** `{message}`",
    ]
    if context:
        serialized_context = json.dumps(context, sort_keys=True)
        lines.append(f"- **Context:** `{serialized_context}`")
    lines.append("")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with SUMMARY_PATH.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    except Exception:
        # Diagnostics should never fail the caller.
        pass


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Append error section to diagnostics/ingestion/summary.md"
    )
    parser.add_argument("--section", required=True, help="Section title (Markdown H2)")
    parser.add_argument("--error-type", required=True, help="Exception type/class")
    parser.add_argument("--message", required=True, help="Exception message")
    parser.add_argument(
        "--context",
        default="{}",
        help="JSON string for extra context to include in the summary",
    )
    args = parser.parse_args(argv)

    context = _load_context(args.context)
    append_error(args.section, args.error_type, args.message, context)


if __name__ == "__main__":
    main()
