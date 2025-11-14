"""Append structured error sections to diagnostics summaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


SUMMARY_PATH = Path("diagnostics/ingestion/summary.md")


def _load_context(raw_context: str) -> Dict[str, Any]:
    """Return a JSON-serialisable context dict from raw CLI input.

    The helper is intentionally forgiving: any parsing error results in a
    singleton dictionary with the raw string so that diagnostics remain
    useful while never raising.
    """

    if not raw_context:
        return {}

    try:
        parsed = json.loads(raw_context)
    except Exception:
        return {"raw_context": raw_context}

    if isinstance(parsed, dict):
        return parsed

    # Preserve non-dict payloads under a synthetic key so the caller still sees
    # the value while maintaining a consistent shape.
    return {"value": parsed}


def _format_section(section: str, error_type: str, message: str, context: Dict[str, Any]) -> str:
    lines = [f"## {section}", "", f"- **Error type:** `{error_type}`", f"- **Error message:** `{message}`"]
    if context:
        try:
            context_json = json.dumps(context, sort_keys=True)
        except Exception:
            context_json = json.dumps({"raw_context": str(context)})
        lines.append(f"- **Context:** `{context_json}`")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Append an error block to diagnostics summary")
    parser.add_argument("--section", required=True, help="Section title (Markdown heading)")
    parser.add_argument("--error-type", required=True, help="Error class name")
    parser.add_argument("--message", required=True, help="Error message")
    parser.add_argument(
        "--context",
        default="{}",
        help="JSON object with additional context; best-effort parsed",
    )

    args = parser.parse_args(argv)

    context = _load_context(args.context)
    section_text = _format_section(args.section, args.error_type, args.message, context)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with SUMMARY_PATH.open("a", encoding="utf-8") as handle:
            handle.write(section_text)
    except Exception:
        # Never raise to keep diagnostics helpers resilient.
        return


if __name__ == "__main__":
    main()
