#!/usr/bin/env python3
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Normalize workflow YAML files for CI lint compliance."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

WORKFLOW_DIR = Path(".github/workflows")
BOOLEAN_PATTERN = re.compile(
    r"(?P<prefix>:\s*)(?P<value>yes|no)(?P<suffix>\b)", re.IGNORECASE
)
REQUIRED_PATTERN = re.compile(
    r"(?P<prefix>\b(required|enabled)\s*:\s*)(?P<value>yes|no)\b",
    re.IGNORECASE,
)
LIST_PATTERN = re.compile(r"(?P<prefix>:\s*|\-\s*)\[(?P<body>[^\[\]]+)\]")


def _normalize_booleans(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        value = match.group("value")
        replacement = "true" if value.lower() == "yes" else "false"
        return f"{match.group('prefix')}{replacement}{match.group('suffix')}"

    def replace_required(match: re.Match[str]) -> str:
        value = match.group("value")
        replacement = "true" if value.lower() == "yes" else "false"
        return f"{match.group('prefix')}{replacement}"

    text = BOOLEAN_PATTERN.sub(replace, text)
    text = REQUIRED_PATTERN.sub(replace_required, text)
    return text


def _normalize_lists(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        body = match.group("body")
        items = [part.strip() for part in body.split(",")]
        items = [item for item in items if item]
        return f"{match.group('prefix')}[" + ", ".join(items) + "]"

    return LIST_PATTERN.sub(repl, text)


def _ensure_document_start(lines: Iterable[str]) -> str:
    lines = list(lines)
    if not lines:
        return "---\n"
    if not lines[0].startswith("---"):
        lines.insert(0, "---\n")
    return "".join(lines)


def _strip_trailing_spaces(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


def process_file(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    content = _normalize_booleans(content)
    content = _normalize_lists(content)
    content = _ensure_document_start(content.splitlines(True))
    content = _strip_trailing_spaces(content)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    if not WORKFLOW_DIR.exists():
        return 0
    for path in sorted(WORKFLOW_DIR.glob("**/*.yml")) + sorted(WORKFLOW_DIR.glob("**/*.yaml")):
        process_file(path)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
