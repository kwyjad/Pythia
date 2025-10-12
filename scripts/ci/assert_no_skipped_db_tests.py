#!/usr/bin/env python3
"""Fail CI runs when DuckDB-backed tests are silently skipped."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

DB_SKIP_PATTERNS = (
    "duckdb not installed",
    "duckdb optional",
    "requires duckdb",
)

ALLOWED_SKIP_PATTERNS = (
    "yamllint not available",
    "fixture files not present",
    "resolver/staging is empty",
)

MAX_UNACCOUNTED_SKIPS = 0
MAX_TOTAL_SKIPS = 5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "junit_xml",
        type=Path,
        help="Path to the pytest JUnit XML report",
    )
    return parser.parse_args()


def _normalise(text: str | None) -> str:
    return (text or "").strip().lower()


def main() -> int:
    args = _parse_args()
    junit_path = args.junit_xml
    if not junit_path.exists():
        print(f"JUnit XML report not found: {junit_path}", file=sys.stderr)
        return 1

    try:
        tree = ET.parse(junit_path)
    except ET.ParseError as exc:  # pragma: no cover - defensive guard
        print(f"Failed to parse {junit_path}: {exc}", file=sys.stderr)
        return 1

    root = tree.getroot()
    total_skips = 0
    allowed_skips = 0
    unexpected: list[tuple[str, str]] = []

    for testcase in root.iterfind(".//testcase"):
        classname = testcase.attrib.get("classname", "")
        name = testcase.attrib.get("name", "")
        identifier = f"{classname}::{name}".strip(":")
        for skipped in testcase.findall("skipped"):
            total_skips += 1
            message = skipped.attrib.get("message") or skipped.text or ""
            normalised = _normalise(message)
            if any(pattern in normalised for pattern in DB_SKIP_PATTERNS) or "duckdb" in classname.lower():
                unexpected.append((identifier, message.strip() or "(no reason provided)"))
                continue
            if any(pattern in normalised for pattern in ALLOWED_SKIP_PATTERNS):
                allowed_skips += 1
                continue
            unexpected.append((identifier, message.strip() or "(no reason provided)"))

    errors: list[str] = []

    if unexpected:
        header = "DuckDB-related tests were skipped unexpectedly:"
        details = "\n".join(f" - {ident}: {reason}" for ident, reason in unexpected)
        errors.append(f"{header}\n{details}")

    unaccounted = total_skips - allowed_skips - len(unexpected)
    if unaccounted > MAX_UNACCOUNTED_SKIPS:
        errors.append(
            "Too many tests were skipped overall: "
            f"{total_skips} total, {allowed_skips} allowed, {len(unexpected)} unexpected."
        )

    if total_skips > MAX_TOTAL_SKIPS:
        errors.append(
            f"Too many skipped tests overall: {total_skips} (allowed maximum {MAX_TOTAL_SKIPS})."
        )

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
