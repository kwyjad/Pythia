"""Validate syntax for critical CI modules via ``py_compile``.

This script is intentionally lightweight and should run before pytest in CI.
"""
from __future__ import annotations

import py_compile
import sys
from pathlib import Path

CRITICAL_MODULES = [
    Path("scripts/ci/run_connectors.py"),
    Path("scripts/ci/summarize_connectors.py"),
    Path("resolver/ingestion/_shared/error_report.py"),
]


def main() -> int:
    failures = []
    for module in CRITICAL_MODULES:
        try:
            py_compile.compile(str(module), doraise=True)
            print(f"Syntax OK: {module}")
        except py_compile.PyCompileError as exc:  # pragma: no cover - failure path
            print(f"Syntax error in {module}: {exc}")
            failures.append(module)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
