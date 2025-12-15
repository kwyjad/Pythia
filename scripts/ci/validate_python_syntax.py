#!/usr/bin/env python3
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Compile critical CI modules to catch syntax errors early."""
from __future__ import annotations

import py_compile
import sys
from pathlib import Path

FILES = [
    Path("scripts/ci/summarize_connectors.py"),
    Path("scripts/ci/run_connectors.py"),
]


def main() -> int:
    ok = True
    for path in FILES:
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:  # pragma: no cover - just prints errors
            ok = False
            print(f"[syntax-error] {path}: {exc}", file=sys.stderr)
        else:
            print(f"[syntax-ok] {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
