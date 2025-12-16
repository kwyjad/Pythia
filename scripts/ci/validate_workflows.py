#!/usr/bin/env python3
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Validate all GitHub workflow YAML files."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import yaml


def iter_workflow_paths(root: Path) -> list[Path]:
    patterns: Iterable[str] = ("*.yml", "*.yaml", "*.YML", "*.YAML")
    seen: set[Path] = set()
    paths: list[Path] = []
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(path)
    return paths


def format_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def validate(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        print(f"::error file={format_path(path)}::YAML parse failed: {exc}")
        return False
    except Exception as exc:  # pragma: no cover - unexpected I/O error
        print(f"::error file={format_path(path)}::Unable to read workflow: {exc}")
        return False
    else:
        print(f"{format_path(path)}: OK")
        return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional directories to scan (defaults to .github/workflows)",
    )
    args = parser.parse_args(argv)

    roots = args.paths or [Path(".github/workflows")]
    workflows: list[Path] = []
    for root in roots:
        if root.is_file():
            workflows.append(root)
            continue
        if not root.exists():
            print(f"::warning::Workflow directory {format_path(root)} missing")
            continue
        workflows.extend(iter_workflow_paths(root))

    if not workflows:
        print("::error::No workflow files found to validate")
        return 1

    had_error = False
    for path in sorted(workflows):
        if not validate(path):
            had_error = True

    if had_error:
        return 1

    print(
        "Validated workflows:",
        ", ".join(format_path(path) for path in sorted(workflows)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
