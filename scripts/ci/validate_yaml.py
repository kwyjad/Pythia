#!/usr/bin/env python3
"""Validate GitHub workflow YAML files."""
from __future__ import annotations

import argparse
import sys
from glob import glob
from pathlib import Path
from typing import Iterable

import yaml


def iter_paths(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = [Path(match) for match in glob(pattern, recursive=True)]
        if not matches:
            print(f"warning: pattern '{pattern}' matched no files", file=sys.stderr)
            continue
        for match in matches:
            try:
                resolved = match.resolve()
            except FileNotFoundError:
                # Broken symlink, treat as missing file.
                print(f"warning: unable to resolve '{match}'", file=sys.stderr)
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(resolved)
    return sorted(paths)


def format_location(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def emit_error(path: Path, exc: yaml.YAMLError) -> None:
    mark = getattr(exc, "problem_mark", None)
    if mark is None:
        print(f"{format_location(path)}: YAML error: {exc}")
        return
    line = mark.line + 1
    column = mark.column + 1
    problem = getattr(exc, "problem", None) or str(exc)
    print(f"{format_location(path)}:{line}:{column}: {problem}")
    try:
        snippet = path.read_text(encoding="utf-8").splitlines()[line - 1]
    except Exception:
        snippet = None
    if snippet is not None:
        print(f"    {snippet}")
        indicator = " " * (column - 1) + "^"
        print(f"    {indicator}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        help="Files or glob patterns to validate (e.g. .github/workflows/*.yml)",
    )
    args = parser.parse_args(argv)

    paths = iter_paths(args.paths)
    if not paths:
        print("No YAML files to validate.", file=sys.stderr)
        return 1

    had_error = False
    for path in paths:
        if not path.is_file():
            print(f"warning: skipping non-file path '{format_location(path)}'", file=sys.stderr)
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                yaml.safe_load(handle)
        except yaml.YAMLError as exc:  # pragma: no cover - CLI tool
            had_error = True
            emit_error(path, exc)
            continue
        except Exception as exc:  # pragma: no cover - diagnostics only
            had_error = True
            print(f"{format_location(path)}: unable to read YAML ({exc})")
            continue
        print(f"{format_location(path)}: OK")

    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
