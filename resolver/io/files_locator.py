# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Helpers for locating file-based resolver snapshots and tables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

__all__ = [
    "discover_files_root",
    "load_table",
    "series_to_column",
]


@dataclass(frozen=True)
class TableSelection:
    """Represents a located table candidate used for diagnostics."""

    path: Path
    relative_depth: int
    mtime_ns: int

    @classmethod
    def from_path(cls, root: Path, path: Path) -> "TableSelection":
        try:
            rel_parts = len(path.relative_to(root).parts)
        except ValueError:
            rel_parts = len(path.parts)
        try:
            mtime_ns = path.stat().st_mtime_ns
        except FileNotFoundError:
            mtime_ns = 0
        return cls(path=path, relative_depth=rel_parts, mtime_ns=mtime_ns)


def _unique_paths(paths: Iterable[Path]) -> List[Path]:
    seen: set[Path] = set()
    unique: List[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def discover_files_root(preferred: Optional[str] = None) -> Path:
    """Return the most suitable root directory for files/csv backends.

    The search order is:
      1. ``preferred`` when provided and exists.
      2. ``RESOLVER_SNAPSHOTS_DIR`` when set and exists.
      3. ``$CWD/data/snapshots`` when it exists.
      4. ``resolver/tests/data`` shipped with the repository for smoke tests.

    Raises ``FileNotFoundError`` with the searched locations when none exist.
    """

    candidates: List[Path] = []

    def add_candidate(value: Optional[str | os.PathLike[str]]) -> None:
        if not value:
            return
        path = Path(value).expanduser()
        if path in candidates:
            return
        candidates.append(path)

    add_candidate(preferred)
    env_dir = os.environ.get("RESOLVER_SNAPSHOTS_DIR")
    add_candidate(env_dir)
    add_candidate(Path.cwd() / "data" / "snapshots")
    add_candidate(Path(__file__).resolve().parents[1] / "tests" / "data")

    searched = _unique_paths(candidates)
    for path in searched:
        if path.exists():
            return path

    formatted = ", ".join(str(path) for path in searched) or "<none>"
    raise FileNotFoundError(
        "No files backend root found; searched: " + formatted
    )


def _candidate_paths(root: Path, table: str, suffixes: Sequence[str]) -> List[Path]:
    matches: List[Path] = []
    for suffix in suffixes:
        pattern = f"**/{table}{suffix}"
        matches.extend(root.rglob(pattern))
    return matches


def _select_best_candidate(root: Path, candidates: Sequence[Path]) -> Optional[Path]:
    if not candidates:
        return None
    selections = [TableSelection.from_path(root, path) for path in candidates]
    selections.sort(key=lambda item: (item.relative_depth, -item.mtime_ns, str(item.path)))
    return selections[0].path


def _read_parquet(path: Path) -> pd.DataFrame:
    errors: List[str] = []
    try:
        import pyarrow.parquet as pq  # type: ignore

        table = pq.read_table(path)
        return table.to_pandas()
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        errors.append(f"pyarrow missing ({exc})")
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"pyarrow failed ({exc})")
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"pandas.read_parquet failed ({exc})")
        raise RuntimeError("; ".join(errors))


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, encoding="latin-1").fillna("")


def load_table(root: Path, table: str) -> pd.DataFrame:
    """Load ``table`` (csv/parquet) from ``root`` for the files backend.

    Returns an empty ``DataFrame`` with a ``locator_reason`` attribute when the
    table cannot be found.
    """

    root = root.resolve()
    search_order = [".parquet", ".parq", ".csv"]
    candidates = _candidate_paths(root, table, search_order)
    best = _select_best_candidate(root, candidates)
    if best is None:
        df = pd.DataFrame()
        df.attrs["locator_reason"] = (
            f"No files located for table '{table}' under {root}"
        )
        return df

    try:
        if best.suffix.lower() in {".parquet", ".parq"}:
            df = _read_parquet(best)
        else:
            df = _read_csv(best)
    except Exception as exc:
        df = pd.DataFrame()
        df.attrs["locator_reason"] = f"Failed to read {best}: {exc}"
        return df

    df.attrs["locator_path"] = str(best)
    return df


def series_to_column(series: str) -> str:
    """Map resolver series semantics to the expected value column."""

    normalized = (series or "").strip().lower()
    if normalized == "new":
        return "value_new"
    if normalized == "stock":
        return "value_stock"
    return normalized or "value"
