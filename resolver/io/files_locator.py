"""Helpers for locating file-based resolver snapshots and tables."""

from __future__ import annotations

import logging
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


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:  # pragma: no cover - keep logging quiet without configuration
    LOGGER.addHandler(logging.NullHandler())


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


def discover_files_root(
    preferred: Optional[Path | str | os.PathLike[str]] = None,
) -> Path:
    """Return the most suitable root directory for files/csv backends."""

    package_root = Path(__file__).resolve().parents[1]
    repo_root = package_root.parent
    searched: list[tuple[str, Path]] = []

    def _normalise(
        value: Optional[Path | str | os.PathLike[str]],
    ) -> Optional[Path]:
        if value is None or value == "":
            return None
        path = value if isinstance(value, Path) else Path(value)
        try:
            return path.expanduser().resolve(strict=False)
        except (FileNotFoundError, RuntimeError):
            # ``resolve`` may fail for certain path types (e.g., Windows drives on
            # POSIX). Fall back to an absolute representation for diagnostics.
            return path.expanduser().absolute()

    def _consider(
        source: str,
        value: Optional[Path | str | os.PathLike[str]],
        *,
        sub: Optional[str] = None,
    ) -> Optional[Path]:
        base = _normalise(value)
        if base is None:
            return None
        candidate = base / sub if sub else base
        try:
            candidate = candidate.resolve(strict=False)
        except (FileNotFoundError, RuntimeError):
            candidate = candidate.absolute()
        searched.append((source, candidate))
        if candidate.exists():
            LOGGER.debug("discover_files_root: using %s (source=%s)", candidate, source)
            return candidate
        return None

    # 1) Explicit ``preferred`` argument wins when it exists on disk.
    result = _consider("preferred", preferred)
    if result is not None:
        return result

    # 2) Respect the direct environment override.
    result = _consider("RESOLVER_FILES_ROOT", os.environ.get("RESOLVER_FILES_ROOT"))
    if result is not None:
        return result

    # Allow callers to steer discovery via snapshot hints before consulting
    # fast-fixture fallbacks.
    result = _consider("RESOLVER_SNAPSHOTS_DIR", os.environ.get("RESOLVER_SNAPSHOTS_DIR"))
    if result is not None:
        return result

    # 3) Fall back to fast-fixture hints and bootstrap artifacts.
    for source, value, sub in [
        ("RESOLVER_STAGING_DIR/exports", os.environ.get("RESOLVER_STAGING_DIR"), "exports"),
        ("RESOLVER_STAGING_DIR", os.environ.get("RESOLVER_STAGING_DIR"), None),
        ("RESOLVER_TEST_DATA_DIR/exports", os.environ.get("RESOLVER_TEST_DATA_DIR"), "exports"),
        ("RESOLVER_TEST_DATA_DIR", os.environ.get("RESOLVER_TEST_DATA_DIR"), None),
        ("RESOLVER_FAST_EXPORTS_DIR", os.environ.get("RESOLVER_FAST_EXPORTS_DIR"), None),
        ("FAST_EXPORTS_ROOT", os.environ.get("FAST_EXPORTS_ROOT"), None),
        ("cwd/data/exports", Path.cwd() / "data" / "exports", None),
        ("cwd/data/snapshots", Path.cwd() / "data" / "snapshots", None),
        ("repo/resolver/exports", repo_root / "resolver" / "exports", None),
        ("repo/resolver/exports/backfill", repo_root / "resolver" / "exports" / "backfill", None),
        ("repo/resolver/snapshots", repo_root / "resolver" / "snapshots", None),
    ]:
        result = _consider(source, value, sub=sub)
        if result is not None:
            return result

    # 4) Repository test data is the final fallback before we fail.
    result = _consider("resolver/tests/data", package_root / "tests" / "data")
    if result is not None:
        return result

    formatted = ", ".join(f"{label}: {path}" for label, path in searched) or "<none>"
    LOGGER.debug("discover_files_root: no viable candidates; searched=%s", formatted)
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
