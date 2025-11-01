"""Utilities for preparing IDMC staging outputs."""
from __future__ import annotations

import csv
import os
import pathlib
from typing import Iterable, Sequence

DEFAULT_DIR = "resolver/staging/idmc"
DEFAULT_FLOW = f"{DEFAULT_DIR}/flow.csv"
DEFAULT_STOCK = f"{DEFAULT_DIR}/stock.csv"

_HEADER_COLUMNS: Iterable[str] = (
    "iso3",
    "as_of_date",
    "metric",
    "value",
    "series_semantics",
    "source",
)


def ensure_staging(dirpath: str = DEFAULT_DIR) -> str:
    """Ensure the staging directory for IDMC exists and return its path."""

    pathlib.Path(dirpath).mkdir(parents=True, exist_ok=True)
    return dirpath


def _write_header(path: pathlib.Path, header: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))


def write_header_if_empty(path: str = DEFAULT_FLOW, *, header: Sequence[str] | None = None) -> str:
    """Write an empty CSV with ``header`` if ``path`` does not exist."""

    dest = pathlib.Path(path)
    if dest.exists():
        return dest.as_posix()
    _write_header(dest, header or tuple(_HEADER_COLUMNS))
    return dest.as_posix()


def ensure_header_pair(dirpath: str = DEFAULT_DIR) -> tuple[str, str]:
    """Ensure both flow and stock CSV headers exist in ``dirpath``."""

    ensure_staging(dirpath)
    flow_path = write_header_if_empty(os.path.join(dirpath, "flow.csv"))
    stock_path = write_header_if_empty(os.path.join(dirpath, "stock.csv"))
    return flow_path, stock_path


__all__ = ["ensure_staging", "write_header_if_empty", "ensure_header_pair"]
