"""Utilities for preparing IDMC staging outputs."""
from __future__ import annotations

import os
import pathlib
from typing import Iterable

import pandas as pd

DEFAULT_DIR = "resolver/staging/idmc"
DEFAULT_FLOW = f"{DEFAULT_DIR}/flow.csv"
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


def write_header_if_empty(path: str = DEFAULT_FLOW) -> str:
    """Write an empty CSV with the expected header if ``path`` is missing."""

    if not os.path.exists(path):
        pd.DataFrame(columns=list(_HEADER_COLUMNS)).to_csv(path, index=False)
    return path
