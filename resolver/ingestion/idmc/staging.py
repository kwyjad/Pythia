# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Utilities for preparing IDMC staging outputs."""
from __future__ import annotations

import os
import pathlib

import pandas as pd

from .export import FLOW_EXPORT_COLUMNS

DEFAULT_DIR = "resolver/staging/idmc"
DEFAULT_FLOW = f"{DEFAULT_DIR}/flow.csv"


def ensure_staging(dirpath: str = DEFAULT_DIR) -> str:
    """Ensure the staging directory for IDMC exists and return its path."""

    pathlib.Path(dirpath).mkdir(parents=True, exist_ok=True)
    return dirpath


def write_header_if_empty(path: str = DEFAULT_FLOW) -> str:
    """Write an empty CSV with the expected header if ``path`` is missing."""

    if not os.path.exists(path):
        pd.DataFrame(columns=list(FLOW_EXPORT_COLUMNS)).to_csv(path, index=False)
    return path
