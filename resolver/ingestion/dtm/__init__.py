# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Utilities for DTM ingestion."""

__all__ = [
    "normalize_admin0",
    "detect_value_column",
]

from .normalize import detect_value_column, normalize_admin0  # noqa: E402,F401
