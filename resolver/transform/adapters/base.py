# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Base classes for canonical normalization adapters."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Final

import pandas as pd

LOGGER = logging.getLogger(__name__)

CANONICAL_COLUMNS: Final[list[str]] = [
    "event_id",
    "country_name",
    "iso3",
    "hazard_code",
    "hazard_label",
    "hazard_class",
    "metric",
    "unit",
    "as_of_date",
    "value",
    "series_semantics",
    "source",
]


class BaseAdapter(ABC):
    """Shared helpers for canonical normalization adapters."""

    canonical_source: str = ""
    raw_slug: str | None = None

    def __init__(self, cli_source: str) -> None:
        self.cli_source = cli_source

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------
    @abstractmethod
    def load(self, raw_path: Path) -> pd.DataFrame:
        """Load a raw staging dataframe from ``raw_path``."""

    @abstractmethod
    def map(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Transform a staging dataframe into the canonical schema."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def resolve_raw_path(self, input_dir: Path) -> Path:
        """Return the expected raw CSV path for this adapter."""

        input_dir = input_dir.expanduser().resolve()
        expected = input_dir / f"{self.raw_slug or self.cli_source}.csv"
        if expected.exists():
            return expected

        fallback = input_dir / f"{self.cli_source}.csv"
        if fallback.exists():
            LOGGER.debug(
                "%s: raw file missing for slug %s, using %s",
                self.cli_source,
                self.raw_slug,
                fallback,
            )
            return fallback

        raise FileNotFoundError(f"No raw CSV found for '{self.cli_source}' in {input_dir}")

    def normalize(self, input_dir: Path) -> pd.DataFrame:
        """Run the full normalization pipeline for ``cli_source``."""

        raw_path = self.resolve_raw_path(input_dir)
        LOGGER.info("%s: loading raw rows from %s", self.cli_source, raw_path)
        frame = self.load(raw_path)
        before = len(frame)
        LOGGER.info("%s: loaded %s rows", self.cli_source, before)
        canonical = self.map(frame)
        after = len(canonical)
        LOGGER.info("%s: mapped %s rows → %s rows", self.cli_source, before, after)
        return self._validate_canonical(canonical)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_canonical(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Ensure the mapped dataframe matches the canonical schema."""

        missing = [col for col in CANONICAL_COLUMNS if col not in frame.columns]
        if missing:
            raise ValueError(f"{self.cli_source}: missing canonical columns: {missing}")

        extra = [col for col in frame.columns if col not in CANONICAL_COLUMNS]
        if extra:
            raise ValueError(f"{self.cli_source}: unexpected canonical columns: {extra}")

        return frame.loc[:, CANONICAL_COLUMNS]

    @property
    def canonical_slug(self) -> str:
        """Return the canonical source slug for the ``source`` column."""

        return self.canonical_source or self.cli_source


__all__ = ["BaseAdapter", "CANONICAL_COLUMNS", "LOGGER"]
