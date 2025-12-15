# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import pandas as pd


def dump_json(obj: Any, path: Path) -> None:
    """Serialize *obj* as JSON to *path*, creating parent directories."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=_json_default)


def write_sample_csv(df: pd.DataFrame, path: Path, n: int = 200) -> None:
    """Persist up to *n* rows from *df* as CSV to *path*."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.head(max(n, 0)).to_csv(target, index=False)


@dataclass
class TimingResult:
    label: str
    start_ns: int
    elapsed_ms: Optional[int] = None

    def stop(self) -> None:
        if self.elapsed_ms is None:
            self.elapsed_ms = max(0, int((time.perf_counter_ns() - self.start_ns) / 1_000_000))


@contextmanager
def timing(label: str) -> Iterator[TimingResult]:
    """Measure a code block, yielding a :class:`TimingResult`."""

    result = TimingResult(label=label, start_ns=time.perf_counter_ns())
    try:
        yield result
    finally:
        result.stop()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")
