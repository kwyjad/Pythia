from __future__ import annotations

import csv
import json
from pathlib import Path


def ensure_parent(path: Path) -> None:
    """Ensure the parent directory for ``path`` exists."""

    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: dict) -> None:
    """Serialise ``obj`` to ``path`` with stable formatting."""

    ensure_parent(path)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def count_csv_rows(csv_path: Path) -> int:
    """Return the number of data rows (excluding the header) in ``csv_path``."""

    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)
