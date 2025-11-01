"""Utility helpers shared across summarizer modules."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Tuple


__all__ = [
    "safe_load_json",
    "safe_load_jsonl",
    "status_histogram",
    "reason_histogram",
    "gather_log_files",
    "gather_meta_json_files",
    "top_value_counts_from_csv",
]


def safe_load_json(path: Path | str) -> Any:
    """Return parsed JSON data from *path* or ``None`` on failure."""

    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def safe_load_jsonl(path: Path | str) -> List[Any]:
    """Return parsed JSON objects from a JSONL file, skipping malformed rows."""

    target = Path(path)
    if not target.exists():
        return []
    entries: List[Any] = []
    try:
        with target.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return entries


def status_histogram(entries: Iterable[Mapping[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for entry in entries:
        status = None
        extras = entry.get("extras") if isinstance(entry, Mapping) else None
        if isinstance(extras, Mapping):
            status = extras.get("status_raw")
        if status is None and isinstance(entry, Mapping):
            status = entry.get("status")
        if status is None:
            continue
        text = str(status).strip()
        if text:
            counter[text] += 1
    return counter


def reason_histogram(entries: Iterable[Mapping[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        reason = entry.get("reason")
        if reason is None and isinstance(entry.get("extras"), Mapping):
            reason = entry["extras"].get("reason")
        if reason is None:
            continue
        text = str(reason).strip()
        if text:
            counter[text] += 1
    return counter


def gather_log_files(root: Path | str) -> List[Path]:
    base = Path(root)
    logs_dir = base / "logs"
    if not logs_dir.exists():
        return []
    return sorted(p for p in logs_dir.glob("*.log") if p.is_file())


def gather_meta_json_files(root: Path | str) -> List[Path]:
    base = Path(root)
    return sorted(p for p in base.rglob("*.meta.json") if p.is_file())


def top_value_counts_from_csv(
    path: Path | str,
    column: str,
    *,
    limit: int = 5,
    sample_rows: int = 5000,
) -> List[Tuple[str, int]]:
    target = Path(path)
    if not target.exists():
        return []
    counts: Counter[str] = Counter()
    try:
        with target.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader):
                if idx >= sample_rows:
                    break
                if not isinstance(row, Mapping):
                    continue
                value = row.get(column)
                if value is None:
                    continue
                text = str(value).strip()
                if not text:
                    continue
                counts[text] += 1
    except (OSError, csv.Error, UnicodeDecodeError):
        return []
    return counts.most_common(limit)
