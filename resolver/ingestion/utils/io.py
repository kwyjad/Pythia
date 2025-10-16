"""I/O helpers shared across ingestion connectors."""

from __future__ import annotations

import csv
import datetime as dt
import os
import re
from pathlib import Path
from typing import Sequence

_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9_.-]+")


def ensure_headers(path: Path, headers: Sequence[str]) -> None:
    """Write a header-only CSV to ``path`` ensuring parent directories exist."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(headers))


def _parse_iso_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return dt.date.fromisoformat(text[:10])
    except ValueError:
        return None


def _clean_segment(segment: str | None, fallback: str = "unknown") -> str:
    text = (segment or "").strip()
    if not text:
        return fallback
    sanitized = _SAFE_SEGMENT.sub("_", text)
    sanitized = sanitized.strip("._-")
    return sanitized or fallback


def resolve_period_label() -> str:
    """Return the effective staging period label used for output directories."""

    explicit = os.getenv("RESOLVER_PERIOD") or os.getenv("RESOLVER_STAGING_PERIOD")
    if explicit:
        return _clean_segment(explicit)

    start = _parse_iso_date(os.getenv("RESOLVER_START_ISO"))
    end = _parse_iso_date(os.getenv("RESOLVER_END_ISO"))

    if start and end:
        if start == end:
            return _clean_segment(start.isoformat())
        start_q = (start.month - 1) // 3 + 1
        end_q = (end.month - 1) // 3 + 1
        if start.year == end.year and start_q == end_q:
            return _clean_segment(f"{start.year}Q{start_q}")
        return _clean_segment(f"{start.isoformat()}_{end.isoformat()}")

    if start or end:
        solo = start or end
        if solo:
            return _clean_segment(solo.isoformat())

    return "unknown"


def resolve_staging_dir(default_root: Path) -> Path:
    """Return the base staging directory for connector outputs."""

    override_dir = os.getenv("RESOLVER_OUTPUT_DIR")
    if override_dir:
        return Path(override_dir).expanduser()

    root_env = os.getenv("RESOLVER_STAGING_DIR")
    if root_env:
        base = Path(root_env).expanduser()
        period = resolve_period_label()
        return (base / period / "raw").resolve()

    return default_root


def resolve_output_path(default_path: Path) -> Path:
    """Return the resolved CSV output path for a connector."""

    override_path = os.getenv("RESOLVER_OUTPUT_PATH")
    if override_path:
        return Path(override_path).expanduser()

    default_path = default_path.expanduser()
    output_dir = os.getenv("RESOLVER_OUTPUT_DIR")
    if output_dir:
        return Path(output_dir).expanduser() / default_path.name

    staging_dir = resolve_staging_dir(default_path.parent)
    return staging_dir / default_path.name
