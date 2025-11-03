"""Helpers for working with ISO3 country codes."""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List, Optional, Sequence

LOGGER = logging.getLogger(__name__)

_MASTER_PATH = Path("resolver/ingestion/static/iso3_master.csv")


def _ensure_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        return (Path.cwd() / candidate).resolve()
    return candidate.resolve()


def load_all_iso3(path: str | Path = _MASTER_PATH) -> List[str]:
    """Return the ordered list of ISO3 codes from ``path``.

    The helper tolerates files without headers and silently skips blank rows.
    """

    resolved = _ensure_path(path)
    codes: List[str] = []
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                code = str(row[0]).strip().upper()
                if not code:
                    continue
                if code not in codes:
                    codes.append(code)
    except FileNotFoundError:
        LOGGER.warning("ISO3 master list missing at %s", resolved)
    except OSError as exc:  # pragma: no cover - defensive logging
        LOGGER.error("Failed to read ISO3 master list at %s: %s", resolved, exc)
    return codes


def _normalize_requested(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return []
    normalised: List[str] = []
    for raw in values:
        text = str(raw).strip().upper()
        if not text:
            continue
        if text not in normalised:
            normalised.append(text)
    return normalised


def resolve_countries(cfg_list: Optional[Sequence[str]]) -> List[str]:
    """Return validated ISO3 codes, defaulting to the master list when empty."""

    requested = _normalize_requested(cfg_list)
    master = load_all_iso3()
    master_set = set(master)

    if not requested:
        LOGGER.debug(
            "No countries configured; falling back to ISO3 master list (%d entries)",
            len(master),
        )
        return master

    unknown = [code for code in requested if code not in master_set]
    if unknown:
        LOGGER.warning("Ignoring %d unrecognised ISO3 codes: %s", len(unknown), ", ".join(unknown))
    resolved = [code for code in requested if code in master_set]

    LOGGER.debug(
        "Resolved countries: requested=%d valid=%d invalid=%d sample=%s",
        len(requested),
        len(resolved),
        len(unknown),
        ", ".join(resolved[:10]),
    )
    return resolved if resolved else master
