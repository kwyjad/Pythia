"""Lightweight helpers for validating connector configuration entries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence, Tuple

__all__ = ["validate_required_fields", "write_json"]


def _normalise_entry(entry: Any) -> MutableMapping[str, Any]:
    if isinstance(entry, MutableMapping):
        return dict(entry)
    if isinstance(entry, Mapping):
        return dict(entry)
    return {"_value": entry}


def validate_required_fields(
    items: Iterable[Mapping[str, Any]] | None,
    required: Sequence[str] = ("id_or_path",),
) -> Tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split configuration entries into valid and invalid collections."""

    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    if not items:
        return valid, invalid
    required_fields = tuple(str(field) for field in required if str(field).strip())
    for raw in items:
        entry = _normalise_entry(raw)
        missing = [field for field in required_fields if not str(entry.get(field) or "").strip()]
        if missing:
            record = dict(entry)
            record["_missing_required"] = missing
            invalid.append(record)
            continue
        valid.append(dict(entry))
    return valid, invalid


def write_json(path: str | Path, payload: Any) -> None:
    """Serialise *payload* to ``path`` ensuring directories exist."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
