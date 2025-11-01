from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Union


log = logging.getLogger(__name__)
PathLike = Union[str, Path]


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def write_text(path: PathLike, text: str, *, encoding: str = "utf-8") -> Path:
    p = Path(path)
    _ensure_parent(p)
    p.write_text(text, encoding=encoding)
    log.debug("write_text: %s (%d bytes)", p, len(text.encode(encoding, errors="ignore")))
    return p


def write_bytes(path: PathLike, data: bytes) -> Path:
    p = Path(path)
    _ensure_parent(p)
    p.write_bytes(data)
    log.debug("write_bytes: %s (%d bytes)", p, len(data))
    return p


def write_json(path: PathLike, obj: Any, *, encoding: str = "utf-8", indent: int = 2) -> Path:
    p = Path(path)
    _ensure_parent(p)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=str(p.parent), encoding=encoding
    ) as tmp:
        json.dump(obj, tmp, ensure_ascii=False, indent=indent, default=str)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, p)
    log.debug(
        "write_json: %s (keys=%s)",
        p,
        list(obj.keys()) if isinstance(obj, dict) else type(obj).__name__,
    )
    return p


def append_jsonl(path: PathLike, obj: Any, *, encoding: str = "utf-8") -> Path:
    p = Path(path)
    _ensure_parent(p)
    line = json.dumps(obj, ensure_ascii=False)
    with p.open("a", encoding=encoding) as fh:
        fh.write(line + "\n")
    log.debug("append_jsonl: %s (len=%d)", p, len(line))
    return p


def attach_config_source(
    diagnostics: Mapping[str, Any] | None, connector_name: str
) -> Dict[str, Any]:
    """Return ``diagnostics`` enriched with config source details."""

    payload: Dict[str, Any] = dict(diagnostics or {})
    try:  # Import lazily to avoid circular dependencies during startup
        from resolver.ingestion._shared.config_loader import get_config_details
    except Exception:  # pragma: no cover - defensive import guard
        return payload

    details = get_config_details(connector_name)
    if not details:
        return payload

    payload["config_source"] = details.source
    payload["config_path"] = details.path.as_posix()

    config_block = payload.get("config")
    if isinstance(config_block, Mapping):
        config_payload = dict(config_block)
    else:
        config_payload = {}
    config_payload.setdefault("config_source_label", details.source)
    config_payload.setdefault("config_path_used", details.path.as_posix())
    config_payload.setdefault("ingestion_config_path", details.ingestion_path.as_posix())
    config_payload.setdefault("legacy_config_path", details.fallback_path.as_posix())
    if details.warnings:
        existing = config_payload.get("config_warnings")
        warnings: list[str] = []
        if isinstance(existing, Sequence) and not isinstance(existing, (str, bytes)):
            warnings.extend(str(item) for item in existing if str(item))
        warnings.extend(str(item) for item in details.warnings if str(item))
        if warnings:
            seen: set[str] = set()
            deduped: list[str] = []
            for entry in warnings:
                if entry not in seen:
                    deduped.append(entry)
                    seen.add(entry)
            config_payload["config_warnings"] = deduped
    payload["config"] = config_payload

    existing_warnings = payload.get("warnings")
    combined: list[str] = []
    if isinstance(existing_warnings, Sequence) and not isinstance(
        existing_warnings, (str, bytes)
    ):
        combined.extend(str(item) for item in existing_warnings if str(item))
    combined.extend(str(item) for item in details.warnings if str(item))
    if combined:
        seen_messages: set[str] = set()
        deduped_messages: list[str] = []
        for message in combined:
            if message not in seen_messages:
                deduped_messages.append(message)
                seen_messages.add(message)
        payload["warnings"] = deduped_messages

    return payload


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
