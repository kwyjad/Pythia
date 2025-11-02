"""Helpers for writing connector error diagnostics payloads.

This module is intentionally lightweight so it can be imported from
``scripts/ci`` utilities without pulling in heavy optional dependencies.

The helpers normalise payloads and make sure callers always receive a JSON
structure containing enough information for the summariser to surface useful
context (exit codes, log tails, hints, etc.).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

DEFAULT_ERROR_FILENAME = "error.json"
DEFAULT_LOG_TAIL_KB = 64


def _ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _clip_text(text: str, *, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def read_log_tail(path: os.PathLike[str] | str, *, limit_kb: int = DEFAULT_LOG_TAIL_KB) -> str:
    """Return the trailing ``limit_kb`` KB of ``path`` (best-effort)."""

    try:
        target = Path(path)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return ""
    try:
        size = target.stat().st_size
    except OSError:
        return ""
    limit_bytes = max(limit_kb, 1) * 1024
    try:
        with target.open("rb") as handle:
            if size > limit_bytes:
                handle.seek(-limit_bytes, os.SEEK_END)
            chunk = handle.read()
    except OSError:
        return ""
    try:
        return chunk.decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover - best effort
        return ""


def _normalise_hints(hints: Iterable[str] | None) -> list[str]:
    if not hints:
        return []
    results: list[str] = []
    for hint in hints:
        text = str(hint).strip()
        if text:
            results.append(text)
    return results


def _normalise_extras(extras: Mapping[str, object] | None) -> dict[str, object]:
    if not isinstance(extras, Mapping):
        return {}
    normalised: MutableMapping[str, object] = {}
    for key, value in extras.items():
        try:
            key_text = str(key)
        except Exception:  # pragma: no cover - defensive
            continue
        normalised[key_text] = value
    return dict(normalised)


def write_error_report(
    directory: os.PathLike[str] | str,
    *,
    exit_code: int | None = None,
    message: str | None = None,
    stderr_tail: str | None = None,
    log_path: os.PathLike[str] | str | None = None,
    log_tail_kb: int = DEFAULT_LOG_TAIL_KB,
    hints: Iterable[str] | None = None,
    extras: Mapping[str, object] | None = None,
) -> str:
    """Serialise an error payload to ``directory / error.json``.

    ``log_path`` may be provided to fetch a trailing snippet of the log file
    automatically. ``stderr_tail`` can be used by callers that already captured
    the tail. Both snippets are stored (when available) so that the summary
    generator can surface them to developers.
    """

    try:
        dest_dir = Path(directory)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        raise TypeError("directory must be path-like")

    payload: dict[str, object | None] = {
        "exit_code": int(exit_code) if exit_code is not None else None,
        "message": message.strip() if isinstance(message, str) else None,
        "stderr_tail": _clip_text(stderr_tail, limit=log_tail_kb * 1024)
        if isinstance(stderr_tail, str)
        else None,
        "log_path": str(log_path) if log_path else None,
        "log_tail": None,
        "hints": _normalise_hints(hints),
    }

    if log_path:
        payload["log_tail"] = read_log_tail(log_path, limit_kb=log_tail_kb)

    extra_payload = _normalise_extras(extras)
    if extra_payload:
        payload["extras"] = extra_payload

    target = dest_dir / DEFAULT_ERROR_FILENAME
    _ensure_directory(target)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return target.as_posix()


__all__ = ["write_error_report", "read_log_tail", "DEFAULT_ERROR_FILENAME"]
