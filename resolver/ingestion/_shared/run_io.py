from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Union


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
