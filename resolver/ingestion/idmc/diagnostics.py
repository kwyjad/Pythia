"""Diagnostics helpers for the IDMC connector skeleton."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def diagnostics_dir() -> str:
    """Return the diagnostics directory for IDMC runs."""

    return _ensure_dir(os.path.join("diagnostics", "ingestion", "idmc"))


def connectors_log_path() -> str:
    """Return the shared connectors log path, creating parent directories."""

    return os.path.join(_ensure_dir(os.path.join("diagnostics", "ingestion")), "connectors.jsonl")


def write_connectors_line(payload: Dict[str, Any]) -> None:
    """Append a diagnostics line for the connector run."""

    line = {"connector": "idmc", "ts": int(time.time())}
    line.update(payload)
    with open(connectors_log_path(), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, ensure_ascii=False) + "\n")


def write_sample_preview(name: str, csv_head: str) -> str:
    """Persist a sample preview CSV and return its path."""

    path = os.path.join(diagnostics_dir(), f"{name}_preview.csv")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(csv_head)
    return path
