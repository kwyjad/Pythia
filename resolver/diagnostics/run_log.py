# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Run-scoped JSONL evidence streams.

Every claim a log makes in prose should also exist as a row somewhere. A
line saying "252 cells were inconclusive" is not a substitute for 252 rows
saying which cells and why — so the code paths that KNOW those facts append
them here, and :mod:`scripts.build_resolver_debug_bundle` turns the streams
into the bundle's ledgers.

Three properties this module must keep:

* **Off by default.** Nothing is written unless ``PYTHIA_RUN_LOG_DIR`` names
  a directory. A local run, a unit test and every existing caller behave
  exactly as before.
* **Never raises.** A diagnostic that can fail an ingest is worse than no
  diagnostic. Every public function swallows its own errors and disables
  itself after the first failure, so a full disk costs one warning rather
  than a corrupted phase.
* **Append-only, one line per fact.** In-process writers are serialised by
  a lock; separate connector subprocesses append whole lines to the same
  file, and the reader skips any line it cannot parse. So a run killed
  mid-write, or an unlucky interleave, costs one record rather than the
  file.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Iterator

LOG = logging.getLogger(__name__)

#: Directory for this run's streams. Unset -> recording is off.
ENV_DIR = "PYTHIA_RUN_LOG_DIR"

#: Stream names the bundle knows how to read.
STREAM_HTTP = "http_requests"
STREAM_ENVELOPE = "http_envelopes"
STREAM_CELLS = "cell_ledger"
STREAM_FIGURES = "figures_ledger"

_LOCK = threading.Lock()
_DISABLED = False


def run_log_dir() -> Path | None:
    """The directory streams are written to, or None when recording is off."""

    if _DISABLED:
        return None
    raw = os.environ.get(ENV_DIR, "").strip()
    if not raw:
        return None
    return Path(raw)


def enabled() -> bool:
    return run_log_dir() is not None


def stream_path(stream: str) -> Path | None:
    directory = run_log_dir()
    if directory is None:
        return None
    return directory / f"{stream}.jsonl"


def record(stream: str, payload: dict[str, Any]) -> None:
    """Append one fact to ``stream``. Never raises."""

    global _DISABLED
    path = stream_path(stream)
    if path is None:
        return
    try:
        line = json.dumps(payload, default=str, ensure_ascii=False)
    except Exception as exc:  # pragma: no cover - default=str makes this rare
        LOG.debug("[run_log] could not serialise a %s record: %s", stream, exc)
        return
    try:
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception as exc:  # noqa: BLE001 - a diagnostic must never fail a run
        _DISABLED = True
        LOG.warning(
            "[run_log] disabling run-scoped evidence streams after a write "
            "failure (%s); the run itself is unaffected",
            exc,
        )


def read_stream(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    """Yield the records in a stream file, skipping any unparseable line.

    A run killed mid-write leaves a truncated final line. Losing that one
    record is right; refusing to read the other 40,000 is not.
    """

    file = Path(path)
    if not file.is_file():
        return
    with open(file, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def reset_for_tests() -> None:
    """Re-enable recording after a deliberate write failure in a test."""

    global _DISABLED
    _DISABLED = False
