# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.
"""Edition gate for the forecast's CrisisWatch inject (hs_submit stage).

The edition the scan should read is ``crisiswatch.expected_edition``: the
previous calendar month once ICG has normally published it (from the 10th),
the month before that earlier. This step refreshes from the Wayback Machine and stores the
edition; if the expected edition is still absent it retries within its time
cap, then lets the run PROCEED. It never blocks a forecast: an older edition
is described in every prompt as older (``format_crisiswatch_for_prompt``),
and the debug bundle raises a FAIL anomaly naming the edition held.

Writes ``diagnostics/crisiswatch_edition_gate.json`` and always exits 0.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

log = logging.getLogger("crisiswatch_edition_gate")


def expected_edition(today: date) -> tuple[int, int]:
    """``horizon_scanner.crisiswatch.expected_edition``: one rule, one place."""
    from horizon_scanner.crisiswatch import expected_edition as _expected  # noqa: PLC0415

    return _expected(today)


def verdict(held: Iterable[tuple[int, int]], today: date) -> dict:
    held = sorted(set(held))
    want = expected_edition(today)
    newest = held[-1] if held else None
    return {
        "expected_edition": f"{want[0]}-{want[1]:02d}",
        "newest_edition_held": f"{newest[0]}-{newest[1]:02d}" if newest else None,
        "ok": want in held,
    }


def run_gate(
    *,
    today: date,
    held_fn: Callable[[], list[tuple[int, int]]],
    refresh_fn: Callable[[], None],
    deadline_sec: float,
    retry_sec: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Refresh, then retry until the expected edition lands or time runs out."""
    start = clock()
    attempts = 0
    while True:
        attempts += 1
        try:
            refresh_fn()
        except Exception as exc:  # noqa: BLE001 - a failed refresh is retried, never fatal
            log.warning("refresh attempt %d failed: %s", attempts, exc)
        v = verdict(held_fn(), today)
        if v["ok"]:
            break
        if clock() - start + retry_sec > deadline_sec:
            break
        log.info(
            "expected edition %s not held after attempt %d; retrying in %.0fs",
            v["expected_edition"], attempts, retry_sec,
        )
        sleep(retry_sec)
    v["attempts"] = attempts
    v["elapsed_sec"] = round(clock() - start, 1)
    return v


# ------------------------------------------------------------------ live I/O


def _held_from_db() -> list[tuple[int, int]]:
    from horizon_scanner.crisiswatch import editions_held  # noqa: PLC0415

    return editions_held()


def _refresh_and_store() -> None:
    subprocess.run(
        [sys.executable, "-m", "scripts.refresh_crisiswatch", "--source", "wayback",
         "--only-if-newer", "--max-age-days", "60", "--verbose"],
        check=False,
    )
    from horizon_scanner.crisiswatch import bulk_store_crisiswatch  # noqa: PLC0415

    print(f"[crisiswatch] stored {bulk_store_crisiswatch()} entries")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--deadline-min", type=float, default=20.0)
    p.add_argument("--retry-min", type=float, default=5.0)
    p.add_argument("--out", default="diagnostics/crisiswatch_edition_gate.json")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    today = datetime.now(timezone.utc).date()
    try:
        v = run_gate(
            today=today, held_fn=_held_from_db, refresh_fn=_refresh_and_store,
            deadline_sec=args.deadline_min * 60, retry_sec=args.retry_min * 60,
        )
    except Exception as exc:  # noqa: BLE001
        v = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(v, indent=2))
    if v.get("ok"):
        line = f"CrisisWatch edition gate: {v['expected_edition']} is held."
        print(line)
    else:
        line = (
            f"CrisisWatch edition gate: expected edition {v.get('expected_edition')} "
            f"is NOT held after {v.get('attempts')} attempt(s); the scan proceeds on the "
            f"{v.get('newest_edition_held')} edition."
        )
        print(f"::error title=CrisisWatch edition gate::{line}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(f"### CrisisWatch edition gate\n\n{line}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
