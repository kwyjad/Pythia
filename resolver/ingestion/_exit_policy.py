# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

from typing import Iterable, List

BENIGN_SKIP_REASONS = {
    "disabled: config",
    "disabled: ci",
}


def compute_exit_code(results: Iterable[dict]) -> int:
    """
    results: iterable of connector result dicts with keys:
      - status: {"ok","error","skipped"}
      - reason: str or None
    Exit rules:
      - Any 'error' => 1
      - If any 'ok' (i.e., ran), return 0
      - Else (no 'ok'):
          * If all are 'skipped' and every reason is in BENIGN_SKIP_REASONS => 0
          * Otherwise => 1   (e.g., skipped due to missing secret or unknown reason)
    """

    seen_ok = False
    seen_error = False
    reasons: List[str] = []
    for result in results:
        status = (str(result.get("status") or "")).lower()
        if status == "ok":
            seen_ok = True
        elif status == "error":
            seen_error = True
        elif status == "skipped":
            reason = str(result.get("reason") or "").lower()
            reasons.append(reason)
    if seen_error:
        return 1
    if seen_ok:
        return 0
    if reasons and all(reason in BENIGN_SKIP_REASONS for reason in reasons):
        return 0
    return 1
