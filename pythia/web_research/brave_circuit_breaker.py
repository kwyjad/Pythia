# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Brave Search circuit breaker.

Tracks consecutive Brave Search API errors across all call sites.
When 3 consecutive raw API calls return errors (HTTP 429, 402, or
other non-200 status codes), the breaker trips and all subsequent
Brave calls are short-circuited (return None immediately).

The breaker is run-scoped: it resets at the start of each HS run.
"""

from __future__ import annotations

import logging
import threading

LOG = logging.getLogger(__name__)

_CONSECUTIVE_FAILURE_THRESHOLD = 3


class BraveCircuitBreaker:
    """Thread-safe circuit breaker for Brave Search API calls."""

    def __init__(self, threshold: int = _CONSECUTIVE_FAILURE_THRESHOLD):
        self._threshold = threshold
        self._consecutive_failures = 0
        self._tripped = False
        self._lock = threading.Lock()
        self._total_failures = 0
        self._total_successes = 0

    def reset(self):
        """Reset at the start of each run."""
        with self._lock:
            self._consecutive_failures = 0
            self._tripped = False
            self._total_failures = 0
            self._total_successes = 0

    def is_tripped(self) -> bool:
        with self._lock:
            return self._tripped

    def record_success(self):
        with self._lock:
            self._consecutive_failures = 0
            self._total_successes += 1

    def record_failure(self, status_code: int | None = None, error: str | None = None):
        with self._lock:
            self._consecutive_failures += 1
            self._total_failures += 1
            if self._consecutive_failures >= self._threshold and not self._tripped:
                self._tripped = True
                LOG.error(
                    "[BRAVE_CIRCUIT_BREAKER] Tripped after %d consecutive failures "
                    "(total: %d failures, %d successes). Last error: %s (HTTP %s). "
                    "All subsequent Brave calls will be skipped. "
                    "Questions without completed grounding will NOT be forecasted.",
                    self._consecutive_failures,
                    self._total_failures,
                    self._total_successes,
                    error or "unknown",
                    status_code or "?",
                )

    def restore(self, stats: dict) -> None:
        """Seed state from a persisted stats() dict.

        The staged pipeline runs each HS stage in a separate process; the
        breaker state is persisted at the end of the grounding stages and
        restored here so hs_finalize's blocked-no-grounding gate still sees
        a trip that happened during hs_submit/hs_rc_collect (and the Brave
        budget stays cumulative across the run instead of resetting 3x).
        """
        with self._lock:
            self._consecutive_failures = int(stats.get("consecutive_failures") or 0)
            self._total_failures = int(stats.get("total_failures") or 0)
            self._total_successes = int(stats.get("total_successes") or 0)
            self._tripped = bool(stats.get("tripped"))
        if self._tripped:
            LOG.error(
                "[BRAVE_CIRCUIT_BREAKER] Restored TRIPPED state from a previous "
                "pipeline stage — Brave calls stay short-circuited and ungrounded "
                "hazards will be blocked from forecasting."
            )

    def stats(self) -> dict:
        with self._lock:
            return {
                "tripped": self._tripped,
                "consecutive_failures": self._consecutive_failures,
                "total_failures": self._total_failures,
                "total_successes": self._total_successes,
                "threshold": self._threshold,
            }


# Module-level singleton
_breaker = BraveCircuitBreaker()


def get_breaker() -> BraveCircuitBreaker:
    """Return the module-level singleton breaker."""
    return _breaker


def reset():
    """Reset the module-level singleton breaker."""
    _breaker.reset()


def is_tripped() -> bool:
    """Check if the module-level singleton breaker is tripped."""
    return _breaker.is_tripped()
