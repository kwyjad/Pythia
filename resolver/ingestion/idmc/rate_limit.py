"""Rate limiting helpers for the IDMC connector."""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

__all__ = ["TokenBucket", "parse_retry_after"]


def _now() -> float:
    return time.monotonic()


class TokenBucket:
    """Thread-safe token bucket implementation.

    The bucket supports dependency injection of both the monotonic clock and the
    sleep function which keeps tests fully deterministic. ``rate_per_sec``
    controls the sustained rate while ``burst`` allows short spikes before the
    limiter begins queueing.
    """

    def __init__(
        self,
        rate_per_sec: float,
        *,
        burst: float = 1.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_fn: Callable[[], float] = _now,
    ) -> None:
        self.rate = max(0.0, float(rate_per_sec))
        self.capacity = max(float(burst), 1.0)
        self.tokens = self.capacity
        self.last = now_fn()
        self.lock = threading.Lock()
        self.sleep_fn = sleep_fn
        self.now_fn = now_fn

    def _refill(self) -> None:
        now = self.now_fn()
        delta = max(0.0, now - self.last)
        self.last = now
        if self.rate <= 0.0:
            self.tokens = self.capacity
            return
        self.tokens = min(self.capacity, self.tokens + delta * self.rate)

    def acquire(self) -> float:
        """Acquire a single token from the bucket.

        Returns the planned sleep duration (seconds). A zero return value
        indicates that the caller was allowed to proceed immediately. Tests can
        stub ``sleep_fn`` to avoid real sleeps while still observing the planned
        pacing.
        """

        with self.lock:
            self._refill()
            if self.rate <= 0.0:
                self.tokens = max(0.0, self.tokens - 1.0)
                return 0.0
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return 0.0
            need = 1.0 - self.tokens
            wait_s = need / self.rate if self.rate > 0 else 0.0
        if wait_s > 0:
            self.sleep_fn(wait_s)
        with self.lock:
            self._refill()
            self.tokens = max(0.0, self.tokens - 1.0)
        return max(0.0, wait_s)


def parse_retry_after(header: Optional[str]) -> Optional[float]:
    """Parse a ``Retry-After`` header into seconds, if possible."""

    if not header:
        return None
    value = header.strip()
    if not value:
        return None
    try:
        seconds = float(value)
        if seconds < 0:
            return None
        return seconds
    except ValueError:
        pass
    # ``Retry-After`` may also be an HTTP date. Attempt to parse using the
    # standard library and fall back to ``None`` when parsing fails.
    try:  # pragma: no cover - very uncommon; exercised defensively
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(value)
        if dt is None:
            return None
        now = time.time()
        delta = dt.timestamp() - now
        return max(0.0, delta)
    except Exception:  # pragma: no cover - defensive parsing
        return None
