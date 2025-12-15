# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Offline tests for the IDMC rate limiter."""
from __future__ import annotations

import pytest

from resolver.ingestion.idmc.rate_limit import TokenBucket


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.planned: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.planned.append(duration)
        self.now += duration


def test_token_bucket_respects_rate_without_sleeping() -> None:
    clock = FakeClock()
    bucket = TokenBucket(rate_per_sec=2.0, sleep_fn=clock.sleep, now_fn=clock.monotonic)

    assert bucket.acquire() == pytest.approx(0.0)
    assert bucket.acquire() == pytest.approx(0.5, rel=1e-3)
    assert bucket.acquire() == pytest.approx(0.5, rel=1e-3)

    assert len(clock.planned) == 2
    assert clock.planned[0] == pytest.approx(0.5, rel=1e-3)
    assert clock.planned[1] == pytest.approx(0.5, rel=1e-3)
    assert clock.monotonic() == pytest.approx(1.0, rel=1e-3)
