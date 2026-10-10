# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The /v1/costs/* routes build once per DB version, one at a time."""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("fastapi")

from pythia.api import core
from pythia.api.routes import costs


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(costs, "_con", lambda: object())
    monkeypatch.setattr(core, "_READ_CON_MTIME", 100.0)
    costs._COSTS_CACHE.clear()
    yield
    costs._COSTS_CACHE.clear()


def test_a_second_request_is_served_from_the_cache() -> None:
    calls = []

    def build(con):
        calls.append(1)
        return {"rows": [1]}

    assert costs._cached_costs("total", None, False, build) == {"rows": [1]}
    assert costs._cached_costs("total", None, False, build) == {"rows": [1]}
    assert len(calls) == 1


def test_a_swapped_db_is_never_answered_from_the_old_one(monkeypatch: pytest.MonkeyPatch) -> None:
    costs._cached_costs("total", None, False, lambda con: {"v": "old"})
    monkeypatch.setattr(core, "_READ_CON_MTIME", 200.0)
    assert costs._cached_costs("total", None, False, lambda con: {"v": "new"}) == {"v": "new"}
    assert all(v[0] == 200.0 for v in costs._COSTS_CACHE.values())


def test_an_unexpected_track_is_computed_but_never_cached() -> None:
    costs._cached_costs("total", 99, False, lambda con: {"v": 1})
    assert costs._COSTS_CACHE == {}


def test_parallel_requests_build_one_at_a_time() -> None:
    active = []
    peak = []
    lock = threading.Lock()

    def build(con):
        with lock:
            active.append(1)
            peak.append(len(active))
        threading.Event().wait(0.05)
        with lock:
            active.pop()
        return {}

    threads = [
        threading.Thread(target=costs._cached_costs, args=(name, None, False, build))
        for name in ("total", "monthly", "runs", "latencies", "run_runtimes")
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert max(peak) == 1
