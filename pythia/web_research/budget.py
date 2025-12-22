# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict


class BudgetExceededError(Exception):
    """Raised when a hard budget or call cap would be exceeded."""


def _env_float(name: str, default: float | None) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        val = float(raw)
        if val < 0:
            return default
        return val
    except Exception:
        return default


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        val = int(raw)
        if val < 0:
            return default
        return val
    except Exception:
        return default


@dataclass
class BudgetState:
    calls: int = 0
    cost_usd: float = 0.0


class BudgetGuard:
    """Per-run budget + call guard to prevent runaway web research spend."""

    _STATE: Dict[str, BudgetState] = {}

    def __init__(self, run_id: str | None):
        self.run_id = run_id or "default"
        self.budget_usd = _env_float("PYTHIA_WEB_RESEARCH_BUDGET_USD_PER_RUN", None)
        self.max_calls = _env_int("PYTHIA_WEB_RESEARCH_MAX_CALLS_PER_QUESTION", None)
        self.state = self._STATE.setdefault(self.run_id, BudgetState())

    def check_and_reserve(self, *, cost_estimate: float = 0.0) -> None:
        """Raise BudgetExceededError if adding this call would cross the cap."""

        projected_calls = self.state.calls + 1
        projected_cost = self.state.cost_usd + float(cost_estimate or 0.0)

        if self.max_calls is not None and projected_calls > self.max_calls:
            raise BudgetExceededError(
                f"web research call cap exceeded: {projected_calls}/{self.max_calls}"
            )

        if self.budget_usd is not None and projected_cost > self.budget_usd:
            raise BudgetExceededError(
                f"web research budget exceeded: ${projected_cost:.2f}/${self.budget_usd:.2f}"
            )

        self.state.calls = projected_calls
        self.state.cost_usd = projected_cost

    def record_actual(self, *, cost_usd: float) -> None:
        """Record actual cost after a call completes."""

        try:
            self.state.cost_usd += float(cost_usd or 0.0)
        except Exception:
            return
