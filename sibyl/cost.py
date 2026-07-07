# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl cost accounting and the run budget guardrail.

Costs are recorded in two places:

* Pythia's existing ledger — every Opus call and Brave query is logged to
  the ``llm_calls`` table via ``log_forecaster_llm_call`` with
  ``phase='sibyl'`` (so the /v1/costs dashboard itemises Sibyl spend, and
  the by-model grouping separates Opus tokens from Brave credits);
* an in-process :class:`CostTracker` — the authoritative running total for
  the hard run cut-off, checked at every question boundary and between the
  K trials of a question.

Budget semantics (load-bearing, not decorative): once ``run_cost_usd >=
RUN_HARD_CAP_USD`` no new question or trial is STARTED. The unit in flight
when the cap is crossed runs to completion (token cost is only known after
a call returns), so realized spend can exceed the cap by roughly one
question's cost. Aborting mid-trial and discarding partial work is
deliberately NOT done.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from sibyl.config import BUDGET_USD_PER_QUESTION, RUN_HARD_CAP_USD

logger = logging.getLogger(__name__)

COST_KIND_OPUS = "opus"
COST_KIND_BRAVE = "brave"


@dataclass
class CostBreakdown:
    opus_usd: float = 0.0
    brave_usd: float = 0.0

    @property
    def total_usd(self) -> float:
        return self.opus_usd + self.brave_usd

    def add(self, kind: str, cost_usd: float) -> None:
        if kind == COST_KIND_BRAVE:
            self.brave_usd += cost_usd
        else:
            self.opus_usd += cost_usd

    def to_dict(self) -> Dict[str, float]:
        return {
            "opus_usd": round(self.opus_usd, 6),
            "brave_usd": round(self.brave_usd, 6),
            "total_usd": round(self.total_usd, 6),
        }


class CostTracker:
    """Thread-safe run/question/trial cost accumulator + budget checks."""

    def __init__(
        self,
        run_hard_cap_usd: float = RUN_HARD_CAP_USD,
        budget_usd_per_question: Optional[float] = BUDGET_USD_PER_QUESTION,
    ) -> None:
        self.run_hard_cap_usd = float(run_hard_cap_usd)
        self.budget_usd_per_question = (
            float(budget_usd_per_question) if budget_usd_per_question else None
        )
        self._lock = threading.Lock()
        self._run = CostBreakdown()
        self._questions: Dict[str, CostBreakdown] = {}

    def add(self, question_id: str, kind: str, cost_usd: float) -> None:
        cost = max(0.0, float(cost_usd or 0.0))
        with self._lock:
            self._run.add(kind, cost)
            self._questions.setdefault(question_id, CostBreakdown()).add(kind, cost)

    @property
    def run_cost_usd(self) -> float:
        with self._lock:
            return self._run.total_usd

    def run_breakdown(self) -> CostBreakdown:
        with self._lock:
            return CostBreakdown(self._run.opus_usd, self._run.brave_usd)

    def question_breakdown(self, question_id: str) -> CostBreakdown:
        with self._lock:
            b = self._questions.get(question_id, CostBreakdown())
            return CostBreakdown(b.opus_usd, b.brave_usd)

    def question_cost_usd(self, question_id: str) -> float:
        return self.question_breakdown(question_id).total_usd

    def hard_cap_reached(self) -> bool:
        """True once cumulative run spend has reached the hard cap."""
        return self.run_cost_usd >= self.run_hard_cap_usd

    def question_cap_reached(self, question_id: str) -> bool:
        """Secondary per-question guard (only when configured)."""
        if self.budget_usd_per_question is None:
            return False
        return self.question_cost_usd(question_id) >= self.budget_usd_per_question


def log_sibyl_call(
    *,
    run_id: str,
    question_id: str,
    prompt_text: str,
    response_text: str,
    provider: str,
    model_id: str,
    usage: Dict[str, Any],
    iso3: str = "",
    hazard_code: str = "",
    metric: str = "",
    error_text: str = "",
    hs_run_id: str = "",
    call_type: str = "sibyl_agent_step",
) -> None:
    """Log one Sibyl call (Opus step or Brave query) to ``llm_calls``.

    Thin sync wrapper over the repo's async ledger writer; logging failures
    never crash a forecast (same guarantee as the forecaster).
    """
    from forecaster.llm_logging import log_forecaster_llm_call  # noqa: PLC0415

    try:
        asyncio.run(
            log_forecaster_llm_call(
                run_id=run_id,
                question_id=question_id,
                prompt_text=prompt_text,
                response_text=response_text,
                provider=provider,
                model_id=model_id,
                model_name="sibyl",
                phase="sibyl",
                call_type=call_type,
                iso3=iso3,
                hazard_code=hazard_code,
                metric=metric,
                usage=usage,
                error_text=error_text or None,
                hs_run_id=hs_run_id,
                is_test=_is_test_mode(),
            )
        )
    except Exception as exc:  # noqa: BLE001 - ledger failures must not stop the run
        logger.warning("sibyl.cost: llm_calls logging failed: %s", exc)


def _is_test_mode() -> bool:
    from pythia.test_mode import is_test_mode  # noqa: PLC0415

    return bool(is_test_mode())
