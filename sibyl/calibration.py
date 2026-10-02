# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl calibration: the advice loader, and a deferred statistical hook.

``load_advice`` (live since Oct 2026) returns the newest
``sibyl_calibration_advice`` text for a question's class, which
``sibyl.agent`` shows in a YOUR TRACK RECORD section. ``calibrate`` is the
statistical correction, still an identity pass-through: six scored
questions cannot fit one, and the findings rows ``sibyl.advice`` writes now
hold what a later fit will need.

``calibrate`` is guarded by ``CALIBRATION_ENABLED = False``.

Intended implementation (once a resolved Sibyl track record exists):
PIT-based recalibration fitted per hazard x horizon over standard Pythia's
resolved numeric history. For each resolved question, evaluate the pooled
CDF at the realized value to get a PIT sample u = F(x_realized); the
empirical PIT distribution diagnoses miscalibration:

* U-shaped PIT histogram => distributions too narrow => inflate spread
  (e.g. widen the CDF around its median by a fitted factor);
* skewed PIT => systematic bias => shift the distribution;
* uniform PIT => leave alone.

This is the distributional analogue of per-source hierarchical calibration
and requires resolved history — which is why it ships disabled until the
track record accumulates. Everything needed to fit it later is persisted by
``sibyl/spd.py`` into ``sibyl_forecasts`` (per-trial quantiles, pooled
quantiles, asOf, K, aggregation method), so enabling it is a pure addition
here — no refactor of the write path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Optional, Union

from sibyl.aggregate import PooledDistribution
from sibyl.config import BACKTEST_MODE, CALIBRATION_ENABLED

logger = logging.getLogger(__name__)


@dataclass
class SibylAdvice:
    text: str
    as_of_month: str
    scope: str  # 'group' | 'pooled'
    n_questions: int


def _month_key(as_of: Union[date, str, None]) -> str:
    if as_of is None:
        return date.today().strftime("%Y-%m")
    if isinstance(as_of, str):
        return as_of[:7]
    return as_of.strftime("%Y-%m")


def load_advice(
    hazard: str,
    metric: str,
    as_of: Union[date, str, None] = None,
    *,
    con: Any = None,
    backtest: Optional[bool] = None,
) -> Optional[SibylAdvice]:
    """The advice in force for (hazard, metric) at *as_of*, or None.

    The newest generation dated on or before the as-of month; within it the
    class's own row first, the pooled row second, and only a row whose
    ``advice`` is non-empty counts. A class in ``PYTHIA_ADVICE_BLOCK_GROUPS`` gets nothing.

    In backtest mode it returns None: advice learned from outcomes after the
    as-of date is leakage, and a row's month says when it was WRITTEN, not
    which outcomes it learned from. Never raises (an absent table is "no
    advice").
    """
    if BACKTEST_MODE if backtest is None else backtest:
        return None
    hz, m = (hazard or "").upper(), (metric or "").upper()
    try:
        from pythia.tools.generate_calibration_advice import advice_blocked_groups

        if (hz, m) in advice_blocked_groups():
            return None
    except Exception:  # noqa: BLE001 - the block list is advisory here
        pass
    month = _month_key(as_of)
    own = con is None
    try:
        if own:
            from pythia.db.schema import connect

            con = connect(read_only=False)
        # The newest generation on or before the as-of month decides; within
        # it, the class's own row before the pooled row. An older class row
        # never outranks a newer month: the generator rewrites every class
        # each month, so a class that went quiet has been re-measured.
        rows = con.execute(
            """
            WITH newest AS (
                SELECT MAX(as_of_month) AS m FROM sibyl_calibration_advice
                WHERE as_of_month <= ?
            )
            SELECT advice, as_of_month, scope, n_questions, hazard_code
            FROM sibyl_calibration_advice, newest
            WHERE as_of_month = newest.m
              AND COALESCE(advice, '') <> ''
              AND ((upper(hazard_code) = ? AND upper(metric) = ?)
                   OR (hazard_code = '*' AND metric = '*'))
            ORDER BY CASE WHEN hazard_code = '*' THEN 1 ELSE 0 END
            LIMIT 1
            """,
            [month, hz, m],
        ).fetchall()
    except Exception as exc:  # noqa: BLE001
        logger.debug("sibyl.calibration: no advice table or read failed: %s", exc)
        return None
    finally:
        if own and con is not None:
            try:
                con.close()
            except Exception:  # noqa: BLE001
                pass
    if not rows:
        return None
    text, row_month, scope, n, _ = rows[0]
    return SibylAdvice(text=str(text), as_of_month=str(row_month), scope=str(scope or ""),
                       n_questions=int(n or 0))


def calibrate(
    spd: PooledDistribution, hazard: str, horizon: int
) -> PooledDistribution:
    """Recalibrate a pooled distribution for (hazard, horizon).

    Identity while ``CALIBRATION_ENABLED`` is False (and while no fitted
    parameters exist). See the module docstring for the planned PIT-based
    implementation.
    """
    if not CALIBRATION_ENABLED:
        return spd
    # Placeholder: no fitted parameters exist yet. When implemented, load
    # per-(hazard, horizon) PIT-fit parameters and transform spd here.
    return spd
