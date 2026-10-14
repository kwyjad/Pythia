# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The label for a Sibyl forecast carried forward from an earlier run.

A question re-asked in the same epoch keeps its earlier Sibyl forecast when
the new Sibyl run did not pick it (the forecast-of-record rule, #995). On
13 October 2026 COL_DR_PHASE3PLUS_IN_NEED_2026-11 and MEX_FL_PA_2026-11
showed Sibyl results from sibyl_1790890510860 (1 October) beside the
13 October ensemble with nothing to say so. Import-light (stdlib only): the
API process and the bundles both call it.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def _date(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text[:10] if len(text) >= 10 else text


def carried_forward(
    *,
    sibyl_run_id: Optional[str],
    sibyl_forecast_run_id: Optional[str],
    current_run_id: Optional[str],
    sibyl_created_at: Any = None,
) -> Optional[Dict[str, Any]]:
    """``{"carried_forward": True, ...}`` when the Sibyl forecast belongs to an
    earlier forecaster run than *current_run_id*; None when it is this run's
    or nothing can be compared."""
    if not sibyl_forecast_run_id or not current_run_id:
        return None
    if str(sibyl_forecast_run_id) == str(current_run_id):
        return None
    day = _date(sibyl_created_at)
    return {
        "carried_forward": True,
        "sibyl_run_id": sibyl_run_id,
        "sibyl_date": day,
        "sibyl_forecast_run_id": str(sibyl_forecast_run_id),
        "current_run_id": str(current_run_id),
        "label": (
            f"Carried forward: Sibyl forecast from {sibyl_run_id or 'an earlier run'}"
            + (f" ({day})" if day else "")
            + f", made beside forecaster run {sibyl_forecast_run_id}; Sibyl did not "
            f"forecast this question in run {current_run_id}."
        ),
    }
