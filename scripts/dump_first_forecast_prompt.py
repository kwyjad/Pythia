# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import logging

from scripts.dump_pythia_debug_bundle import main as debug_bundle_main


logger = logging.getLogger(__name__)


def _load_spd_from_forecasts_ensemble(
    con,
    run_id: str,
    question_id: str,
    *,
    n_buckets: int = 5,
) -> Tuple[Dict[int, List[float]], Dict[int, float]]:
    """
    Load ensemble SPD rows for a single (run_id, question_id) from forecasts_ensemble.

    Returns:
      - ensemble_probs: {month_index: [p_bucket_1, ..., p_bucket_n]}
      - ensemble_ev:    {month_index: ev_value}

    Rows with NULL month_index/bucket_index/probability are ignored so that
    'no_forecast' summary rows do not contaminate the SPD.
    """
    logger.debug(
        "Loading SPD from forecasts_ensemble for run_id=%s question_id=%s",
        run_id,
        question_id,
    )

    rows = con.execute(
        """
        SELECT month_index, bucket_index, probability, ev_value
        FROM forecasts_ensemble
        WHERE run_id = ? AND question_id = ?
        """,
        [run_id, question_id],
    ).fetchall()

    ensemble_probs: Dict[int, List[float]] = {}
    ensemble_ev: Dict[int, float] = {}

    for month_index, bucket_index, probability, ev_value in rows:
        # Skip non-SPD / no-forecast rows (null indices or probability)
        if month_index is None or bucket_index is None or probability is None:
            continue

        m = int(month_index)
        b = int(bucket_index)

        # Initialise probability vector for this month if needed
        vec = ensemble_probs.setdefault(m, [0.0] * n_buckets)

        # Treat bucket_index as 1-based, clamp into [0, n_buckets-1] for safety
        idx = max(0, min(n_buckets - 1, b - 1))
        vec[idx] += float(probability)

        # Pick up EV if present (last value wins; they should agree per month)
        if ev_value is not None:
            ensemble_ev[m] = float(ev_value)

    logger.debug(
        "Loaded SPD for %d month(s) from forecasts_ensemble for run_id=%s question_id=%s",
        len(ensemble_probs),
        run_id,
        question_id,
    )

    return ensemble_probs, ensemble_ev


DEPRECATION_NOTE = (
    "[info] dump_first_forecast_prompt is deprecated; use dump_pythia_debug_bundle instead."
)


def main() -> None:
    print(DEPRECATION_NOTE)
    debug_bundle_main()


if __name__ == "__main__":
    main()
