# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl calibration hook — deferred (identity pass-through).

Guarded by ``CALIBRATION_ENABLED = False``.

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

from sibyl.aggregate import PooledDistribution
from sibyl.config import CALIBRATION_ENABLED


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
