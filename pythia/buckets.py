# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class BucketSpec:
    idx: int
    label: str
    centroid: float
    lower: float | None = None
    upper: float | None = None


PA_BUCKETS: tuple[BucketSpec, ...] = (
    BucketSpec(idx=1, label="<10k", centroid=0.0, lower=0.0, upper=10_000.0),
    BucketSpec(idx=2, label="10k-<50k", centroid=30_000.0, lower=10_000.0, upper=50_000.0),
    BucketSpec(idx=3, label="50k-<250k", centroid=150_000.0, lower=50_000.0, upper=250_000.0),
    BucketSpec(idx=4, label="250k-<500k", centroid=375_000.0, lower=250_000.0, upper=500_000.0),
    BucketSpec(idx=5, label=">=500k", centroid=700_000.0, lower=500_000.0, upper=None),
)

FATALITIES_BUCKETS: tuple[BucketSpec, ...] = (
    BucketSpec(idx=1, label="<5", centroid=0.0, lower=0.0, upper=5.0),
    BucketSpec(idx=2, label="5-<25", centroid=15.0, lower=5.0, upper=25.0),
    BucketSpec(idx=3, label="25-<100", centroid=62.0, lower=25.0, upper=100.0),
    BucketSpec(idx=4, label="100-<500", centroid=300.0, lower=100.0, upper=500.0),
    BucketSpec(idx=5, label=">=500", centroid=700.0, lower=500.0, upper=None),
)

DR_PHASE3_BUCKETS: tuple[BucketSpec, ...] = (
    BucketSpec(idx=1, label="<100k", centroid=50_000.0, lower=0.0, upper=100_000.0),
    BucketSpec(idx=2, label="100k-<1M", centroid=500_000.0, lower=100_000.0, upper=1_000_000.0),
    BucketSpec(idx=3, label="1M-<5M", centroid=2_500_000.0, lower=1_000_000.0, upper=5_000_000.0),
    BucketSpec(idx=4, label="5M-<15M", centroid=10_000_000.0, lower=5_000_000.0, upper=15_000_000.0),
    BucketSpec(idx=5, label=">=15M", centroid=20_000_000.0, lower=15_000_000.0, upper=None),
)

BUCKET_SPECS: Mapping[str, Sequence[BucketSpec]] = {
    "PA": PA_BUCKETS,
    "FATALITIES": FATALITIES_BUCKETS,
    "PHASE3PLUS_IN_NEED": DR_PHASE3_BUCKETS,
}


def get_bucket_specs(metric: str) -> Sequence[BucketSpec]:
    return BUCKET_SPECS.get(metric.upper(), ())


def thresholds_for(metric: str) -> list[float]:
    """Bucket boundary thresholds ``[0, b1, .., b4, +inf]`` for a metric.

    Derived from ``BUCKET_SPECS`` so scoring/calibration modules share one
    source of truth (duplicated literal lists have already caused one
    documented drift bug). Empty list for unknown metrics.
    """
    specs = get_bucket_specs(metric)
    if not specs:
        return []
    out: list[float] = [float(specs[0].lower) if specs[0].lower is not None else 0.0]
    for s in specs:
        out.append(float(s.upper) if s.upper is not None else float("inf"))
    return out


def interior_thresholds_for(metric: str) -> list[float]:
    """The finite bucket boundaries only (no leading 0 / trailing +inf)."""
    return thresholds_for(metric)[1:-1]


def labels_for(metric: str) -> list[str]:
    """Bucket display labels for a metric, in bucket order."""
    return [s.label for s in get_bucket_specs(metric)]
