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

BUCKET_SPECS: Mapping[str, Sequence[BucketSpec]] = {
    "PA": PA_BUCKETS,
    "FATALITIES": FATALITIES_BUCKETS,
}


def get_bucket_specs(metric: str) -> Sequence[BucketSpec]:
    return BUCKET_SPECS.get(metric.upper(), ())
