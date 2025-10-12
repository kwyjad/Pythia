"""Unit tests for resolver.tools.build_forecaster_features."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from resolver.tools import build_forecaster_features as features


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_forecaster_features_basic_aggregations(tmp_path: Path) -> None:
    resolved = _frame(
        [
            {
                "ym": "2025-01",
                "iso3": "ETH",
                "hazard_code": "DR",
                "metric": "in_need",
                "as_of_date": "2025-01-10",
                "hazard_class": "natural",
                "precedence_tier": "tier1",
            },
            {
                "ym": "2025-03",
                "iso3": "ETH",
                "hazard_code": "DR",
                "metric": "in_need",
                "as_of_date": "2025-03-18",
                "hazard_class": "natural",
                "precedence_tier": "tier1",
            },
            {
                "ym": "2025-04",
                "iso3": "ETH",
                "hazard_code": "DR",
                "metric": "in_need",
                "as_of_date": "2025-04-22",
                "hazard_class": "natural",
                "precedence_tier": "tier2",
            },
            {
                "ym": "2025-04",
                "iso3": "ETH",
                "hazard_code": "DR",
                "metric": "affected",
                "as_of_date": "2025-04-22",
                "hazard_class": "natural",
                "precedence_tier": "tier2",
            },
        ]
    )

    deltas = _frame(
        [
            {
                "ym": "2025-01",
                "iso3": "ETH",
                "hazard_code": "DR",
                "metric": "in_need",
                "value_new": 100,
                "as_of": "2025-01-15",
            },
            {
                "ym": "2025-03",
                "iso3": "ETH",
                "hazard_code": "DR",
                "metric": "in_need",
                "value_new": 200,
                "as_of": "2025-03-20",
            },
            {
                "ym": "2025-04",
                "iso3": "ETH",
                "hazard_code": "DR",
                "metric": "in_need",
                "value_new": 500,
                "as_of": "2025-04-25",
            },
            {
                "ym": "2025-04",
                "iso3": "ETH",
                "hazard_code": "DR",
                "metric": "affected",
                "value_new": 50,
                "as_of": "2025-04-25",
            },
        ]
    )

    generated = dt.datetime(2025, 4, 30, tzinfo=dt.timezone.utc)

    frame = features.compute_feature_frame(
        resolved=resolved,
        deltas=deltas,
        metrics=("in_need", "affected"),
        generated_at=generated,
    )

    assert set(frame.columns) >= {
        "country_iso3",
        "hazard_code",
        "metric",
        "ym",
        "delta_m1",
        "delta_m3_sum",
        "delta_m6_sum",
        "delta_m12_sum",
        "delta_zscore_6m",
        "sudden_spike_flag",
        "missing_month_flag",
        "as_of_date",
        "as_of_recency_days",
        "source_tier",
        "hazard_class",
    }

    pin_apr = frame[(frame["metric"] == "in_need") & (frame["ym"] == "2025-04")].iloc[0]
    assert pin_apr["delta_m1"] == 500
    assert pin_apr["delta_m3_sum"] == 800
    assert pin_apr["delta_m6_sum"] == 800
    assert pin_apr["delta_m12_sum"] == 800
    assert pin_apr["source_tier"] == "tier2"
    assert pin_apr["hazard_class"] == "natural"
    assert pin_apr["as_of_date"] == "2025-04-25"
    assert pin_apr["as_of_recency_days"] == 5
    assert not pin_apr["missing_month_flag"]

    pin_mar = frame[(frame["metric"] == "in_need") & (frame["ym"] == "2025-03")].iloc[0]
    assert pin_mar["delta_m1"] == 200
    assert pin_mar["missing_month_flag"]
    assert pin_mar["delta_zscore_6m"] == 1.0

    pa_apr = frame[(frame["metric"] == "affected") & (frame["ym"] == "2025-04")].iloc[0]
    assert pa_apr["delta_m1"] == 50
    assert pa_apr["delta_m3_sum"] == 50

    out_path = tmp_path / "resolver_features.parquet"
    features.write_feature_artifacts(frame, out_path)
    assert out_path.exists()
    csv_path = out_path.with_suffix(".csv")
    assert csv_path.exists()
    roundtrip = pd.read_parquet(out_path)
    assert len(roundtrip) == len(frame)
