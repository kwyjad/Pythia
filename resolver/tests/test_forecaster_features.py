"""Unit tests for resolver.tools.build_forecaster_features."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from resolver.tools import build_forecaster_features as features
from resolver.tools import build_llm_context as llm_context


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


def test_llm_context_bundle_rounding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    months = ["2024-01", "2024-02"]

    january = _frame(
        [
            {
                "iso3": "eth",
                "hazard_code": "dr",
                "metric": "in_need",
                "unit": "persons",
                "value_new": "1249.6",
            },
            {
                "iso3": "eth",
                "hazard_code": "dr",
                "metric": "affected",
                "unit": "households",
                "value_new": 75.5,
            },
        ]
    )
    february = _frame(
        [
            {
                "iso3": "Ken",
                "hazard_code": "FL",
                "metric": "in_need",
                "unit": "persons",
                "value": 200.2,
            }
        ]
    )
    fixtures = {"2024-01": january, "2024-02": february}

    def fake_loader(ym: str, is_current_month: bool, requested_series: str, backend: str):
        assert not is_current_month
        assert requested_series == "new"
        assert backend == "db"
        df = fixtures.get(ym)
        if df is None:
            return None, "fake", requested_series
        return df, "fake", requested_series

    monkeypatch.setattr(llm_context.selectors, "load_series_for_month", fake_loader)

    frame, counts = llm_context.build_context_frame(months)
    assert list(frame.columns) == list(llm_context.TARGET_COLUMNS)
    assert counts == {"2024-01": 2, "2024-02": 1}
    assert (frame["series"] == "new").all()

    bundle = llm_context.write_context_bundle(frame, tmp_path)
    assert bundle.jsonl.exists()
    assert bundle.parquet.exists()

    with bundle.jsonl.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    assert len(records) == 3
    persons_values = [row["value"] for row in records if row["unit"].lower().startswith("person")]
    assert persons_values and all(isinstance(value, int) for value in persons_values)
    assert persons_values[0] == 1250

    households = next(row for row in records if row["unit"] == "households")
    assert isinstance(households["value"], float)
    assert households["value"] == pytest.approx(75.5)

    parquet_frame = pd.read_parquet(bundle.parquet).sort_values(llm_context.TARGET_COLUMNS).reset_index(drop=True)
    expected = frame.sort_values(list(frame.columns)).reset_index(drop=True)
    pd.testing.assert_frame_equal(parquet_frame, expected, check_dtype=False)
