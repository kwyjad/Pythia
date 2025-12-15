# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd

from resolver.ingestion.dtm_client import compute_monthly_flows


def _frame(values: list[tuple[str, int]]) -> pd.DataFrame:
    records = []
    for month_iso, value in values:
        bucket = date.fromisoformat(month_iso)
        records.append(
            {
                "country_iso3": "AAA",
                "admin1": "Test",  # same admin for ordering
                "month_start": bucket,
                "as_of": datetime.combine(bucket, datetime.min.time(), tzinfo=UTC),
                "value": value,
                "raw_event_id": f"raw:{month_iso}",
                "raw_fields": {"reportingDate": month_iso, "value": value},
            }
        )
    return pd.DataFrame.from_records(records)


def test_monotone_increase_produces_positive_flows() -> None:
    frame = _frame([("2024-01-01", 100), ("2024-02-01", 180), ("2024-03-01", 240)])
    flows, has_negative = compute_monthly_flows(frame)
    assert not has_negative
    assert list(flows["value"]) == [80, 60]


def test_gaps_do_not_interpolate() -> None:
    frame = _frame([("2024-01-01", 50), ("2024-03-01", 70)])
    flows, has_negative = compute_monthly_flows(frame)
    assert not has_negative
    assert list(flows["value"]) == [20]
    assert flows.iloc[0]["month_start"].isoformat() == "2024-03-01"


def test_first_observation_is_dropped() -> None:
    frame = _frame([("2024-01-01", 10), ("2024-02-01", 30)])
    flows, _ = compute_monthly_flows(frame)
    assert len(flows) == 1
    assert flows.iloc[0]["value"] == 20


def test_negative_deltas_are_preserved() -> None:
    frame = _frame([("2024-01-01", 300), ("2024-02-01", 250), ("2024-03-01", 275)])
    flows, has_negative = compute_monthly_flows(frame)
    assert not has_negative
    assert list(flows["value"]) == [0, 25]
