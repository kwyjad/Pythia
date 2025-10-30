"""Tests for the IDMC resolution-ready facts export adapter."""

from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from resolver.ingestion.idmc.export import (
    FACT_COLUMNS,
    FLOW_METRIC,
    FLOW_SERIES_SEMANTICS,
    build_resolution_ready_facts,
    summarise_facts,
)
from resolver.ingestion.idmc.normalize import normalize_all


def build_normalized_fixture() -> pd.DataFrame:
    """Return a deterministic normalised frame with representative rows."""

    raw = pd.DataFrame(
        [
            {"iso3": "sdn", "displacement_date": "2024-01-05", "figure": 120},
            {"iso3": "SDN", "displacement_end_date": "2024-01-31", "figure": 200},
            {"iso3": "uga", "displacement_date": "2024-02-10", "figure": 50},
            {"iso3": "UGA", "displacement_date": "2024-02-11", "figure": 30},
            {"iso3": None, "displacement_date": "2024-03-01", "figure": 5},
        ]
    )

    aliases = {
        "value_flow": ["figure"],
        "value_stock": [],
        "date": [
            "displacement_date",
            "displacement_start_date",
            "displacement_end_date",
        ],
        "iso3": ["iso3"],
    }
    normalized, _ = normalize_all(
        {"monthly_flow": raw}, aliases, {"start": None, "end": None}, selected_series=["flow"]
    )
    return normalized


def test_build_resolution_ready_facts_contract():
    normalized = build_normalized_fixture()
    facts = build_resolution_ready_facts(normalized)

    assert list(facts.columns) == FACT_COLUMNS
    assert len(facts) == 2
    assert (facts["metric"] == FLOW_METRIC).all()
    assert (facts["series_semantics"] == FLOW_SERIES_SEMANTICS).all()
    assert (facts["source"] == "IDMC").all()
    assert not facts.duplicated(["iso3", "as_of_date", "metric"]).any()
    assert (facts["value"] >= 0).all()

    expected = facts.sort_values(["iso3", "as_of_date"]).reset_index(drop=True)
    actual = facts.sort_values(["iso3", "as_of_date"]).reset_index(drop=True)
    assert_frame_equal(expected, actual)


def test_build_resolution_ready_facts_empty_frame():
    empty = pd.DataFrame(columns=FACT_COLUMNS)
    facts = build_resolution_ready_facts(empty)
    assert facts.empty
    assert list(facts.columns) == FACT_COLUMNS


def test_summarise_facts_reports_uniques():
    normalized = build_normalized_fixture()
    facts = build_resolution_ready_facts(normalized)
    summary = summarise_facts(facts)

    assert summary["rows"] == len(facts)
    assert summary["metrics"] == [FLOW_METRIC]
    assert summary["series_semantics"] == [FLOW_SERIES_SEMANTICS]
    assert summary["countries"] == sorted(facts["iso3"].unique().tolist())
    assert summary["as_of_dates"] == sorted(facts["as_of_date"].unique().tolist())


def test_summarise_facts_empty_frame():
    empty = pd.DataFrame(columns=FACT_COLUMNS)
    summary = summarise_facts(empty)

    assert summary == {
        "rows": 0,
        "metrics": [],
        "series_semantics": [],
        "countries": [],
        "as_of_dates": [],
    }
