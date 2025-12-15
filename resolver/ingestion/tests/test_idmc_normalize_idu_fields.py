# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import pandas as pd

from resolver.ingestion.idmc.normalize import normalize_all


def test_idmc_normalize_idu_fields():
    raw = pd.DataFrame(
        [
            {"ISO3": "SDN", "displacement_date": "2024-02-12", "figure": 800},
            {"iso3": "COD", "displacement_start_date": "2024-01-03", "figure": 500},
            {"iso3": "SDN", "displacement_date": "2024-02-05", "figure": 700},
            {"iso3": "XYZ", "displacement_date": "2024-02-05", "figure": 100},
            {"iso3": "SDN", "displacement_date": "bad-date", "figure": 200},
        ]
    )

    tidy, drops = normalize_all(
        {"monthly_flow": raw},
        {
            "value_flow": ["figure"],
            "value_stock": [],
            "date": ["displacement_date", "displacement_start_date"],
            "iso3": ["ISO3", "iso3"],
        },
        {"start": None, "end": None},
    )

    assert list(tidy.columns) == [
        "iso3",
        "as_of_date",
        "metric",
        "value",
        "series_semantics",
        "source",
    ]
    assert len(tidy) == 2
    assert set(tidy["iso3"]) == {"SDN", "COD"}
    assert set(pd.to_datetime(tidy["as_of_date"], errors="raise").dt.date) == {
        pd.Timestamp("2024-02-29").date(),
        pd.Timestamp("2024-01-31").date(),
    }
    assert set(tidy["metric"]) == {"new_displacements"}
    assert set(tidy["series_semantics"]) == {"new"}
    assert tidy.loc[tidy["iso3"] == "SDN", "value"].item() == 1500

    assert drops["no_iso3"] >= 1
    assert drops["date_parse_failed"] >= 1
    assert drops["dup_event"] == 0


def test_idmc_normalize_monthly_rollup():
    raw = pd.DataFrame(
        [
            {
                "iso3": "AFG",
                "displacement_start_date": "2024-04-01",
                "displacement_end_date": "2024-04-30",
                "figure": 100,
            },
            {
                "iso3": "AFG",
                "displacement_end_date": "2024-04-15",
                "figure": 50,
            },
            {
                "iso3": "PAK",
                "displacement_start_date": "2024-04-10",
                "figure": 25,
            },
        ]
    )

    tidy, drops = normalize_all(
        {"monthly_flow": raw},
        {
            "value_flow": ["figure"],
            "value_stock": [],
            "date": ["displacement_start_date"],
            "iso3": ["iso3"],
        },
        {"start": "2024-04-01", "end": "2024-04-30"},
    )

    assert len(tidy) == 2
    afg_row = tidy[tidy["iso3"] == "AFG"].iloc[0]
    assert afg_row["value"] == 150
    assert afg_row["as_of_date"] == pd.Timestamp("2024-04-30")
    assert drops["date_parse_failed"] == 0
