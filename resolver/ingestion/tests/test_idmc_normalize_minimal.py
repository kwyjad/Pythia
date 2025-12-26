# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Normalization unit tests for the IDMC skeleton."""
import pandas as pd

from resolver.ingestion.idmc.normalize import normalize_all


def test_idmc_normalize_happy_and_drops():
    monthly = pd.DataFrame(
        {
            "iso3": ["SDN", None, "COD", "XYZ"],
            "month": ["2024-01", "2024-01", "2024-02", "2024-01"],
            "New displacements": [100, 200, 300, None],
        }
    )
    stock = pd.DataFrame(
        {
            "ISO3": ["SDN", "COD"],
            "Date": ["2024", "2024"],
            "IDPs": [3500000, 6000000],
        }
    )
    by_series = {"monthly_flow": monthly, "stock": stock}
    aliases = {
        "value_flow": ["New displacements"],
        "value_stock": ["IDPs"],
        "date": ["month", "Date"],
        "iso3": ["iso3", "ISO3"],
    }

    tidy, drops = normalize_all(by_series, aliases, {"start": None, "end": None})
    assert set(tidy.columns) == {
        "iso3",
        "as_of_date",
        "metric",
        "value",
        "series_semantics",
        "source",
        "ym",
        "record_id",
    }
    assert (tidy["iso3"].isin(["SDN", "COD"])).all()
    assert set(tidy["metric"].unique()) == {"new_displacements"}
    assert drops["no_iso3"] >= 1


def test_idmc_normalize_flow_maps_figure_to_value() -> None:
    raw = pd.DataFrame(
        {
            "iso3": ["AFG", "AFG", "PAK"],
            "displacement_end_date": ["2024-03-31", "2024-03-15", "2024-02-28"],
            "figure": [25, 10, 5],
        }
    )
    aliases = {
        "value_flow": ["figure"],
        "value_stock": [],
        "date": ["displacement_end_date"],
        "iso3": ["iso3"],
    }

    tidy, drops = normalize_all({"monthly_flow": raw}, aliases, {"start": None, "end": None})

    assert drops["no_value_col"] == 0
    assert set(tidy["metric"].unique()) == {"new_displacements"}
    assert tidy["value"].dtype == pd.Int64Dtype()
    assert pd.api.types.is_datetime64_ns_dtype(tidy["as_of_date"])
    afg_value = tidy.loc[tidy["iso3"] == "AFG", "value"].item()
    assert afg_value == 35
