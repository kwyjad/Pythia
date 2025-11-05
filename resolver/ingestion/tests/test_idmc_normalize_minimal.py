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
    }
    assert (tidy["iso3"].isin(["SDN", "COD"])).all()
    assert set(tidy["metric"].unique()) == {"new_displacements"}
    assert drops["no_iso3"] >= 1
