"""Connector contract for the IDMC (IDU) normalisation pipeline."""

import pandas as pd
from pandas.testing import assert_frame_equal

from resolver.ingestion.idmc.normalize import normalize_all


def build_idmc_fixture() -> pd.DataFrame:
    """Return a deterministic DataFrame covering edge cases."""

    return pd.DataFrame(
        [
            {"iso3": "cod", "displacement_date": "2023-01-05", "figure": 10},
            {"ISO3": "COD", "displacement_start_date": "2023-01-20", "figure": 15},
            {"ISO3": "UGA", "displacement_start_date": "2023-02-01", "figure": 5},
            {"ISO3": "SDN", "displacement_end_date": "2023-03-31", "figure": 0},
            {"ISO3": "sdn", "displacement_date": "2023-03-12", "figure": -3},
            {"ISO3": "  mex", "displacement_date": "2023-04-17", "figure": 7},
            {"ISO3": None, "displacement_date": "2023-05-01", "figure": 4},
        ]
    )


def run_normalize(frame: pd.DataFrame):
    aliases = {
        "value_flow": ["figure"],
        "value_stock": [],
        "date": [
            "displacement_date",
            "displacement_start_date",
            "displacement_end_date",
        ],
        "iso3": ["iso3", "ISO3"],
    }
    return normalize_all({"monthly_flow": frame}, aliases, {"start": None, "end": None})


def test_idmc_connector_contract_monthly_flow():
    raw = build_idmc_fixture()

    tidy_first, drops_first = run_normalize(raw.copy())
    tidy_second, drops_second = run_normalize(raw.copy())

    assert list(tidy_first.columns) == [
        "iso3",
        "as_of_date",
        "metric",
        "value",
        "series_semantics",
        "source",
    ]

    assert (tidy_first["metric"] == "new_displacements").all()
    assert (tidy_first["series_semantics"] == "new").all()
    assert (tidy_first["value"] >= 0).all()
    assert tidy_first["iso3"].str.match(r"^[A-Z]{3}$").all()

    as_of = pd.to_datetime(tidy_first["as_of_date"], errors="raise")
    assert as_of.dt.is_month_end.all()

    assert not tidy_first.duplicated(["iso3", "as_of_date", "metric"]).any()
    assert drops_first["dup_event"] in {0, 1}
    assert drops_first["negative_value"] == 1
    assert drops_first["no_iso3"] == 1

    tidy_sorted = tidy_first.sort_values(["iso3", "as_of_date"]).reset_index(drop=True)
    tidy_second_sorted = tidy_second.sort_values(["iso3", "as_of_date"]).reset_index(drop=True)
    assert_frame_equal(tidy_sorted, tidy_second_sorted)
    assert drops_first == drops_second
