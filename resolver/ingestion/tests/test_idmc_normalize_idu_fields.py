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
    assert set(tidy["as_of_date"]) == {"2024-02-29", "2024-01-31"}
    assert set(tidy["metric"]) == {"idp_displacement_new_idmc"}
    assert set(tidy["series_semantics"]) == {"new"}
    assert tidy.loc[tidy["iso3"] == "SDN", "value"].item() == 800

    assert drops["no_iso3"] >= 1
    assert drops["date_parse_failed"] >= 1
    assert drops["duplicates_dropped"] == 1
