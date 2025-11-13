import datetime as dt

import pandas as pd

from resolver.tools.export_facts import _enrich_facts_for_validation


def test_enrich_facts_maps_emdat_hazards_and_required_fields():
    today = dt.date.today()
    frame = pd.DataFrame(
        [
            {
                "iso3": "BGD",
                "ym": "2023-03",
                "as_of_date": "2023-03-31",
                "hazard_code": "tropical_cyclone",
                "shock_type": "tropical_cyclone",
                "metric": "affected",
                "value": "1200",
                "publication_date": "",
                "event_id": "",
                "disno_first": "2023-0001",
            },
            {
                "iso3": "BGD",
                "ym": "2023-04",
                "as_of_date": "2023-04-30",
                "hazard_code": "flood",
                "shock_type": "flood",
                "metric": "affected",
                "value": "900",
                "publication_date": "",
                "event_id": "",
                "disno_first": "",
            },
            {
                "iso3": "BGD",
                "ym": "2023-05",
                "as_of_date": "2023-05-31",
                "hazard_code": "",
                "shock_type": "drought",
                "metric": "cases",
                "value": "42",
                "publication_date": "2023-06",
                "event_id": "",
                "disno_first": "",
            },
        ]
    )

    enriched = _enrich_facts_for_validation(frame)

    assert enriched["hazard_code"].tolist() == ["TC", "FL", "DR"]
    assert enriched["hazard_label"].tolist() == ["Tropical Cyclone", "Flood", "Drought"]
    assert enriched["hazard_class"].tolist() == ["natural", "natural", "natural"]

    assert enriched["country_name"].str.strip().tolist() == ["Bangladesh"] * 3

    pub_dates = enriched["publication_date"].tolist()
    for original, pub in zip(frame["as_of_date"], pub_dates):
        pub_date = dt.date.fromisoformat(pub)
        as_of = dt.date.fromisoformat(original)
        assert as_of <= pub_date <= today

    assert enriched.loc[0, "event_id"] == "2023-0001"
    assert enriched.loc[1, "event_id"] == "BGD-FL-2023-04-30"
    assert enriched.loc[2, "event_id"] == "BGD-DR-2023-05-31"

    defaults = {
        "publisher",
        "source_type",
        "source_url",
        "doc_title",
        "definition_text",
        "method",
        "confidence",
        "ingested_at",
    }
    for column in defaults:
        assert enriched[column].str.strip().ne("").all(), column

    assert enriched["revision"].tolist() == [1, 1, 1]
    assert enriched.loc[2, "unit"] == "persons_cases"
    assert enriched.loc[0, "unit"] == "persons"
    assert enriched.loc[1, "unit"] == "persons"

    ingested_dates = enriched["ingested_at"].tolist()
    assert ingested_dates == [today.isoformat()] * 3
