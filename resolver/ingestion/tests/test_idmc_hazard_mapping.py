import pandas as pd

from resolver.ingestion.idmc.hazards import apply_hazard_mapping


def test_apply_hazard_mapping_basic():
    df = pd.read_csv(
        "resolver/ingestion/tests/fixtures/idmc_idu_hazard_samples.csv",
        comment="#",
    )
    mapped = apply_hazard_mapping(df)

    for column in ["hazard_code", "hazard_label", "hazard_class"]:
        assert column in mapped.columns

    mapped_codes = set(mapped.loc[mapped["hazard_code"].notna(), "hazard_code"])
    assert mapped_codes.issuperset({"FL", "DR", "TC", "HW", "PHE", "CU", "ACE"})

    flood_row = mapped[mapped["hazard_code"] == "FL"].iloc[0]
    assert flood_row["hazard_label"] == "Flood"
    assert flood_row["hazard_class"] == "natural"

    riots = mapped[
        mapped["violence_type"].astype(str).str.contains("riot", case=False, na=False)
    ]
    assert set(riots["hazard_code"]) == {"CU"}

    conflict_codes = mapped.loc[
        mapped["displacement_type"].str.lower() == "conflict", "hazard_code"
    ].dropna()
    assert "ACE" in set(conflict_codes)
