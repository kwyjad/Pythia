import pandas as pd

from resolver.tools.export_facts import _map_dtm_admin0_with_flows


def test_dtm_dual_series_basic():
    df = pd.DataFrame(
        {
            "country_iso3": ["AAA", "AAA", "AAA"],
            "as_of": ["2025-01-15", "2025-02-15", "2025-03-15"],
            "value": [100, 150, 130],
            "source": ["IOM DTM"] * 3,
        }
    )
    out = _map_dtm_admin0_with_flows(df)

    assert set(out["metric"].unique()) == {
        "idp_displacement_stock_dtm",
        "idp_displacement_new_dtm",
    }

    stock = (
        out[out["metric"] == "idp_displacement_stock_dtm"]
        .sort_values(["iso3", "as_of_date"])
        .reset_index(drop=True)
    )
    new = (
        out[out["metric"] == "idp_displacement_new_dtm"]
        .sort_values(["iso3", "as_of_date"])
        .reset_index(drop=True)
    )

    assert [int(v) for v in stock["value"].tolist()] == [100, 150, 130]
    assert [int(v) for v in new["value"].tolist()] == [0, 50, 0]
