import pandas as pd

from resolver.tools.export_facts import _ensure_export_contract, _prepare_deltas_for_db


def test_prepare_deltas_for_db_fills_ym_from_as_of_date():
    df = pd.DataFrame(
        {
            "iso3": ["PHL", "PHL"],
            "metric": ["affected", "in_need"],
            "as_of_date": ["2024-01-10", "2024-01-15"],
            "value": ["500", "1000"],
            "series_semantics": ["new", "new"],
            "ym": ["", ""],
        }
    )

    prepared = _prepare_deltas_for_db(df)
    assert prepared is not None
    ym_values = sorted(set(prepared["ym"].astype(str)))
    assert ym_values == ["2024-01"]


def test_prepare_deltas_for_db_fills_ym_from_publication_date():
    df = pd.DataFrame(
        {
            "iso3": ["PHL"],
            "metric": ["affected"],
            "publication_date": ["2024-02-02"],
            "value": ["42"],
            "series_semantics": ["new"],
            "ym": [""],
        }
    )

    prepared = _prepare_deltas_for_db(df)
    assert prepared is not None
    assert prepared.loc[prepared.index[0], "ym"] == "2024-02"


def test_ensure_export_contract_fills_blank_ym_from_dates():
    df = pd.DataFrame(
        {
            "iso3": ["PHL"],
            "metric": ["affected"],
            "value": ["500"],
            "as_of_date": ["2024-01-31"],
            "publication_date": ["2024-02-01"],
            "ym": [""],
        }
    )
    result = _ensure_export_contract(df)
    assert "ym" in result.columns
    assert result.loc[0, "ym"] == "2024-01"
