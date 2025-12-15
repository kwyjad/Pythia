# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import pandas as pd

from resolver.ingestion.dtm.normalize import normalize_admin0


def _iso_lookup_factory(mapping):
    def _lookup(row):
        explicit = str(row.get("CountryISO3") or "").strip()
        if explicit:
            return explicit[:3].upper(), None
        name = str(row.get("CountryName") or "").strip()
        return mapping.get(name), None

    return _lookup


def test_chosen_value_column_counts_raw_pre_filter():
    iso_lookup = _iso_lookup_factory({"Democratic Republic of the Congo": "COD"})
    df = pd.DataFrame(
        [
            {
                "CountryName": "Democratic Republic of the Congo",
                "ReportingDate": "2023-03-15",
                "TotalIDPs": 120,
            },
            {
                "CountryName": "Atlantis",
                "ReportingDate": "2023-03-15",
                "TotalIDPs": 15,
            },
        ]
    )
    result = normalize_admin0(
        df,
        idp_aliases=["TotalIDPs", "IDPTotal"],
        start_iso=None,
        end_iso=None,
        iso3_lookup=iso_lookup,
    )
    assert result["counters"]["chosen_value_columns"] == [
        {"column": "TotalIDPs", "count": 2}
    ]
    assert result["counters"]["drop_reasons"]["no_iso3"] == 1

    df_no_value = pd.DataFrame(
        [
            {
                "CountryName": "Côte d'Ivoire",
                "ReportingDate": "2023-03-01",
                "OtherValue": 5,
            }
        ]
    )
    result_no_value = normalize_admin0(
        df_no_value,
        idp_aliases=["TotalIDPs", "IDPTotal"],
        start_iso=None,
        end_iso=None,
        iso3_lookup=_iso_lookup_factory({"Côte d'Ivoire": "CIV"}),
    )
    assert result_no_value["zero_rows_reason"] == "invalid_indicator"
    assert result_no_value["counters"]["drop_reasons"]["no_value_col"] == 1


def test_date_window_and_out_of_window_counter():
    iso_lookup = _iso_lookup_factory({"Democratic Republic of the Congo": "COD"})
    df = pd.DataFrame(
        [
            {
                "CountryName": "Democratic Republic of the Congo",
                "ReportingDate": "2023-02-01",
                "TotalIDPs": 200,
            },
            {
                "CountryName": "Democratic Republic of the Congo",
                "ReportingDate": "2022-12-01",
                "TotalIDPs": 300,
            },
        ]
    )
    result = normalize_admin0(
        df,
        idp_aliases=["TotalIDPs"],
        start_iso="2023-01-01",
        end_iso="2023-03-31",
        iso3_lookup=iso_lookup,
    )
    assert result["counters"]["drop_reasons"]["date_out_of_window"] == 1
    assert len(result["df"]) == 1


def test_date_parse_failures_increment_counter():
    iso_lookup = _iso_lookup_factory({"Democratic Republic of the Congo": "COD"})
    df = pd.DataFrame(
        [
            {
                "CountryName": "Democratic Republic of the Congo",
                "ReportingDate": "not-a-date",
                "TotalIDPs": 50,
            }
        ]
    )
    result = normalize_admin0(
        df,
        idp_aliases=["TotalIDPs"],
        start_iso=None,
        end_iso=None,
        iso3_lookup=iso_lookup,
    )
    assert result["counters"]["drop_reasons"]["date_parse_failed"] == 1
    assert result["df"].empty
