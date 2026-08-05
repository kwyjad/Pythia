# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Population denominator loading into haz_raw_population (no network)."""

from __future__ import annotations

import duckdb

from resolver.hazard_resolution.population import (
    _known_iso3s,
    load_population_into_db,
    load_population_records,
)


def test_known_iso3s_covers_countries_not_aggregates():
    known = _known_iso3s()
    assert known is not None
    assert "AFG" in known
    # World Bank aggregate codes are ISO3-shaped but are not countries.
    assert "AFE" not in known


def test_parses_repo_population_csv():
    records = load_population_records()
    assert len(records) > 150
    assert set(records.columns) >= {"iso3", "year", "population", "source"}
    afg = records[records["iso3"] == "AFG"]
    assert len(afg) == 1
    assert int(afg.iloc[0]["population"]) > 1_000_000
    # Aggregates from the curated CSV are filtered out.
    assert (records["iso3"] == "AFE").sum() == 0


def test_parses_year_column_variant(tmp_path):
    csv_path = tmp_path / "population.csv"
    csv_path.write_text(
        "iso3,year,population,source,source_url\n"
        "ken,2025,55000000,WorldPop,https://example.org/ken\n"
        "XXX,2025,1,NotACountry,\n",
        encoding="utf-8",
    )
    records = load_population_records(csv_path)
    assert len(records) == 1
    row = records.iloc[0]
    assert row["iso3"] == "KEN"
    assert row["year"] == 2025
    assert row["population"] == 55000000
    assert row["source_url"] == "https://example.org/ken"


def test_parses_as_of_column_and_formatted_numbers(tmp_path):
    # Mirrors the curated CSV's shape: padded headers, comma-grouped numbers,
    # the year living in as_of.
    csv_path = tmp_path / "population.csv"
    csv_path.write_text(
        'iso3,country_name, population ,as_of,source\n'
        'PHL,Philippines," 117,337,368 ",2024,World Bank\n',
        encoding="utf-8",
    )
    records = load_population_records(csv_path)
    assert len(records) == 1
    row = records.iloc[0]
    assert row["iso3"] == "PHL"
    assert row["year"] == 2024
    assert row["population"] == 117337368


def test_load_into_db_is_idempotent(tmp_path):
    csv_path = tmp_path / "population.csv"
    csv_path.write_text(
        "iso3,year,population,source\n"
        "KEN,2025,55000000,WorldPop\n"
        "PHL,2024,117000000,World Bank\n",
        encoding="utf-8",
    )
    conn = duckdb.connect(str(tmp_path / "haz.duckdb"))
    try:
        assert load_population_into_db(conn, csv_path) == 2
        assert load_population_into_db(conn, csv_path) == 2  # replaces, no duplicates

        rows = conn.execute(
            "SELECT iso3, year, population, source, retrieved_at FROM haz_raw_population ORDER BY iso3"
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 2
    assert rows[0][0] == "KEN"
    assert all(row[4] is not None for row in rows)  # retrieval timestamp recorded
