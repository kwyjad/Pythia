# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations
from resolver.ingestion import emdat_stub
from resolver.ingestion.emdat_normalize import normalize_emdat_pa
from resolver.ingestion.utils.hazard_map import CLASSIF_TO_SHOCK

import pandas as pd


def _derive_iso(row: dict[str, object]) -> str | None:
    iso = str(row.get("iso") or "").strip().upper()
    if iso:
        return iso
    disno = str(row.get("disno") or "").strip()
    if not disno:
        return None
    parts = disno.split("-")
    if len(parts) < 3:
        return None
    candidate = parts[-1].strip().upper()
    return candidate or None


def _parse_int(value: object) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text.replace(",", "")))
    except (TypeError, ValueError):
        return None


def test_emdat_normalize_pa_groups_and_sums() -> None:
    raw = emdat_stub.fetch_raw(2015, 2023)

    extra_flash = raw.loc[raw["classif_key"] == "nat-hyd-flo-fla"].iloc[0].to_dict()
    extra_flash.update(
        {
            "disno": "2022-9999-BGD",
            "iso": "",
            "country": "Bangladesh",
            "classif_key": "nat-hyd-flo-fla",
            "start_year": 2022,
            "start_month": 5,
            "total_affected": 20000,
            "entry_date": "2022-05-22",
            "last_update": "2022-05-30",
        }
    )

    flood_end_fallback = extra_flash.copy()
    flood_end_fallback.update(
        {
            "disno": "2022-9998-BGD",
            "start_month": pd.NA,
            "end_month": 6,
            "total_affected": 1234,
            "entry_date": "2022-06-01",
            "last_update": "2022-06-20",
        }
    )

    flood_components = extra_flash.copy()
    flood_components.update(
        {
            "disno": "2022-9997-BGD",
            "start_month": 7,
            "end_month": 7,
            "total_affected": pd.NA,
            "affected": "1,000",
            "injured": "200",
            "homeless": 300,
            "entry_date": "2022-07-05",
            "last_update": "2022-07-18",
        }
    )

    drought_missing_month = raw.iloc[0].to_dict()
    drought_missing_month.update(
        {
            "disno": "2020-0001-KEN",
            "iso": "KEN",
            "classif_key": "nat-cli-dro-dro",
            "start_month": pd.NA,
            "end_month": pd.NA,
            "total_affected": 5000,
            "entry_date": "2020-02-15",
            "last_update": "2020-03-01",
        }
    )

    augmented = pd.concat(
        [
            raw,
            pd.DataFrame(
                [extra_flash, flood_end_fallback, flood_components, drought_missing_month]
            ),
        ],
        ignore_index=True,
    )

    normalized = normalize_emdat_pa(augmented, info={"timestamp": "2024-01-15T00:00:00Z"})

    expected_totals: dict[tuple[str, str, str], int] = {}
    for record in augmented.to_dict(orient="records"):
        iso3 = _derive_iso(record)
        if not iso3:
            continue
        classif_key = str(record.get("classif_key") or "").strip().lower()
        shock = CLASSIF_TO_SHOCK.get(classif_key)
        if not shock:
            continue
        start_year = pd.to_numeric(record.get("start_year"), errors="coerce")
        if pd.isna(start_year):
            continue
        start_month = pd.to_numeric(record.get("start_month"), errors="coerce")
        end_year = pd.to_numeric(record.get("end_year"), errors="coerce")
        end_month = pd.to_numeric(record.get("end_month"), errors="coerce")

        if pd.isna(start_month) and shock in {"flood", "tropical_cyclone"}:
            start_month = end_month

        if pd.isna(start_month):
            continue

        start_year_int = int(start_year)
        start_month_int = int(start_month)
        end_year_int = start_year_int if pd.isna(end_year) else int(end_year)
        end_month_int = start_month_int if pd.isna(end_month) else int(end_month)
        if (end_year_int, end_month_int) < (start_year_int, start_month_int):
            end_year_int, end_month_int = start_year_int, start_month_int

        value = _parse_int(record.get("total_affected"))
        if value is None:
            value = 0
            for column in ("affected", "injured", "homeless"):
                value += _parse_int(record.get(column)) or 0
        if value <= 0:
            continue

        y, m = start_year_int, start_month_int
        while (y < end_year_int) or (y == end_year_int and m <= end_month_int):
            ym = f"{y:04d}-{m:02d}"
            key = (iso3, ym, shock)
            expected_totals[key] = expected_totals.get(key, 0) + int(value)
            if m == 12:
                y += 1
                m = 1
            else:
                m += 1

    actual_totals = {
        (row.iso3, row.ym, row.shock_type): int(row.pa)
        for row in normalized.itertuples(index=False)
    }

    assert actual_totals == expected_totals

    assert set(normalized["as_of_date"]) == {"2024-01-15"}
    assert set(normalized["source_id"]) == {"emdat"}
    assert set(normalized["metric"]) == {"total_affected"}
    assert set(normalized["series_semantics"]) == {"new"}
    assert set(normalized["semantics"]) == {"new"}
    assert set(normalized["unit"]) == {"persons"}
    assert normalized["value"].equals(normalized["pa"])
    assert normalized["hazard_code"].tolist() == normalized["shock_type"].tolist()
    assert "KEN" not in set(normalized["iso3"])

    bgd_row = normalized[(normalized["iso3"] == "BGD") & (normalized["ym"] == "2022-05")].iloc[0]
    assert bgd_row["pa"] == 1680000 + 20000
    assert bgd_row["publication_date"] == "2022-06-15"
    assert bgd_row["disno_first"] == "2022-0005-BGD"

    june_key = ("BGD", "2022-06", "flood")
    if june_key in expected_totals:
        june_row = normalized[
            (normalized["iso3"] == "BGD")
            & (normalized["ym"] == "2022-06")
            & (normalized["shock_type"] == "flood")
        ].iloc[0]
        assert int(june_row["pa"]) == expected_totals[june_key]

    july_row = normalized[
        (normalized["iso3"] == "BGD")
        & (normalized["ym"] == "2022-07")
        & (normalized["shock_type"] == "flood")
    ].iloc[0]
    assert int(july_row["pa"]) == 1500

    assert set(normalized["shock_type"]) == {"drought", "tropical_cyclone", "flood"}

    stats = normalized.attrs.get("normalize_stats")
    assert stats is not None
    assert stats["kept_rows"] > 0
    assert stats["drop_counts"]["missing_month"] >= 1
    assert stats["fallback_counts"]["used_end_month"] >= 1


def test_emdat_normalize_sticky_multi_month() -> None:
    df = pd.DataFrame(
        [
            {
                "disno": "2024-0001-ETH",
                "classif_key": "nat-hyd-flo-riv",
                "iso": "ETH",
                "country": "Ethiopia",
                "start_year": 2024,
                "start_month": 1,
                "end_year": 2024,
                "end_month": 2,
                "total_affected": 100000,
                "entry_date": "2024-03-01",
                "last_update": "2024-03-10",
            }
        ]
    )

    normalized = normalize_emdat_pa(df, info={"timestamp": "2024-03-15T00:00:00Z"})

    assert set(normalized["ym"]) == {"2024-01", "2024-02"}
    assert all(normalized["pa"] == 100000)


def test_emdat_normalize_flood_end_month_only() -> None:
    df = pd.DataFrame(
        [
            {
                "disno": "2024-0002-ETH",
                "classif_key": "nat-hyd-flo-fla",
                "iso": "ETH",
                "country": "Ethiopia",
                "start_year": 2024,
                "start_month": pd.NA,
                "end_year": 2024,
                "end_month": 5,
                "total_affected": 50000,
                "entry_date": "2024-06-01",
                "last_update": "2024-06-05",
            }
        ]
    )

    normalized = normalize_emdat_pa(df, info={"timestamp": "2024-06-10T00:00:00Z"})

    assert len(normalized) == 1
    row = normalized.iloc[0]
    assert row["ym"] == "2024-05"
    assert int(row["pa"]) == 50000


def test_emdat_normalize_single_month_no_end() -> None:
    df = pd.DataFrame(
        [
            {
                "disno": "2024-0003-ETH",
                "classif_key": "nat-hyd-flo-riv",
                "iso": "ETH",
                "country": "Ethiopia",
                "start_year": 2024,
                "start_month": 7,
                "end_year": pd.NA,
                "end_month": pd.NA,
                "total_affected": 123,
                "entry_date": "2024-07-15",
                "last_update": "2024-07-16",
            }
        ]
    )

    normalized = normalize_emdat_pa(df, info={"timestamp": "2024-07-20T00:00:00Z"})

    assert len(normalized) == 1
    row = normalized.iloc[0]
    assert row["ym"] == "2024-07"
    assert int(row["pa"]) == 123


def test_emdat_normalize_diagnostics_includes_expanded_rows() -> None:
    df = pd.DataFrame(
        [
            {
                "disno": "2024-0004-ETH",
                "classif_key": "nat-hyd-flo-riv",
                "iso": "ETH",
                "country": "Ethiopia",
                "start_year": 2024,
                "start_month": 1,
                "end_year": 2024,
                "end_month": 2,
                "total_affected": 10,
                "entry_date": "2024-02-01",
                "last_update": "2024-02-02",
            },
            {
                "disno": "2024-0005-ETH",
                "classif_key": "nat-met-sto-tro",
                "iso": "ETH",
                "country": "Ethiopia",
                "start_year": 2024,
                "start_month": 3,
                "end_year": pd.NA,
                "end_month": pd.NA,
                "total_affected": 5,
                "entry_date": "2024-03-01",
                "last_update": "2024-03-02",
            },
        ]
    )

    normalized = normalize_emdat_pa(df, info={"timestamp": "2024-03-10T00:00:00Z"})

    stats = normalized.attrs.get("normalize_stats")
    assert stats is not None
    for key in (
        "raw_rows",
        "kept_rows",
        "dropped_rows",
        "drop_counts",
        "dropped_sample",
        "fallback_counts",
        "expanded_rows",
        "month_buckets",
    ):
        assert key in stats

    assert stats["expanded_rows"] >= stats["kept_rows"]
    assert stats["month_buckets"] == len(normalized)
