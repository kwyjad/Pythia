from __future__ import annotations

import pandas as pd

from resolver.ingestion import emdat_stub
from resolver.ingestion.emdat_normalize import normalize_emdat_pa
from resolver.ingestion.utils.hazard_map import CLASSIF_TO_SHOCK


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

    missing_month = raw.iloc[0].to_dict()
    missing_month.update(
        {
            "disno": "2020-0001-KEN",
            "iso": "KEN",
            "start_month": pd.NA,
            "total_affected": 5000,
            "entry_date": "2020-02-15",
            "last_update": "2020-03-01",
        }
    )

    augmented = pd.concat(
        [raw, pd.DataFrame([extra_flash, missing_month])],
        ignore_index=True,
    )

    normalized = normalize_emdat_pa(augmented, info={"timestamp": "2024-01-15T00:00:00Z"})

    expected_totals: dict[tuple[str, str, str], int] = {}
    for record in augmented.to_dict(orient="records"):
        iso3 = _derive_iso(record)
        if not iso3:
            continue
        month = record.get("start_month")
        year = record.get("start_year")
        if pd.isna(month) or pd.isna(year):
            continue
        classif_key = str(record.get("classif_key") or "").strip().lower()
        shock = CLASSIF_TO_SHOCK.get(classif_key)
        if not shock:
            continue
        ym = f"{int(year):04d}-{int(month):02d}"
        value = pd.to_numeric(record.get("total_affected"), errors="coerce")
        if pd.isna(value) or value < 0:
            continue
        key = (iso3, ym, shock)
        expected_totals[key] = expected_totals.get(key, 0) + int(value)

    actual_totals = {
        (row.iso3, row.ym, row.shock_type): int(row.pa)
        for row in normalized.itertuples(index=False)
    }

    assert actual_totals == expected_totals

    assert set(normalized["as_of_date"]) == {"2024-01-15"}
    assert set(normalized["source_id"]) == {"emdat"}
    assert "KEN" not in set(normalized["iso3"])

    bgd_row = normalized[(normalized["iso3"] == "BGD") & (normalized["ym"] == "2022-05")].iloc[0]
    assert bgd_row["pa"] == 1680000 + 20000
    assert bgd_row["publication_date"] == "2022-06-15"
    assert bgd_row["disno_first"] == "2022-0005-BGD"

    assert set(normalized["shock_type"]) == {"drought", "tropical_cyclone", "flood"}
