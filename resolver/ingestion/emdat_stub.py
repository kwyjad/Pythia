#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

try:
    from resolver.ingestion._stub_utils import load_registries, now_dates, write_staging
except ImportError:  # pragma: no cover - legacy script invocation
    from _stub_utils import load_registries, now_dates, write_staging

OUT = Path(__file__).resolve().parents[1] / "staging" / "emdat.csv"

STUB_COLUMNS: Sequence[str] = (
    "disno",
    "classif_key",
    "type",
    "subtype",
    "iso",
    "country",
    "start_year",
    "start_month",
    "start_day",
    "end_year",
    "end_month",
    "end_day",
    "total_affected",
    "entry_date",
    "last_update",
)

STUB_ROWS: Iterable[dict[str, object]] = (
    {
        "disno": "2015-0003-ETH",
        "classif_key": "nat-cli-dro-dro",
        "type": "Drought",
        "subtype": "Meteorological",
        "iso": "ETH",
        "country": "Ethiopia",
        "start_year": 2015,
        "start_month": 6,
        "start_day": 1,
        "end_year": 2016,
        "end_month": 3,
        "end_day": 31,
        "total_affected": 10500000,
        "entry_date": "2015-06-15",
        "last_update": "2016-04-12",
    },
    {
        "disno": "2019-0007-PHL",
        "classif_key": "nat-met-sto-tro",
        "type": "Storm",
        "subtype": "Tropical cyclone",
        "iso": "PHL",
        "country": "Philippines",
        "start_year": 2019,
        "start_month": 12,
        "start_day": 3,
        "end_year": 2019,
        "end_month": 12,
        "end_day": 7,
        "total_affected": 257000,
        "entry_date": "2019-12-09",
        "last_update": "2020-01-05",
    },
    {
        "disno": "2022-0005-BGD",
        "classif_key": "nat-hyd-flo-riv",
        "type": "Flood",
        "subtype": "Riverine flood",
        "iso": "BGD",
        "country": "Bangladesh",
        "start_year": 2022,
        "start_month": 5,
        "start_day": 14,
        "end_year": 2022,
        "end_month": 6,
        "end_day": 2,
        "total_affected": 1680000,
        "entry_date": "2022-05-20",
        "last_update": "2022-06-15",
    },
    {
        "disno": "2023-0004-PER",
        "classif_key": "nat-hyd-flo-fla",
        "type": "Flood",
        "subtype": "Flash flood",
        "iso": "PER",
        "country": "Peru",
        "start_year": 2023,
        "start_month": 3,
        "start_day": 10,
        "end_year": 2023,
        "end_month": 3,
        "end_day": 12,
        "total_affected": 48000,
        "entry_date": "2023-03-13",
        "last_update": "2023-03-20",
    },
)


def fetch_raw(
    from_year: int,
    to_year: int,
    *,
    iso: Sequence[str] | None = None,
    classif: Sequence[str] | None = None,
    include_hist: bool = False,  # noqa: ARG001 - parity with live client signature
    limit: int | None = None,
) -> pd.DataFrame:
    """Return a pandas ``DataFrame`` mirroring the live EM-DAT client output."""

    df = pd.DataFrame(STUB_ROWS, columns=STUB_COLUMNS)
    df = df[(df["start_year"] >= int(from_year)) & (df["start_year"] <= int(to_year))]

    if iso:
        iso_set = {str(value).strip().upper() for value in iso if str(value).strip()}
        if iso_set:
            df = df[df["iso"].isin(iso_set)]

    if classif:
        classif_set = {str(value).strip() for value in classif if str(value).strip()}
        if classif_set:
            df = df[df["classif_key"].isin(classif_set)]

    if limit is not None and limit >= 0:
        df = df.head(int(limit))

    return df.reset_index(drop=True)[list(STUB_COLUMNS)]

def make_rows():
    countries, shocks = load_registries()
    as_of, pub, ing = now_dates()
    # EM-DAT: standardized disaster impacts (lagged). Use natural hazards (no earthquakes in scope).
    hz = shocks[shocks["hazard_code"].isin(["FL", "TC", "HW", "DR"])]
    rows = []
    for _, c in countries.head(2).iterrows():
        for _, h in hz.head(1).iterrows():
            event_id = f"{c.iso3}-{h.hazard_code}-emdat-stub-r1"
            rows.append([
                event_id, c.country_name, c.iso3,
                h.hazard_code, h.hazard_label, h.hazard_class,
                "affected", "56000", "persons",
                as_of, pub,
                "EM-DAT", "agency", "https://example.org/emdat", f"{h.hazard_label} Event Record",
                "People affected per standardized EM-DAT disaster record.",
                "api", "med", 1, ing
            ])
    return rows

if __name__ == "__main__":
    print(write_staging(make_rows(), OUT, series_semantics="stock"))
