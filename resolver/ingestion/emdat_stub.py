#!/usr/bin/env python3
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Sequence

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

_CLASSIF = [
    "nat-cli-dro-dro",  # drought
    "nat-met-sto-tro",  # tropical cyclone
    "nat-hyd-flo-riv",  # riverine flood
    "nat-hyd-flo-fla",  # flash flood
]

_SUBTYPE_LABEL = {
    "nat-cli-dro-dro": "Drought",
    "nat-met-sto-tro": "Tropical cyclone",
    "nat-hyd-flo-riv": "Riverine flood",
    "nat-hyd-flo-fla": "Flash flood",
}


def _synth_rows(
    from_year: int,
    to_year: int,
    *,
    iso_list: Sequence[str] | None,
    classif: Sequence[str] | None,
) -> pd.DataFrame:
    keys = [str(value).strip() for value in (classif or []) if str(value).strip()]
    if not keys:
        keys = list(_CLASSIF)

    iso = (iso_list[0] if iso_list else "BEL").upper()

    if from_year == to_year:
        year = max(int(from_year), int(to_year))
    else:
        year = int(from_year)

    rows_needed = max(3, len(keys))

    records: list[dict[str, object]] = []
    for i in range(rows_needed):
        key = keys[i % len(keys)]
        month = (i % 3) + 1
        disno = f"{year:04d}-{1000 + i:04d}-{iso}"
        subtype = _SUBTYPE_LABEL.get(key, "Unknown")

        if key.startswith("nat-hyd-flo"):
            hazard_type = "Flood"
        elif "dro" in key:
            hazard_type = "Drought"
        else:
            hazard_type = "Storm"

        records.append(
            {
                "disno": disno,
                "classif_key": key,
                "type": hazard_type,
                "subtype": subtype,
                "iso": iso,
                "country": "Belgium",
                "start_year": year,
                "start_month": month,
                "start_day": 1,
                "end_year": year,
                "end_month": month,
                "end_day": 2,
                "total_affected": 1000 + (10 * i),
                "entry_date": date(year, month, 1).isoformat(),
                "last_update": date(year, month, 2).isoformat(),
            }
        )

    return pd.DataFrame.from_records(records, columns=STUB_COLUMNS)


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

    iso_list: Sequence[str] | None = None
    if iso:
        if isinstance(iso, (list, tuple)):
            iso_list = [str(value).strip().upper() for value in iso if str(value).strip()]
        else:
            iso_list = [str(iso).strip().upper()]

    df = _synth_rows(
        int(from_year),
        int(to_year),
        iso_list=iso_list,
        classif=classif,
    )

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
