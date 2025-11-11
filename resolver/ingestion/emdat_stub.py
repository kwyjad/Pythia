#!/usr/bin/env python3
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Sequence

import pandas as pd

try:
    from resolver.ingestion._stub_utils import load_registries, now_dates, write_staging
except ImportError:  # pragma: no cover - legacy script invocation
    from _stub_utils import load_registries, now_dates, write_staging  # type: ignore[assignment]

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

_ISO_COUNTRY = {
    "BGD": "Bangladesh",
    "BEL": "Belgium",
    "KEN": "Kenya",
    "PHL": "Philippines",
}

_ROW_HINTS = {
    "nat-cli-dro-dro": {
        "base_year": 2021,
        "month": 1,
        "event": 1,
        "total_affected": 52000,
        "iso": "BEL",
        "country": "Belgium",
        "entry_day": 5,
        "end_day": 28,
        "last_update_day": 20,
    },
    "nat-met-sto-tro": {
        "base_year": 2021,
        "month": 9,
        "event": 12,
        "total_affected": 145000,
        "iso": "BEL",
        "country": "Belgium",
        "entry_day": 10,
        "end_day": 25,
        "last_update_day": 18,
    },
    "nat-hyd-flo-riv": {
        "base_year": 2022,
        "month": 5,
        "event": 5,
        "total_affected": 1680000,
        "iso": "BGD",
        "entry_day": 18,
        "end_month": 6,
        "end_day": 5,
        "last_update_month": 6,
        "last_update_day": 15,
    },
    "nat-hyd-flo-fla": {
        "base_year": 2022,
        "month": 6,
        "event": 11,
        "total_affected": 18000,
        "iso": "BGD",
        "entry_day": 22,
        "end_day": 30,
        "last_update_day": 30,
    },
}


def _pick_year(hints: dict[str, int], from_year: int, to_year: int) -> int:
    base_year = int(hints.get("base_year", from_year))
    if base_year < from_year:
        return from_year
    if base_year > to_year:
        return to_year
    return base_year


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

    rows_needed = max(3, len(keys))
    iso_cycle: Sequence[str] | None = None
    if iso_list:
        iso_cycle = [value.strip().upper() for value in iso_list if str(value).strip()]
        if not iso_cycle:
            iso_cycle = None

    records: list[dict[str, object]] = []
    for i in range(rows_needed):
        key = keys[i % len(keys)]
        hints = dict(_ROW_HINTS.get(key, {}))

        year = _pick_year(hints, from_year, to_year)
        month = int(hints.get("month", (i % 12) + 1))
        month = max(1, min(12, month))

        iso_hint = str(hints.get("iso") or "BEL").upper()
        if iso_cycle:
            iso_value = iso_cycle[i % len(iso_cycle)]
        else:
            iso_value = iso_hint
        country = hints.get("country") or _ISO_COUNTRY.get(iso_value, iso_value)

        subtype = _SUBTYPE_LABEL.get(key, "Unknown")
        if key.startswith("nat-hyd-flo"):
            hazard_type = "Flood"
        elif "dro" in key:
            hazard_type = "Drought"
        else:
            hazard_type = "Storm"

        start_day = int(hints.get("entry_day", 1))
        end_month = int(hints.get("end_month", month))
        end_month = max(1, min(12, end_month))
        end_day = int(hints.get("end_day", max(start_day + 1, 2)))
        last_update_month = int(hints.get("last_update_month", end_month))
        last_update_month = max(1, min(12, last_update_month))
        last_update_day = int(hints.get("last_update_day", end_day))

        event_base = int(hints.get("event", 1000 + i))
        event_offset = i // max(1, len(keys))
        disno = f"{year:04d}-{event_base + event_offset:04d}-{iso_value}"

        entry_date = date(year, month, min(start_day, 28)).isoformat()
        end_year = year
        if end_month < month:
            # Ensure end month never precedes start month in year arithmetic.
            end_year = year + 1

        update_year = year
        if last_update_month < month:
            update_year = year + 1
        last_update = date(update_year, last_update_month, min(last_update_day, 28)).isoformat()

        total = hints.get("total_affected")
        if total is None:
            total = 1000 + (10 * i)

        records.append(
            {
                "disno": disno,
                "classif_key": key,
                "type": hazard_type,
                "subtype": subtype,
                "iso": iso_value,
                "country": country,
                "start_year": year,
                "start_month": month,
                "start_day": start_day,
                "end_year": end_year,
                "end_month": end_month,
                "end_day": end_day,
                "total_affected": int(total),
                "entry_date": entry_date,
                "last_update": last_update,
            }
        )

    frame = pd.DataFrame.from_records(records, columns=STUB_COLUMNS)

    if iso_cycle:
        frame = frame[frame["iso"].isin(iso_cycle)]
    frame = frame[(frame["start_year"] >= from_year) & (frame["start_year"] <= to_year)]

    return frame.reset_index(drop=True)


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
                event_id,
                c.country_name,
                c.iso3,
                h.hazard_code,
                h.hazard_label,
                h.hazard_class,
                "affected",
                "56000",
                "persons",
                as_of,
                pub,
                "EM-DAT",
                "agency",
                "https://example.org/emdat",
                f"{h.hazard_label} Event Record",
                "People affected per standardized EM-DAT disaster record.",
                "api",
                "med",
                1,
                ing,
            ])
    return rows


if __name__ == "__main__":
    print(write_staging(make_rows(), OUT, series_semantics="stock"))
