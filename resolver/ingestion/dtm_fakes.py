# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Helpers for writing clearly marked fake DTM staging data."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

__all__ = ["write_fake_admin0_staging"]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _filter_iso3(values: Sequence[str] | None) -> List[str]:
    if not values:
        return []
    seen: set[str] = set()
    filtered: List[str] = []
    for value in values:
        iso = str(value or "").strip().upper()
        if len(iso) != 3:
            continue
        if iso in seen:
            continue
        seen.add(iso)
        filtered.append(iso)
    return filtered


def write_fake_admin0_staging(
    out_csv: str | Path,
    out_meta: str | Path,
    *,
    window_start: str | None,
    window_end: str | None,
    iso3_whitelist: Iterable[str] | None = None,
    reason: str = "network_timeout",
    diagnostics: dict | None = None,
) -> int:
    """Write clearly marked fake admin0 rows to staging outputs."""

    static = Path(__file__).with_name("static") / "dtm_admin0_fake.csv"
    frame = pd.read_csv(static)

    whitelist = _filter_iso3(list(iso3_whitelist) if iso3_whitelist is not None else None)
    if whitelist:
        frame = frame[frame["CountryISO3"].isin(whitelist)]

    if window_start and window_end:
        window_start_ts = pd.to_datetime(window_start, errors="coerce")
        window_end_ts = pd.to_datetime(window_end, errors="coerce")
        frame["ReportingDate"] = pd.to_datetime(frame["ReportingDate"], errors="coerce")
        if pd.notna(window_start_ts) and pd.notna(window_end_ts):
            frame = frame[(frame["ReportingDate"] >= window_start_ts) & (frame["ReportingDate"] <= window_end_ts)]
    frame = frame.sort_values(["CountryISO3", "ReportingDate"], ignore_index=True)
    if pd.api.types.is_datetime64_any_dtype(frame["ReportingDate"]):
        frame["ReportingDate"] = frame["ReportingDate"].dt.strftime("%Y-%m-%d")

    out_csv = Path(out_csv)
    out_meta = Path(out_meta)
    _ensure_parent(out_csv)
    _ensure_parent(out_meta)

    frame.to_csv(out_csv, index=False)
    try:
        repo_root = Path(__file__).resolve().parents[2]
        fake_source = str(static.resolve().relative_to(repo_root))
    except Exception:
        fake_source = str(static)

    payload = {
        "fake_data": True,
        "fake_source": fake_source,
        "reason": reason,
        "schema": {"required": ["CountryISO3", "ReportingDate", "idp_count"]},
        "window": {"start": window_start, "end": window_end},
        "iso3_filter": whitelist,
        "written_rows": int(frame.shape[0]),
        "generated_at": date.today().isoformat(),
    }
    if diagnostics:
        payload["diagnostics"] = diagnostics

    out_meta.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return int(frame.shape[0])
