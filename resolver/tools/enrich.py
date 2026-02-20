# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Enrichment helpers for canonical connector output.

Fills in registry-backed fields (country names, hazard labels/classes)
and normalises dates, defaults, and identifiers before the DataFrame
is handed to the precedence engine.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

LOG = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_COUNTRIES_PATH = _ROOT / "data" / "countries.csv"
_SHOCKS_PATH = _ROOT / "data" / "shocks.csv"


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        LOG.warning("registry file not found: %s", path)
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str).fillna("")


def derive_ym(df: pd.DataFrame) -> pd.DataFrame:
    """Add or fill the ``ym`` column from ``as_of_date``.

    ``ym`` is the YYYY-MM month key used throughout the Resolver.
    It is derived from ``as_of_date`` (e.g. ``2025-09-30`` → ``2025-09``).
    """
    df = df.copy()
    if "ym" not in df.columns:
        df["ym"] = ""
    df["ym"] = df["ym"].fillna("").astype(str)
    mask = df["ym"].str.strip() == ""
    if mask.any() and "as_of_date" in df.columns:
        df.loc[mask, "ym"] = df.loc[mask, "as_of_date"].astype(str).str.slice(0, 7)
    return df


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich a canonical DataFrame with registry lookups and defaults.

    - Fills ``country_name`` from ``data/countries.csv`` where missing
    - Fills ``hazard_label`` and ``hazard_class`` from ``data/shocks.csv``
    - Normalises ``iso3`` to uppercase
    - Defaults ``metric`` to ``"affected"`` and ``unit`` to ``"persons"``
    - Fixes ``publication_date`` (must be >= as_of_date, <= today)
    - Generates ``event_id`` for rows that lack one
    """
    if df is None or df.empty:
        return df

    facts = df.copy()

    # --- ISO3 normalisation ---
    facts["iso3"] = facts["iso3"].fillna("").astype(str).str.strip().str.upper()
    facts["hazard_code"] = facts["hazard_code"].fillna("").astype(str).str.strip().str.upper()

    # --- Hazard registry enrichment ---
    shocks = _load_csv(_SHOCKS_PATH)
    if not shocks.empty:
        shocks["hazard_code"] = shocks["hazard_code"].fillna("").astype(str).str.upper()
        registry = shocks[["hazard_code", "hazard_label", "hazard_class"]].copy()
        registry.columns = ["hazard_code", "_reg_label", "_reg_class"]
        facts = facts.merge(registry, on="hazard_code", how="left")
        for col, reg_col in [("hazard_label", "_reg_label"), ("hazard_class", "_reg_class")]:
            reg_vals = facts.pop(reg_col).fillna("").astype(str)
            if col in facts.columns:
                current = facts[col].fillna("").astype(str)
                empty = current.str.strip() == ""
                facts.loc[empty, col] = reg_vals[empty]
            else:
                facts[col] = reg_vals

    # --- Country registry enrichment ---
    countries = _load_csv(_COUNTRIES_PATH)
    if not countries.empty:
        countries["iso3"] = countries["iso3"].fillna("").astype(str).str.upper()
        reg_country = countries[["iso3", "country_name"]].rename(
            columns={"country_name": "_reg_country"}
        )
        facts = facts.merge(reg_country, on="iso3", how="left")
        reg_vals = facts.pop("_reg_country").fillna("").astype(str)
        if "country_name" in facts.columns:
            current = facts["country_name"].fillna("").astype(str)
            empty = current.str.strip() == ""
            facts.loc[empty, "country_name"] = reg_vals[empty]
        else:
            facts["country_name"] = reg_vals

    # --- Metric and unit defaults ---
    facts["metric"] = facts["metric"].fillna("").astype(str).str.strip()
    empty_metric = facts["metric"] == ""
    if empty_metric.any():
        facts.loc[empty_metric, "metric"] = "affected"

    facts["unit"] = facts["unit"].fillna("").astype(str).str.strip()
    empty_unit = facts["unit"] == ""
    if empty_unit.any():
        facts.loc[empty_unit, "unit"] = "persons"

    # --- Publication date fix ---
    today = dt.date.today()
    facts["publication_date"] = facts["publication_date"].fillna("").astype(str)
    facts["as_of_date"] = facts["as_of_date"].fillna("").astype(str)

    def _fix_pub(row: pd.Series) -> str:
        pub = _parse_date(row.get("publication_date", ""))
        as_of = _parse_date(row.get("as_of_date", ""))
        if pub is None:
            pub = as_of or today
        if as_of and pub < as_of:
            pub = as_of
        if pub > today:
            pub = today
        return pub.isoformat()

    if len(facts):
        facts["publication_date"] = facts.apply(_fix_pub, axis=1)

    # --- Revision default ---
    if "revision" in facts.columns:
        rev = facts["revision"].fillna("").astype(str)
        empty_rev = rev.str.strip() == ""
        if empty_rev.any():
            facts.loc[empty_rev, "revision"] = "1"
    else:
        facts["revision"] = "1"

    # --- Ingested-at default ---
    facts["ingested_at"] = facts["ingested_at"].fillna("").astype(str)
    empty_ing = facts["ingested_at"].str.strip() == ""
    if empty_ing.any():
        facts.loc[empty_ing, "ingested_at"] = today.isoformat()

    # --- Event ID fallback ---
    facts["event_id"] = facts["event_id"].fillna("").astype(str)
    missing_eid = facts["event_id"].str.strip() == ""
    if missing_eid.any():
        fallback = (
            facts.loc[missing_eid, "iso3"].fillna("UNK")
            + "-"
            + facts.loc[missing_eid, "hazard_code"].fillna("UNK")
            + "-"
            + facts.loc[missing_eid, "as_of_date"].fillna("")
        )
        facts.loc[missing_eid, "event_id"] = fallback

    return facts


def _parse_date(text: Any) -> Optional[dt.date]:
    """Best-effort parse of a date string."""
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    try:
        if len(s) == 10:
            return dt.date.fromisoformat(s)
    except Exception:
        pass
    if len(s) == 7 and s[4:5] == "-":
        try:
            year, month = int(s[:4]), int(s[5:7])
            if month == 12:
                return dt.date(year, 12, 31)
            return dt.date(year, month + 1, 1) - dt.timedelta(days=1)
        except Exception:
            return None
    return None
