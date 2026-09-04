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


def enrich(df: pd.DataFrame, *, today: dt.date | None = None) -> pd.DataFrame:
    """Enrich a canonical DataFrame with registry lookups and defaults.

    - Fills ``country_name`` from ``data/countries.csv`` where missing
    - Fills ``hazard_label`` and ``hazard_class`` from ``data/shocks.csv``
    - Normalises ``iso3`` to uppercase
    - Defaults ``metric`` to ``"affected"`` and ``unit`` to ``"persons"``
    - Fixes ``publication_date`` (see :func:`fix_publication_date`): a date the
      source stated is kept; a missing one is filled from ``as_of_date`` only
      when that date has passed; and nothing is ever dated after ``today``
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
    today = today or dt.date.today()
    facts["publication_date"] = facts["publication_date"].fillna("").astype(str)
    facts["as_of_date"] = facts["as_of_date"].fillna("").astype(str)

    counts = {"supplied": 0, "filled": 0, "raised_to_as_of": 0, "clamped_to_today": 0}

    def _fix_pub(row: pd.Series) -> str:
        pub, why = fix_publication_date(
            row.get("publication_date", ""), row.get("as_of_date", ""), today
        )
        counts[why] += 1
        return pub

    if len(facts):
        facts["publication_date"] = facts.apply(_fix_pub, axis=1)
        if counts["clamped_to_today"] or counts["raised_to_as_of"]:
            LOG.info(
                "[enrich] publication_date: %d supplied, %d filled from the period "
                "end, %d raised to as_of_date, %d clamped to today",
                counts["supplied"], counts["filled"],
                counts["raised_to_as_of"], counts["clamped_to_today"],
            )

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
        # Empty components produce degenerate, collision-prone IDs like
        # "AFG--2026-01" — flag them so the upstream connector gets fixed.
        for col in ("iso3", "hazard_code", "as_of_date"):
            blank = facts.loc[missing_eid, col].fillna("").astype(str).str.strip() == ""
            if blank.any():
                LOG.warning(
                    "event_id fallback: %d row(s) have empty %s; generated "
                    "IDs may collide", int(blank.sum()), col,
                )
        fallback = (
            facts.loc[missing_eid, "iso3"].fillna("UNK")
            + "-"
            + facts.loc[missing_eid, "hazard_code"].fillna("UNK")
            + "-"
            + facts.loc[missing_eid, "as_of_date"].fillna("")
        )
        facts.loc[missing_eid, "event_id"] = fallback

    return facts


def fix_publication_date(publication: Any, as_of: Any, today: dt.date) -> tuple[str, str]:
    """The publication date a row is stored with, and why.

    Returns ``(iso_date, reason)`` with reason one of ``supplied``, ``filled``,
    ``raised_to_as_of``, ``clamped_to_today``.

    Three rules, in order. A date the source stated is kept as stated. A
    missing date is filled from ``as_of_date`` when that date has passed (an
    observation cannot be published before it is made); when ``as_of_date``
    is still in the future the row describes a period that has not ended (an
    IPC projection window, the current month) and the fill is ``today``, the
    day this run first saw it. And no row is ever dated after ``today``.

    The rule this replaces raised EVERY publication date to ``as_of_date``
    and then clamped it to today. For a FEWS NET projection whose window ends
    in March 2027 that turned the reporting date the connector supplied into
    the run date, on every run, so the row said it was published today each
    time it was rewritten — and the freshness report read a projection about
    next year as the newest publication in the table.
    """

    pub = _parse_date(publication)
    as_of_date = _parse_date(as_of)
    why = "supplied"
    if pub is None:
        why = "filled"
        if as_of_date is not None and as_of_date <= today:
            pub = as_of_date
        else:
            pub = today
    elif as_of_date is not None and as_of_date <= today and pub < as_of_date:
        # An observation dated before it was observed: the source's period
        # end is the earliest it can have been published.
        pub = as_of_date
        why = "raised_to_as_of"
    if pub > today:
        pub = today
        why = "clamped_to_today"
    return pub.isoformat(), why


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
