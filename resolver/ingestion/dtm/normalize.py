"""Normalization helpers for DTM admin0 staging."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

LOG = logging.getLogger(__name__)

ValueColumnEntry = Dict[str, Any]

IsoLookup = Union[
    Mapping[str, str],
    Callable[[pd.Series], Union[str, Tuple[Optional[str], Optional[str]], Tuple[Optional[str], Optional[str], Any]]],
]


def detect_value_column(
    df: pd.DataFrame, aliases: Iterable[str]
) -> Tuple[Optional[str], Sequence[ValueColumnEntry]]:
    """Return the first matching value column and its raw match count.

    Parameters
    ----------
    df:
        Raw DataFrame returned by the connector.
    aliases:
        Ordered sequence of candidate column names.

    Returns
    -------
    tuple
        (column_name, chosen_entries). ``chosen_entries`` is an empty
        sequence when no column matched, or a one-item list describing the
        selected column and the number of non-null numeric values found on the
        unfiltered frame.
    """

    chosen: list[ValueColumnEntry] = []
    for alias in aliases:
        if alias not in df.columns:
            continue
        numeric = pd.to_numeric(df[alias], errors="coerce")
        count = int(numeric.notna().sum())
        LOG.debug("normalize.detect_value_column candidate=%s count=%d", alias, count)
        if count > 0:
            chosen.append({"column": alias, "count": count})
            LOG.debug("normalize.detect_value_column selected=%s", alias)
            return alias, chosen
    LOG.debug("normalize.detect_value_column no column matched from %s", list(aliases))
    return None, []


def _ensure_drop_counters() -> Dict[str, int]:
    return {
        "no_iso3": 0,
        "no_value_col": 0,
        "date_parse_failed": 0,
        "date_out_of_window": 0,
        "other": 0,
    }


def _to_timestamp(value: Optional[str]) -> Optional[pd.Timestamp]:
    if not value:
        return None
    try:
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:  # pragma: no cover - defensive guard
        LOG.debug("normalize_admin0: failed to parse boundary %s", value, exc_info=True)
        return None
    if isinstance(parsed, pd.Series):
        parsed = parsed.iloc[0] if not parsed.empty else None
    if parsed is None or pd.isna(parsed):
        return None
    return parsed


def _lookup_iso3(
    row: pd.Series, lookup: IsoLookup
) -> Tuple[Optional[str], Optional[str]]:
    explicit = str(row.get("CountryISO3") or "").strip()
    if explicit:
        return explicit[:3].upper(), None

    iso: Optional[str] = None
    reason: Optional[str] = None
    if callable(lookup):
        try:
            value = lookup(row)
        except Exception:  # pragma: no cover - defensive guard
            LOG.debug("normalize_admin0: iso3 lookup callable failed", exc_info=True)
            value = None
        if isinstance(value, tuple):
            if len(value) >= 1:
                iso = value[0]
            if len(value) >= 2:
                reason = value[1]
        else:
            iso = value  # type: ignore[assignment]
    else:
        # mapping lookup by admin0 code or country name
        for key in ("admin0Pcode", "CountryName"):
            candidate = str(row.get(key) or "").strip()
            if not candidate:
                continue
            iso = lookup.get(candidate) or lookup.get(candidate.upper()) or lookup.get(candidate.lower())
            if iso:
                break
    if not iso:
        return None, reason
    return str(iso).strip().upper()[:3] or None, reason


def normalize_admin0(
    df_raw: pd.DataFrame,
    *,
    idp_aliases: Sequence[str],
    start_iso: Optional[str],
    end_iso: Optional[str],
    iso3_lookup: IsoLookup,
) -> Dict[str, Any]:
    """Normalize a raw admin0 frame and return diagnostics counters."""

    counters: Dict[str, Any] = {
        "drop_reasons": _ensure_drop_counters(),
        "chosen_value_columns": [],
    }
    if df_raw is None or df_raw.empty:
        LOG.debug("normalize_admin0: empty frame received")
        return {"df": pd.DataFrame(), "counters": counters, "zero_rows_reason": None}

    value_column, chosen_entries = detect_value_column(df_raw, idp_aliases)
    counters["chosen_value_columns"] = list(chosen_entries)
    if not value_column:
        rows = int(df_raw.shape[0]) if hasattr(df_raw, "shape") else 0
        counters["drop_reasons"]["no_value_col"] += rows or 1
        LOG.debug("normalize_admin0: no value column found; rows=%d", rows)
        return {
            "df": pd.DataFrame(columns=df_raw.columns),
            "counters": counters,
            "zero_rows_reason": "invalid_indicator",
        }

    working = df_raw.copy()
    working["idp_count"] = pd.to_numeric(working[value_column], errors="coerce")
    invalid_value = working["idp_count"].isna()
    if invalid_value.any():
        dropped = int(invalid_value.sum())
        counters["drop_reasons"]["other"] += dropped
        LOG.debug("normalize_admin0: dropping %d rows with NaN idp_count", dropped)
        working = working.loc[~invalid_value]

    non_positive = working["idp_count"] <= 0
    if non_positive.any():
        dropped = int(non_positive.sum())
        counters["drop_reasons"]["other"] += dropped
        LOG.debug("normalize_admin0: dropping %d rows with non-positive idp_count", dropped)
        working = working.loc[~non_positive]

    if working.empty:
        LOG.debug("normalize_admin0: no rows remain after value filtering")
        return {"df": working, "counters": counters, "zero_rows_reason": None}

    if "ReportingDate" in working.columns:
        parsed_dates = pd.to_datetime(working["ReportingDate"], errors="coerce", utc=True)
    else:
        parsed_dates = pd.Series([pd.NaT] * len(working), index=working.index)
    parse_failed = parsed_dates.isna()
    if parse_failed.any():
        dropped = int(parse_failed.sum())
        counters["drop_reasons"]["date_parse_failed"] += dropped
        LOG.debug("normalize_admin0: dropping %d rows with unparsable dates", dropped)
        working = working.loc[~parse_failed].copy()
        parsed_dates = parsed_dates.loc[~parse_failed]
    if working.empty:
        LOG.debug("normalize_admin0: no rows remain after date parsing")
        return {"df": working, "counters": counters, "zero_rows_reason": None}

    working = working.copy()
    working["ReportingDate"] = parsed_dates

    start_ts = _to_timestamp(start_iso)
    end_ts = _to_timestamp(end_iso)
    if start_ts is not None or end_ts is not None:
        outside_mask = pd.Series(False, index=working.index)
        if start_ts is not None:
            outside_mask |= working["ReportingDate"] < start_ts
        if end_ts is not None:
            outside_mask |= working["ReportingDate"] > end_ts
        if outside_mask.any():
            dropped = int(outside_mask.sum())
            counters["drop_reasons"]["date_out_of_window"] += dropped
            LOG.debug(
                "normalize_admin0: dropping %d rows outside window start=%s end=%s",
                dropped,
                start_ts,
                end_ts,
            )
            working = working.loc[~outside_mask]
    if working.empty:
        LOG.debug("normalize_admin0: no rows remain after window filtering")
        return {"df": working, "counters": counters, "zero_rows_reason": None}

    iso_values = []
    keep_mask = []
    for _, row in working.iterrows():
        iso, reason = _lookup_iso3(row, iso3_lookup)
        if not iso:
            counters["drop_reasons"]["no_iso3"] += 1
            if reason:
                LOG.debug("normalize_admin0: iso lookup failed (%s) for %s", reason, row.to_dict())
            else:
                LOG.debug("normalize_admin0: iso lookup failed for %s", row.to_dict())
            keep_mask.append(False)
            iso_values.append(None)
            continue
        keep_mask.append(True)
        iso_values.append(str(iso).strip().upper())
    if not any(keep_mask):
        LOG.debug("normalize_admin0: all rows dropped due to missing ISO3")
        return {"df": pd.DataFrame(columns=working.columns), "counters": counters, "zero_rows_reason": None}

    filtered = working.loc[keep_mask].copy()
    filtered["CountryISO3"] = [iso for iso, keep in zip(iso_values, keep_mask) if keep]
    filtered.reset_index(drop=True, inplace=True)

    return {"df": filtered, "counters": counters, "zero_rows_reason": None}
