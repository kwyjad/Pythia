"""Normalization helpers for EM-DAT People Affected (PA) pulls."""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import pandas as pd

from resolver.db.schema_keys import EMDAT_PA_KEY_COLUMNS
from resolver.ingestion.utils.hazard_map import CLASSIF_TO_SHOCK

LOG = logging.getLogger("resolver.ingestion.emdat.normalize")

SOURCE_ID = "emdat"

_OUTPUT_COLUMNS: Sequence[str] = (
    "iso3",
    "ym",
    "shock_type",
    "pa",
    "as_of_date",
    "publication_date",
    "source_id",
    "disno_first",
)

if TYPE_CHECKING:
    import duckdb
    from resolver.db.duckdb_io import UpsertResult


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "iso3": pd.Series(dtype="string"),
            "ym": pd.Series(dtype="string"),
            "shock_type": pd.Series(dtype="string"),
            "pa": pd.Series(dtype="Int64"),
            "as_of_date": pd.Series(dtype="string"),
            "publication_date": pd.Series(dtype="string"),
            "source_id": pd.Series(dtype="string"),
            "disno_first": pd.Series(dtype="string"),
        }
    )


def _iso_from_disno(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    parts = text.split("-")
    if len(parts) < 3:
        return None
    candidate = parts[-1].strip().upper()
    if len(candidate) == 3 and candidate.isalpha():
        return candidate
    return None


def _coerce_publication(series: pd.Series, fallback: pd.Series) -> pd.Series:
    primary = pd.to_datetime(series, errors="coerce")
    fallback_ts = pd.to_datetime(fallback, errors="coerce")
    resolved = primary.fillna(fallback_ts)
    if resolved.dtype == "datetime64[ns, UTC]":
        resolved = resolved.dt.tz_convert(None)
    try:
        resolved = resolved.dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    return resolved.dt.date.astype("string")


def _resolve_as_of(info: Mapping[str, Any] | None) -> str:
    if info and isinstance(info, Mapping):
        for key in ("timestamp", "Timestamp", "as_of", "asOf"):
            raw = info.get(key)
            if raw:
                parsed = pd.to_datetime(raw, errors="coerce")
                if pd.notna(parsed):
                    try:
                        parsed = parsed.tz_convert(None)
                    except (TypeError, AttributeError):
                        pass
                    try:
                        parsed = parsed.tz_localize(None)
                    except (TypeError, AttributeError):
                        pass
                    return parsed.date().isoformat()
    return date.today().isoformat()


def normalize_emdat_pa(
    df_raw: pd.DataFrame | None,
    *,
    info: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Aggregate raw EM-DAT rows into Resolver's country-month PA series."""

    if df_raw is None or df_raw.empty:
        LOG.debug("emdat.normalize.empty_input")
        return _empty_frame()

    raw_rows = len(df_raw)
    working = df_raw.copy()

    disno_series = working.get("disno")
    if disno_series is None:
        disno_series = pd.Series(pd.NA, index=working.index, dtype="string")
    working["disno"] = disno_series.astype("string")

    iso_series = working.get("iso")
    if iso_series is None:
        iso_series = pd.Series(pd.NA, index=working.index, dtype="string")
    iso_series = iso_series.astype("string").str.strip().str.upper()
    missing_iso = iso_series.isna() | iso_series.eq("")
    for idx in working.index[missing_iso]:
        derived = _iso_from_disno(working.at[idx, "disno"])
        if derived:
            iso_series.at[idx] = derived
            LOG.debug(
                "emdat.normalize.iso_from_disno|disno=%s|iso3=%s",
                working.at[idx, "disno"],
                derived,
            )
    iso_series = iso_series.where(iso_series.str.fullmatch(r"[A-Z]{3}"), other=pd.NA)
    missing_iso = iso_series.isna()
    dropped_iso = int(missing_iso.sum())
    if dropped_iso:
        for idx in working.index[missing_iso]:
            LOG.debug("emdat.normalize.drop_no_iso|disno=%s", working.at[idx, "disno"])
    working = working.loc[~missing_iso].copy()
    working["iso3"] = iso_series.loc[~missing_iso].astype("string")

    year_series = pd.to_numeric(
        working.get("start_year", pd.Series(pd.NA, index=working.index)),
        errors="coerce",
    )
    month_series = pd.to_numeric(
        working.get("start_month", pd.Series(pd.NA, index=working.index)),
        errors="coerce",
    )
    working["start_year"] = year_series.astype("Int64")
    working["start_month"] = month_series.astype("Int64")

    missing_month_mask = working["start_month"].isna() | working["start_year"].isna()
    dropped_missing_month = int(missing_month_mask.sum())
    if dropped_missing_month:
        for idx in working.index[missing_month_mask]:
            LOG.debug(
                "emdat.normalize.missing_month|disno=%s",
                working.at[idx, "disno"],
            )
    working = working.loc[~missing_month_mask].copy()

    working["ym"] = (
        working["start_year"].astype(int).astype(str)
        + "-"
        + working["start_month"].astype(int).map(lambda value: f"{value:02d}")
    )

    classif_series = working.get("classif_key")
    if classif_series is None:
        classif_series = pd.Series(pd.NA, index=working.index, dtype="string")
    classif = classif_series.astype("string")
    classif = classif.str.strip().str.lower()
    working["shock_type"] = classif.map(CLASSIF_TO_SHOCK)
    unmapped_mask = working["shock_type"].isna()
    dropped_unmapped = int(unmapped_mask.sum())
    if dropped_unmapped:
        for idx in working.index[unmapped_mask]:
            LOG.debug(
                "emdat.normalize.unmapped_classif|disno=%s|classif=%s",
                working.at[idx, "disno"],
                working.at[idx, "classif_key"],
            )
    working = working.loc[~unmapped_mask].copy()

    affected = pd.to_numeric(
        working.get("total_affected", pd.Series(pd.NA, index=working.index)),
        errors="coerce",
    )
    na_affected = affected.isna()
    if na_affected.any():
        LOG.debug("emdat.normalize.aff_na_to_zero|count=%d", int(na_affected.sum()))
    affected = affected.fillna(0)
    negative_mask = affected < 0
    if negative_mask.any():
        LOG.debug("emdat.normalize.aff_negative_to_zero|count=%d", int(negative_mask.sum()))
        affected = affected.where(~negative_mask, other=0)
    working["total_affected"] = affected

    working["publication_date"] = _coerce_publication(
        working.get("last_update", pd.Series(pd.NA, index=working.index)),
        working.get("entry_date", pd.Series(pd.NA, index=working.index)),
    )

    as_of_date = _resolve_as_of(info)

    grouped = working.groupby(["iso3", "ym", "shock_type"], dropna=False, as_index=False).agg(
        pa=("total_affected", "sum"),
        publication_date=("publication_date", lambda s: s.dropna().max() if not s.dropna().empty else pd.NA),
        disno_first=("disno", lambda s: min((str(v) for v in s if str(v).strip()), default="")),
    )

    grouped["pa"] = grouped["pa"].round().astype("Int64")
    grouped["publication_date"] = grouped["publication_date"].astype("string")
    grouped["as_of_date"] = as_of_date
    grouped["source_id"] = SOURCE_ID

    grouped = grouped[list(_OUTPUT_COLUMNS)].sort_values(["iso3", "ym", "shock_type"]).reset_index(drop=True)

    kept_rows = len(working)
    LOG.info(
        "emdat.normalize.summary|raw=%s|kept=%s|groups=%s|dropped_missing_month=%s|dropped_unmapped=%s|dropped_iso=%s",
        raw_rows,
        kept_rows,
        len(grouped),
        dropped_missing_month,
        dropped_unmapped,
        dropped_iso,
    )

    return grouped


def write_emdat_pa_to_duckdb(
    conn: "duckdb.DuckDBPyConnection",
    frame: pd.DataFrame | None,
) -> "duckdb_io.UpsertResult":
    """Upsert normalised EM-DAT PA rows into DuckDB using canonical keys."""

    from resolver.db import duckdb_io

    if frame is None:
        working = _empty_frame()
    else:
        working = frame

    ordered = working.reindex(columns=list(_OUTPUT_COLUMNS))

    return duckdb_io.upsert_dataframe(
        conn,
        "emdat_pa",
        ordered,
        keys=EMDAT_PA_KEY_COLUMNS,
    )
