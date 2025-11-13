"""Normalization helpers for EM-DAT People Affected (PA) pulls."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, MutableMapping, Sequence

import pandas as pd

from resolver.db.schema_keys import EMDAT_PA_KEY_COLUMNS
from resolver.ingestion._shared.run_io import write_json
from resolver.ingestion.utils.hazard_map import CLASSIF_TO_SHOCK
from resolver.ingestion.utils import stable_digest

LOG = logging.getLogger("resolver.ingestion.emdat.normalize")

SOURCE_ID = "emdat"

try:  # Import lazily to avoid hard dependency during tests
    from resolver.ingestion.emdat_client import (  # type: ignore
        EMDAT_NORMALIZE_DEBUG_PATH as _NORMALIZE_DEBUG_PATH,
    )
except Exception:  # pragma: no cover - defensive fallback
    _NORMALIZE_DEBUG_PATH = Path("diagnostics/ingestion/emdat/normalize_debug.json")

_FACTS_COLUMNS: Sequence[str] = (
    "iso3",
    "as_of_date",
    "ym",
    "metric",
    "value",
    "unit",
    "series_semantics",
    "semantics",
    "hazard_code",
    "hazard_label",
    "hazard_class",
    "country_name",
    "source_id",
    "publication_date",
    "publisher",
    "source_type",
    "source_url",
    "doc_title",
    "definition_text",
    "method",
    "confidence",
    "revision",
    "ingested_at",
    "event_id",
    "disno_first",
)

_DUCKDB_COLUMNS: Sequence[str] = (
    "iso3",
    "ym",
    "shock_type",
    "pa",
    "as_of_date",
    "publication_date",
    "source_id",
    "disno_first",
)

_DROPPED_SAMPLE_LIMIT = 10

if TYPE_CHECKING:
    import duckdb
    from resolver.db.duckdb_io import UpsertResult


def _empty_frame() -> pd.DataFrame:
    columns: dict[str, pd.Series] = {
        "iso3": pd.Series(dtype="string"),
        "as_of_date": pd.Series(dtype="string"),
        "ym": pd.Series(dtype="string"),
        "metric": pd.Series(dtype="string"),
        "value": pd.Series(dtype="Int64"),
        "unit": pd.Series(dtype="string"),
        "series_semantics": pd.Series(dtype="string"),
        "semantics": pd.Series(dtype="string"),
        "hazard_code": pd.Series(dtype="string"),
        "hazard_label": pd.Series(dtype="string"),
        "hazard_class": pd.Series(dtype="string"),
        "country_name": pd.Series(dtype="string"),
        "source_id": pd.Series(dtype="string"),
        "publication_date": pd.Series(dtype="string"),
        "publisher": pd.Series(dtype="string"),
        "source_type": pd.Series(dtype="string"),
        "source_url": pd.Series(dtype="string"),
        "doc_title": pd.Series(dtype="string"),
        "definition_text": pd.Series(dtype="string"),
        "method": pd.Series(dtype="string"),
        "confidence": pd.Series(dtype="string"),
        "revision": pd.Series(dtype="Int64"),
        "ingested_at": pd.Series(dtype="string"),
        "event_id": pd.Series(dtype="string"),
        "disno_first": pd.Series(dtype="string"),
        "shock_type": pd.Series(dtype="string"),
        "pa": pd.Series(dtype="Int64"),
    }
    return pd.DataFrame(columns)


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


def _parse_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.upper() == "NA":
        return None
    normalized = text.replace(",", "")
    try:
        number = float(normalized)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    try:
        return int(number)
    except (TypeError, ValueError, OverflowError):
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


def _record_drops(
    frame: pd.DataFrame,
    mask: pd.Series,
    *,
    reason: str,
    drop_counts: MutableMapping[str, int],
    dropped_sample: list[dict[str, str]],
) -> None:
    indexes = frame.index[mask]
    if not len(indexes):
        return
    drop_counts[reason] += int(len(indexes))
    for idx in indexes:
        if len(dropped_sample) >= _DROPPED_SAMPLE_LIMIT:
            break
        row = frame.loc[idx]
        dropped_sample.append(
            {
                "disno": str(row.get("disno") or ""),
                "classif_key": str(row.get("classif_key") or ""),
                "reason": reason,
            }
        )


def _write_normalize_diagnostics(payload: Mapping[str, Any]) -> None:
    try:
        write_json(_NORMALIZE_DEBUG_PATH, payload)
    except Exception:  # pragma: no cover - diagnostics best-effort
        LOG.debug("emdat.normalize.debug_write_failed", exc_info=True)


def normalize_emdat_pa(
    df_raw: pd.DataFrame | None,
    *,
    info: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Aggregate raw EM-DAT rows into Resolver's country-month PA series."""

    drop_counts: MutableMapping[str, int] = defaultdict(int)
    dropped_sample: list[dict[str, str]] = []
    fallback_counts: MutableMapping[str, int] = defaultdict(int)

    if df_raw is None or df_raw.empty:
        LOG.debug("emdat.normalize.empty_input")
        stats_payload = {
            "raw_rows": 0,
            "kept_rows": 0,
            "dropped_rows": 0,
            "drop_counts": {
                "missing_iso": 0,
                "missing_month": 0,
                "unmapped_classif": 0,
                "non_positive_value": 0,
            },
            "dropped_sample": [],
            "fallback_counts": {},
        }
        _write_normalize_diagnostics(stats_payload)
        empty = _empty_frame()
        empty.attrs["normalize_stats"] = stats_payload
        LOG.info(
            "emdat.normalize.stats|kept=0|dropped=0|missing_iso=0|missing_month=0|unmapped_classif=0|non_positive=0"
        )
        return empty

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
    missing_iso_initial = iso_series.isna() | iso_series.eq("")
    for idx in working.index[missing_iso_initial]:
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
    if missing_iso.any():
        for idx in working.index[missing_iso]:
            LOG.debug("emdat.normalize.drop_no_iso|disno=%s", working.at[idx, "disno"])
        _record_drops(
            working,
            missing_iso,
            reason="missing_iso",
            drop_counts=drop_counts,
            dropped_sample=dropped_sample,
        )
    working = working.loc[~missing_iso].copy()
    iso_series = iso_series.loc[working.index]
    working["iso3"] = iso_series.astype("string")

    year_series = pd.to_numeric(
        working.get("start_year", pd.Series(pd.NA, index=working.index)),
        errors="coerce",
    )
    month_series = pd.to_numeric(
        working.get("start_month", pd.Series(pd.NA, index=working.index)),
        errors="coerce",
    )
    end_month_series = pd.to_numeric(
        working.get("end_month", pd.Series(pd.NA, index=working.index)),
        errors="coerce",
    )
    working["start_year"] = year_series.astype("Int64")
    working["start_month"] = month_series.astype("Int64")
    working["end_month"] = end_month_series.astype("Int64")

    for column in ("start_month", "end_month"):
        series = working[column]
        valid_mask = series.between(1, 12, inclusive="both").fillna(False)
        working[column] = series.where(valid_mask, other=pd.NA)

    classif_series = working.get("classif_key")
    if classif_series is None:
        classif_series = pd.Series(pd.NA, index=working.index, dtype="string")
    classif_clean = classif_series.astype("string").str.strip().str.lower()
    working = working.assign(shock_type=classif_clean.map(CLASSIF_TO_SHOCK))

    start_missing_month = working["start_month"].isna()
    sudden_mask = working["shock_type"].isin({"flood", "tropical_cyclone"})
    fallback_mask = start_missing_month & sudden_mask & working["end_month"].notna()
    if fallback_mask.any():
        fallback_counts["used_end_month"] += int(fallback_mask.sum())
        working.loc[fallback_mask, "start_month"] = working.loc[fallback_mask, "end_month"]

    working["start_month"] = working["start_month"].where(
        working["start_month"].between(1, 12, inclusive="both").fillna(False),
        other=pd.NA,
    )

    missing_month_mask = working["start_month"].isna() | working["start_year"].isna()
    if missing_month_mask.any():
        for idx in working.index[missing_month_mask]:
            LOG.debug(
                "emdat.normalize.missing_month|disno=%s",
                working.at[idx, "disno"],
            )
        _record_drops(
            working,
            missing_month_mask,
            reason="missing_month",
            drop_counts=drop_counts,
            dropped_sample=dropped_sample,
        )
    working = working.loc[~missing_month_mask].copy()

    unmapped_mask = working["shock_type"].isna()
    if unmapped_mask.any():
        for idx in working.index[unmapped_mask]:
            LOG.debug(
                "emdat.normalize.unmapped_classif|disno=%s|classif=%s",
                working.at[idx, "disno"],
                working.at[idx, "classif_key"],
            )
        _record_drops(
            working,
            unmapped_mask,
            reason="unmapped_classif",
            drop_counts=drop_counts,
            dropped_sample=dropped_sample,
        )
    working = working.loc[~unmapped_mask].copy()
    flood_row_count = int((working["shock_type"] == "flood").sum())

    working["ym"] = (
        working["start_year"].astype(int).astype(str)
        + "-"
        + working["start_month"].astype(int).map(lambda value: f"{value:02d}")
    )

    def _resolve_pa(row: pd.Series) -> int | None:
        total = _parse_int(row.get("total_affected"))
        if total is None:
            running = 0
            observed = False
            for field in ("affected", "injured", "homeless"):
                value = _parse_int(row.get(field))
                if value is not None:
                    observed = True
                    running += value
            if not observed:
                return None
            total = running
        return total

    pa_series = pd.Series(
        pd.array(working.apply(_resolve_pa, axis=1), dtype="Int64"),
        index=working.index,
    )
    non_positive_mask = pa_series.isna() | (pa_series <= 0)
    if non_positive_mask.any():
        _record_drops(
            working,
            non_positive_mask,
            reason="non_positive_value",
            drop_counts=drop_counts,
            dropped_sample=dropped_sample,
        )
    working = working.loc[~non_positive_mask].copy()
    pa_series = pa_series.loc[working.index]
    working["total_affected"] = pa_series.astype("Int64")

    working["publication_date"] = _coerce_publication(
        working.get("last_update", pd.Series(pd.NA, index=working.index)),
        working.get("entry_date", pd.Series(pd.NA, index=working.index)),
    )

    as_of_date = _resolve_as_of(info)
    kept_rows = len(working)

    grouped = working.groupby(["iso3", "ym", "shock_type"], dropna=False, as_index=False).agg(
        pa=("total_affected", "sum"),
        publication_date=(
            "publication_date",
            lambda s: s.dropna().max() if not s.dropna().empty else pd.NA,
        ),
        disno_first=(
            "disno",
            lambda s: min((str(v) for v in s if str(v).strip()), default=""),
        ),
        country_name=(
            "country",
            lambda s: next((str(v).strip() for v in s if str(v).strip()), ""),
        ),
    )

    grouped["pa"] = grouped["pa"].round().astype("Int64")
    grouped["publication_date"] = grouped["publication_date"].astype("string")
    grouped["country_name"] = grouped["country_name"].astype("string")
    grouped["as_of_date"] = str(as_of_date)
    grouped["source_id"] = SOURCE_ID

    grouped["value"] = grouped["pa"].astype("Int64")
    grouped["metric"] = "total_affected"
    grouped["unit"] = "persons"
    grouped["series_semantics"] = "new"
    grouped["semantics"] = "new"
    grouped["hazard_code"] = grouped["shock_type"].astype("string")
    grouped["hazard_label"] = grouped["hazard_code"].str.replace("_", " ").str.title()
    grouped["hazard_class"] = grouped["hazard_label"]
    grouped["publisher"] = "CRED/EM-DAT"
    grouped["source_type"] = "api"
    grouped["source_url"] = ""
    grouped["doc_title"] = ""
    grouped["definition_text"] = "EM-DAT total affected persons (monthly aggregate)."
    grouped["method"] = "EM-DAT API monthly aggregation"
    grouped["confidence"] = ""
    grouped["revision"] = pd.Series(0, index=grouped.index, dtype="Int64")
    ingested_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    grouped["ingested_at"] = ingested_at

    digests = [
        stable_digest(
            [
                row.iso3,
                row.shock_type,
                "total_affected",
                row.ym,
                row.disno_first,
            ]
        )
        for row in grouped.itertuples(index=False)
    ]
    grouped["event_id"] = [
        f"{row.iso3}-EMDAT-{row.shock_type}-total_affected-{row.ym.replace('-', '')}-{digest}"
        for row, digest in zip(grouped.itertuples(index=False), digests, strict=False)
    ]

    string_columns = [
        "iso3",
        "as_of_date",
        "ym",
        "metric",
        "unit",
        "series_semantics",
        "semantics",
        "hazard_code",
        "hazard_label",
        "hazard_class",
        "country_name",
        "source_id",
        "publication_date",
        "publisher",
        "source_type",
        "source_url",
        "doc_title",
        "definition_text",
        "method",
        "confidence",
        "ingested_at",
        "event_id",
        "disno_first",
    ]
    for column in string_columns:
        grouped[column] = grouped[column].astype("string").fillna("")

    grouped = grouped.sort_values(["iso3", "ym", "shock_type"]).reset_index(drop=True)

    for column in _FACTS_COLUMNS:
        if column not in grouped.columns:
            grouped[column] = pd.NA
    ordered_columns = list(_FACTS_COLUMNS) + [
        column for column in grouped.columns if column not in _FACTS_COLUMNS
    ]
    grouped = grouped.reindex(columns=ordered_columns)

    drop_counts_payload: dict[str, int] = {
        "missing_iso": int(drop_counts.get("missing_iso", 0)),
        "missing_month": int(drop_counts.get("missing_month", 0)),
        "unmapped_classif": int(drop_counts.get("unmapped_classif", 0)),
        "non_positive_value": int(drop_counts.get("non_positive_value", 0)),
    }
    for reason, count in drop_counts.items():
        if reason not in drop_counts_payload:
            drop_counts_payload[reason] = int(count)

    diagnostics_payload = {
        "raw_rows": int(raw_rows),
        "kept_rows": int(kept_rows),
        "dropped_rows": int(raw_rows - kept_rows),
        "drop_counts": drop_counts_payload,
        "dropped_sample": dropped_sample,
        "fallback_counts": {k: int(v) for k, v in fallback_counts.items()},
    }

    grouped.attrs["normalize_stats"] = diagnostics_payload

    _write_normalize_diagnostics(diagnostics_payload)

    LOG.debug(
        "emdat.normalize.grouped|rows_in=%s|rows_out=%s|flood_rows=%s",
        raw_rows,
        len(grouped),
        flood_row_count,
    )

    dropped_total = sum(drop_counts.values())
    LOG.info(
        "emdat.normalize.stats|kept=%s|dropped=%s|missing_iso=%s|missing_month=%s|unmapped_classif=%s|non_positive=%s",
        kept_rows,
        dropped_total,
        drop_counts_payload.get("missing_iso", 0),
        drop_counts_payload.get("missing_month", 0),
        drop_counts_payload.get("unmapped_classif", 0),
        drop_counts_payload.get("non_positive_value", 0),
    )

    return grouped


def write_emdat_pa_to_duckdb(
    conn: "duckdb.DuckDBPyConnection",
    frame: pd.DataFrame | None,
) -> "duckdb_io.UpsertResult":
    """Upsert normalised EM-DAT PA rows into DuckDB using canonical keys."""

    from resolver.db import duckdb_io

    if frame is None or frame.empty:
        working = _empty_frame()
    else:
        working = frame.copy()

    if "pa" not in working.columns and "value" in working.columns:
        working["pa"] = working["value"]
    if "shock_type" not in working.columns and "hazard_code" in working.columns:
        working["shock_type"] = working["hazard_code"]

    ordered = working.reindex(columns=list(_DUCKDB_COLUMNS))

    return duckdb_io.upsert_dataframe(
        conn,
        "emdat_pa",
        ordered,
        keys=EMDAT_PA_KEY_COLUMNS,
    )
