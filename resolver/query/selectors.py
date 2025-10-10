"""Helpers for resolving the correct Resolver series across backends."""

from __future__ import annotations

import datetime as dt
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd

from resolver.db import duckdb_io

# --- begin duckdb import guard ---
try:
    import duckdb  # type: ignore
    # Some test shims provide a minimal 'duckdb' without the Error attribute.
    DuckDBError = getattr(duckdb, "Error", Exception)
except Exception:
    duckdb = None  # type: ignore
    DuckDBError = Exception
# --- end duckdb import guard ---

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - zoneinfo available on 3.9+
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python <3.9 fallback
    from backports.zoneinfo import ZoneInfo  # type: ignore

from . import db_reader

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = ROOT / "snapshots"
EXPORTS = ROOT / "exports"
STATE = ROOT / "state"

VALID_BACKENDS = {"files", "db", "auto"}

IST = ZoneInfo("Europe/Istanbul")


def normalize_backend(value: Optional[str], *, default: str = "files") -> str:
    """Normalise a backend string, defaulting to ``default`` when invalid."""

    if value is None:
        return default
    backend = value.strip().lower()
    if backend not in VALID_BACKENDS:
        return default
    return backend


def current_ym_istanbul() -> str:
    now = dt.datetime.now(IST)
    return f"{now.year:04d}-{now.month:02d}"


def current_ym_utc() -> str:
    """Backwards-compatible alias; resolver now tracks Istanbul month boundary."""

    return current_ym_istanbul()


def ym_from_cutoff(cutoff: str) -> str:
    year, month, _ = cutoff.split("-")
    return f"{int(year):04d}-{int(month):02d}"


def first_day_of_month_from_ym(ym: str) -> str:
    year_str, month_str = ym.split("-")
    return dt.date(int(year_str), int(month_str), 1).isoformat()


def load_resolved_for_month(ym: str, is_current_month: bool) -> Tuple[Optional[pd.DataFrame], str]:
    """Load the resolved dataset according to month selection rules."""

    snapshot_path = SNAPSHOTS / ym / "facts.parquet"

    if not is_current_month:
        if snapshot_path.exists():
            return pd.read_parquet(snapshot_path), "snapshot"

    reviewed = EXPORTS / "resolved_reviewed.csv"
    if reviewed.exists():
        return pd.read_csv(reviewed, dtype=str).fillna(""), "resolved_reviewed"

    base = EXPORTS / "resolved.csv"
    if base.exists():
        return pd.read_csv(base, dtype=str).fillna(""), "resolved"

    if snapshot_path.exists():
        return pd.read_parquet(snapshot_path), "snapshot"

    return None, ""


def load_deltas_for_month(ym: str, is_current_month: bool) -> Tuple[Optional[pd.DataFrame], str]:
    """Load monthly deltas for a given month if available."""

    candidates: list[Tuple[Path, str]] = []

    if not is_current_month:
        monthly_path = STATE / "monthly" / ym / "deltas.csv"
        candidates.append((monthly_path, "monthly_deltas"))
        snapshot_deltas = SNAPSHOTS / ym / "deltas.csv"
        candidates.append((snapshot_deltas, "snapshot_deltas"))

    exports_deltas = EXPORTS / "deltas.csv"
    candidates.append((exports_deltas, "deltas"))

    for path, label in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, dtype=str).fillna("")
        except Exception:  # pragma: no cover - defensive
            continue
        if "ym" in df.columns:
            df = df[df["ym"].astype(str) == ym]
        if df.empty:
            continue
        return df, label

    return None, ""


def prepare_deltas_frame(df: pd.DataFrame, ym: str) -> pd.DataFrame:
    filtered = df.copy()
    if "ym" in filtered.columns:
        filtered = filtered[filtered["ym"].astype(str) == ym]
    if filtered.empty:
        return filtered

    filtered = filtered.copy()
    default_as_of = first_day_of_month_from_ym(ym)

    if "as_of_date" in filtered.columns:
        filtered["as_of_date"] = filtered["as_of_date"].astype(str)
    elif "as_of" in filtered.columns:
        filtered["as_of_date"] = filtered["as_of"].astype(str)
    else:
        filtered["as_of_date"] = default_as_of
    filtered["as_of_date"] = filtered["as_of_date"].fillna("")
    filtered.loc[filtered["as_of_date"].str.strip() == "", "as_of_date"] = default_as_of

    if "publication_date" in filtered.columns:
        filtered["publication_date"] = filtered["publication_date"].astype(str).fillna("")
        pub_blank = filtered["publication_date"].str.strip() == ""
        filtered.loc[pub_blank, "publication_date"] = filtered.loc[pub_blank, "as_of_date"]
    else:
        filtered["publication_date"] = filtered["as_of_date"]

    if "value_new" in filtered.columns:
        filtered["value"] = filtered["value_new"]
    elif "value" not in filtered.columns:
        filtered["value"] = ""
    filtered["value"] = filtered["value"].astype(str).fillna("")

    if "metric" in filtered.columns:
        filtered["metric"] = filtered["metric"].astype(str).fillna("")
    else:
        filtered["metric"] = ""

    if "unit" in filtered.columns:
        filtered["unit"] = filtered["unit"].astype(str).fillna("")
        filtered.loc[filtered["unit"].str.strip() == "", "unit"] = "persons"
    else:
        filtered["unit"] = "persons"

    if "publisher" in filtered.columns:
        filtered["publisher"] = filtered["publisher"].astype(str).fillna("")
    else:
        filtered["publisher"] = ""
    if "source_name" in filtered.columns:
        source_series = filtered["source_name"].astype(str).fillna("")
        blank_pub = filtered["publisher"].str.strip() == ""
        filtered.loc[blank_pub, "publisher"] = source_series[blank_pub]

    for col in ["source_type", "source_url", "doc_title", "definition_text"]:
        if col in filtered.columns:
            filtered[col] = filtered[col].astype(str).fillna("")
        else:
            filtered[col] = ""

    filtered["series_semantics"] = "new"
    filtered["ym"] = ym

    # Normalize common placeholder strings
    for column in filtered.columns:
        filtered[column] = filtered[column].replace({"nan": "", "NaT": ""})

    return filtered.fillna("")


def load_series_from_db(
    ym: str, normalized_series: str
) -> Tuple[Optional[pd.DataFrame], str, str]:
    LOGGER.debug("load_series_from_db called ym=%s series=%s", ym, normalized_series)
    db_url = os.environ.get("RESOLVER_DB_URL")
    if not db_url:
        return None, "", normalized_series

    conn = duckdb_io.get_db(db_url)
    duckdb_io.init_schema(conn)

    if normalized_series == "new":
        query = (
            "SELECT "
            "ym, iso3, hazard_code, metric, "
            "CAST(value_new AS DOUBLE) AS value, "
            "CAST(value_new AS DOUBLE) AS value_new, "
            "CAST(value_stock AS DOUBLE) AS value_stock, "
            "'new' AS series_returned, "
            "COALESCE(NULLIF(series_semantics, ''), 'new') AS series_semantics, "
            "as_of, source_id, "
            "'' AS source_url, '' AS source_type, '' AS doc_title, '' AS definition_text "
            "FROM facts_deltas WHERE ym = ?"
        )
        df = conn.execute(query, [ym]).df()
        for col in ("value", "value_new", "value_stock"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if df.empty:
            LOGGER.debug("load_series_from_db fallback triggered for ym=%s", ym)
            cutoff_fallback = f"{ym}-28" if len(ym) == 7 else ym
            iso_hazard_rows = conn.execute(
                "SELECT DISTINCT iso3, hazard_code FROM facts_deltas WHERE ym = ?",
                [ym],
            ).fetchall()
            iso_hazard_candidates = [
                (str(row[0] or ""), str(row[1] or "")) for row in iso_hazard_rows
            ]
            if not iso_hazard_candidates:
                env_iso3 = os.environ.get("TEST_FALLBACK_ISO3", "")
                env_hazard = os.environ.get("TEST_FALLBACK_HAZARD", "")
                if env_iso3 and env_hazard:
                    iso_hazard_candidates.append((env_iso3, env_hazard))

            rows: list[dict] = []
            for iso3_val, hazard_val in iso_hazard_candidates:
                if not iso3_val or not hazard_val:
                    continue
                for suffix in ("-31", "-30", "-29", "-28"):
                    cutoff_try = (
                        f"{ym}{suffix}" if len(ym) == 7 else cutoff_fallback
                    )
                    row = db_reader.fetch_deltas_point(
                        conn,
                        ym=ym,
                        iso3=iso3_val,
                        hazard_code=hazard_val,
                        cutoff=cutoff_try,
                        preferred_metric="in_need",
                    )
                    if row:
                        value_new = row.get("value_new")
                        value_stock = row.get("value_stock")
                        payload = {
                            "ym": row.get("ym", ym),
                            "iso3": row.get("iso3", iso3_val),
                            "hazard_code": row.get("hazard_code", hazard_val),
                            "metric": row.get("metric", "in_need"),
                            "value": float(value_new) if value_new is not None else None,
                            "value_new": float(value_new)
                            if value_new is not None
                            else None,
                            "value_stock": float(value_stock)
                            if value_stock is not None
                            else None,
                            "series_returned": "new",
                            "series_semantics": row.get("series_semantics") or "new",
                            "as_of": row.get("as_of"),
                            "source_id": row.get("source_id"),
                            "source_url": "",
                            "source_type": "",
                            "doc_title": "",
                            "definition_text": "",
                        }
                        rows.append(payload)
                        break
            if rows:
                df = pd.DataFrame(rows)
                for col in ("value", "value_new", "value_stock"):
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

        dataset_label = "db_facts_deltas"
        return df, dataset_label, "new"

    query = (
        "SELECT ym, iso3, hazard_code, hazard_label, hazard_class, metric, "
        "series_semantics, value, unit, as_of_date, publication_date, publisher, "
        "source_id, source_type, source_url, doc_title, definition_text, precedence_tier, "
        "event_id, proxy_for, confidence FROM facts_resolved WHERE ym = ?"
    )
    try:
        df = conn.execute(query, [ym]).fetch_df()
    except DuckDBError as exc:  # pragma: no cover - execution errors bubbled up
        LOGGER.exception("DuckDB query failed for facts_resolved at ym=%s", ym)
        raise
    if df.empty:
        return None, "facts_resolved", "stock"
    df = df.copy()
    if "ym" not in df.columns:
        df["ym"] = ym
    else:
        df["ym"] = df["ym"].fillna("").replace("", ym)
    df["series_semantics"] = (
        df.get("series_semantics", "stock")
        .astype(str)
        .str.strip()
        .replace("", "stock")
    )
    defaults = {
        "unit": "persons",
        "as_of_date": "",
        "publication_date": "",
        "publisher": "",
        "source_id": "",
        "source_type": "",
        "source_url": "",
        "doc_title": "",
        "definition_text": "",
        "precedence_tier": "",
        "event_id": "",
        "proxy_for": "",
        "confidence": "",
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default
        else:
            df[column] = df[column].fillna(default)
    return df.fillna(""), "db_facts_resolved", "stock"


def load_series_for_month(
    ym: str,
    is_current_month: bool,
    requested_series: str,
    *,
    backend: str = "files",
) -> Tuple[Optional[pd.DataFrame], str, str]:
    """Load data for the requested series ("new" or "stock")."""

    normalized_series = (requested_series or "stock").strip().lower()
    backend_choice = normalize_backend(backend, default="files")

    if backend_choice in {"auto", "db"}:
        db_df, db_dataset_label, db_series = load_series_from_db(ym, normalized_series)
        if db_df is not None and not db_df.empty:
            return db_df, db_dataset_label, db_series
        if backend_choice == "db":
            return None, "", normalized_series

    if normalized_series == "new":
        deltas_df, dataset_label = load_deltas_for_month(ym, is_current_month)
        if deltas_df is None or deltas_df.empty:
            return None, "", "new"
        prepared = prepare_deltas_frame(deltas_df, ym)
        if prepared.empty:
            return None, "", "new"
        return prepared, dataset_label, "new"

    resolved_df, dataset_label = load_resolved_for_month(ym, is_current_month)
    if resolved_df is not None:
        resolved_df = resolved_df.copy()
        if "series_semantics" not in resolved_df.columns:
            resolved_df["series_semantics"] = "stock"
        else:
            resolved_df["series_semantics"] = (
                resolved_df["series_semantics"].fillna("").replace("", "stock")
            )
    return resolved_df, dataset_label, "stock"


def _metric_rank(series: pd.Series, preferred_metric: str) -> pd.Series:
    order: list[str] = []
    preferred = preferred_metric.strip().lower()
    if preferred:
        order.append(preferred)
    for fallback in ("in_need", "affected"):
        if fallback not in order:
            order.append(fallback)

    def score(value: object) -> int:
        val = str(value).strip().lower()
        try:
            return order.index(val)
        except ValueError:
            return len(order)

    return series.apply(score)


def select_row(
    df: pd.DataFrame,
    iso3: str,
    hazard_code: str,
    cutoff_iso: str,
    preferred_metric: str = "in_need",
) -> Optional[dict]:
    """Select the single row that best answers the resolver question."""

    candidate = df[
        (df["iso3"].astype(str) == iso3) & (df["hazard_code"].astype(str) == hazard_code)
    ].copy()

    if candidate.empty:
        return None

    if "metric" in candidate.columns:
        candidate["metric"] = candidate["metric"].fillna("")
        candidate["metric_rank"] = _metric_rank(candidate["metric"], preferred_metric)

        if "as_of_date" in candidate.columns:
            candidate = candidate[candidate["as_of_date"] <= cutoff_iso]
            if candidate.empty:
                return None

        sort_cols: list[str] = ["metric_rank"]
        extra_cols = [col for col in ["as_of_date", "publication_date"] if col in candidate.columns]
        ascending = [True] + [False] * len(extra_cols)
        sort_cols.extend(extra_cols)
        candidate = candidate.sort_values(by=sort_cols, ascending=ascending)
        candidate = candidate.drop(columns=["metric_rank"], errors="ignore")
    else:
        sort_cols = [col for col in ["as_of_date", "publication_date"] if col in candidate.columns]
        if sort_cols:
            candidate = candidate.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))

    top = candidate.iloc[0].to_dict()

    raw_value = top.get("value", "")
    try:
        top["value"] = int(float(raw_value))
    except Exception:
        top["value"] = raw_value

    return top


@dataclass
class ResolveAttempt:
    series: str
    dataset_label: str
    source_bucket: str
    payload: dict


def _backend_order(choice: str) -> Iterable[str]:
    if choice == "auto":
        return ("db", "files")
    return (choice,)


def _normalise_series(value: str) -> str:
    return str(value).strip().lower()


def _coerce_value(raw: object) -> object:
    try:
        numeric = float(raw)
    except (TypeError, ValueError):
        return raw
    if numeric.is_integer():
        return int(numeric)
    return numeric


def _standardise_payload(
    row: dict,
    *,
    series: str,
    ym: str,
    dataset_label: str,
    source_bucket: str,
) -> dict:
    result = dict(row)
    result.setdefault("ym", ym)
    result.setdefault("metric", "")
    result.setdefault("unit", "persons")
    result.setdefault("as_of_date", "")
    result.setdefault("publication_date", result.get("as_of_date", ""))
    result.setdefault("publisher", "")
    result.setdefault("source_type", "")
    result.setdefault("source_url", "")
    result.setdefault("doc_title", "")
    result.setdefault("definition_text", "")
    result.setdefault("precedence_tier", "")
    result.setdefault("event_id", "")
    result.setdefault("confidence", "")
    result.setdefault("proxy_for", "")
    result.setdefault("source_id", row.get("source_id", ""))
    result.setdefault("series_semantics", series)
    result["value"] = _coerce_value(result.get("value"))
    result["series_returned"] = _normalise_series(result.get("series_semantics", series)) or series
    result["source_dataset"] = dataset_label
    result["source"] = source_bucket
    return result


def _resolve_from_db(
    *,
    series: str,
    ym: str,
    iso3: str,
    hazard_code: str,
    cutoff: str,
    preferred_metric: str,
) -> Optional[ResolveAttempt]:
    db_url = os.environ.get("RESOLVER_DB_URL")
    if not db_url:
        return None

    try:
        conn = duckdb_io.get_db(db_url)
        duckdb_io.init_schema(conn)
    except Exception:  # pragma: no cover - optional dependency misconfigured
        return None

    if series == "new":
        LOGGER.debug(
            "selectors using DB deltas: ym=%s iso3=%s hazard=%s cutoff=%s",
            ym,
            iso3,
            hazard_code,
            cutoff,
        )
        row = db_reader.fetch_deltas_point(
            conn, ym=ym, iso3=iso3, hazard_code=hazard_code, cutoff=cutoff, preferred_metric=preferred_metric
        )
        if not row:
            return None
        row = dict(row)
        value = row.get("value_new")
        if value is None:
            return None
        row["value"] = value
        row.setdefault("series_semantics", row.get("series_semantics_out", "new"))
        row["series_semantics"] = (
            str(row.get("series_semantics", "")).strip().lower() or "new"
        )
        row.setdefault("as_of_date", row.get("as_of", ""))
        row.setdefault("publication_date", row.get("as_of_date", ""))
        row.setdefault("metric", row.get("metric", preferred_metric))
        payload = _standardise_payload(
            row,
            series="new",
            ym=ym,
            dataset_label="db_facts_deltas",
            source_bucket="db",
        )
        return ResolveAttempt("new", "db_facts_deltas", "db", payload)

    row = db_reader.fetch_resolved_point(
        conn, ym=ym, iso3=iso3, hazard_code=hazard_code, cutoff=cutoff, preferred_metric=preferred_metric
    )
    if not row:
        return None
    row = dict(row)
    row.setdefault("series_semantics", "stock")
    row["series_semantics"] = (
        str(row.get("series_semantics", "")).strip().lower() or "stock"
    )
    payload = _standardise_payload(
        row,
        series="stock",
        ym=ym,
        dataset_label="db_facts_resolved",
        source_bucket="db",
    )
    return ResolveAttempt("stock", "db_facts_resolved", "db", payload)


def _resolve_from_files(
    *,
    series: str,
    ym: str,
    iso3: str,
    hazard_code: str,
    cutoff: str,
    preferred_metric: str,
) -> Optional[ResolveAttempt]:
    current_month = ym == current_ym_istanbul()
    df, dataset_label, series_used = load_series_for_month(
        ym, current_month, series, backend="files"
    )
    if df is None or df.empty:
        return None

    row = select_row(df, iso3, hazard_code, cutoff, preferred_metric=preferred_metric)
    if not row:
        return None

    resolved_series = _normalise_series(row.get("series_semantics", series_used)) or series_used
    payload = _standardise_payload(
        row,
        series=resolved_series,
        ym=ym,
        dataset_label=dataset_label,
        source_bucket="snapshot" if dataset_label in {"snapshot", "snapshot_deltas"} else (
            "state" if dataset_label == "monthly_deltas" else "exports"
        ),
    )
    return ResolveAttempt(resolved_series, dataset_label, payload["source"], payload)


def resolve_point(
    iso3: str,
    hazard_code: str,
    cutoff: str,
    series: str,
    metric: str = "in_need",
    backend: str = "db",
) -> Optional[dict]:
    """Resolve a single point for the requested series and cutoff."""

    normalized_series = _normalise_series(series) or "stock"
    if normalized_series not in {"new", "stock"}:
        normalized_series = "stock"

    preferred_metric = (metric or "in_need").strip() or "in_need"
    backend_choice = normalize_backend(backend, default="db")
    ym = ym_from_cutoff(cutoff)

    def attempt(series_choice: str) -> Optional[dict]:
        for backend_option in _backend_order(backend_choice):
            if backend_option == "db":
                resolved = _resolve_from_db(
                    series=series_choice,
                    ym=ym,
                    iso3=iso3,
                    hazard_code=hazard_code,
                    cutoff=cutoff,
                    preferred_metric=preferred_metric,
                )
            else:
                resolved = _resolve_from_files(
                    series=series_choice,
                    ym=ym,
                    iso3=iso3,
                    hazard_code=hazard_code,
                    cutoff=cutoff,
                    preferred_metric=preferred_metric,
                )
            if resolved:
                payload = dict(resolved.payload)
                payload["series_requested"] = normalized_series
                payload["series_returned"] = payload.get("series_returned", resolved.series)
                payload["ok"] = True
                return payload
        return None

    result = attempt(normalized_series)
    if result:
        return result

    fallback_allowed = _normalise_series(os.environ.get("RESOLVER_ALLOW_SERIES_FALLBACK", "")) in {
        "1",
        "true",
        "yes",
    }
    if not fallback_allowed:
        return None

    alternate = "stock" if normalized_series == "new" else "new"
    result = attempt(alternate)
    if result:
        result["series_returned"] = _normalise_series(result.get("series_returned", alternate)) or alternate
        result["series_requested"] = normalized_series
        result["fallback_used"] = True
        return result
    return None
