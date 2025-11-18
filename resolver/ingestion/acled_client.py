#!/usr/bin/env python3
"""ACLED connector — monthly-first aggregation with conflict onset detection."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import pandas as pd
import requests
import yaml
from urllib.parse import urlencode, urlsplit, urlunsplit

from . import acled_auth
from resolver.ingestion._manifest import ensure_manifest_for_csv
from resolver.ingestion.utils.io import (
    render_with_context,
    resolve_ingestion_window,
    resolve_output_path,
)
from resolver.ingestion.utils.iso_normalize import to_iso3

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STAGING = ROOT / "staging"
CONFIG = ROOT / "ingestion" / "config" / "acled.yml"
DIAGNOSTICS_ROOT = ROOT / "diagnostics" / "ingestion"
ACLED_DIAGNOSTICS = DIAGNOSTICS_ROOT / "acled"
ACLED_RUN_PATH = DIAGNOSTICS_ROOT / "acled_client" / "acled_client_run.json"
ACLED_HTTP_DIAG_PATH = ACLED_DIAGNOSTICS / "http_diag.json"

ACLED_API_BASE_URL = "https://acleddata.com/api/acled/read"
ACLED_DEFAULT_FORMAT = "json"

COUNTRIES = DATA / "countries.csv"
SHOCKS = DATA / "shocks.csv"

DEFAULT_OUTPUT = ROOT / "staging" / "acled.csv"
OUT_PATH = resolve_output_path(DEFAULT_OUTPUT)

CANONICAL_HEADERS = [
    "event_id",
    "country_name",
    "iso3",
    "hazard_code",
    "hazard_label",
    "hazard_class",
    "metric",
    "series_semantics",
    "value",
    "unit",
    "as_of_date",
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
]

SERIES_SEMANTICS = "new"
DOC_TITLE = "ACLED monthly aggregation"
CONFLICT_METRIC = "fatalities_battle_month"

HAZARD_KEY_TO_CODE = {
    "armed_conflict_onset": "ACO",
    "armed_conflict_escalation": "ACE",
    "civil_unrest": "CU",
}

DEBUG = os.getenv("RESOLVER_DEBUG", "0") == "1"
LOG = logging.getLogger("resolver.ingestion.acled.client")


def dbg(message: str) -> None:
    if DEBUG:
        print(f"[acled] {message}")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "y", "yes", "on"}


def _env_int(name: str) -> Optional[int]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def load_config() -> Dict[str, Any]:
    if not CONFIG.exists():
        return {}
    with open(CONFIG, "r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    return data


def load_registries() -> Tuple[pd.DataFrame, pd.DataFrame]:
    countries = pd.read_csv(COUNTRIES, dtype=str).fillna("")
    shocks = pd.read_csv(SHOCKS, dtype=str).fillna("")
    return countries, shocks


def _safe_base_url(url: str) -> str:
    try:
        parsed = urlsplit(str(url))
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    except Exception:
        return str(url).split("?")[0]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _clear_zero_rows_diagnostic() -> None:
    try:
        (ACLED_DIAGNOSTICS / "zero_rows.json").unlink()
    except FileNotFoundError:
        return
    except OSError:
        pass


def _clear_http_diagnostic() -> None:
    try:
        ACLED_HTTP_DIAG_PATH.unlink()
    except FileNotFoundError:
        return
    except OSError:
        pass


def _write_acled_http_diag(*, status: int, url: str) -> None:
    try:
        ACLED_HTTP_DIAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    except OSError:  # pragma: no cover - diagnostics best effort
        return
    if ACLED_HTTP_DIAG_PATH.exists():
        return
    payload = {"status": int(status), "url": str(url)}
    try:
        ACLED_HTTP_DIAG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:  # pragma: no cover - diagnostics best effort
        return


def _write_zero_rows_diagnostic(meta: Dict[str, Any], reason: str) -> None:
    payload = {
        "status": meta.get("http_status"),
        "params_keys": sorted(set(meta.get("params_keys", []))),
        "start": meta.get("start"),
        "end": meta.get("end"),
        "reason": reason,
        "base_url": meta.get("base_url"),
    }
    _write_json(ACLED_DIAGNOSTICS / "zero_rows.json", payload)


def _write_run_summary(meta: Dict[str, Any], *, rows_fetched: int, rows_normalized: int, rows_written: int) -> None:
    payload = {
        "rows_fetched": int(rows_fetched),
        "rows_normalized": int(rows_normalized),
        "rows_written": int(rows_written),
        "http_status": meta.get("http_status"),
        "base_url": meta.get("base_url"),
        "source_url": meta.get("source_url"),
        "window": {"start": meta.get("start"), "end": meta.get("end")},
        "params_keys": sorted(set(meta.get("params_keys", []))),
    }
    _write_json(ACLED_RUN_PATH, payload)


def _normalise_month(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        parsed = pd.to_datetime(value, errors="coerce")
    except Exception:
        parsed = pd.NaT
    if pd.isna(parsed):
        return None
    return parsed.to_period("M").strftime("%Y-%m")


def _to_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return 0
        return int(value)
    text = str(value).strip()
    if not text:
        return 0
    text = text.replace(",", "").replace(" ", "")
    try:
        return int(float(text))
    except Exception:
        return 0


def compute_conflict_onset_flags(
    df: pd.DataFrame,
    *,
    iso_col: str = "iso3",
    date_col: str = "month",
    event_type_col: str = "event_type",
    fatalities_col: str = "fatalities",
    battle_event_types: Sequence[str] = ("Battles",),
    lookback_months: int = 12,
    threshold: int = 25,
) -> pd.DataFrame:
    """Return battle fatalities with rolling lookback totals and onset flags."""

    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "iso3",
                "month",
                "battle_fatalities",
                "prev12_battle_fatalities",
                "is_onset",
            ]
        )

    lookback = max(int(lookback_months or 0), 1)
    threshold_value = int(threshold or 0)

    work = df[[iso_col, date_col, event_type_col, fatalities_col]].copy()
    work[iso_col] = work[iso_col].astype(str).str.strip().str.upper()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[iso_col, date_col])
    if work.empty:
        return pd.DataFrame(
            columns=[
                "iso3",
                "month",
                "battle_fatalities",
                "prev12_battle_fatalities",
                "is_onset",
            ]
        )

    battle_types = {str(v).strip().lower() for v in battle_event_types if str(v).strip()}
    work["event_type_lower"] = work[event_type_col].astype(str).str.strip().str.lower()
    if battle_types:
        work = work[work["event_type_lower"].isin(battle_types)]
    if work.empty:
        return pd.DataFrame(
            columns=[
                "iso3",
                "month",
                "battle_fatalities",
                "prev12_battle_fatalities",
                "is_onset",
            ]
        )

    work[fatalities_col] = work[fatalities_col].map(_to_int)
    work["month_period"] = work[date_col].dt.to_period("M")
    work = work.dropna(subset=["month_period"])
    if work.empty:
        return pd.DataFrame(
            columns=[
                "iso3",
                "month",
                "battle_fatalities",
                "prev12_battle_fatalities",
                "is_onset",
            ]
        )

    grouped = (
        work.groupby([iso_col, "month_period"], as_index=False)[fatalities_col].sum()
    )
    grouped.rename(
        columns={iso_col: "iso3", "month_period": "month", fatalities_col: "battle_fatalities"},
        inplace=True,
    )

    rows: List[pd.DataFrame] = []
    for iso3, group in grouped.groupby("iso3"):
        group = group.sort_values("month")
        start = group["month"].min()
        end = group["month"].max()
        idx = pd.period_range(start, end, freq="M")
        series = pd.Series(0, index=idx, dtype="int64")
        for record in group.itertuples(index=False):
            series.loc[record.month] = int(record.battle_fatalities)
        prev_window = (
            series.shift(1).rolling(window=lookback, min_periods=1).sum().fillna(0).astype(int)
        )
        frame = pd.DataFrame(
            {
                "iso3": iso3,
                "month": idx.strftime("%Y-%m"),
                "battle_fatalities": series.astype(int).to_list(),
                "prev12_battle_fatalities": prev_window.to_list(),
            }
        )
        frame["is_onset"] = (
            (frame["prev12_battle_fatalities"] < threshold_value)
            & (frame["battle_fatalities"] >= threshold_value)
        )
        rows.append(frame)

    if not rows:
        return pd.DataFrame(
            columns=[
                "iso3",
                "month",
                "battle_fatalities",
                "prev12_battle_fatalities",
                "is_onset",
            ]
        )

    result = pd.concat(rows, ignore_index=True)
    return result


def _digest(parts: Iterable[str]) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:12]


def _build_source_url(base_url: str, params: Dict[str, Any], token_keys: Sequence[str]) -> str:
    safe_params = {}
    for key, value in params.items():
        if key in token_keys:
            continue
        safe_params[key] = value
    if not safe_params:
        return base_url
    return f"{base_url}?{urlencode(safe_params, doseq=True)}"


def _resolve_query_params(config: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    raw = config.get("query") if isinstance(config, dict) else None
    if not isinstance(raw, dict):
        return params
    for key, value in raw.items():
        if isinstance(value, list):
            resolved = [render_with_context(str(item)) for item in value]
            params[key] = [item for item in resolved if item != ""]
        elif isinstance(value, (str, int, float)):
            if isinstance(value, str):
                rendered = render_with_context(value)
                if rendered == "":
                    continue
                params[key] = rendered
            else:
                params[key] = value
        else:
            params[key] = value
    return params


def _apply_query_auth(params: Dict[str, Any], config: Dict[str, Any]) -> None:
    auth = config.get("auth") if isinstance(config, dict) else None
    if not isinstance(auth, dict):
        return
    if str(auth.get("type") or "").strip().lower() != "query":
        return
    auth_params = auth.get("params")
    if not isinstance(auth_params, dict):
        return
    for key, value in auth_params.items():
        if isinstance(value, list):
            rendered = [render_with_context(str(item)) for item in value]
            filtered = [item for item in rendered if item != ""]
            if filtered:
                params[key] = filtered
        elif isinstance(value, str):
            rendered = render_with_context(value)
            if rendered != "":
                params[key] = rendered
        elif value is not None:
            params[key] = value


def fetch_events(config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    base_url = os.getenv("ACLED_BASE", config.get("base_url", ACLED_API_BASE_URL))

    window_days = int(os.getenv("ACLED_WINDOW_DAYS", config.get("window_days", 450)))
    limit = int(os.getenv("ACLED_MAX_LIMIT", config.get("limit", 1000)))
    max_pages = _env_int("RESOLVER_MAX_PAGES")
    max_results = _env_int("RESOLVER_MAX_RESULTS")

    override_start, override_end = resolve_ingestion_window()
    end_date = override_end or date.today()
    start_date = override_start or (end_date - timedelta(days=window_days))
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    params: Dict[str, Any] = {
        "limit": limit,
        "page": 1,
    }

    params.update(_resolve_query_params(config))
    params.setdefault("_format", ACLED_DEFAULT_FORMAT)
    params["event_date"] = f"{start_date:%Y-%m-%d}|{end_date:%Y-%m-%d}"
    params.setdefault("event_date_where", "BETWEEN")
    params.setdefault("limit", limit)
    params.setdefault("page", 1)

    token_keys = {"access_token", "key", "token", "email", "username", "password"}
    source_url = _build_source_url(base_url, params, token_keys)
    safe_base_url = _safe_base_url(base_url)

    diagnostics_meta: Dict[str, Any] = {
        "base_url": safe_base_url,
        "params_keys": sorted(params.keys()),
        "start": f"{start_date:%Y-%m-%d}",
        "end": f"{end_date:%Y-%m-%d}",
        "http_status": None,
    }

    _clear_http_diagnostic()
    records: List[Dict[str, Any]] = []
    session = requests.Session()
    access_token = acled_auth.get_access_token()
    headers = {"Authorization": f"Bearer {access_token}"}

    page = 1
    last_status: Optional[int] = None
    last_url: Optional[str] = None
    while True:
        if max_pages is not None and page > max_pages:
            dbg(f"max pages reached at page {page}")
            break
        params["page"] = page
        dbg(f"fetching page {page}")
        resp = session.get(base_url, params=params, headers=headers, timeout=60)
        status = resp.status_code
        last_status = status
        last_url = str(resp.url)
        _write_acled_http_diag(status=status, url=str(resp.url))
        diagnostics_meta["http_status"] = status
        LOG.info(
            "ACLED HTTP request",
            extra={
                "status": status,
                "url": f"{safe_base_url}?...",
                "page": page,
                "window": f"{diagnostics_meta['start']}→{diagnostics_meta['end']}",
                "params_keys": sorted(params.keys()),
            },
        )
        if status != 200:
            try:
                body_snippet = resp.text[:500]
            except Exception:  # pragma: no cover - defensive fallback
                body_snippet = "<unreadable response>"
            LOG.error(
                "ACLED HTTP error",
                extra={
                    "status": status,
                    "url": f"{safe_base_url}?...",
                    "body_snippet": body_snippet,
                },
            )
            raise RuntimeError(f"ACLED read failed: HTTP {status}")
        try:
            payload = resp.json() or {}
        except ValueError as exc:  # pragma: no cover - JSON decode errors
            try:
                body_snippet = resp.text[:500]
            except Exception:  # pragma: no cover - defensive
                body_snippet = "<unreadable response>"
            LOG.error(
                "ACLED JSON decode error",
                extra={"status": status, "url": f"{safe_base_url}?...", "body_snippet": body_snippet},
            )
            raise RuntimeError("ACLED payload was not valid JSON") from exc
        if not isinstance(payload, dict):
            LOG.error(
                "ACLED payload unexpected type",
                extra={"status": status, "type": type(payload).__name__},
            )
            raise RuntimeError("ACLED payload missing expected fields (data/results/count)")
        if page == 1:
            expected = {"data", "results", "count"}
            present = sorted(key for key in expected if key in payload)
            if not present:
                try:
                    body_snippet = resp.text[:500]
                except Exception:  # pragma: no cover - defensive
                    body_snippet = "<unreadable response>"
                LOG.error(
                    "ACLED payload missing expected keys",
                    extra={
                        "status": status,
                        "keys": sorted(payload.keys()),
                        "body_snippet": body_snippet,
                    },
                )
                raise RuntimeError("ACLED payload missing expected fields (data/results/count)")
            dbg(f"ACLED connectivity ok; payload keys include: {', '.join(present)}")
        data = payload.get("data") or payload.get("results") or []
        if payload.get("status") not in (200, "200", None):
            LOG.error(
                "ACLED API returned non-200 status in JSON",
                extra={
                    "json_status": payload.get("status"),
                    "url": f"{safe_base_url}?...",
                },
            )
            raise RuntimeError(f"ACLED read failed: JSON status={payload.get('status')}")
        if not isinstance(data, list):
            LOG.error(
                "ACLED payload unexpected structure",
                extra={"status": status, "data_type": type(data).__name__},
            )
            raise RuntimeError("Unexpected ACLED payload structure")
        if not data:
            LOG.warning(
                "ACLED returned zero rows",
                extra={
                    "status": status,
                    "url": f"{safe_base_url}?...",
                    "params_keys": sorted(params.keys()),
                    "window": f"{diagnostics_meta['start']}→{diagnostics_meta['end']}",
                },
            )
            break
        records.extend(data)
        dbg(f"page {page} returned {len(data)} rows (total={len(records)})")
        if max_results is not None and len(records) >= max_results:
            dbg("max results reached; truncating")
            records = records[:max_results]
            break
        if len(data) < limit:
            break
        page += 1

    diagnostics_meta["rows_fetched"] = len(records)
    diagnostics_meta["source_url"] = last_url or source_url
    if records:
        _clear_zero_rows_diagnostic()
    else:
        _write_zero_rows_diagnostic(diagnostics_meta, "empty dataframe")

    return records, source_url, diagnostics_meta


def _extract_first(record: MutableMapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in record:
            value = record.get(key)
            if value is not None:
                return value
    return None


def _prepare_dataframe(
    records: Sequence[MutableMapping[str, Any]],
    config: Dict[str, Any],
    countries: pd.DataFrame,
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["iso3", "country_name", "event_type", "month", "fatalities", "notes"])

    keys = config.get("keys", {})
    iso_keys = keys.get("iso3", [])
    country_keys = keys.get("country", [])
    date_keys = keys.get("date", [])
    event_type_keys = keys.get("event_type", [])
    fatalities_keys = keys.get("fatalities", [])
    notes_keys = keys.get("notes", [])

    iso_lookup = {str(row.iso3).strip().upper(): str(row.country_name)
                  for row in countries.itertuples(index=False)}
    name_lookup = {str(row.country_name).strip().lower(): str(row.iso3).strip().upper()
                   for row in countries.itertuples(index=False)}

    rows: List[Dict[str, Any]] = []
    for record in records:
        event_date = _extract_first(record, date_keys)
        month = _normalise_month(event_date)
        if not month:
            continue
        iso = str(_extract_first(record, iso_keys) or "").strip().upper()
        if not iso:
            country_name = str(_extract_first(record, country_keys) or "").strip()
            iso = name_lookup.get(country_name.lower(), "") if country_name else ""
        if not iso or iso not in iso_lookup:
            continue
        country_name = iso_lookup[iso]
        event_type = str(_extract_first(record, event_type_keys) or "").strip()
        fatalities = _to_int(_extract_first(record, fatalities_keys))
        notes = str(_extract_first(record, notes_keys) or "").strip()
        rows.append(
            {
                "iso3": iso,
                "country_name": country_name,
                "event_type": event_type,
                "month": month,
                "fatalities": fatalities,
                "notes": notes,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["iso3", "country_name", "event_type", "month", "fatalities", "notes"])

    df = pd.DataFrame(rows)
    df["event_type_lower"] = df["event_type"].str.lower()
    return df


def _parse_participants(df: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
    participants_cfg = config.get("participants", {})
    enabled_cfg = bool(participants_cfg.get("enabled", False))
    enabled_env = _env_bool("ACLED_PARSE_PARTICIPANTS", enabled_cfg)
    if not enabled_env:
        return pd.Series([None] * len(df))

    regex_text = participants_cfg.get("regex", "")
    if not regex_text:
        return pd.Series([None] * len(df))
    pattern = re.compile(regex_text, flags=re.IGNORECASE)

    values: List[Optional[int]] = []
    for note in df["notes"].fillna(""):
        match = pattern.search(str(note))
        if not match:
            values.append(None)
            continue
        raw = match.group(1)
        if raw is None:
            values.append(None)
            continue
        raw = raw.replace(",", "").replace(".", "")
        try:
            values.append(int(raw))
        except Exception:
            values.append(None)
    return pd.Series(values)


def _make_conflict_rows(
    conflict_stats: pd.DataFrame,
    shocks: pd.DataFrame,
    source_url: str,
    publication_date: str,
    ingested_at: str,
    method_base: str,
    definition_base: str,
    threshold: int,
    lookback_months: int,
    onset_enabled: bool,
) -> List[Dict[str, Any]]:
    if conflict_stats.empty:
        return []

    shocks_index = shocks.set_index("hazard_code")
    lookback = max(int(lookback_months or 0), 1)
    threshold_value = int(threshold or 0)

    rows: List[Dict[str, Any]] = []
    sorted_stats = conflict_stats.sort_values(["iso3", "month"])
    for record in sorted_stats.itertuples(index=False):
        fatalities = int(getattr(record, "battle_fatalities", 0))
        if fatalities <= 0:
            continue
        prev_value = int(getattr(record, "prev12_battle_fatalities", 0))
        hazard_code = HAZARD_KEY_TO_CODE["armed_conflict_escalation"]
        hazard_row = shocks_index.loc[hazard_code]
        digest = _digest(
            [
                record.iso3,
                hazard_code,
                CONFLICT_METRIC,
                record.month,
                str(fatalities),
                source_url,
            ]
        )
        year, month = record.month.split("-")
        definition_text = (
            f"{definition_base} Prev{lookback}m battle fatalities={prev_value}; "
            f"current month battle fatalities={fatalities}; threshold={threshold_value}."
        )
        method_parts = [
            method_base,
            f"battle_fatalities={fatalities}",
            f"prev{lookback}m_battle_fatalities={prev_value}",
            f"threshold={threshold_value}",
        ]
        method = "; ".join(method_parts)
        common_row = {
            "country_name": record.country_name,
            "iso3": record.iso3,
            "metric": CONFLICT_METRIC,
            "series_semantics": SERIES_SEMANTICS,
            "value": fatalities,
            "unit": "persons",
            "as_of_date": record.month,
            "publication_date": publication_date,
            "publisher": record.publisher,
            "source_type": record.source_type,
            "source_url": source_url,
            "doc_title": DOC_TITLE,
            "definition_text": definition_text,
            "method": method,
            "confidence": "",
            "revision": 0,
            "ingested_at": ingested_at,
        }
        rows.append(
            {
                "event_id": f"{record.iso3}-ACLED-{hazard_code}-{CONFLICT_METRIC}-{year}-{month}-{digest}",
                "hazard_code": hazard_code,
                "hazard_label": hazard_row["hazard_label"],
                "hazard_class": hazard_row["hazard_class"],
                **common_row,
            }
        )

        if onset_enabled and bool(getattr(record, "is_onset", False)):
            onset_code = HAZARD_KEY_TO_CODE["armed_conflict_onset"]
            onset_row = shocks_index.loc[onset_code]
            onset_digest = _digest(
                [
                    record.iso3,
                    onset_code,
                    CONFLICT_METRIC,
                    record.month,
                    str(fatalities),
                    source_url,
                    "onset",
                ]
            )
            onset_method = method + "; onset_rule_v1"
            onset_definition = definition_text + " Onset rule triggered."
            onset_common = dict(common_row)
            onset_common.update({"definition_text": onset_definition, "method": onset_method})
            rows.append(
                {
                    "event_id": f"{record.iso3}-ACLED-{onset_code}-{CONFLICT_METRIC}-{year}-{month}-{onset_digest}",
                    "hazard_code": onset_code,
                    "hazard_label": onset_row["hazard_label"],
                    "hazard_class": onset_row["hazard_class"],
                    **onset_common,
                }
            )
    return rows


def _make_unrest_rows(
    unrest_counts: pd.DataFrame,
    shocks: pd.DataFrame,
    source_url: str,
    publication_date: str,
    ingested_at: str,
    method_base: str,
    definition_base: str,
) -> List[Dict[str, Any]]:
    if unrest_counts.empty:
        return []
    shocks_index = shocks.set_index("hazard_code")
    hazard_code = HAZARD_KEY_TO_CODE["civil_unrest"]
    hazard_row = shocks_index.loc[hazard_code]
    rows: List[Dict[str, Any]] = []
    for record in unrest_counts.sort_values(["iso3", "month"]).itertuples(index=False):
        value = int(record.events)
        digest = _digest([record.iso3, hazard_code, "events", record.month, str(value), source_url])
        year, month = record.month.split("-")
        definition_text = f"{definition_base} Metric=events aggregated for unrest types."
        rows.append(
            {
                "event_id": f"{record.iso3}-ACLED-{hazard_code}-events-{year}-{month}-{digest}",
                "country_name": record.country_name,
                "iso3": record.iso3,
                "hazard_code": hazard_code,
                "hazard_label": hazard_row["hazard_label"],
                "hazard_class": hazard_row["hazard_class"],
                "metric": "events",
                "series_semantics": SERIES_SEMANTICS,
                "value": value,
                "unit": "events",
                "as_of_date": record.month,
                "publication_date": publication_date,
                "publisher": record.publisher,
                "source_type": record.source_type,
                "source_url": source_url,
                "doc_title": DOC_TITLE,
                "definition_text": definition_text,
                "method": method_base,
                "confidence": "",
                "revision": 0,
                "ingested_at": ingested_at,
            }
        )
    return rows


def _make_participant_rows(
    participants: pd.DataFrame,
    shocks: pd.DataFrame,
    source_url: str,
    publication_date: str,
    ingested_at: str,
    method_base: str,
    definition_base: str,
) -> List[Dict[str, Any]]:
    if participants.empty:
        return []
    shocks_index = shocks.set_index("hazard_code")
    hazard_code = HAZARD_KEY_TO_CODE["civil_unrest"]
    hazard_row = shocks_index.loc[hazard_code]
    rows: List[Dict[str, Any]] = []
    for record in participants.sort_values(["iso3", "month"]).itertuples(index=False):
        value = int(record.participants)
        digest = _digest([record.iso3, hazard_code, "participants", record.month, str(value), source_url])
        year, month = record.month.split("-")
        rows.append(
            {
                "event_id": f"{record.iso3}-ACLED-{hazard_code}-participants-{year}-{month}-{digest}",
                "country_name": record.country_name,
                "iso3": record.iso3,
                "hazard_code": hazard_code,
                "hazard_label": hazard_row["hazard_label"],
                "hazard_class": hazard_row["hazard_class"],
                "metric": "participants",
                "series_semantics": SERIES_SEMANTICS,
                "value": value,
                "unit": "persons",
                "as_of_date": record.month,
                "publication_date": publication_date,
                "publisher": record.publisher,
                "source_type": record.source_type,
                "source_url": source_url,
                "doc_title": DOC_TITLE,
                "definition_text": f"{definition_base} Metric=participants from event notes heuristic.",
                "method": method_base,
                "confidence": "",
                "revision": 0,
                "ingested_at": ingested_at,
            }
        )
    return rows


def _aggregate_participants(df: pd.DataFrame, participants_values: pd.Series, aggregate: str) -> pd.DataFrame:
    df = df.copy()
    df["participants_value"] = participants_values
    df = df.dropna(subset=["participants_value"])
    if df.empty:
        return pd.DataFrame(columns=["iso3", "country_name", "month", "participants"])
    if aggregate == "median":
        grouped = df.groupby(["iso3", "country_name", "month"], as_index=False)["participants_value"].median()
    else:
        grouped = df.groupby(["iso3", "country_name", "month"], as_index=False)["participants_value"].sum()
    grouped.rename(columns={"participants_value": "participants"}, inplace=True)
    grouped["participants"] = grouped["participants"].round().astype(int)
    return grouped


def _build_rows(
    records: Sequence[MutableMapping[str, Any]],
    config: Dict[str, Any],
    countries: pd.DataFrame,
    shocks: pd.DataFrame,
    source_url: str,
    publication_date: str,
    ingested_at: str,
) -> List[Dict[str, Any]]:
    df = _prepare_dataframe(records, config, countries)
    if df.empty:
        return []

    publisher = config.get("publisher", "ACLED")
    source_type = config.get("source_type", "other")
    df["publisher"] = publisher
    df["source_type"] = source_type

    unrest_types = {str(v).strip().lower() for v in config.get("unrest_types", []) if v}
    df_unrest = df[df["event_type_lower"].isin(unrest_types)]
    unrest_counts = (
        df_unrest.groupby(["iso3", "country_name", "month", "publisher", "source_type"], as_index=False)
        .size()
        .rename(columns={"size": "events"})
    )

    onset_cfg = config.get("onset", {})
    onset_enabled = bool(onset_cfg.get("enabled", True))
    lookback_months = int(onset_cfg.get("lookback_months", 12) or 12)
    threshold = int(onset_cfg.get("threshold_battle_deaths", 25) or 25)
    battle_event_types_cfg = onset_cfg.get("battle_event_types", ["Battles"])
    if not battle_event_types_cfg:
        battle_event_types_cfg = ["Battles"]
    battle_types_lower = {str(v).strip().lower() for v in battle_event_types_cfg if str(v).strip()}
    if not battle_types_lower:
        battle_types_lower = {"battles"}

    battle_events = df[df["event_type_lower"].isin(battle_types_lower)]
    battle_totals = (
        battle_events.groupby(
            ["iso3", "country_name", "month", "publisher", "source_type"], as_index=False
        )["fatalities"].sum()
    )
    battle_totals.rename(columns={"fatalities": "battle_fatalities"}, inplace=True)

    onset_flags = pd.DataFrame(
        columns=["iso3", "month", "battle_fatalities", "prev12_battle_fatalities", "is_onset"]
    )
    if onset_enabled and not df.empty:
        onset_input = df[["iso3", "month", "event_type", "fatalities"]].copy()
        onset_flags = compute_conflict_onset_flags(
            onset_input,
            iso_col="iso3",
            date_col="month",
            event_type_col="event_type",
            fatalities_col="fatalities",
            battle_event_types=tuple(battle_event_types_cfg),
            lookback_months=lookback_months,
            threshold=threshold,
        )

    conflict_stats = battle_totals.copy()
    if conflict_stats.empty:
        conflict_stats = pd.DataFrame(
            columns=[
                "iso3",
                "country_name",
                "month",
                "publisher",
                "source_type",
                "battle_fatalities",
                "prev12_battle_fatalities",
                "is_onset",
            ]
        )
    else:
        if not onset_flags.empty:
            conflict_stats = conflict_stats.merge(
                onset_flags[["iso3", "month", "prev12_battle_fatalities", "is_onset"]],
                on=["iso3", "month"],
                how="left",
            )
        if "prev12_battle_fatalities" not in conflict_stats.columns:
            conflict_stats["prev12_battle_fatalities"] = 0
        conflict_stats["prev12_battle_fatalities"] = (
            conflict_stats["prev12_battle_fatalities"].fillna(0).astype(int)
        )
        if "is_onset" not in conflict_stats.columns:
            conflict_stats["is_onset"] = False
        conflict_stats["is_onset"] = conflict_stats["is_onset"].fillna(False).astype(bool)
        if not onset_enabled:
            conflict_stats["is_onset"] = False

    unrest_label = " + ".join(sorted(config.get("unrest_types", []))) or "Protests + Riots"
    battle_label = ", ".join(str(v).strip() for v in battle_event_types_cfg if str(v).strip()) or "Battles"
    definition_base = (
        "ACLED monthly-first aggregation; battle fatalities aggregated from "
        f"{battle_label}; civil unrest events counted from {unrest_label}."
    )
    method_base = (
        "ACLED; monthly-first; battle fatalities aggregated; "
        f"unrest events={unrest_label}; onset rule applied"
    )

    participants_values = _parse_participants(df, config)
    participants_rows: List[Dict[str, Any]] = []
    if participants_values.notna().any():
        aggregate = config.get("participants", {}).get("aggregate", "sum").lower()
        participants_totals = _aggregate_participants(df, participants_values, aggregate)
        if not participants_totals.empty:
            participants_totals["publisher"] = publisher
            participants_totals["source_type"] = source_type
            participants_rows = _make_participant_rows(
                participants_totals,
                shocks,
                source_url,
                publication_date,
                ingested_at,
                method_base,
                definition_base,
            )

    conflict_rows = _make_conflict_rows(
        conflict_stats,
        shocks,
        source_url,
        publication_date,
        ingested_at,
        method_base,
        definition_base,
        threshold,
        lookback_months,
        onset_enabled,
    )
    unrest_rows = _make_unrest_rows(
        unrest_counts,
        shocks,
        source_url,
        publication_date,
        ingested_at,
        method_base,
        definition_base,
    )

    rows = conflict_rows + unrest_rows + participants_rows
    rows.sort(key=lambda r: (r["iso3"], r["as_of_date"], r["metric"]))
    return rows


def collect_rows() -> List[Dict[str, Any]]:
    config = load_config()
    countries, shocks = load_registries()
    ingestion_mode = (os.getenv("RESOLVER_INGESTION_MODE") or "").strip().lower()
    legacy_token = os.getenv("ACLED_TOKEN") or str(config.get("token", ""))
    if legacy_token:
        os.environ.setdefault("ACLED_ACCESS_TOKEN", legacy_token)

    try:
        records, source_url, diagnostics_meta = fetch_events(config)
    except RuntimeError as exc:
        message = f"ACLED auth failed: {exc}"
        if ingestion_mode == "real":
            print(message)
            if os.getenv("RESOLVER_FAIL_ON_STUB_ERROR") == "1":
                raise
            return []
        dbg(message)
        return []
    publication_date = date.today().isoformat()
    ingested_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    rows: List[Dict[str, Any]] = []
    if records:
        rows = _build_rows(records, config, countries, shocks, source_url, publication_date, ingested_at)
    else:
        rows = []

    if not rows and records:
        diagnostics_meta["rows_fetched"] = len(records)
        _write_zero_rows_diagnostic(diagnostics_meta, "normalized dataframe empty")

    _write_run_summary(
        diagnostics_meta,
        rows_fetched=len(records),
        rows_normalized=len(rows),
        rows_written=len(rows),
    )

    if not rows:
        return []
    return rows


def _write_header_only(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(CANONICAL_HEADERS)
    ensure_manifest_for_csv(path)


def _write_rows(rows: Sequence[MutableMapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=CANONICAL_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    ensure_manifest_for_csv(path)


class ACLEDClient:
    """Thin ACLED API client supporting monthly fatalities aggregation."""

    _DEFAULT_FIELDS = ["event_date", "iso3", "country", "fatalities"]

    def __init__(
        self,
        *,
        base_url: str = ACLED_API_BASE_URL,
        endpoint: str = "",
        timeout: int = 30,
        max_retries: int = 4,
        page_size: int = 5000,
        session: Optional[requests.Session] = None,
        logger: Optional[logging.Logger] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = config or load_config().get("client", {})
        raw_base = str(cfg.get("base_url", base_url)).strip() or base_url
        raw_endpoint = str(cfg.get("endpoint", endpoint)).strip()
        if raw_endpoint:
            raw_base = f"{raw_base.rstrip('/')}/{raw_endpoint.lstrip('/')}"
        self.base_url = raw_base.rstrip("/") or ACLED_API_BASE_URL
        self.timeout = int(cfg.get("timeout", timeout))
        self.max_retries = int(cfg.get("max_retries", max_retries))
        self.page_size = int(cfg.get("page_size", page_size))
        self.fields = cfg.get("fields") or list(self._DEFAULT_FIELDS)
        self.use_stub = bool(cfg.get("use_stub", False))
        # Honour global stub toggles if present.
        if os.getenv("RESOLVER_FORCE_STUBS") == "1" or os.getenv("RESOLVER_INCLUDE_STUBS") == "1":
            self.use_stub = True
        self.session = session or requests.Session()
        self.logger = logger or LOG
        self._token = acled_auth.get_access_token()

    # ------------------------------------------------------------------
    # Network helpers
    # ------------------------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    def _fetch_page(self, params: Dict[str, Any]) -> Dict[str, Any]:
        attempt = 0
        params = dict(params)
        if "_format" not in params:
            params["_format"] = ACLED_DEFAULT_FORMAT
        url = self.base_url
        while True:
            attempt += 1
            start = time.time()
            response = self.session.get(url, params=params, headers=self._headers(), timeout=self.timeout)
            elapsed = time.time() - start
            _write_acled_http_diag(status=response.status_code, url=str(response.url))
            safe_url = f"{_safe_base_url(url)}?..."
            self.logger.info(
                "ACLED HTTP request",
                extra={
                    "url": safe_url,
                    "status": response.status_code,
                    "params_keys": sorted(params.keys()),
                },
            )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                try:
                    wait = float(retry_after)
                except (TypeError, ValueError):
                    wait = min(60.0, 2 ** attempt)
                self.logger.debug(
                    "ACLED rate limited",
                    extra={"attempt": attempt, "wait_seconds": wait, "status": response.status_code},
                )
                if attempt >= self.max_retries:
                    response.raise_for_status()
                time.sleep(wait)
                continue
            if response.status_code >= 500:
                wait = min(60.0, 2 ** attempt)
                self.logger.debug(
                    "ACLED server error",
                    extra={
                        "attempt": attempt,
                        "status": response.status_code,
                        "wait_seconds": wait,
                        "elapsed": round(elapsed, 3),
                    },
                )
                if attempt >= self.max_retries:
                    response.raise_for_status()
                time.sleep(wait)
                continue
            if response.status_code != 200:
                snippet = ""
                try:
                    snippet = response.text[:500]
                except Exception:  # pragma: no cover - defensive
                    snippet = "<unreadable response>"
                self.logger.error(
                    "ACLED HTTP error",
                    extra={"status": response.status_code, "url": safe_url, "body_snippet": snippet},
                )
                raise RuntimeError(f"ACLED read failed: HTTP {response.status_code}")
            response.raise_for_status()
            try:
                payload = response.json()
            except ValueError as exc:  # pragma: no cover - requests already validated status
                self.logger.debug("ACLED response was not valid JSON", extra={"error": str(exc)})
                raise RuntimeError("ACLED response was not valid JSON") from exc
            if payload.get("status") not in (200, "200", None):
                self.logger.error(
                    "ACLED API returned non-200 status in JSON",
                    extra={"json_status": payload.get("status"), "url": safe_url},
                )
                raise RuntimeError(f"ACLED read failed: JSON status={payload.get('status')}")
            self.logger.debug(
                "Fetched ACLED page",
                extra={
                    "status": response.status_code,
                    "elapsed": round(elapsed, 3),
                    "content_length": int(response.headers.get("Content-Length", 0) or 0),
                },
            )
            return payload if isinstance(payload, dict) else {}

    def fetch_events(
        self,
        start_date: str | date,
        end_date: str | date,
        *,
        countries: Optional[Sequence[str]] = None,
        fields: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Fetch ACLED events within ``start_date`` → ``end_date`` inclusive."""

        if self.use_stub:
            self.logger.debug("ACLED client in stub mode; returning empty frame")
            return pd.DataFrame(columns=self._DEFAULT_FIELDS)

        start = pd.to_datetime(start_date, utc=True).normalize()
        end = pd.to_datetime(end_date, utc=True).normalize()
        if start > end:
            start, end = end, start

        if isinstance(countries, str):
            countries = [countries]

        selected_fields = list(fields or self.fields or self._DEFAULT_FIELDS)
        params: Dict[str, Any] = {
            "event_date": f"{start.strftime('%Y-%m-%d')}|{end.strftime('%Y-%m-%d')}",
            "event_date_where": "BETWEEN",
            "page": 1,
            "limit": self.page_size,
            "_format": ACLED_DEFAULT_FORMAT,
            "fields": "|".join(selected_fields),
        }
        if countries:
            params["iso3"] = ",".join(sorted({c.strip().upper() for c in countries if c}))

        safe_params = {k: v for k, v in params.items() if k not in {"access_token"}}
        self.logger.debug(
            "Starting ACLED fetch",
            extra={
                "url": self.base_url,
                "start": start.strftime("%Y-%m-%d"),
                "end": end.strftime("%Y-%m-%d"),
                "page_size": self.page_size,
                "countries": params.get("iso3"),
                "query": safe_params,
            },
        )

        records: List[Dict[str, Any]] = []
        page = 1
        while True:
            params["page"] = page
            payload = self._fetch_page(params)
            data = payload.get("data") or payload.get("results") or []
            if not isinstance(data, list):
                raise RuntimeError("Unexpected ACLED payload structure")
            if page == 1 and not data and "fields" in params:
                self.logger.debug(
                    "ACLED empty data with fields set; retrying without fields",
                    extra={"params_keys": sorted(params.keys())},
                )
                params.pop("fields", None)
                payload = self._fetch_page(params)
                data = payload.get("data") or payload.get("results") or []
            if not data:
                break
            records.extend(data)
            self.logger.debug(
                "Fetched ACLED records",
                extra={"page": page, "page_rows": len(data), "total_rows": len(records)},
            )
            if len(data) < self.page_size:
                break
            page += 1

        self.logger.debug(
            "Completed ACLED fetch",
            extra={"pages": page, "rows": len(records)},
        )

        frame = pd.DataFrame(records)
        if frame.empty:
            return frame.reindex(columns=selected_fields)

        for column in selected_fields:
            if column not in frame.columns:
                frame[column] = pd.NA

        frame = frame.reindex(columns=selected_fields)
        frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce", utc=True).dt.tz_convert(None)
        original_iso_missing = frame["iso3"].isna()
        frame["iso3"] = frame["iso3"].astype(str).str.strip().str.upper()
        iso_mask = original_iso_missing | frame["iso3"].isin({"", "NAN", "NONE", "NULL"})
        frame.loc[iso_mask, "iso3"] = pd.NA
        frame["country"] = frame["country"].astype(str).str.strip()
        missing_iso_mask = frame["iso3"].isna()
        if missing_iso_mask.any():
            filled_iso = frame.loc[missing_iso_mask, "country"].map(lambda value: to_iso3(value))
            filled_count = int(filled_iso.notna().sum())
            frame.loc[missing_iso_mask, "iso3"] = filled_iso
            if filled_count > 0:
                self.logger.info(
                    "ACLED iso3 filled from country",
                    extra={"filled": filled_count},
                )
        frame["fatalities"] = (
            pd.to_numeric(frame["fatalities"], errors="coerce").fillna(0).astype("int64")
        )
        frame = frame.dropna(subset=["event_date", "iso3"])
        frame = frame.sort_values(["event_date", "iso3"]).reset_index(drop=True)
        return frame

    def monthly_fatalities(
        self,
        start_date: str | date,
        end_date: str | date,
        *,
        countries: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Aggregate fatalities per ISO3/month bucket."""

        frame = self.fetch_events(start_date, end_date, countries=countries)
        if frame.empty:
            result = pd.DataFrame(
                columns=["iso3", "month", "fatalities", "source", "updated_at"],
            )
            return result

        if "event_date" not in frame.columns:
            raise RuntimeError("ACLED monthly_fatalities expected an 'event_date' column")

        frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce", utc=False)
        frame = frame.dropna(subset=["event_date"]).copy()

        if frame.empty:
            result = pd.DataFrame(
                columns=["iso3", "month", "fatalities", "source", "updated_at"],
            )
            return result

        if "iso3" not in frame.columns:
            raise RuntimeError("ACLED monthly_fatalities expected an 'iso3' column")
        frame["iso3"] = frame["iso3"].astype(str).str.upper().str.strip()
        frame = frame[frame["iso3"] != ""].copy()

        if frame.empty:
            result = pd.DataFrame(
                columns=["iso3", "month", "fatalities", "source", "updated_at"],
            )
            return result

        frame["fatalities"] = (
            pd.to_numeric(frame.get("fatalities"), errors="coerce").fillna(0)
        )

        if countries:
            if isinstance(countries, str):
                countries = [countries]
            allowed = {c.strip().upper() for c in countries if c}
            if allowed:
                frame = frame[frame["iso3"].isin(allowed)]
                if frame.empty:
                    result = pd.DataFrame(
                        columns=["iso3", "month", "fatalities", "source", "updated_at"],
                    )
                    return result

        frame["month"] = (
            frame["event_date"].dt.to_period("M").dt.to_timestamp(how="start")
        )
        grouped = frame.groupby(["iso3", "month"], as_index=False)["fatalities"].sum()
        grouped["fatalities"] = grouped["fatalities"].fillna(0).astype("int64")
        grouped["iso3"] = grouped["iso3"].astype(str).str.upper().str.strip()
        grouped["source"] = "ACLED"
        grouped["updated_at"] = pd.Timestamp.now(tz=timezone.utc)
        grouped = grouped.sort_values(["iso3", "month"]).reset_index(drop=True)

        preview_head = grouped.head(3).to_dict("records")
        preview_tail = grouped.tail(3).to_dict("records") if len(grouped) > 3 else []
        self.logger.debug(
            "Grouped to monthly fatalities",
            extra={
                "rows": len(grouped),
                "head": preview_head,
                "tail": preview_tail,
            },
        )
        return grouped


def main() -> bool:
    if os.getenv("RESOLVER_SKIP_ACLED") == "1":
        dbg("RESOLVER_SKIP_ACLED=1 — skipping ACLED pull")
        _write_header_only(OUT_PATH)
        return False

    try:
        rows = collect_rows()
    except Exception as exc:  # fail-soft
        dbg(f"collect_rows failed: {exc}")
        _write_header_only(OUT_PATH)
        return False

    if not rows:
        dbg("no ACLED rows collected; writing header only")
        _write_header_only(OUT_PATH)
        return False

    _write_rows(rows, OUT_PATH)
    dbg(f"wrote {len(rows)} ACLED rows")
    return True


if __name__ == "__main__":
    main()
