# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Resolver data-explorer routes (/v1/resolver/*).

Endpoint functions moved verbatim from pythia.api.app (July 2026
decomposition); shared helpers come from pythia.api.core.
"""

import logging
import re

from fastapi import APIRouter, HTTPException, Query

from pythia.api.core import (
    _con,
    _rows_from_cursor,
    _table_columns,
    _table_exists,
    _validate_iso3_param,
)
from resolver.query.resolver_ui import get_connector_last_updated, get_country_facts

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/v1/resolver/connector_status")
def get_resolver_connector_status():
    con = _con()
    rows, diagnostics = get_connector_last_updated(con)
    summary = ", ".join(
        f"{row.get('source')}={row.get('last_updated')}" for row in rows
    )
    logger.info("Resolver connector status rows=%s updates=%s", len(rows), summary)
    return {"rows": rows, "diagnostics": diagnostics}


@router.get("/v1/resolver/country_facts")
def get_resolver_country_facts(
    iso3: str = Query(..., description="ISO3 country code"),
    limit: int = Query(5000, description="Maximum rows to return"),
):
    iso3_value = (iso3 or "").strip().upper()
    if not re.fullmatch(r"[A-Z]{3}", iso3_value or ""):
        raise HTTPException(status_code=400, detail="iso3 must be a 3-letter code")
    con = _con()
    rows, diagnostics = get_country_facts(con, iso3_value, limit=limit)
    return {"rows": rows, "iso3": iso3_value, "diagnostics": diagnostics}


# ---------------------------------------------------------------------------
# Resolver data explorer endpoints
# ---------------------------------------------------------------------------

_DB_SUMMARY_TABLES = [
    # (table_name, date_column_candidates, has_iso3)
    # Freshness columns: prefer ingestion-time over data-period columns
    ("facts_resolved", ["created_at", "as_of_date"], True),
    ("facts_deltas", ["created_at"], True),
    ("acled_monthly_fatalities", ["updated_at"], True),
    ("conflict_forecasts", ["created_at", "forecast_issue_date"], True),
    ("reliefweb_reports", ["fetched_at", "published_date"], True),
    ("acled_political_events", ["fetched_at", "event_date"], True),
    ("acaps_inform_severity", ["fetched_at", "snapshot_date"], True),
    ("acaps_risk_radar", ["fetched_at"], True),
    ("acaps_daily_monitoring", ["fetched_at", "entry_date"], True),
    ("acaps_humanitarian_access", ["fetched_at", "snapshot_date"], True),
    ("seasonal_forecasts", ["created_at", "forecast_issue_date"], True),
    ("enso_state", ["created_at", "fetch_date"], False),
    ("seasonal_tc_outlooks", ["fetched_at"], False),
    ("seasonal_tc_context_cache", ["fetched_at"], True),
    ("hdx_signals", ["fetched_at", "signal_date"], True),
    ("crisiswatch_entries", ["fetched_at"], True),
]


@router.get("/v1/resolver/db_summary")
def get_resolver_db_summary():
    con = _con()
    tables = []
    for tbl, date_candidates, has_iso3 in _DB_SUMMARY_TABLES:
        if not _table_exists(con, tbl):
            continue
        try:
            row_count = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        except Exception:
            row_count = 0
        last_updated = None
        cols = _table_columns(con, tbl)
        for cand in date_candidates:
            if cand.lower() in cols:
                try:
                    val = con.execute(
                        f"SELECT MAX({cand}) FROM {tbl} WHERE {cand} IS NOT NULL"
                    ).fetchone()
                    if val and val[0] is not None:
                        last_updated = str(val[0])[:10]
                        break
                except Exception:
                    pass
        tables.append({
            "name": tbl,
            "row_count": row_count,
            "last_updated": last_updated,
            "has_iso3": has_iso3,
        })
    return {"tables": tables}


def _resolver_query(table: str, iso3: str | None, limit: int,
                    order_by: str = "", extra_where: str = "",
                    extra_params: list | None = None,
                    exclude_cols: set[str] | None = None) -> dict:
    """Generic helper for resolver data-explorer endpoints."""
    con = _con()
    if not _table_exists(con, table):
        return {"rows": []}
    cols = _table_columns(con, table)
    if exclude_cols:
        select_cols = ", ".join(
            c for c in sorted(cols) if c not in {e.lower() for e in exclude_cols}
        )
    else:
        select_cols = "*"
    sql = f"SELECT {select_cols} FROM {table}"
    params: list = list(extra_params or [])
    clauses: list[str] = []
    if iso3 and "iso3" in cols:
        clauses.append("iso3 = ?")
        params.append(iso3.upper())
    if extra_where:
        clauses.append(extra_where)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    if order_by:
        sql += f" ORDER BY {order_by}"
    sql += f" LIMIT {min(limit, 5000)}"
    return {"rows": _rows_from_cursor(con.execute(sql, params))}


@router.get("/v1/resolver/facts_deltas")
def get_resolver_facts_deltas(
    iso3: str | None = Query(None), limit: int = Query(500),
):
    return _resolver_query("facts_deltas", iso3, limit, order_by="created_at DESC")


@router.get("/v1/resolver/acled_monthly_fatalities")
def get_resolver_acled_monthly_fatalities(
    iso3: str | None = Query(None), limit: int = Query(500),
):
    return _resolver_query("acled_monthly_fatalities", iso3, limit,
                           order_by="year DESC, month DESC")


@router.get("/v1/resolver/conflict_forecasts")
def get_resolver_conflict_forecasts(
    iso3: str | None = Query(None),
    source: str | None = Query(None),
    limit: int = Query(500),
):
    extra_where = ""
    extra_params: list = []
    if source:
        extra_where = "source = ?"
        extra_params.append(source)
    return _resolver_query("conflict_forecasts", iso3, limit,
                           order_by="forecast_issue_date DESC",
                           extra_where=extra_where, extra_params=extra_params)


@router.get("/v1/resolver/reliefweb_reports")
def get_resolver_reliefweb_reports(
    iso3: str | None = Query(None),
    include_body: bool = Query(False),
    limit: int = Query(100),
):
    exclude = {"body"} if not include_body else set()
    return _resolver_query("reliefweb_reports", iso3, limit,
                           order_by="date DESC", exclude_cols=exclude)


@router.get("/v1/resolver/acled_political_events")
def get_resolver_acled_political_events(
    iso3: str | None = Query(None), limit: int = Query(500),
):
    return _resolver_query("acled_political_events", iso3, limit,
                           order_by="event_date DESC")


@router.get("/v1/resolver/acaps")
def get_resolver_acaps(
    iso3: str | None = Query(None),
    dataset: str = Query("inform_severity",
                         description="inform_severity|risk_radar|daily_monitoring|humanitarian_access"),
    limit: int = Query(500),
):
    table_map = {
        "inform_severity": ("acaps_inform_severity", "snapshot_date DESC"),
        "risk_radar": ("acaps_risk_radar", "fetched_at DESC"),
        "daily_monitoring": ("acaps_daily_monitoring", "entry_date DESC"),
        "humanitarian_access": ("acaps_humanitarian_access", "snapshot_date DESC"),
    }
    tbl, order = table_map.get(dataset, ("acaps_inform_severity", "snapshot_date DESC"))
    return _resolver_query(tbl, iso3, limit, order_by=order)


@router.get("/v1/resolver/seasonal_forecasts")
def get_resolver_seasonal_forecasts(
    iso3: str | None = Query(None), limit: int = Query(500),
):
    return _resolver_query("seasonal_forecasts", iso3, limit,
                           order_by="forecast_issue_date DESC")


@router.get("/v1/resolver/hdx_signals")
def get_resolver_hdx_signals(
    iso3: str | None = Query(None), limit: int = Query(500),
):
    return _resolver_query("hdx_signals", iso3, limit,
                           order_by="signal_date DESC")


@router.get("/v1/resolver/crisiswatch")
def get_resolver_crisiswatch(
    iso3: str | None = Query(None), limit: int = Query(500),
):
    return _resolver_query("crisiswatch_entries", iso3, limit,
                           order_by="year DESC, month DESC")


@router.get("/v1/resolver/enso_state")
def get_resolver_enso_state(limit: int = Query(10)):
    return _resolver_query("enso_state", None, limit,
                           order_by="fetch_date DESC")


@router.get("/v1/resolver/seasonal_tc_outlooks")
def get_resolver_seasonal_tc_outlooks(limit: int = Query(50)):
    return _resolver_query("seasonal_tc_outlooks", None, limit,
                           order_by="fetched_at DESC")


# ---------------------------------------------------------------------------
# Source-level data explorer (accordion page)
# ---------------------------------------------------------------------------

# Best column for "when was this data last ingested" per table.
_FRESHNESS_CANDIDATES = ("fetched_at", "created_at", "stored_at",
                         "ingested_at", "fetch_date", "updated_at")

_SOURCE_REGISTRY: dict[str, dict] = {
    # --- Resolution Data (facts_resolved, filtered by publisher) ---
    "ifrc":             {"table": "facts_resolved",
                         "filter": "LOWER(publisher) IN ('ifrc', 'ifrc_go', 'ifrc_montandon')",
                         "columns": ["iso3", "hazard_code", "metric", "ym",
                                     "value", "as_of_date", "publisher", "source_id"],
                         "order": "created_at DESC"},
    "idmc":             {"table": "facts_resolved",
                         "filter": "LOWER(publisher) IN ('idmc')",
                         "columns": ["iso3", "hazard_code", "metric", "ym",
                                     "value", "as_of_date", "publisher", "source_id"],
                         "order": "created_at DESC"},
    "acled":            {"table": "facts_resolved",
                         "filter": "LOWER(publisher) IN ('acled')",
                         "columns": ["iso3", "hazard_code", "metric", "ym",
                                     "value", "as_of_date", "publisher", "source_id"],
                         "order": "created_at DESC"},
    "gdacs":            {"table": "facts_resolved",
                         "filter": "publisher = 'GDACS / JRC'",
                         "columns": ["iso3", "hazard_code", "metric", "ym",
                                     "value", "alertlevel", "as_of_date", "publisher"],
                         "order": "created_at DESC"},
    "fewsnet":          {"table": "facts_resolved",
                         "filter": "publisher = 'FEWS NET'",
                         "columns": ["iso3", "hazard_code", "metric", "ym",
                                     "value", "as_of_date", "publisher"],
                         "order": "created_at DESC"},
    "ipc_api":          {"table": "facts_resolved",
                         "filter": "publisher = 'IPC'",
                         "columns": ["iso3", "hazard_code", "metric", "ym",
                                     "value", "as_of_date", "publisher"],
                         "order": "created_at DESC"},
    "acled_fatalities":  {"table": "acled_monthly_fatalities",
                         "columns": ["iso3", "month", "fatalities", "source"],
                         "order": "month DESC"},
    # --- Conflict Forecasts ---
    "views":            {"table": "conflict_forecasts",
                         "filter": "source = 'VIEWS'",
                         "columns": ["iso3", "hazard_code", "metric",
                                     "lead_months", "value",
                                     "forecast_issue_date", "target_month"],
                         "order": "forecast_issue_date DESC"},
    "conflictforecast": {"table": "conflict_forecasts",
                         "filter": "source = 'conflictforecast_org'",
                         "columns": ["iso3", "hazard_code", "metric",
                                     "lead_months", "value",
                                     "forecast_issue_date", "target_month"],
                         "order": "forecast_issue_date DESC"},
    "acled_cast":       {"table": "conflict_forecasts",
                         "filter": "source = 'ACLED_CAST'",
                         "columns": ["iso3", "hazard_code", "metric",
                                     "lead_months", "value",
                                     "forecast_issue_date", "target_month"],
                         "order": "forecast_issue_date DESC"},
    "crisiswatch":      {"table": "crisiswatch_entries",
                         "columns": ["iso3", "year", "month", "arrow",
                                     "alert_type", "country_name"],
                         "order": "year DESC, month DESC"},
    # --- Weather and Climate ---
    "nmme":             {"table": "seasonal_forecasts",
                         "columns": ["iso3", "variable", "lead_months",
                                     "anomaly_value", "tercile_category",
                                     "forecast_issue_date"],
                         "order": "forecast_issue_date DESC"},
    "enso":             {"table": "enso_state", "has_iso3": False,
                         "columns": ["fetch_date", "enso_phase",
                                     "nino34_anomaly", "iod_phase"],
                         "order": "fetch_date DESC"},
    "seasonal_tc":      {"table": "seasonal_tc_outlooks", "has_iso3": False,
                         "columns": ["basin", "source", "forecast_season",
                                     "named_storms_forecast", "category",
                                     "fetched_at"],
                         "order": "fetched_at DESC"},
    "tc_context":       {"table": "seasonal_tc_context_cache",
                         "columns": ["iso3", "context_text"],
                         "order": "fetched_at DESC"},
    # --- Situation Reports ---
    "reliefweb":        {"table": "reliefweb_reports",
                         "columns": ["iso3", "title", "sources",
                                     "published_date", "url"],
                         "order": "published_date DESC",
                         "exclude_default": {"body_excerpt"}},
    "acaps_daily":      {"table": "acaps_daily_monitoring",
                         "columns": ["iso3", "entry_date",
                                     "latest_developments", "source"],
                         "order": "entry_date DESC"},
    "acled_political":  {"table": "acled_political_events",
                         "columns": ["iso3", "event_date", "event_type",
                                     "sub_event_type", "fatalities",
                                     "actor1", "location"],
                         "order": "event_date DESC"},
    # --- Other Alerts ---
    "hdx_signals":      {"table": "hdx_signals",
                         "columns": ["iso3", "hazard_code", "indicator",
                                     "concern_level", "indicator_value",
                                     "signal_date"],
                         "order": "signal_date DESC"},
    "acaps_risk_radar": {"table": "acaps_risk_radar",
                         "columns": ["iso3", "risk_title", "risk_level",
                                     "risk_type", "risk_trend"],
                         "order": "fetched_at DESC"},
    "gdelt":            {"table": "gdelt_conflict_indicators",
                         "columns": ["iso3", "event_date", "total_events",
                                     "tier1_events", "tier2_events",
                                     "tier3_events", "avg_goldstein",
                                     "avg_tone_conflict"],
                         "order": "event_date DESC"},
    # --- Other ---
    "acaps_inform":     {"table": "acaps_inform_severity",
                         "columns": ["iso3", "crisis_name", "severity_score",
                                     "severity_category", "snapshot_date"],
                         "order": "snapshot_date DESC"},
    "acaps_access":     {"table": "acaps_humanitarian_access",
                         "columns": ["iso3", "access_score",
                                     "access_category", "snapshot_date"],
                         "order": "snapshot_date DESC"},
}

_SOURCE_LABELS: dict[str, str] = {
    "ifrc": "IFRC", "idmc": "IDMC", "acled": "ACLED",
    "gdacs": "GDACS", "fewsnet": "FEWS NET",
    "acled_fatalities": "ACLED Monthly Fatalities",
    "views": "VIEWS", "conflictforecast": "conflictforecast.org",
    "acled_cast": "ACLED CAST", "crisiswatch": "CrisisWatch",
    "nmme": "NMME Seasonal", "enso": "ENSO State",
    "seasonal_tc": "Seasonal TC Outlooks", "tc_context": "TC Context",
    "reliefweb": "ReliefWeb", "acaps_daily": "ACAPS Daily Monitoring",
    "acled_political": "ACLED Political Events",
    "hdx_signals": "HDX Signals", "acaps_risk_radar": "ACAPS Risk Radar",
    "gdelt": "GDELT Conflict Events",
    "acaps_inform": "ACAPS INFORM Severity",
    "acaps_access": "ACAPS Humanitarian Access",
    "ipc_api": "IPC API",
}

_SOURCE_CATEGORIES: dict[str, str] = {
    "ifrc": "resolution_data", "idmc": "resolution_data",
    "acled": "resolution_data", "gdacs": "resolution_data",
    "fewsnet": "resolution_data", "acled_fatalities": "resolution_data",
    "views": "conflict_forecasts", "conflictforecast": "conflict_forecasts",
    "acled_cast": "conflict_forecasts", "crisiswatch": "conflict_forecasts",
    "nmme": "weather_climate", "enso": "weather_climate",
    "seasonal_tc": "weather_climate", "tc_context": "weather_climate",
    "reliefweb": "situation_reports", "acaps_daily": "situation_reports",
    "acled_political": "situation_reports",
    "hdx_signals": "other_alerts", "acaps_risk_radar": "other_alerts",
    "gdelt": "other_alerts",
    "acaps_inform": "other", "acaps_access": "other",
    "ipc_api": "resolution_data",
}


def _best_freshness_column(con, table: str) -> str | None:
    """Return the best column for 'last ingested' timestamp."""
    cols = _table_columns(con, table)
    for candidate in _FRESHNESS_CANDIDATES:
        if candidate in cols:
            return candidate
    return None


def _source_freshness(con, spec: dict) -> str | None:
    """Compute last-updated date for a source using ingestion-time columns."""
    table = spec["table"]
    if not _table_exists(con, table):
        return None
    ts_col = _best_freshness_column(con, table)
    if not ts_col:
        return None
    filt = spec.get("filter", "")
    sql = f"SELECT MAX({ts_col}) FROM {table}"
    if filt:
        sql += f" WHERE {filt}"
    try:
        val = con.execute(sql).fetchone()
        if val and val[0] is not None:
            return str(val[0])[:10]
    except Exception:
        pass
    return None


def _source_row_count(con, spec: dict, iso3: str | None = None) -> int:
    """Count rows for a source, optionally filtered by iso3."""
    table = spec["table"]
    if not _table_exists(con, table):
        return 0
    filt = spec.get("filter", "")
    clauses: list[str] = []
    params: list = []
    if iso3 and spec.get("has_iso3", True):
        cols = _table_columns(con, table)
        if "iso3" in cols:
            clauses.append("iso3 = ?")
            params.append(iso3.strip().upper())
    if filt:
        clauses.append(filt)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    try:
        return con.execute(f"SELECT COUNT(*) FROM {table}{where}", params).fetchone()[0]
    except Exception:
        return 0


def _validated_columns(con, table: str, curated: list[str]) -> list[str]:
    """Return only curated columns that actually exist in the table schema."""
    actual = _table_columns(con, table)
    return [c for c in curated if c.lower() in actual]


@router.get("/v1/resolver/source_inventory")
def get_resolver_source_inventory(
    iso3: str | None = Query(None, description="ISO3 country code"),
):
    """Per-source metadata for the accordion data explorer."""
    con = _con()
    iso3_val = _validate_iso3_param(iso3)
    sources = []
    for key, spec in _SOURCE_REGISTRY.items():
        has_iso3 = spec.get("has_iso3", True)
        table = spec["table"]
        exists = _table_exists(con, table)
        sources.append({
            "key": key,
            "label": _SOURCE_LABELS.get(key, key),
            "category": _SOURCE_CATEGORIES.get(key, "other"),
            "last_updated": _source_freshness(con, spec) if exists else None,
            "global_rows": _source_row_count(con, spec) if exists else 0,
            "country_rows": (
                _source_row_count(con, spec, iso3_val)
                if exists and iso3_val and has_iso3 else None
            ),
            "has_iso3": has_iso3,
        })
    return {"sources": sources}


@router.get("/v1/resolver/source_data")
def get_resolver_source_data(
    source: str = Query(..., description="Source key from registry"),
    iso3: str | None = Query(None, description="ISO3 country code"),
    limit: int = Query(500, description="Max rows", ge=1, le=5000),
    all_columns: bool = Query(False, description="SELECT * instead of curated"),
    include_body: bool = Query(False, description="Include body text (reliefweb)"),
):
    """Lazy-load rows for a single source. Called when accordion expands."""
    spec = _SOURCE_REGISTRY.get(source)
    if not spec:
        raise HTTPException(status_code=400, detail=f"Unknown source: {source}")
    con = _con()
    table = spec["table"]
    if not _table_exists(con, table):
        return {"rows": [], "columns": []}

    actual_cols = _table_columns(con, table)

    if all_columns:
        exclude = set()
        if source == "reliefweb" and not include_body:
            exclude.add("body_excerpt")
        select_list = sorted(c for c in actual_cols if c not in exclude)
    else:
        select_list = _validated_columns(con, table, spec["columns"])
        if not select_list:
            select_list = sorted(actual_cols)

    select_sql = ", ".join(select_list)
    clauses: list[str] = []
    params: list = []
    filt = spec.get("filter", "")
    if filt:
        clauses.append(filt)
    iso3_val = _validate_iso3_param(iso3)
    if iso3_val and spec.get("has_iso3", True) and "iso3" in actual_cols:
        clauses.append("iso3 = ?")
        params.append(iso3_val)

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    order = spec.get("order", "")
    # Validate order columns exist
    if order:
        order_cols_valid = all(
            col.strip().lower().replace(" desc", "").replace(" asc", "") in actual_cols
            for col in order.split(",")
        )
        if not order_cols_valid:
            order = ""
    order_sql = f" ORDER BY {order}" if order else ""
    sql = f"SELECT {select_sql} FROM {table}{where}{order_sql} LIMIT {min(limit, 5000)}"

    try:
        rows = _rows_from_cursor(con.execute(sql, params))
    except Exception as exc:
        logger.warning("source_data query failed for %s: %s", source, exc)
        return {"rows": [], "columns": select_list}

    return {"rows": rows, "columns": select_list}
