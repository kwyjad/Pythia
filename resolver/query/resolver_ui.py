# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Resolver UI query helpers for lightweight API endpoints."""

from __future__ import annotations

from datetime import date, datetime
import logging
import math
from typing import Any


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:  # pragma: no cover - silence library default
    LOGGER.addHandler(logging.NullHandler())


HAZARD_LABELS = {
    "ACE": "Armed Conflict",
    "CONFLICT": "Armed Conflict",
    "DI": "Displacement Inflow",
    "DR": "Drought",
    "FL": "Flood",
    "HW": "Heatwave",
    "TC": "Tropical Cyclone",
}


def _table_exists(conn, table: str) -> bool:
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE LOWER(table_name) = LOWER(?)",
            [table],
        ).fetchone()
        return bool(row and row[0])
    except Exception:
        pass

    try:
        df = conn.execute("PRAGMA show_tables").fetchdf()
    except Exception:
        return False
    if df.empty:
        return False
    first_col = df.columns[0]
    return df[first_col].astype(str).str.lower().eq(table.lower()).any()


def _table_columns(conn, table: str) -> set[str]:
    try:
        df = conn.execute(f"PRAGMA table_info('{table}')").fetchdf()
    except Exception:
        return set()
    if df.empty or "name" not in df.columns:
        return set()
    return set(df["name"].astype(str).str.lower().tolist())


def _format_date(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if len(text) >= 10:
        return text[:10]
    return text or None


def _timestamp_expr(column: str) -> str:
    return f"TRY_CAST(CAST({column} AS VARCHAR) AS TIMESTAMP)"


def _ym_proxy_expr() -> str:
    return "TRY_CAST(CAST(ym || '-01' AS VARCHAR) AS TIMESTAMP)"


def _table_has_rows(conn, table: str) -> bool:
    try:
        row = conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone()
        return row is not None
    except Exception:
        return False


def _has_acled_monthly_table(conn) -> bool:
    return _table_exists(conn, "acled_monthly_fatalities")


def _acled_monthly_columns(conn) -> set[str]:
    return _table_columns(conn, "acled_monthly_fatalities")


def _pick_facts_source(conn) -> tuple[dict[str, Any], dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "facts_source_table": None,
        "fallback_used": False,
        "missing_tables_checked": [],
        "notes": [],
    }
    if conn is None:
        diagnostics["notes"].append("conn_missing")
        return {}, diagnostics

    candidates = [
        {
            "table": "facts_resolved",
            "value_expr": "value",
            "has_hazard_label": True,
            "required": {"iso3", "ym", "hazard_code", "metric", "value", "source_id"},
        },
        {
            "table": "facts_monthly_deltas",
            "value_expr": "value",
            "has_hazard_label": False,
            "required": {"iso3", "ym", "hazard_code", "metric", "value", "source_id"},
        },
        {
            "table": "facts_deltas",
            "value_expr": None,
            "has_hazard_label": False,
            "required": {"iso3", "ym", "hazard_code", "metric", "source_id"},
        },
    ]

    for candidate in candidates:
        table = candidate["table"]
        if not _table_exists(conn, table):
            diagnostics["missing_tables_checked"].append(table)
            continue
        columns = _table_columns(conn, table)
        if not candidate["required"].issubset(columns):
            diagnostics["notes"].append(f"{table}_columns_missing")
            continue
        value_expr = candidate.get("value_expr")
        if table == "facts_deltas":
            if "value_new" in columns and "value_stock" in columns:
                value_expr = "COALESCE(value_new, value_stock)"
            elif "value_new" in columns:
                value_expr = "value_new"
            elif "value_stock" in columns:
                value_expr = "value_stock"
            else:
                diagnostics["notes"].append("facts_deltas_value_missing")
                continue
        if not _table_has_rows(conn, table):
            diagnostics["notes"].append(f"{table}_empty")
            continue
        diagnostics["facts_source_table"] = table
        diagnostics["fallback_used"] = table != "facts_resolved"
        return {
            "table": table,
            "value_expr": value_expr,
            "has_hazard_label": candidate["has_hazard_label"],
        }, diagnostics

    diagnostics["notes"].append("facts_source_unavailable")
    return {}, diagnostics


def _metric_display(metric: str | None) -> str | None:
    if metric is None:
        return None
    metric_str = str(metric).strip()
    if not metric_str or metric_str.lower() == "nan":
        return None
    if "fatalit" in metric_str.lower():
        return "FATALITIES"
    if metric_str.lower() in {"pa", "people_affected", "affected", "peopleaffected"}:
        return "PA"
    return metric_str.upper()


def _hazard_display(hazard_label: str | None, hazard_code: str | None) -> str | None:
    label = str(hazard_label).strip() if hazard_label is not None else ""
    if label.lower() == "nan":
        label = ""
    if label:
        return label
    code = (hazard_code or "").strip().upper()
    if not code:
        return None
    return HAZARD_LABELS.get(code, code)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _status_from_facts(
    conn, label: str, source_filter_sql: str, params: list[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    status: dict[str, Any] = {
        "source": label,
        "last_updated": None,
        "rows_scanned": 0,
        "diagnostics": {
            "table_used": None,
            "date_expr": "coalesce(created_at, publication_date, as_of_date, as_of, ym_proxy)",
            "filter_used": label,
        },
    }
    diagnostics: dict[str, Any] = {
        "facts_source_table": None,
        "fallback_used": False,
        "missing_tables_checked": [],
        "notes": [],
    }
    if conn is None:
        diagnostics["notes"].append("conn_missing")
        return status, diagnostics

    source_info, diagnostics = _pick_facts_source(conn)
    table = source_info.get("table") if source_info else None
    status["diagnostics"]["table_used"] = table
    if not source_info or not table:
        return status, diagnostics

    columns = _table_columns(conn, table)
    if "source_id" not in columns:
        diagnostics["notes"].append("source_id_missing")
        return status, diagnostics

    date_exprs: list[str] = []
    for candidate in ("created_at", "publication_date", "as_of_date", "as_of"):
        if candidate in columns:
            date_exprs.append(_timestamp_expr(candidate))
    if "ym" in columns:
        date_exprs.append(_ym_proxy_expr())
    date_expr = f"COALESCE({', '.join(date_exprs)})" if date_exprs else "NULL"

    query = f"""
        SELECT
          MAX({date_expr}) AS last_updated,
          COUNT(*) AS rows_scanned
        FROM {table}
        WHERE {source_filter_sql}
    """
    row = conn.execute(query, params).fetchone()
    max_date = row[0] if row else None
    count = int(row[1] or 0) if row else 0
    status["last_updated"] = _format_date(max_date)
    status["rows_scanned"] = count
    return status, diagnostics


def _status_from_acled_table(conn) -> dict[str, Any] | None:
    if conn is None:
        return None
    if not _table_exists(conn, "acled_monthly_fatalities"):
        return None

    columns = _acled_monthly_columns(conn)
    updated_expr = _timestamp_expr("updated_at") if "updated_at" in columns else "NULL"
    created_expr = _timestamp_expr("created_at") if "created_at" in columns else "NULL"
    ingested_expr = _timestamp_expr("ingested_at") if "ingested_at" in columns else "NULL"
    month_expr = "TRY_CAST(CAST(month AS VARCHAR) AS DATE)" if "month" in columns else "NULL"

    query = f"""
        SELECT
          MAX({updated_expr}) AS updated_at_max,
          MAX({created_expr}) AS created_at_max,
          MAX({ingested_expr}) AS ingested_at_max,
          MAX({month_expr}) AS month_max,
          COUNT(*) AS rows_scanned
        FROM acled_monthly_fatalities
    """
    row = conn.execute(query).fetchone()
    updated_at_max, created_at_max, ingested_at_max, month_max, count = row or (
        None,
        None,
        None,
        None,
        0,
    )

    date_column_used = "none"
    last_updated = None
    if updated_at_max is not None:
        date_column_used = "updated_at"
        last_updated = updated_at_max
    elif created_at_max is not None:
        date_column_used = "created_at"
        last_updated = created_at_max
    elif ingested_at_max is not None:
        date_column_used = "ingested_at"
        last_updated = ingested_at_max
    elif month_max is not None:
        date_column_used = "month"
        last_updated = month_max

    return {
        "source": "ACLED",
        "last_updated": _format_date(last_updated),
        "rows_scanned": int(count or 0),
        "diagnostics": {
            "table_used": "acled_monthly_fatalities",
            "date_column_used": date_column_used,
        },
    }


def get_connector_last_updated(conn) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    acled_status = _status_from_acled_table(conn)

    acled_facts_clause = (
        "(lower(source_id) LIKE '%acled%' OR lower(source_id) = 'acled_client' "
        "OR ((source_id IS NULL OR TRIM(CAST(source_id AS VARCHAR)) = '') "
        "AND lower(metric) IN ('events', 'fatalities_battle_month', 'fatalities')))"
    )

    acled_facts_status, facts_diagnostics = _status_from_facts(conn, "ACLED", acled_facts_clause, [])
    idmc_status, _ = _status_from_facts(conn, "IDMC", "lower(source_id) LIKE ?", ["%idmc%"])
    emdat_status, _ = _status_from_facts(
        conn,
        "EM-DAT",
        "(lower(source_id) LIKE ? OR lower(source_id) LIKE ?)",
        ["%em-dat%", "%emdat%"],
    )

    if acled_status is None:
        acled_status = acled_facts_status

    rows = [acled_status, idmc_status, emdat_status]

    diagnostics = facts_diagnostics
    acled_diag = acled_status.get("diagnostics", {})
    diagnostics["acled_status_source_table"] = acled_diag.get("table_used")
    diagnostics["acled_status_date_column_used"] = acled_diag.get("date_column_used")
    diagnostics["rows_total"] = sum(int(row["rows_scanned"]) for row in rows)
    return rows, diagnostics


def get_country_facts(
    conn, iso3: str, limit: int = 5000
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if conn is None:
        return [], {"notes": ["conn_missing"]}
    source_info, diagnostics = _pick_facts_source(conn)
    if not source_info:
        LOGGER.warning("facts source missing; country facts unavailable.")
        return [], diagnostics

    table = source_info["table"]
    value_expr = source_info["value_expr"]
    columns = _table_columns(conn, table)
    select_cols = [
        "iso3",
        "hazard_code",
        "source_id",
        "ym",
        "metric",
        f"{value_expr} AS value",
    ]
    if source_info["has_hazard_label"] and "hazard_label" in columns:
        select_cols.append("hazard_label")

    select_expr = ", ".join(select_cols)
    query = f"""
        SELECT {select_expr}
        FROM {table}
        WHERE UPPER(iso3) = UPPER(?)
        ORDER BY ym DESC, hazard_code, source_id
        LIMIT ?
    """
    df = conn.execute(query, [iso3, limit]).fetchdf()
    if df.empty:
        return [], diagnostics

    rows: list[dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        ym = str(record.get("ym") or "").strip()
        parts = ym.split("-") if ym else []
        year = int(parts[0]) if len(parts) >= 2 and parts[0].isdigit() else None
        month = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else None
        if month is not None and (month < 1 or month > 12):
            month = None

        hazard_code = record.get("hazard_code")
        hazard_label = record.get("hazard_label") if "hazard_label" in record else None
        hazard_display = _hazard_display(hazard_label, hazard_code)
        metric_display = _metric_display(record.get("metric"))

        rows.append(
            {
                "iso3": str(record.get("iso3") or "").upper(),
                "hazard": hazard_display,
                "hazard_code": str(hazard_code).upper()
                if hazard_code is not None
                else None,
                "source_id": record.get("source_id"),
                "year": year,
                "month": month,
                "metric": metric_display,
                "value": _coerce_float(record.get("value")),
            }
        )

    existing_keys = {
        (
            row.get("iso3"),
            row.get("hazard_code"),
            row.get("year"),
            row.get("month"),
            row.get("metric"),
            row.get("source_id"),
        )
        for row in rows
    }

    diagnostics["acled_table_present"] = _has_acled_monthly_table(conn)
    diagnostics["acled_rows_added"] = 0
    diagnostics["acled_country_rows_total"] = 0

    if diagnostics["acled_table_present"]:
        acled_columns = _acled_monthly_columns(conn)
        iso_col = "iso3" if "iso3" in acled_columns else None
        month_col = "month" if "month" in acled_columns else None
        fatalities_col = None
        for candidate in ("fatalities", "fatalities_total", "fatalities_sum"):
            if candidate in acled_columns:
                fatalities_col = candidate
                break
        source_expr = (
            "COALESCE(source, 'ACLED') AS source_id" if "source" in acled_columns else "'ACLED' AS source_id"
        )

        if iso_col and month_col and fatalities_col:
            count_row = conn.execute(
                f"SELECT COUNT(*) FROM acled_monthly_fatalities WHERE UPPER({iso_col}) = UPPER(?)",
                [iso3],
            ).fetchone()
            diagnostics["acled_country_rows_total"] = int(count_row[0] or 0) if count_row else 0

            acled_query = f"""
                SELECT
                  UPPER({iso_col}) AS iso3,
                  STRFTIME(TRY_CAST({month_col} AS DATE), '%Y-%m') AS ym,
                  {source_expr},
                  CAST({fatalities_col} AS DOUBLE) AS value
                FROM acled_monthly_fatalities
                WHERE UPPER({iso_col}) = UPPER(?)
                ORDER BY {month_col} DESC NULLS LAST
                LIMIT ?
            """
            acled_df = conn.execute(acled_query, [iso3, min(limit, 5000)]).fetchdf()
            if not acled_df.empty:
                for record in acled_df.to_dict(orient="records"):
                    ym = str(record.get("ym") or "").strip()
                    parts = ym.split("-") if ym else []
                    year = int(parts[0]) if len(parts) >= 2 and parts[0].isdigit() else None
                    month = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else None
                    if month is not None and (month < 1 or month > 12):
                        month = None
                    source_id = record.get("source_id") or "ACLED"
                    key = (str(iso3).upper(), "ACE", year, month, "FATALITIES", source_id)
                    if key in existing_keys:
                        continue
                    existing_keys.add(key)
                    rows.append(
                        {
                            "iso3": str(iso3 or "").upper(),
                            "hazard": HAZARD_LABELS.get("ACE"),
                            "hazard_code": "ACE",
                            "source_id": source_id,
                            "year": year,
                            "month": month,
                            "metric": "FATALITIES",
                            "value": _coerce_float(record.get("value")),
                        }
                    )
                    diagnostics["acled_rows_added"] += 1

    diagnostics["rows_returned"] = len(rows)
    return rows, diagnostics
