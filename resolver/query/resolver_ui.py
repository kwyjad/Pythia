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


def _max_date_expr(column: str) -> str:
    return f"TRY_CAST(CAST({column} AS VARCHAR) AS DATE)"


def _ym_proxy_expr() -> str:
    return "TRY_CAST(CAST(ym || '-01' AS VARCHAR) AS DATE)"


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


def get_connector_last_updated(conn) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_info, diagnostics = _pick_facts_source(conn)
    acled_table_present = _has_acled_monthly_table(conn)
    diagnostics["connector_status_date_column_priority"] = [
        "created_at",
        "publication_date",
        "as_of_date",
        "as_of",
        "ym_proxy",
    ]
    diagnostics["acled_status_source_table"] = source_info.get("table") if source_info else None
    diagnostics["acled_status_date_column_used"] = None
    diagnostics["acled_facts_rows_scanned"] = 0
    diagnostics["acled_facts_clause_used"] = None
    if not source_info:
        LOGGER.warning("facts source missing; connector status unavailable.")
        return [
            {"source": "ACLED", "last_updated": None, "rows_scanned": 0},
            {"source": "IDMC", "last_updated": None, "rows_scanned": 0},
            {"source": "EM-DAT", "last_updated": None, "rows_scanned": 0},
        ], diagnostics

    table = source_info["table"]
    columns = _table_columns(conn, table)
    if "source_id" not in columns:
        LOGGER.warning("%s missing source_id; connector status unavailable.", table)
        return [
            {"source": "ACLED", "last_updated": None, "rows_scanned": 0},
            {"source": "IDMC", "last_updated": None, "rows_scanned": 0},
            {"source": "EM-DAT", "last_updated": None, "rows_scanned": 0},
        ], diagnostics

    date_column = None
    for candidate in ("created_at", "publication_date", "as_of_date", "as_of"):
        if candidate in columns:
            date_column = candidate
            break
    if date_column:
        date_expr = _max_date_expr(date_column)
        diagnostics["date_column_used"] = date_column
    else:
        date_expr = _ym_proxy_expr()
        diagnostics["date_column_used"] = "ym_proxy"

    def fetch_acled_status() -> dict[str, Any] | None:
        if not acled_table_present:
            return None
        if not _table_has_rows(conn, "acled_monthly_fatalities"):
            diagnostics["acled_status_source_table"] = "acled_monthly_fatalities"
            diagnostics["acled_status_date_column_used"] = "none"
            return {
                "source": "ACLED",
                "last_updated": None,
                "rows_scanned": 0,
            }
        acled_columns = _acled_monthly_columns(conn)
        if "updated_at" in acled_columns:
            acled_date_expr = _max_date_expr("updated_at")
            diagnostics["acled_status_date_column_used"] = "updated_at"
        elif "month" in acled_columns:
            acled_date_expr = _max_date_expr("month")
            diagnostics["acled_status_date_column_used"] = "month"
        else:
            acled_date_expr = "NULL"
            diagnostics["acled_status_date_column_used"] = "none"
        diagnostics["acled_status_source_table"] = "acled_monthly_fatalities"
        query = f"""
            SELECT
              MAX({acled_date_expr}) AS last_updated,
              COUNT(*) AS rows_scanned
            FROM acled_monthly_fatalities
        """
        row = conn.execute(query).fetchone()
        max_date = row[0] if row else None
        count = int(row[1] or 0) if row else 0
        return {
            "source": "ACLED",
            "last_updated": _format_date(max_date),
            "rows_scanned": count,
        }

    def fetch_status(source: str, clause: str, params: list[str]) -> dict[str, Any]:
        query = f"""
            SELECT
              MAX({date_expr}) AS last_updated,
              COUNT(*) AS rows_scanned
            FROM {table}
            WHERE {clause}
        """
        row = conn.execute(query, params).fetchone()
        max_date = row[0] if row else None
        count = int(row[1] or 0) if row else 0
        return {
            "source": source,
            "last_updated": _format_date(max_date),
            "rows_scanned": count,
        }

    acled_facts_clause = (
        "(lower(source_id) LIKE '%acled%' OR lower(source_id) = 'acled_client' "
        "OR ((source_id IS NULL OR TRIM(CAST(source_id AS VARCHAR)) = '') "
        "AND lower(metric) IN ('events', 'fatalities_battle_month', 'fatalities')))"
    )
    diagnostics["acled_facts_clause_used"] = "source_id_or_signature_blank"
    acled_from_facts = fetch_status("ACLED", acled_facts_clause, [])
    diagnostics["acled_facts_rows_scanned"] = int(acled_from_facts.get("rows_scanned") or 0)
    if acled_from_facts["rows_scanned"] > 0:
        acled_status = acled_from_facts
        diagnostics["acled_status_source_table"] = table
        diagnostics["acled_status_date_column_used"] = diagnostics.get("date_column_used")
    else:
        acled_status = fetch_acled_status()
        if acled_status is None:
            acled_status = acled_from_facts
            diagnostics["acled_status_source_table"] = table
            diagnostics["acled_status_date_column_used"] = diagnostics.get("date_column_used")

    rows = [
        acled_status,
        fetch_status("IDMC", "lower(source_id) LIKE ?", ["%idmc%"]),
        fetch_status(
            "EM-DAT",
            "(lower(source_id) LIKE ? OR lower(source_id) LIKE ?)",
            ["%em-dat%", "%emdat%"],
        ),
    ]
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
