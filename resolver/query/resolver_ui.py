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


def get_connector_last_updated(conn) -> list[dict[str, Any]]:
    if conn is None:
        return []
    if not _table_exists(conn, "facts_resolved"):
        LOGGER.warning("facts_resolved missing; connector status unavailable.")
        return []

    columns = _table_columns(conn, "facts_resolved")
    if "source_id" not in columns:
        LOGGER.warning("facts_resolved missing source_id; connector status unavailable.")
        return [
            {"source": "ACLED", "last_updated": None, "rows_scanned": 0},
            {"source": "IDMC", "last_updated": None, "rows_scanned": 0},
            {"source": "EM-DAT", "last_updated": None, "rows_scanned": 0},
        ]

    date_column = None
    for candidate in ("as_of_date", "publication_date", "as_of"):
        if candidate in columns:
            date_column = candidate
            break

    date_expr = _max_date_expr(date_column) if date_column else "NULL"

    def fetch_status(source: str, clause: str, params: list[str]) -> dict[str, Any]:
        query = f"""
            SELECT
              MAX({date_expr}) AS last_updated,
              COUNT(*) AS rows_scanned
            FROM facts_resolved
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

    return [
        fetch_status("ACLED", "lower(source_id) LIKE ?", ["%acled%"]),
        fetch_status("IDMC", "lower(source_id) LIKE ?", ["%idmc%"]),
        fetch_status(
            "EM-DAT",
            "(lower(source_id) LIKE ? OR lower(source_id) LIKE ?)",
            ["%em-dat%", "%emdat%"],
        ),
    ]


def get_country_facts(conn, iso3: str, limit: int = 5000) -> list[dict[str, Any]]:
    if conn is None:
        return []
    if not _table_exists(conn, "facts_resolved"):
        LOGGER.warning("facts_resolved missing; country facts unavailable.")
        return []

    columns = _table_columns(conn, "facts_resolved")
    required = {"iso3", "ym", "hazard_code", "source_id", "metric", "value"}
    if not required.issubset(columns):
        LOGGER.warning("facts_resolved missing required columns for resolver UI.")
        return []

    select_cols = ["iso3", "hazard_code", "source_id", "ym", "metric", "value"]
    if "hazard_label" in columns:
        select_cols.append("hazard_label")

    select_expr = ", ".join(select_cols)
    query = f"""
        SELECT {select_expr}
        FROM facts_resolved
        WHERE UPPER(iso3) = UPPER(?)
        ORDER BY ym DESC, hazard_code, source_id
        LIMIT ?
    """
    df = conn.execute(query, [iso3, limit]).fetchdf()
    if df.empty:
        return []

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

    return rows
