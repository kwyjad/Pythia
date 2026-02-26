# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime
from typing import Optional

from resolver.db import duckdb_io

from pythia.config import load as load_cfg


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    LOGGER.addHandler(logging.NullHandler())

NUM_HORIZONS = 6

# Hazards without a resolution data source.  Remove entries once a source
# is available (e.g. when a DI resolution connector is added).
UNRESOLVABLE_HAZARDS: set[str] = {"DI"}


def _table_exists(conn, name: str) -> bool:
    try:
        conn.execute(f"PRAGMA table_info('{name}')").fetchall()
        return True
    except Exception:
        return False


def _row_count(conn, name: str) -> int:
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0] or 0
    except Exception:
        return 0


def _get_db_url_from_config() -> str:
    cfg = load_cfg()
    app_cfg = cfg.get("app", {}) if isinstance(cfg, dict) else {}
    db_url = str(app_cfg.get("db_url", "")).strip()
    if not db_url:
        db_url = duckdb_io.DEFAULT_DB_URL
        LOGGER.warning("app.db_url missing in config; falling back to %s", db_url)
    else:
        LOGGER.info("Using app.db_url from config: %s", db_url)
    return db_url


def _open_db(db_url: str | None):
    if not duckdb_io.DUCKDB_AVAILABLE:
        raise RuntimeError(duckdb_io.duckdb_unavailable_reason())
    return duckdb_io.get_db(db_url or duckdb_io.DEFAULT_DB_URL)


def _close_db(conn) -> None:
    try:
        duckdb_io.close_db(conn)
    except Exception:
        pass


def _shift_month(month_anchor: date, delta_months: int) -> date:
    """Shift ``month_anchor`` (assumed day=1) by ``delta_months`` months."""

    year = month_anchor.year + (month_anchor.month - 1 + delta_months) // 12
    month = (month_anchor.month - 1 + delta_months) % 12 + 1
    return date(year, month, 1)


def horizon_to_calendar_month(window_start_date: date, horizon_m: int) -> str:
    """Return 'YYYY-MM' for the calendar month corresponding to horizon_m.

    horizon_m is 1-based: horizon_m=1 corresponds to window_start_date,
    horizon_m=2 corresponds to one month after window_start_date, etc.
    """
    shifted = _shift_month(window_start_date, horizon_m - 1)
    return f"{shifted.year:04d}-{shifted.month:02d}"


def _calendar_cutoff(today: date) -> str:
    """Return the latest calendar month that is fully complete.

    Rule: max resolvable month = ``current_month - 1``.  In February 2026,
    this returns ``"2026-01"`` (January is the last complete month).
    This prevents resolving against partial-month data that the Resolver
    uploads mid-month.
    """
    prev = _shift_month(date(today.year, today.month, 1), -1)
    return f"{prev.year:04d}-{prev.month:02d}"


def _data_freshness_cutoff(conn, metric: str) -> Optional[str]:
    """Return the latest ``YYYY-MM`` for which source data exists.

    This determines which calendar months are eligible for resolution.
    If a source has data covering 2026-01, forecasts for months up to and
    including 2026-01 are eligible.  Months beyond that are not yet
    resolvable because data sources have not been refreshed for them.
    """
    max_yms: list[str] = []

    if metric == "PA":
        for table, filt in [
            ("facts_resolved",
             "lower(metric) IN ('affected','people_affected','pa','displaced')"),
            ("facts_deltas",
             "lower(metric) IN "
             "('new_displacements','affected','people_affected','pa','displaced')"),
        ]:
            if _table_exists(conn, table):
                try:
                    row = conn.execute(
                        f"SELECT MAX(ym) FROM {table} WHERE {filt}"
                    ).fetchone()
                    if row and row[0]:
                        max_yms.append(str(row[0]))
                except Exception:
                    pass
        if _table_exists(conn, "emdat_pa"):
            try:
                row = conn.execute("SELECT MAX(ym) FROM emdat_pa").fetchone()
                if row and row[0]:
                    max_yms.append(str(row[0]))
            except Exception:
                pass

    elif metric == "FATALITIES":
        for table, filt in [
            ("facts_resolved", "lower(metric) = 'fatalities'"),
            ("facts_deltas", "lower(metric) = 'fatalities'"),
        ]:
            if _table_exists(conn, table):
                try:
                    row = conn.execute(
                        f"SELECT MAX(ym) FROM {table} WHERE {filt}"
                    ).fetchone()
                    if row and row[0]:
                        max_yms.append(str(row[0]))
                except Exception:
                    pass
        if _table_exists(conn, "acled_monthly_fatalities"):
            try:
                row = conn.execute(
                    "SELECT MAX(strftime(month, '%Y-%m')) "
                    "FROM acled_monthly_fatalities"
                ).fetchone()
                if row and row[0]:
                    max_yms.append(str(row[0]))
            except Exception:
                pass

    return max(max_yms) if max_yms else None


# Hazard-code → EM-DAT shock_type mapping for emdat_pa table lookups.
_HAZARD_TO_EMDAT_SHOCK: dict[str, str] = {
    "FL": "flood",
    "DR": "drought",
    "TC": "tropical_cyclone",
    "HW": "heatwave",
}


def _try_facts_resolved(
    conn, iso3: str, hazard_code: str, calendar_month: str, metric: str,
) -> Optional[tuple[float, Optional[str]]]:
    """Look up in ``facts_resolved`` (IFRC stock data, highest priority)."""
    if not _table_exists(conn, "facts_resolved"):
        return None
    if metric == "PA":
        metric_filter = "lower(metric) IN ('affected','people_affected','pa','displaced')"
    elif metric == "FATALITIES":
        metric_filter = "lower(metric) = 'fatalities'"
    else:
        return None
    sql = f"""
        SELECT value, created_at
        FROM facts_resolved
        WHERE iso3 = ? AND hazard_code = ? AND ym = ? AND {metric_filter}
        ORDER BY created_at DESC LIMIT 1
    """
    try:
        row = conn.execute(sql, [iso3, hazard_code, calendar_month]).fetchone()
    except Exception:
        return None
    if not row:
        return None
    return float(row[0]), (str(row[1]) if row[1] is not None else None)


def _try_facts_deltas(
    conn, iso3: str, hazard_code: str, calendar_month: str, metric: str,
) -> Optional[tuple[float, Optional[str]]]:
    """Look up in ``facts_deltas`` (IDMC flow data, etc.)."""
    if not _table_exists(conn, "facts_deltas"):
        return None
    if metric == "PA":
        metric_filter = (
            "lower(metric) IN "
            "('new_displacements','affected','people_affected','pa','displaced')"
        )
    elif metric == "FATALITIES":
        metric_filter = "lower(metric) = 'fatalities'"
    else:
        return None
    sql = f"""
        SELECT COALESCE(value_new, value_stock) AS value, created_at
        FROM facts_deltas
        WHERE iso3 = ? AND hazard_code = ? AND ym = ? AND {metric_filter}
        ORDER BY created_at DESC LIMIT 1
    """
    try:
        row = conn.execute(sql, [iso3, hazard_code, calendar_month]).fetchone()
    except Exception:
        return None
    if not row or row[0] is None:
        return None
    return float(row[0]), (str(row[1]) if row[1] is not None else None)


def _try_emdat_pa(
    conn, iso3: str, hazard_code: str, calendar_month: str,
) -> Optional[tuple[float, Optional[str]]]:
    """Look up people-affected in the ``emdat_pa`` table."""
    if not _table_exists(conn, "emdat_pa"):
        return None
    shock_type = _HAZARD_TO_EMDAT_SHOCK.get(hazard_code)
    if not shock_type:
        return None
    sql = """
        SELECT pa, as_of_date
        FROM emdat_pa
        WHERE iso3 = ? AND ym = ? AND shock_type = ?
        ORDER BY as_of_date DESC LIMIT 1
    """
    try:
        row = conn.execute(sql, [iso3, calendar_month, shock_type]).fetchone()
    except Exception:
        return None
    if not row or row[0] is None:
        return None
    return float(row[0]), (str(row[1]) if row[1] is not None else None)


def _try_acled_fatalities(
    conn, iso3: str, calendar_month: str,
) -> Optional[tuple[float, Optional[str]]]:
    """Look up fatalities in ``acled_monthly_fatalities``."""
    if not _table_exists(conn, "acled_monthly_fatalities"):
        return None
    sql = """
        SELECT fatalities, updated_at
        FROM acled_monthly_fatalities
        WHERE iso3 = ? AND strftime(month, '%Y-%m') = ?
        LIMIT 1
    """
    try:
        row = conn.execute(sql, [iso3, calendar_month]).fetchone()
    except Exception:
        return None
    if not row or row[0] is None:
        return None
    return float(row[0]), (str(row[1]) if row[1] is not None else None)


def _resolve_value(
    conn,
    iso3: str,
    hazard_code: str,
    calendar_month: str,
    metric: str,
) -> Optional[tuple[float, Optional[str]]]:
    """Resolve a single metric for (iso3, hazard_code, calendar_month).

    Checks multiple Resolver tables in priority order:
      1. ``facts_resolved`` — IFRC stock data (highest source priority)
      2. ``facts_deltas``   — IDMC flow data and derived deltas
      3. ``emdat_pa``       — EM-DAT people-affected (PA metric only)
      4. ``acled_monthly_fatalities`` — ACLED fatalities (FATALITIES only)
    """

    # 1. facts_resolved (IFRC stock rows, highest priority)
    result = _try_facts_resolved(conn, iso3, hazard_code, calendar_month, metric)
    if result is not None:
        return result

    # 2. facts_deltas (IDMC new_displacements, derived deltas, etc.)
    result = _try_facts_deltas(conn, iso3, hazard_code, calendar_month, metric)
    if result is not None:
        return result

    # 3. emdat_pa for PA metric on natural hazards
    if metric == "PA":
        result = _try_emdat_pa(conn, iso3, hazard_code, calendar_month)
        if result is not None:
            return result

    # 4. acled_monthly_fatalities for FATALITIES metric
    if metric == "FATALITIES":
        result = _try_acled_fatalities(conn, iso3, calendar_month)
        if result is not None:
            return result

    return None


def _llm_derived_window_starts(conn) -> dict[str, date]:
    """Return ``{question_id: window_start_date}`` derived from the earliest
    LLM forecast timestamp for each question.

    ``window_start`` is the first day of the month **after** the earliest
    forecast.  E.g. a forecast made on 2025-12-15 → window_start = 2026-01-01.

    This is immune to HS-run overwrites because ``llm_calls`` is append-only
    and never deleted or replaced by newer Horizon Scanner runs.
    """
    sql = """
        SELECT question_id,
               MIN(timestamp) AS earliest_ts
        FROM llm_calls
        WHERE question_id IS NOT NULL
          AND timestamp IS NOT NULL
        GROUP BY question_id
    """
    try:
        rows = conn.execute(sql).fetchall()
    except Exception:
        # llm_calls table may not exist in test or early-stage DBs
        return {}

    result: dict[str, date] = {}
    for qid, earliest_ts in rows:
        try:
            if isinstance(earliest_ts, str):
                dt = datetime.fromisoformat(earliest_ts)
            else:
                dt = earliest_ts
            # window_start = first day of month AFTER the forecast
            y, m = dt.year, dt.month + 1
            if m > 12:
                y += 1
                m = 1
            result[str(qid)] = date(y, m, 1)
        except Exception:
            continue
    return result


def _ensure_resolutions_table(conn) -> None:
    """Create the resolutions table if it does not exist, and add horizon_m
    column if missing (migration for existing databases)."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS resolutions (
          question_id TEXT,
          horizon_m INTEGER,
          observed_month TEXT,
          value DOUBLE,
          source_snapshot_ym TEXT,
          created_at TIMESTAMP DEFAULT now(),
          PRIMARY KEY (question_id, horizon_m)
        )
        """
    )
    # Migration: add horizon_m column if table predates this change
    existing = set()
    try:
        for row in conn.execute("PRAGMA table_info('resolutions')").fetchall():
            existing.add(str(row[1]).lower())
    except Exception:
        pass
    if "horizon_m" not in existing:
        try:
            conn.execute("ALTER TABLE resolutions ADD COLUMN horizon_m INTEGER")
        except Exception:
            pass


def compute_resolutions(db_url: str, today: Optional[date] = None) -> None:
    """
    Compute and upsert resolutions for eligible questions.

    For each question, resolves all 6 horizon months independently.  Each
    horizon month maps to a distinct calendar month derived from the
    question's ``window_start_date``.

    Rules:
      - Only metrics PA and FATALITIES are processed.
      - Hazards in ``UNRESOLVABLE_HAZARDS`` are skipped (no data source).
      - Eligibility is **data-driven**: a calendar month is eligible when
        at least one source table has data covering that month (determined
        via ``_data_freshness_cutoff``).
      - When an eligible month has no matching row in any source table the
        resolution defaults to ``0.0`` (no humanitarian impact reported).
      - Checks facts_resolved, facts_deltas, emdat_pa, and
        acled_monthly_fatalities.
    """

    if today is None:
        today = date.today()

    conn = _open_db(db_url)

    try:
        _ensure_resolutions_table(conn)

        # Early exit if questions table doesn't exist or is empty
        if not _table_exists(conn, "questions"):
            LOGGER.info("compute_resolutions: questions table not found; nothing to do.")
            return

        q_count = _row_count(conn, "questions")
        if q_count == 0:
            LOGGER.info("compute_resolutions: questions table is empty; nothing to do.")
            return

        # Calendar cutoff: previous complete month (prevents partial-month data).
        cal_cutoff = _calendar_cutoff(today)

        # Data-driven guard: don't resolve beyond what sources actually cover.
        pa_data_cutoff = _data_freshness_cutoff(conn, "PA")
        fat_data_cutoff = _data_freshness_cutoff(conn, "FATALITIES")

        # Effective cutoff: min(calendar, data) per metric.
        # Calendar prevents partial-month resolution; data guard prevents
        # resolving months for which no source data has been loaded yet.
        pa_cutoff = min(cal_cutoff, pa_data_cutoff) if pa_data_cutoff else cal_cutoff
        fat_cutoff = min(cal_cutoff, fat_data_cutoff) if fat_data_cutoff else cal_cutoff
        LOGGER.info(
            "Effective cutoffs: PA=%s (calendar=%s, data=%s), "
            "FATALITIES=%s (calendar=%s, data=%s)",
            pa_cutoff, cal_cutoff, pa_data_cutoff or "<none>",
            fat_cutoff, cal_cutoff, fat_data_cutoff or "<none>",
        )

        query_sql = """
            SELECT
              q.question_id,
              q.iso3,
              q.hazard_code,
              upper(q.metric) AS metric,
              q.target_month,
              q.window_start_date
            FROM questions q
            JOIN hs_runs h ON q.hs_run_id = h.hs_run_id
            WHERE q.status IN ('active','resolved')
              AND upper(q.metric) IN ('PA','FATALITIES')
            ORDER BY q.question_id
        """
        rows = conn.execute(query_sql).fetchall()
        LOGGER.info(
            "Found %d candidate questions for resolution.",
            len(rows),
        )

        # LLM-derived window starts: immune to HS-run overwrites because
        # llm_calls is append-only.
        llm_windows = _llm_derived_window_starts(conn)
        if llm_windows:
            LOGGER.info(
                "LLM-derived window_start_date available for %d questions.",
                len(llm_windows),
            )

        written = 0
        resolved_from_source = 0
        resolved_as_zero = 0
        skipped_no_data_coverage = 0
        skipped_unresolvable_hazard = 0
        llm_window_overrides = 0

        for question_id, iso3, hazard_code, metric, target_month, window_start_date in rows:
            iso3_norm = (iso3 or "").upper()
            hazard_norm = (hazard_code or "").upper()
            metric_norm = (metric or "").upper()

            if metric_norm not in ("PA", "FATALITIES"):
                continue

            # Skip hazards without a resolution data source.
            if hazard_norm in UNRESOLVABLE_HAZARDS:
                skipped_unresolvable_hazard += 1
                continue

            # ── 3-tier window_start_date derivation ──────────────────────
            #
            # Priority 1: LLM-derived window (from llm_calls.timestamp).
            #   This is immune to HS-run overwrites because llm_calls is
            #   append-only and never modified by Horizon Scanner runs.
            #
            # Priority 2: q.window_start_date from the questions table.
            #
            # Priority 3: Derive from target_month (the 6th horizon month).
            # ─────────────────────────────────────────────────────────────

            ws_date: Optional[date] = llm_windows.get(question_id)

            if ws_date is not None:
                # Check if this overrides a different value from the questions table
                q_ws: Optional[date] = None
                if window_start_date is not None:
                    if isinstance(window_start_date, str):
                        try:
                            parts = window_start_date.split("-")
                            q_ws = date(int(parts[0]), int(parts[1]), int(parts[2]))
                        except Exception:
                            pass
                    elif isinstance(window_start_date, date):
                        q_ws = window_start_date
                if q_ws is not None and q_ws != ws_date:
                    llm_window_overrides += 1

            # Priority 2: questions table window_start_date
            if ws_date is None and window_start_date is not None:
                if isinstance(window_start_date, str):
                    try:
                        parts = window_start_date.split("-")
                        ws_date = date(int(parts[0]), int(parts[1]), int(parts[2]))
                    except Exception:
                        ws_date = None
                elif isinstance(window_start_date, date):
                    ws_date = window_start_date
                else:
                    ws_date = None

            # Priority 3: derive from target_month
            if ws_date is None and target_month:
                try:
                    parts = target_month.split("-")
                    tm_date = date(int(parts[0]), int(parts[1]), 1)
                    ws_date = _shift_month(tm_date, -(NUM_HORIZONS - 1))
                except Exception:
                    LOGGER.warning(
                        "Cannot derive window_start_date for %s; skipping.", question_id
                    )
                    continue

            if ws_date is None:
                LOGGER.warning(
                    "No window_start_date or target_month for %s; skipping.", question_id
                )
                continue

            # Select the per-metric effective cutoff.
            data_cutoff = (
                pa_cutoff if metric_norm == "PA" else fat_cutoff
            )

            for horizon_m in range(1, NUM_HORIZONS + 1):
                cal_month = horizon_to_calendar_month(ws_date, horizon_m)

                # Only resolve months for which source data exists.
                if data_cutoff is None or cal_month > data_cutoff:
                    skipped_no_data_coverage += 1
                    continue

                resolved = _resolve_value(
                    conn, iso3_norm, hazard_norm, cal_month, metric_norm,
                )
                if resolved is None:
                    # No impact data found — default to zero (no reported
                    # humanitarian impact for this hazard/country/month).
                    value: float = 0.0
                    source_ts: Optional[str] = None
                    resolved_as_zero += 1
                else:
                    value, source_ts = resolved
                    resolved_from_source += 1

                conn.execute(
                    """
                    INSERT OR REPLACE INTO resolutions (
                      question_id,
                      horizon_m,
                      observed_month,
                      value,
                      source_snapshot_ym,
                      created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        question_id,
                        horizon_m,
                        cal_month,
                        float(value),
                        source_ts,
                        datetime.utcnow(),
                    ],
                )
                written += 1
                LOGGER.info(
                    "Resolved %s h%d (%s/%s/%s %s) -> value=%.1f source_ts=%s",
                    question_id,
                    horizon_m,
                    iso3_norm,
                    hazard_norm,
                    cal_month,
                    metric_norm,
                    value,
                    source_ts or "<zero-default>",
                )

        # Update question status for fully-resolved questions.
        try:
            conn.execute(
                """
                UPDATE questions SET status = 'resolved'
                WHERE question_id IN (
                    SELECT question_id FROM resolutions
                    GROUP BY question_id
                    HAVING COUNT(DISTINCT horizon_m) = ?
                ) AND status = 'active'
                """,
                [NUM_HORIZONS],
            )
        except Exception as exc:
            LOGGER.warning("Failed to update question statuses: %s", exc)

        LOGGER.info(
            "compute_resolutions: %d questions processed, %d resolution rows "
            "written (%d from source data, %d defaulted to 0.0), "
            "%d horizon-months skipped (no data coverage yet), "
            "%d questions skipped (unresolvable hazard), "
            "%d window_start overrides from llm_calls.",
            len(rows),
            written,
            resolved_from_source,
            resolved_as_zero,
            skipped_no_data_coverage,
            skipped_unresolvable_hazard,
            llm_window_overrides,
        )
    finally:
        _close_db(conn)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Pythia resolutions from Resolver.")
    parser.add_argument(
        "--db-url",
        default=None,
        help="DuckDB URL (default: app.db_url from pythia.config, or resolver default)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    db_url = args.db_url or _get_db_url_from_config()
    compute_resolutions(db_url=db_url)


if __name__ == "__main__":
    main()
