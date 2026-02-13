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


def _eligible_cutoff_month(today: date, lag_day: int = 15) -> str:
    """Return the latest calendar month eligible for resolution.

    A month is eligible once we reach the ``lag_day`` of the following month.
    """

    month_anchor = date(today.year, today.month, 1)
    months_back = 1 if today.day >= lag_day else 2
    eligible_month = _shift_month(month_anchor, -months_back)
    cutoff = f"{eligible_month.year:04d}-{eligible_month.month:02d}"
    LOGGER.info(
        "Resolution cutoff using lag_day=%d and today=%s is calendar_month <= %s",
        lag_day,
        today.isoformat(),
        cutoff,
    )
    return cutoff


def _resolve_value(
    conn,
    iso3: str,
    hazard_code: str,
    calendar_month: str,
    metric: str,
) -> Optional[tuple[float, Optional[str]]]:
    """Resolve a single metric for (iso3, hazard_code, calendar_month) from Resolver.

    Uses facts_resolved as canonical and trusts Resolver's source prioritisation.
    """

    if metric == "PA":
        metric_filter = "lower(metric) IN ('affected','people_affected','pa')"
    elif metric == "FATALITIES":
        metric_filter = "lower(metric) = 'fatalities'"
    else:
        return None

    sql = f"""
        SELECT value, snapshot_ym
        FROM facts_resolved
        WHERE iso3 = ?
          AND hazard_code = ?
          AND ym = ?
          AND {metric_filter}
        ORDER BY snapshot_ym DESC
        LIMIT 1
    """
    try:
        row = conn.execute(sql, [iso3, hazard_code, calendar_month]).fetchone()
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.error(
            "Resolver %s query failed for %s/%s/%s: %r",
            metric, iso3, hazard_code, calendar_month, exc,
        )
        return None
    if not row:
        return None
    value, snapshot_ym = row
    return float(value), (snapshot_ym if snapshot_ym is not None else None)


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
      - A calendar month is eligible once we reach the 15th of the following month.
      - Uses Resolver's facts_resolved as canonical (no source arbitration here).
    """

    if today is None:
        today = date.today()

    cutoff_month = _eligible_cutoff_month(today)
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
            "Found %d candidate questions for resolution (cutoff_month=%s).",
            len(rows), cutoff_month,
        )

        written = 0
        for question_id, iso3, hazard_code, metric, target_month, window_start_date in rows:
            iso3_norm = (iso3 or "").upper()
            hazard_norm = (hazard_code or "").upper()
            metric_norm = (metric or "").upper()

            if metric_norm not in ("PA", "FATALITIES"):
                continue

            # Derive the window_start_date as a Python date
            if window_start_date is not None:
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
            else:
                ws_date = None

            # Fallback: if window_start_date is not available, derive from
            # target_month (which is the 6th month in the window).
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

            for horizon_m in range(1, NUM_HORIZONS + 1):
                cal_month = horizon_to_calendar_month(ws_date, horizon_m)

                # Only resolve months that are past the eligibility cutoff
                if cal_month > cutoff_month:
                    continue

                resolved = _resolve_value(
                    conn, iso3_norm, hazard_norm, cal_month, metric_norm,
                )
                if resolved is None:
                    LOGGER.debug(
                        "No resolution for %s h%d (%s/%s/%s).",
                        question_id, horizon_m, iso3_norm, hazard_norm, cal_month,
                    )
                    continue

                value, snapshot_ym = resolved
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
                        snapshot_ym,
                        datetime.utcnow(),
                    ],
                )
                written += 1
                LOGGER.info(
                    "Resolved %s h%d (%s/%s/%s %s) -> value=%.1f snapshot_ym=%s",
                    question_id,
                    horizon_m,
                    iso3_norm,
                    hazard_norm,
                    cal_month,
                    metric_norm,
                    value,
                    snapshot_ym or "<none>",
                )

        LOGGER.info("compute_resolutions: wrote %d resolution rows.", written)
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
