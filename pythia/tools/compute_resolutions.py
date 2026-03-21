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
UNRESOLVABLE_HAZARDS: set[str] = {"DI", "HW"}

# Metrics where absence of source data genuinely means zero impact.
# All other metrics: absence = unknown (do NOT resolve).
_ZERO_DEFAULT_RULES: dict[str, set[str]] = {
    # ACLED covers all countries continuously — no record = zero fatalities
    "FATALITIES": {"ACE", "ACO"},
    # GDACS binary event occurrence: no event = 0
    "EVENT_OCCURRENCE": {"FL", "DR", "TC"},
}


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


def _purge_stale_resolutions(conn, cutoff: str) -> None:
    """Delete resolutions (and orphaned scores) with observed_month beyond *cutoff*.

    This prevents stale rows — written by earlier pipeline runs before the
    calendar-cutoff guard was in place — from persisting indefinitely and
    blocking correct ``INSERT OR REPLACE`` when the calendar advances.
    """
    if not _table_exists(conn, "resolutions"):
        return

    stale = conn.execute(
        "SELECT COUNT(*) FROM resolutions WHERE observed_month > ?", [cutoff]
    ).fetchone()[0]
    if stale == 0:
        return

    # Delete orphaned scores first (referential-integrity-safe order).
    if _table_exists(conn, "scores"):
        conn.execute(
            """
            DELETE FROM scores
            WHERE (question_id, horizon_m) IN (
                SELECT question_id, horizon_m FROM resolutions
                WHERE observed_month > ?
            )
            """,
            [cutoff],
        )

    # Delete the stale resolutions themselves.
    conn.execute("DELETE FROM resolutions WHERE observed_month > ?", [cutoff])

    # Revert question status: questions that no longer have all 6 horizons
    # resolved should go back to 'active'.
    if _table_exists(conn, "questions"):
        conn.execute(
            """
            UPDATE questions SET status = 'active'
            WHERE status = 'resolved'
              AND question_id NOT IN (
                  SELECT question_id FROM resolutions
                  GROUP BY question_id
                  HAVING COUNT(DISTINCT horizon_m) = ?
              )
            """,
            [NUM_HORIZONS],
        )

    LOGGER.info(
        "Purged %d stale resolution rows (observed_month > %s).", stale, cutoff
    )


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

    elif metric == "EVENT_OCCURRENCE":
        if _table_exists(conn, "facts_resolved"):
            try:
                row = conn.execute(
                    "SELECT MAX(ym) FROM facts_resolved "
                    "WHERE lower(metric) = 'event_occurrence'"
                ).fetchone()
                if row and row[0]:
                    max_yms.append(str(row[0]))
            except Exception:
                pass

    elif metric == "PHASE3PLUS_IN_NEED":
        if _table_exists(conn, "facts_resolved"):
            try:
                row = conn.execute(
                    "SELECT MAX(ym) FROM facts_resolved "
                    "WHERE lower(metric) = 'phase3plus_in_need'"
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


def _try_gdacs_binary(
    conn, iso3: str, hazard_code: str, calendar_month: str,
) -> Optional[tuple[float, Optional[str]]]:
    """Look up GDACS binary event occurrence in ``facts_resolved``."""
    if not _table_exists(conn, "facts_resolved"):
        return None
    sql = """
        SELECT value, created_at
        FROM facts_resolved
        WHERE iso3 = ? AND hazard_code = ?
          AND ym = ?
          AND lower(metric) = 'event_occurrence'
        ORDER BY created_at DESC LIMIT 1
    """
    try:
        row = conn.execute(sql, [iso3, hazard_code, calendar_month]).fetchone()
    except Exception:
        return None
    if not row or row[0] is None:
        return None
    return float(row[0]), (str(row[1]) if row[1] is not None else None)


def _try_fewsnet_ipc(
    conn, iso3: str, hazard_code: str, calendar_month: str,
) -> Optional[tuple[float, Optional[str]]]:
    """Look up Phase 3+ population in ``facts_resolved`` (FEWS NET IPC)."""
    if hazard_code != "DR":
        return None
    if not _table_exists(conn, "facts_resolved"):
        return None
    sql = """
        SELECT value, created_at
        FROM facts_resolved
        WHERE iso3 = ? AND hazard_code = 'DR'
          AND ym = ?
          AND lower(metric) = 'phase3plus_in_need'
        ORDER BY created_at DESC LIMIT 1
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

    Checks multiple Resolver tables in priority order.  The dispatch
    depends on the metric type:

    PA:
      1. ``facts_resolved`` — IFRC stock data (highest source priority)
      2. ``facts_deltas``   — IDMC flow data and derived deltas
      3. ``emdat_pa``       — EM-DAT people-affected
    FATALITIES:
      1. ``facts_resolved``
      2. ``facts_deltas``
      3. ``acled_monthly_fatalities`` — ACLED fatalities
    EVENT_OCCURRENCE:
      1. ``facts_resolved`` (GDACS binary event rows)
    PHASE3PLUS_IN_NEED:
      1. ``facts_resolved`` (FEWS NET IPC Phase 3+ data)
    """

    if metric == "EVENT_OCCURRENCE":
        return _try_gdacs_binary(conn, iso3, hazard_code, calendar_month)

    if metric == "PHASE3PLUS_IN_NEED":
        return _try_fewsnet_ipc(conn, iso3, hazard_code, calendar_month)

    # PA and FATALITIES: existing priority cascade
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


def _should_default_to_zero(metric_norm: str, hazard_norm: str) -> bool:
    """Return True if absence of source data means zero impact for this
    metric/hazard combination."""
    allowed_hazards = _ZERO_DEFAULT_RULES.get(metric_norm)
    if allowed_hazards is not None and hazard_norm in allowed_hazards:
        return True
    return False



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
          is_test BOOLEAN DEFAULT FALSE,
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
    if "is_test" not in existing:
        try:
            conn.execute("ALTER TABLE resolutions ADD COLUMN is_test BOOLEAN DEFAULT FALSE")
        except Exception:
            pass


def compute_resolutions(db_url: str, today: Optional[date] = None) -> None:
    """
    Compute and upsert resolutions for eligible questions.

    For each question, resolves all 6 horizon months independently.  Each
    horizon month maps to a distinct calendar month derived from the
    question's ``window_start_date``.

    Rules:
      - Metrics PA, FATALITIES, EVENT_OCCURRENCE, and PHASE3PLUS_IN_NEED
        are processed.
      - Hazards in ``UNRESOLVABLE_HAZARDS`` are skipped (no data source).
      - Eligibility is **data-driven**: a calendar month is eligible when
        at least one source table has data covering that month (determined
        via ``_data_freshness_cutoff``).
      - Source-aware null handling: when no matching row exists in any
        source table, the behavior depends on the metric and hazard:
          * FATALITIES + ACE/ACO: default to 0.0 (ACLED continuous coverage)
          * EVENT_OCCURRENCE: default to 0.0 (no event = no occurrence)
          * All others (PA, PHASE3PLUS_IN_NEED, etc.): skip the horizon
            (no resolution row written — unresolvable, not zero).
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

        # Purge any stale resolutions beyond the cutoff (left over from
        # earlier pipeline runs before the calendar guard was added).
        _purge_stale_resolutions(conn, cal_cutoff)

        # Data-driven guard: don't resolve beyond what sources actually cover.
        _SUPPORTED_METRICS = ("PA", "FATALITIES", "EVENT_OCCURRENCE", "PHASE3PLUS_IN_NEED")
        metric_cutoffs: dict[str, Optional[str]] = {}
        for m in _SUPPORTED_METRICS:
            data_cut = _data_freshness_cutoff(conn, m)
            metric_cutoffs[m] = min(cal_cutoff, data_cut) if data_cut else cal_cutoff

        cutoff_summary = ", ".join(
            f"{m}={metric_cutoffs[m]} (data={_data_freshness_cutoff(conn, m) or '<none>'})"
            for m in _SUPPORTED_METRICS
        )
        LOGGER.info("Effective cutoffs (calendar=%s): %s", cal_cutoff, cutoff_summary)

        # Supported metrics filter for SQL
        metric_in = "','".join(_SUPPORTED_METRICS)
        query_sql = f"""
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
              AND upper(q.metric) IN ('{metric_in}')
            ORDER BY q.question_id
        """
        rows = conn.execute(query_sql).fetchall()
        LOGGER.info(
            "Found %d candidate questions for resolution.",
            len(rows),
        )

        written = 0
        resolved_from_source = 0
        resolved_as_zero = 0
        skipped_no_data_coverage = 0
        skipped_null_resolution = 0
        skipped_unresolvable_hazard = 0

        for question_id, iso3, hazard_code, metric, target_month, window_start_date in rows:
            iso3_norm = (iso3 or "").upper()
            hazard_norm = (hazard_code or "").upper()
            metric_norm = (metric or "").upper()

            if metric_norm not in _SUPPORTED_METRICS:
                continue

            # Skip hazards without a resolution data source.
            if hazard_norm in UNRESOLVABLE_HAZARDS:
                skipped_unresolvable_hazard += 1
                continue

            # ── 2-tier window_start_date derivation ──────────────────────
            #
            # Priority 1: q.window_start_date from the questions table.
            #   This is authoritative — each question carries its own
            #   window dates set at creation time.
            #
            # Priority 2: Derive from target_month (the 6th horizon month).
            # ─────────────────────────────────────────────────────────────

            ws_date: Optional[date] = None

            # Priority 1: questions table window_start_date
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

            # Priority 2: derive from target_month
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
            data_cutoff = metric_cutoffs.get(metric_norm)

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
                    # Source-aware null handling: only default to zero for
                    # sources where absence genuinely means zero impact.
                    if _should_default_to_zero(metric_norm, hazard_norm):
                        value: float = 0.0
                        source_ts: Optional[str] = None
                        resolved_as_zero += 1
                    else:
                        # All other sources: no record = unresolvable.
                        # Do NOT write a resolution row — leave this
                        # horizon unresolved so scoring skips it.
                        skipped_null_resolution += 1
                        continue
                else:
                    value, source_ts = resolved
                    resolved_from_source += 1

                try:
                    q_test = conn.execute(
                        "SELECT COALESCE(is_test, FALSE) FROM questions WHERE question_id = ?",
                        [question_id],
                    ).fetchone()
                    is_test_val = q_test[0] if q_test else False
                except Exception:
                    is_test_val = False

                conn.execute(
                    """
                    INSERT OR REPLACE INTO resolutions (
                      question_id,
                      horizon_m,
                      observed_month,
                      value,
                      source_snapshot_ym,
                      created_at,
                      is_test
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        question_id,
                        horizon_m,
                        cal_month,
                        float(value),
                        source_ts,
                        datetime.utcnow(),
                        is_test_val,
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
        # A question moves to "resolved" when all 6 horizons have resolution
        # rows.  Horizons skipped due to null data intentionally remain
        # unresolved — they may become resolvable when new data arrives.
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
            "%d horizon-months skipped (no resolution data), "
            "%d horizon-months skipped (no data coverage yet), "
            "%d questions skipped (unresolvable hazard).",
            len(rows),
            written,
            resolved_from_source,
            resolved_as_zero,
            skipped_null_resolution,
            skipped_no_data_coverage,
            skipped_unresolvable_hazard,
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
