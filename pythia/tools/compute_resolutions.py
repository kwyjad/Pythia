# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import argparse
import logging
import os
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from typing import Optional

from resolver.db import duckdb_io

from pythia.config import load as load_cfg


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    LOGGER.addHandler(logging.NullHandler())

from pythia.buckets import NUM_HORIZONS


def _utcnow_naive() -> datetime:
    """Naive-UTC now (datetime.utcnow is deprecated; DB columns are naive)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

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

# Deterministic ground-truth selection. Multiple candidate rows can exist
# for the same (iso3, hazard, ym) — different metrics within the PA family
# and different publishers — and "ORDER BY created_at DESC" made the
# resolved value depend on ingestion order (observed spreads up to 145x,
# e.g. IFRC 'affected' 19M vs 'displaced' 130k for the same month).
#
# Preference 1 — metric: direct affected counts beat displacement
# components (position in this list = priority; lower wins).
_PA_METRIC_PREFERENCE = [
    "affected",
    "people_affected",
    "pa",
    "new_displacements",
    "displaced",
]

# The PA metrics that can actually resolve, per source table. Exported (not
# underscore-private) because the forecaster's base-rate builders must never
# show a model a series the resolver will not use to score it — the two are
# pinned together by forecaster/tests/test_base_rate_matches_resolution_source.py.
#
# Note what is deliberately ABSENT: 'in_need'. GDACS writes 'in_need' for
# FL/DR/TC, and it is modelled population exposure (hazard footprint x
# population), not reported impact. It is orders of magnitude larger than an
# IFRC 'affected' figure and is a different quantity in kind, so it must never
# enter a PA series. See docs/montandon_assessment.md.
PA_FACTS_RESOLVED_METRICS: tuple[str, ...] = (
    "affected",
    "people_affected",
    "pa",
    "displaced",
)
PA_FACTS_DELTAS_METRICS: tuple[str, ...] = ("new_displacements",) + PA_FACTS_RESOLVED_METRICS


# ACE/FATALITIES resolves from ONE series: ``acled_monthly_fatalities``, the
# monthly sum of ACLED deaths over ALL event types. It is the series the
# question is worded against, the series the prompt base rate
# (forecaster ``_build_conflict_base_rate``) is drawn from, and the series the
# climatology reference (``pythia.tools.base_rate_spd``) is built from.
#
# Until Sept 2026 FATALITIES read ``facts_resolved``/``facts_deltas`` FIRST,
# where the ACLED adapter had stored the connector's BATTLES-ONLY series under
# the name ``fatalities``: 27 of 32 August ACE/FATALITIES questions resolved
# to a battle-only count, a median 0.42 of the all-types level the model was
# shown. A question's base rate and its resolution must be one quantity.
ACE_FATALITIES_TABLE = "acled_monthly_fatalities"
ACE_FATALITIES_SERIES = f"{ACE_FATALITIES_TABLE}:all_event_types"


def _metric_in_clause(metrics: tuple[str, ...], column: str = "metric") -> str:
    """SQL ``lower(col) IN (...)`` over a metric tuple."""
    joined = ",".join(f"'{m}'" for m in metrics)
    return f"lower({column}) IN ({joined})"

# Preference 2 — publisher tier, mirroring
# resolver/tools/precedence_config.yml (tier 0: IFRC Montandon + ACLED;
# tier 1: IDMC; everything else tier 2). facts_resolved stores publisher
# in mixed case/slug forms ('IFRC', 'ifrc_go', 'ifrc_montandon', ...).
_PUBLISHER_TIER_CASE = (
    "CASE"
    " WHEN lower(COALESCE(publisher, '')) IN"
    " ('ifrc', 'ifrc_go', 'ifrc_montandon', 'acled') THEN 0"
    " WHEN lower(COALESCE(publisher, '')) = 'idmc' THEN 1"
    " ELSE 2 END"
)


def _metric_preference_case(column: str = "metric") -> str:
    """SQL CASE ranking a metric column by ``_PA_METRIC_PREFERENCE``."""
    parts = " ".join(
        f"WHEN lower({column}) = '{m}' THEN {i}"
        for i, m in enumerate(_PA_METRIC_PREFERENCE)
    )
    return f"CASE {parts} ELSE {len(_PA_METRIC_PREFERENCE)} END"


# Shared rollback-safe DuckDB helpers (see pythia/tools/_db_utils.py).
from pythia.tools._db_utils import (
    apply_compute_memory_guard,
    column_exists as _has_column,
    row_count as _row_count,
    table_exists as _table_exists,
)
from pythia.tools.scoring_class import INDICATIVE_REASON_SHORT, conflict_scoring_class
from pythia.tools.base_rate_spd import (
    CONFLICT_DISPLACEMENT_SERIES,
    CONFLICT_STATUS_QUIET,
    CONFLICT_STATUS_REPORTED,
    CONFLICT_STATUS_UNSETTLED,
    conflict_displacement_coverage,
    conflict_displacement_series,
    resolve_conflict_month,
)
from pythia.tools.source_coverage import (
    acled_complete_clause as _acled_complete_clause,
    countries_with_source_data as _coverage_countries,
    months_with_source_data as _coverage_months,
    refresh_source_coverage as _refresh_source_coverage,
)


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
    conn = duckdb_io.get_db(db_url or duckdb_io.DEFAULT_DB_URL)
    apply_compute_memory_guard(conn)
    return conn


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
            ("facts_resolved", _metric_in_clause(PA_FACTS_RESOLVED_METRICS)),
            ("facts_deltas", _metric_in_clause(PA_FACTS_DELTAS_METRICS)),
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
        # The ACE/PA series (IDMC conflict displacement) lives in
        # facts_resolved under a metric the PA filter above leaves out.
        live, _universe = conflict_displacement_coverage(conn)
        if live:
            max_yms.append(max(live))
        if _table_exists(conn, "emdat_pa"):
            try:
                row = conn.execute("SELECT MAX(ym) FROM emdat_pa").fetchone()
                if row and row[0]:
                    max_yms.append(str(row[0]))
            except Exception:
                pass

    elif metric == "FATALITIES":
        # The resolution series alone decides freshness: a facts row named
        # 'fatalities' is an IFRC natural-hazard death count or a legacy
        # battle-only ACLED row, and neither can resolve ACE/FATALITIES.
        if _table_exists(conn, ACE_FATALITIES_TABLE):
            try:
                # Complete rows only: a month whose only rows were written
                # before it ended has not happened yet as far as resolution
                # is concerned (the prompt readers' rule, base_rate_spd).
                complete = _acled_complete_clause(conn, ACE_FATALITIES_TABLE)
                row = conn.execute(
                    "SELECT MAX(strftime(month, '%Y-%m')) "
                    f"FROM acled_monthly_fatalities WHERE {complete}"
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


def _months_with_source_data(conn, metric: str) -> set[str]:
    """Months ('YYYY-MM') where the metric's sources have ANY row globally.

    Gates zero-defaulting: the absence of a row for a (country, month) only
    means "zero impact" when the source actually reported that month at
    all. A month missing from every source table is an ingestion gap (or
    predates source coverage) — horizons in it must stay unresolved rather
    than become false zeros. The freshness cutoff only guards the right
    edge; this guards the left edge and interior gaps.

    Reads the ``source_coverage`` table (rebuilt at run start by
    :func:`pythia.tools.source_coverage.refresh_source_coverage`).
    """
    return _coverage_months(conn, metric)


def _countries_with_source_data(conn, metric: str) -> set[str]:
    """ISO3 codes that appear at least once (any month) in the metric's
    source tables.

    Gates zero-defaulting alongside :func:`_months_with_source_data`: a
    country the source has NEVER reported is outside the source's coverage
    universe, so absence of a row is "unknown", not "zero impact". Countries
    the source does cover still zero-default normally for covered months
    (e.g. a peaceful country appears in ACLED via protests/riots even when
    it has no conflict fatalities).

    Reads the ``source_coverage`` table (rebuilt at run start).
    """
    return _coverage_countries(conn, metric)


# Hazard-code → EM-DAT shock_type mapping for emdat_pa table lookups.
_HAZARD_TO_EMDAT_SHOCK: dict[str, str] = {
    "FL": "flood",
    "DR": "drought",
    "TC": "tropical_cyclone",
    "HW": "heatwave",
}


def _try_facts_resolved(
    conn, iso3: str, hazard_code: str, calendar_month: str, metric: str,
) -> Optional[tuple[float, Optional[str], str]]:
    """Look up in ``facts_resolved`` (IFRC stock data, highest priority).

    Candidate rows are ranked deterministically: metric preference first
    (direct affected counts beat displacement components), then publisher
    tier (IFRC/ACLED > IDMC > others), then recency — never recency alone.
    """
    if not _table_exists(conn, "facts_resolved"):
        return None
    if metric == "PA":
        metric_filter = _metric_in_clause(PA_FACTS_RESOLVED_METRICS)
    else:
        # FATALITIES resolves from ACE_FATALITIES_TABLE alone (see above).
        return None
    # Legacy DBs / minimal test fixtures may lack the publisher column;
    # degrade to metric-preference + recency ordering.
    has_publisher = _has_column(conn, "facts_resolved", "publisher")
    publisher_col = "publisher" if has_publisher else "NULL AS publisher"
    publisher_order = f"{_PUBLISHER_TIER_CASE}, " if has_publisher else ""
    sql = f"""
        SELECT value, created_at, metric, {publisher_col}
        FROM facts_resolved
        WHERE iso3 = ? AND hazard_code = ? AND ym = ? AND {metric_filter}
        ORDER BY {_metric_preference_case()}, {publisher_order}created_at DESC
        LIMIT 1
    """
    try:
        row = conn.execute(sql, [iso3, hazard_code, calendar_month]).fetchone()
    except Exception:
        return None
    if not row:
        return None
    source_desc = f"facts_resolved:{row[3] or '?'}:{row[2]}"
    return float(row[0]), (str(row[1]) if row[1] is not None else None), source_desc


def _try_facts_deltas(
    conn, iso3: str, hazard_code: str, calendar_month: str, metric: str,
) -> Optional[tuple[float, Optional[str], str]]:
    """Look up in ``facts_deltas`` (IDMC flow data, etc.).

    facts_deltas is unique per (ym, iso3, hazard_code, metric), so ranking
    by metric preference fully determines the winner; created_at only
    breaks ties among legacy duplicates.
    """
    if not _table_exists(conn, "facts_deltas"):
        return None
    if metric == "PA":
        metric_filter = _metric_in_clause(PA_FACTS_DELTAS_METRICS)
    else:
        # FATALITIES resolves from ACE_FATALITIES_TABLE alone (see above).
        return None
    sql = f"""
        SELECT COALESCE(value_new, value_stock) AS value, created_at, metric
        FROM facts_deltas
        WHERE iso3 = ? AND hazard_code = ? AND ym = ? AND {metric_filter}
        ORDER BY {_metric_preference_case()}, created_at DESC
        LIMIT 1
    """
    try:
        row = conn.execute(sql, [iso3, hazard_code, calendar_month]).fetchone()
    except Exception:
        return None
    if not row or row[0] is None:
        return None
    return float(row[0]), (str(row[1]) if row[1] is not None else None), f"facts_deltas:{row[2]}"


def _try_emdat_pa(
    conn, iso3: str, hazard_code: str, calendar_month: str,
) -> Optional[tuple[float, Optional[str], str]]:
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
    return float(row[0]), (str(row[1]) if row[1] is not None else None), "emdat_pa"


def _try_acled_fatalities(
    conn, iso3: str, calendar_month: str,
) -> Optional[tuple[float, Optional[str], str]]:
    """Look up all-event-type fatalities in ``acled_monthly_fatalities``.

    Only a COMPLETE row resolves: one written after its month ended. A row
    written before then is a partial count (the ingest used to write the
    month in progress), and resolving against it scores a forecast against
    a fraction of the month — see :func:`_acled_partial_row_exists`.
    """
    if not _table_exists(conn, ACE_FATALITIES_TABLE):
        return None
    complete = _acled_complete_clause(conn, ACE_FATALITIES_TABLE)
    sql = f"""
        SELECT fatalities, updated_at
        FROM {ACE_FATALITIES_TABLE}
        WHERE iso3 = ? AND strftime(month, '%Y-%m') = ?
          AND {complete}
        LIMIT 1
    """
    try:
        row = conn.execute(sql, [iso3, calendar_month]).fetchone()
    except Exception:
        return None
    if not row or row[0] is None:
        return None
    return (
        float(row[0]),
        (str(row[1]) if row[1] is not None else None),
        ACE_FATALITIES_SERIES,
    )


def _acled_partial_row_exists(conn, iso3: str, calendar_month: str) -> bool:
    """True when the series holds a row for this cell that was written
    before its month ended (and so was refused by :func:`_try_acled_fatalities`).

    Such a horizon stays UNRESOLVED: it must neither resolve to the partial
    figure nor zero-default, because "the source has a row and we will not
    read it" is not "the source reported nothing".
    """
    if not _table_exists(conn, ACE_FATALITIES_TABLE):
        return False
    complete = _acled_complete_clause(conn, ACE_FATALITIES_TABLE)
    if complete == "TRUE":
        return False
    try:
        row = conn.execute(
            f"""
            SELECT 1 FROM {ACE_FATALITIES_TABLE}
            WHERE iso3 = ? AND strftime(month, '%Y-%m') = ?
              AND NOT ({complete})
            LIMIT 1
            """,
            [iso3, calendar_month],
        ).fetchone()
    except Exception:
        return False
    return row is not None


def _try_gdacs_binary(
    conn, iso3: str, hazard_code: str, calendar_month: str,
) -> Optional[tuple[float, Optional[str], str]]:
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
    return (
        float(row[0]),
        (str(row[1]) if row[1] is not None else None),
        "facts_resolved:event_occurrence",
    )


def _try_phase3plus(
    conn, iso3: str, hazard_code: str, calendar_month: str,
) -> Optional[tuple[float, Optional[str], str]]:
    """Look up Phase 3+ population in ``facts_resolved`` (FEWS NET or IPC)."""
    if hazard_code != "DR":
        return None
    if not _table_exists(conn, "facts_resolved"):
        return None
    publisher_col = (
        "publisher"
        if _has_column(conn, "facts_resolved", "publisher")
        else "NULL AS publisher"
    )
    sql = f"""
        SELECT value, created_at, {publisher_col}
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
    return (
        float(row[0]),
        (str(row[1]) if row[1] is not None else None),
        f"facts_resolved:{row[2] or '?'}:phase3plus_in_need",
    )


def _resolve_value(
    conn,
    iso3: str,
    hazard_code: str,
    calendar_month: str,
    metric: str,
) -> Optional[tuple[float, Optional[str], str]]:
    """Resolve a single metric for (iso3, hazard_code, calendar_month).

    Returns (value, source_timestamp, source_description) or None.

    Checks multiple Resolver tables in priority order.  The dispatch
    depends on the metric type:

    PA:
      1. ``facts_resolved`` — IFRC stock data (highest source priority)
      2. ``facts_deltas``   — IDMC flow data and derived deltas
      3. ``emdat_pa``       — EM-DAT people-affected
    FATALITIES:
      1. ``acled_monthly_fatalities`` — ACLED deaths over ALL event types,
         the series the question and its base rate are defined on. Nothing
         else: the battle-only series and IFRC death counts are different
         quantities.
    EVENT_OCCURRENCE:
      1. ``facts_resolved`` (GDACS binary event rows)
    PHASE3PLUS_IN_NEED:
      1. ``facts_resolved`` (FEWS NET or IPC Phase 3+ data)
    """

    if metric == "EVENT_OCCURRENCE":
        return _try_gdacs_binary(conn, iso3, hazard_code, calendar_month)

    if metric == "PHASE3PLUS_IN_NEED":
        return _try_phase3plus(conn, iso3, hazard_code, calendar_month)

    if metric == "FATALITIES":
        return _try_acled_fatalities(conn, iso3, calendar_month)

    # PA: priority cascade
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

    return None


def _should_default_to_zero(metric_norm: str, hazard_norm: str) -> bool:
    """Return True if absence of source data means zero impact for this
    metric/hazard combination."""
    allowed_hazards = _ZERO_DEFAULT_RULES.get(metric_norm)
    if allowed_hazards is not None and hazard_norm in allowed_hazards:
        return True
    return False



def _purge_non_series_fatalities(conn) -> int:
    """Delete FATALITIES resolutions not drawn from ``ACE_FATALITIES_TABLE``.

    Zero defaults are kept (they are re-derived from the same series'
    coverage). Idempotent: a DB with no such rows is untouched.
    """
    if not (_table_exists(conn, "resolutions") and _table_exists(conn, "questions")):
        return 0
    where = f"""
        question_id IN (
            SELECT question_id FROM questions WHERE upper(metric) = 'FATALITIES'
        )
        AND COALESCE(source_desc, '') <> 'zero_default'
        AND COALESCE(source_desc, '') NOT LIKE '{ACE_FATALITIES_TABLE}%'
    """
    try:
        n = int(conn.execute(f"SELECT COUNT(*) FROM resolutions WHERE {where}").fetchone()[0])
        if n:
            conn.execute(f"DELETE FROM resolutions WHERE {where}")
            LOGGER.warning(
                "compute_resolutions: purged %d FATALITIES resolution(s) not drawn "
                "from %s (battle-only facts rows); they are re-resolved below.",
                n, ACE_FATALITIES_SERIES,
            )
        return n
    except Exception as exc:  # noqa: BLE001 - the purge never blocks resolution
        LOGGER.warning("compute_resolutions: FATALITIES purge failed: %r", exc)
        return 0


# ---------------------------------------------------------------------------
# Resolution vintages (ACE/FATALITIES)
# ---------------------------------------------------------------------------

#: ACLED revises a month's counts for weeks after it ends, and every run of
#: this module re-resolves every horizon and REPLACES the row, so a revision
#: used to be invisible: the resolution a forecast was first scored against
#: was gone. The first resolution and the ones taken at 60 and 90 days after
#: month end are kept here and never overwritten.
#:
#: The resolver runs on the 11th (``resolver_update.yml``; the 28th until
#: 2026-10-05), so a month is first resolved ~11 days after it ends and again
#: at ~41, ~72 and ~103 days. A milestone is recorded by the first run at or
#: after ``days - VINTAGE_TOLERANCE_DAYS``, so on the 11th cycle ``d60`` lands
#: at ~72 days and ``d90`` at ~103 (the ~41-day run records no milestone),
#: and the row carries the ACTUAL day count, never the milestone's. The
#: milestones keep their meaning (a month's count about two and three months
#: on) rather than following the calendar, so vintages written under the two
#: cycles stay comparable; ``days_after_month_end`` says how far each was.
VINTAGE_MILESTONES: tuple[tuple[str, int], ...] = (("d60", 60), ("d90", 90))
VINTAGE_TOLERANCE_DAYS = 5
VINTAGE_METRICS = frozenset({"FATALITIES"})
#: Conflict displacement (ACE/PA) settles far more slowly: of a month's
#: eventual recommended IDMC figure, 68% had arrived 90 days after month end
#: and 90% at 180 (probe of 6 Oct 2026). The first reading is the one taken
#: at the 90-day settle period; the later ones at ~180 and ~270 days.
VINTAGE_MILESTONES_BY_GROUP: dict[tuple[str, str], tuple[tuple[str, int], ...]] = {
    ("*", "FATALITIES"): VINTAGE_MILESTONES,
    ("ACE", "PA"): (("d180", 180), ("d270", 270)),
}


def vintage_milestones(hazard: str, metric: str) -> Optional[tuple[tuple[str, int], ...]]:
    """The later readings kept for a (hazard, metric), or None for none."""
    key = (str(hazard or "").upper(), str(metric or "").upper())
    return VINTAGE_MILESTONES_BY_GROUP.get(key) or VINTAGE_MILESTONES_BY_GROUP.get(("*", key[1]))


def reading_label(hazard: str, metric: str, observed_month: str, as_of: date) -> str:
    """Which reading a resolution taken on ``as_of`` is: ``first`` or the
    latest milestone reached (the scored bundle states it per score)."""
    milestones = vintage_milestones(hazard, metric) or ()
    days = _days_after_month_end(observed_month, as_of)
    label = "first"
    if days is not None:
        for name, milestone in milestones:
            if days >= milestone - VINTAGE_TOLERANCE_DAYS:
                label = name
    return label


def _ensure_vintage_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS resolution_vintages (
          question_id TEXT,
          horizon_m INTEGER,
          vintage TEXT,
          observed_month TEXT,
          days_after_month_end INTEGER,
          value DOUBLE,
          source_desc TEXT,
          acled_snapshot_date DATE,
          recorded_at TIMESTAMP,
          is_test BOOLEAN DEFAULT FALSE,
          PRIMARY KEY (question_id, horizon_m, vintage)
        )
        """
    )


def _days_after_month_end(observed_month: str, today: date) -> Optional[int]:
    try:
        y, m = int(observed_month[:4]), int(observed_month[5:7])
    except (TypeError, ValueError):
        return None
    first_next = date(y + (m // 12), m % 12 + 1, 1)
    return (today - first_next).days + 1


def _snapshot_date(source_ts: Optional[str]) -> Optional[str]:
    if not source_ts:
        return None
    text = str(source_ts)[:10]
    try:
        date.fromisoformat(text)
    except ValueError:
        return None
    return text


def record_vintages(
    conn,
    *,
    question_id: str,
    horizon_m: int,
    observed_month: str,
    value: float,
    source_desc: str,
    source_ts: Optional[str],
    today: date,
    is_test: bool,
    milestones: Optional[tuple[tuple[str, int], ...]] = None,
) -> list[str]:
    """Insert the vintages this resolution completes; returns their labels.

    ``first`` is written the first time the horizon resolves; ``d60``/``d90``
    once the month is old enough. Existing vintages are never touched
    (``INSERT OR IGNORE``), which is the whole point.
    """
    days = _days_after_month_end(observed_month, today)
    labels = ["first"]
    if days is not None:
        labels += [
            label for label, milestone in (milestones or VINTAGE_MILESTONES)
            if days >= milestone - VINTAGE_TOLERANCE_DAYS
        ]
    written: list[str] = []
    for label in labels:
        before = conn.execute(
            "SELECT COUNT(*) FROM resolution_vintages "
            "WHERE question_id = ? AND horizon_m = ? AND vintage = ?",
            [question_id, horizon_m, label],
        ).fetchone()[0]
        if before:
            continue
        conn.execute(
            """
            INSERT INTO resolution_vintages (
              question_id, horizon_m, vintage, observed_month,
              days_after_month_end, value, source_desc, acled_snapshot_date,
              recorded_at, is_test
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [question_id, horizon_m, label, observed_month, days, float(value),
             source_desc, _snapshot_date(source_ts), _utcnow_naive(), is_test],
        )
        written.append(label)
    return written


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
          source_desc TEXT,
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
    if "acled_snapshot_date" not in existing:
        try:
            conn.execute("ALTER TABLE resolutions ADD COLUMN acled_snapshot_date DATE")
        except Exception:
            pass
    if "source_desc" not in existing:
        try:
            conn.execute("ALTER TABLE resolutions ADD COLUMN source_desc TEXT")
        except Exception:
            pass
    # ACE/PA scoring class (Oct 2026, pythia/tools/scoring_class.py): NULL
    # for every other hazard and metric, which reads as scored.
    for col in ("scoring_class", "scoring_class_reason"):
        if col not in existing:
            try:
                conn.execute(f"ALTER TABLE resolutions ADD COLUMN {col} TEXT")
            except Exception:
                pass


def compute_resolutions(db_url: str, today: Optional[date] = None) -> dict:
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
        _ensure_vintage_table(conn)

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

        # Purge ACE/FATALITIES resolutions read from a facts table: every one
        # of them is a battle-only count (see ACE_FATALITIES_SERIES). This run
        # rewrites each horizon from the all-types series; deleting first
        # means a horizon it can no longer resolve is left unresolved rather
        # than keeping the wrong figure.
        _purge_non_series_fatalities(conn)

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
        vintages_written = 0
        resolved_from_source = 0
        resolved_as_zero = 0
        skipped_no_data_coverage = 0
        skipped_null_resolution = 0
        skipped_unresolvable_hazard = 0
        skipped_outside_source_universe = 0
        skipped_partial_month = 0
        skipped_conflict_displacement_gate = 0
        # The IDMC conflict displacement series (reported and held-out
        # months per country), read once per run on the first ACE/PA horizon.
        conflict_series = None
        conflict_status_counts: Counter = Counter()
        scoring_counts: Counter = Counter()
        # Per (hazard, metric): how many horizon-months resolved from a
        # source, defaulted to zero, or stayed unresolved and why. A group
        # resolved mostly by zero-defaults is a group whose outcomes are
        # mostly the resolver's inference (CLAUDE.md, the 2026-10-06 entry).
        outcome_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)

        # A production question a same-epoch test scan re-pointed goes back
        # to its production scan before anything is resolved from it.
        try:
            from pythia.tools.question_repairs import repair_questions_pointing_at_test_scans

            repair_questions_pointing_at_test_scans(conn)
        except Exception as exc:  # noqa: BLE001 - a repair must never stop resolution
            LOGGER.warning("question provenance repair skipped: %s", exc)

        # Rebuild the source_coverage table from the metric source tables so
        # the gates below (and any dashboard consumer) see current coverage.
        _refresh_source_coverage(conn)

        # Per-month global source coverage for zero-default metrics: a month
        # absent from every source table is an ingestion gap, not "no impact".
        zero_default_coverage: dict[str, set[str]] = {
            m: _months_with_source_data(conn, m)
            for m in _ZERO_DEFAULT_RULES
        }

        # Per-country source universe: countries the source has NEVER
        # reported stay unresolved instead of zero-defaulting. Applies to
        # FATALITIES only — ACLED has a defined coverage universe that
        # expanded over time. EVENT_OCCURRENCE is exempt: GDACS coverage is
        # satellite-global and only writes rows where events occurred, so a
        # country with no GDACS rows genuinely had no qualifying events.
        universe_gated_metrics = {"FATALITIES"}
        zero_default_universe: dict[str, set[str]] = {
            m: _countries_with_source_data(conn, m)
            for m in _ZERO_DEFAULT_RULES
            if m in universe_gated_metrics
        }

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

            # is_test is per question — look it up once, not per horizon.
            try:
                q_test = conn.execute(
                    "SELECT COALESCE(is_test, FALSE) FROM questions WHERE question_id = ?",
                    [question_id],
                ).fetchone()
                is_test_val = q_test[0] if q_test else False
            except Exception:
                is_test_val = False

            for horizon_m in range(1, NUM_HORIZONS + 1):
                cal_month = horizon_to_calendar_month(ws_date, horizon_m)
                group = (hazard_norm, metric_norm)
                scoring = None

                if hazard_norm == "ACE" and metric_norm == "PA":
                    # ACE/PA resolves from the IDMC conflict displacement
                    # series alone, under the one rule the prompt, the anchor
                    # and the references read (base_rate_spd.
                    # resolve_conflict_month): a month resolves once SETTLED;
                    # a missing month is zero only for a regular reporter
                    # with a later report. Anything else is unknown, and a
                    # row an earlier rule wrote for it is deleted, because
                    # IDMC reports late and a trailing month is not quiet.
                    if cal_month > cal_cutoff:
                        skipped_no_data_coverage += 1
                        outcome_counts[group]["unresolved_future"] += 1
                        continue
                    if conflict_series is None:
                        conflict_series = conflict_displacement_series(conn)
                    reported_all, held_all = conflict_series
                    value_cd, status = resolve_conflict_month(
                        reported_all.get(iso3_norm, {}), cal_month, today,
                        held=held_all.get(iso3_norm, set()),
                    )
                    conflict_status_counts[status] += 1
                    scoring = conflict_scoring_class(reported_all.get(iso3_norm, {}), cal_month)
                    if status not in (CONFLICT_STATUS_REPORTED, CONFLICT_STATUS_QUIET):
                        skipped_conflict_displacement_gate += 1
                        outcome_counts[group][
                            "unresolved_for_lag" if status == CONFLICT_STATUS_UNSETTLED
                            else status
                        ] += 1
                        conn.execute(
                            "DELETE FROM resolutions WHERE question_id = ? AND horizon_m = ?",
                            [question_id, horizon_m],
                        )
                        continue
                    if status == CONFLICT_STATUS_QUIET:
                        resolved_as_zero += 1
                        outcome_counts[group]["zero_default"] += 1
                        resolved = (0.0, None, "zero_default")
                    else:
                        resolved_from_source += 1
                        outcome_counts[group]["sourced"] += 1
                        resolved = (float(value_cd), None, CONFLICT_DISPLACEMENT_SERIES)
                    scoring_counts[scoring[0]] += 1

                # A FATALITIES cell whose only row was written before its
                # month ended stays unresolved and is counted as such, not
                # filed under "no data coverage yet" — the reader should see
                # that the row exists and was refused.
                elif metric_norm == "FATALITIES" and (
                    data_cutoff is None or cal_month > data_cutoff
                ) and _acled_partial_row_exists(conn, iso3_norm, cal_month):
                    skipped_partial_month += 1
                    outcome_counts[group]["unresolved_partial_month"] += 1
                    continue

                # Only resolve months for which source data exists.
                elif data_cutoff is None or cal_month > data_cutoff:
                    skipped_no_data_coverage += 1
                    outcome_counts[group]["unresolved_no_coverage"] += 1
                    continue

                else:
                    resolved = _resolve_value(
                        conn, iso3_norm, hazard_norm, cal_month, metric_norm,
                    )
                if resolved is None and metric_norm == "FATALITIES" and (
                    _acled_partial_row_exists(conn, iso3_norm, cal_month)
                ):
                    # A row exists but was written before its month ended:
                    # unresolved, never zero (CLAUDE.md, Invariants).
                    skipped_partial_month += 1
                    outcome_counts[group]["unresolved_partial_month"] += 1
                    continue
                if resolved is None:
                    # Source-aware null handling: only default to zero for
                    # sources where absence genuinely means zero impact —
                    # and only when the source actually covered this month
                    # at all (ingestion gaps must not become false zeros).
                    if _should_default_to_zero(metric_norm, hazard_norm):
                        covered = zero_default_coverage.get(metric_norm) or set()
                        if cal_month not in covered:
                            skipped_no_data_coverage += 1
                            outcome_counts[group]["unresolved_no_coverage"] += 1
                            continue
                        universe = zero_default_universe.get(metric_norm)
                        if universe is not None and iso3_norm not in universe:
                            skipped_outside_source_universe += 1
                            outcome_counts[group]["unresolved_outside_universe"] += 1
                            continue
                        value: float = 0.0
                        source_ts: Optional[str] = None
                        source_desc = "zero_default"
                        resolved_as_zero += 1
                        outcome_counts[group]["zero_default"] += 1
                    else:
                        # All other sources: no record = unresolvable.
                        # Do NOT write a resolution row — leave this
                        # horizon unresolved so scoring skips it.
                        skipped_null_resolution += 1
                        outcome_counts[group]["unresolved_no_data"] += 1
                        continue
                elif hazard_norm == "ACE" and metric_norm == "PA":
                    value, source_ts, source_desc = resolved
                else:
                    value, source_ts, source_desc = resolved
                    resolved_from_source += 1
                    outcome_counts[group]["sourced"] += 1

                conn.execute(
                    """
                    INSERT OR REPLACE INTO resolutions (
                      question_id,
                      horizon_m,
                      observed_month,
                      value,
                      source_snapshot_ym,
                      source_desc,
                      created_at,
                      is_test,
                      acled_snapshot_date,
                      scoring_class,
                      scoring_class_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        question_id,
                        horizon_m,
                        cal_month,
                        float(value),
                        source_ts,
                        source_desc,
                        _utcnow_naive(),
                        is_test_val,
                        # The ACLED pull this figure came from (its
                        # updated_at), stated as a date on the row.
                        _snapshot_date(source_ts) if metric_norm == "FATALITIES" else None,
                        scoring[0] if scoring else None,
                        scoring[1] if scoring else None,
                    ],
                )
                written += 1
                milestones = vintage_milestones(hazard_norm, metric_norm)
                if milestones:
                    try:
                        vintages_written += len(record_vintages(
                            conn,
                            question_id=question_id,
                            horizon_m=horizon_m,
                            observed_month=cal_month,
                            value=float(value),
                            source_desc=source_desc,
                            source_ts=source_ts,
                            today=today,
                            is_test=bool(is_test_val),
                            milestones=milestones,
                        ))
                    except Exception as exc:  # noqa: BLE001 - a vintage never blocks a resolution
                        LOGGER.warning("resolution vintage for %s h%d failed: %r",
                                       question_id, horizon_m, exc)
                LOGGER.info(
                    "Resolved %s h%d (%s/%s/%s %s) -> value=%.1f source=%s source_ts=%s",
                    question_id,
                    horizon_m,
                    iso3_norm,
                    hazard_norm,
                    cal_month,
                    metric_norm,
                    value,
                    source_desc,
                    source_ts or "<none>",
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
            "%d horizon-months skipped (country outside source universe), "
            "%d horizon-months skipped (only a partial-month ACLED row), "
            "%d ACE/PA horizon-months unresolved (unsettled, trailing, held "
            "out, or an irregular reporter's missing month), "
            "%d questions skipped (unresolvable hazard); "
            "%d new FATALITIES resolution vintage(s) recorded.",
            len(rows),
            written,
            resolved_from_source,
            resolved_as_zero,
            skipped_null_resolution,
            skipped_no_data_coverage,
            skipped_outside_source_universe,
            skipped_partial_month,
            skipped_conflict_displacement_gate,
            skipped_unresolvable_hazard,
            vintages_written,
        )
        if scoring_counts:
            LOGGER.info(
                "ACE/PA resolutions by scoring class: %s (indicative = %s; "
                "kept out of calibration, advice, recalibration, centroids, the "
                "Sibyl comparison and headline skill)",
                dict(scoring_counts), INDICATIVE_REASON_SHORT,
            )
        if conflict_status_counts:
            LOGGER.info("ACE/PA horizon-months by conflict displacement status: %s",
                        dict(conflict_status_counts))
        report_outcome_counts(outcome_counts)
        return {f"{hz}/{m}": dict(c) for (hz, m), c in sorted(outcome_counts.items())}
    finally:
        _close_db(conn)


#: Above this share of zero-defaults among a group's resolutions, the group's
#: outcomes are mostly the resolver's inference rather than a source's figure,
#: and every report that reads them says so (CLAUDE.md, the 2026-10-06
#: conflict displacement entry: 64 of 96 ACE/PA resolutions were zeros, 60 of
#: them months IDMC had simply not reported yet).
ZERO_DEFAULT_SHARE_LIMIT = 0.5


def mostly_zero_default_groups(counts) -> list[str]:
    """Groups (``"HAZ/METRIC"`` keys mapping to ``(sourced, zero_default)``)
    whose zero-defaults exceed ``ZERO_DEFAULT_SHARE_LIMIT`` of their
    resolutions. EVENT_OCCURRENCE is exempt: a binary event resolves "no
    event" in most months by design, and the limit is about magnitudes."""
    out: list[str] = []
    for group, (sourced, zeros) in sorted(counts.items()):
        if str(group).upper().endswith("/EVENT_OCCURRENCE"):
            continue
        total = int(sourced) + int(zeros)
        if total and int(zeros) / total > ZERO_DEFAULT_SHARE_LIMIT:
            out.append(str(group))
    return out


def report_outcome_counts(outcome_counts) -> list[str]:
    """Log, and append to the step summary, how each (hazard, metric)
    resolved this run: sourced, zero-default, and unresolved by reason.
    Returns the groups whose zero-defaults exceed half their resolutions."""
    lines = [
        "| hazard/metric | sourced | zero-default | unresolved (lag) | unresolved (other) |",
        "|---|---:|---:|---:|---:|",
    ]
    pairs: dict[str, tuple[int, int]] = {}
    for (hz, m), counts in sorted(outcome_counts.items()):
        sourced = int(counts.get("sourced", 0))
        zeros = int(counts.get("zero_default", 0))
        lag = int(counts.get("unresolved_for_lag", 0))
        other = sum(
            int(v) for k, v in counts.items()
            if k.startswith("unresolved") and k != "unresolved_for_lag"
        )
        lines.append(f"| {hz}/{m} | {sourced} | {zeros} | {lag} | {other} |")
        pairs[f"{hz}/{m}"] = (sourced, zeros)
    flagged = mostly_zero_default_groups(pairs)
    LOGGER.info("Resolution outcomes this run:\n%s", "\n".join(lines))
    for group in flagged:
        print(
            f"::warning title=Resolutions mostly zero-defaults::{group}: more than "
            f"{int(ZERO_DEFAULT_SHARE_LIMIT * 100)}% of this run's resolutions are zero-defaults"
        )
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        try:
            with open(summary, "a", encoding="utf-8") as fh:
                fh.write("### Resolution outcomes\n\n" + "\n".join(lines) + "\n\n")
        except OSError:
            pass
    return flagged


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
