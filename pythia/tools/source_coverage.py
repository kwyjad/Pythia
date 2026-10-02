# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Per-country source-coverage table for resolution gating.

``source_coverage`` records, per (metric, iso3, ym), how many rows the
metric's resolution source tables contain. It is rebuilt at the start of
each ``compute_resolutions`` run and consulted for zero-default gating:

- **month gate** — a horizon month only zero-defaults when the source has
  at least one row globally in that calendar month (months inside ingestion
  gaps or before source coverage began stay unresolved);
- **country-universe gate** (FATALITIES only) — a country must appear at
  least once, in any month, in the metric's source tables; countries the
  source has never reported stay unresolved. EVENT_OCCURRENCE is
  deliberately exempt — GDACS coverage is satellite-global and only writes
  rows where events occurred, so no rows genuinely means no qualifying
  events (see docs/audit_2026-07.md).

This table supersedes the July 2026 audit's in-memory heuristics
(``_months_with_source_data`` / ``_countries_with_source_data``) with the
same gating semantics, while leaving an auditable, queryable coverage map
in the DB.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from pythia.tools._db_utils import column_exists, rollback_quietly, table_exists

LOGGER = logging.getLogger(__name__)

# Per-metric queries yielding (iso3, ym, row_count) from the metric's
# resolution source tables. Mirrors the source set used by
# compute_resolutions' zero-default rules — extend both together.
_COVERAGE_QUERIES: dict[str, list[tuple[str, str]]] = {
    # FATALITIES coverage is the resolution series' own coverage. A facts row
    # named 'fatalities' is an IFRC natural-hazard death count or a legacy
    # battle-only ACLED row, and counting either would widen the month and
    # country gates with rows that cannot resolve a question (Sept 2026).
    "FATALITIES": [
        ("acled_monthly_fatalities",
         "SELECT upper(iso3) AS iso3, strftime(month, '%Y-%m') AS ym, COUNT(*) AS n "
         "FROM acled_monthly_fatalities WHERE iso3 IS NOT NULL AND month IS NOT NULL "
         "AND {acled_complete} "
         "GROUP BY 1, 2"),
    ],
    "EVENT_OCCURRENCE": [
        ("facts_resolved",
         "SELECT upper(iso3) AS iso3, ym, COUNT(*) AS n FROM facts_resolved "
         "WHERE lower(metric) = 'event_occurrence' AND iso3 IS NOT NULL AND ym IS NOT NULL "
         "GROUP BY 1, 2"),
    ],
}


#: A row of ``acled_monthly_fatalities`` written before its month ended is a
#: PARTIAL count, not a month (Sept 2026, CLAUDE.md "Connectors and upstream
#: data"). It must not make the month "covered", or a country with no row
#: that month would zero-default against a month the source has not finished
#: reporting. The predicate is the one every prompt reader uses
#: (``pythia.tools.base_rate_spd.ACLED_COMPLETE_MONTH_SQL``), imported rather
#: than copied so the two can never drift.
def acled_complete_clause(conn, table: str = "acled_monthly_fatalities") -> str:
    """The complete-month predicate when ``table`` records ``updated_at``,
    else ``TRUE`` (a hand-built test table or a pre-stamp database)."""
    from pythia.tools.base_rate_spd import ACLED_COMPLETE_MONTH_SQL

    try:
        if column_exists(conn, table, "updated_at"):
            return ACLED_COMPLETE_MONTH_SQL
    except Exception:  # noqa: BLE001 - a probe failure keeps the old behaviour
        rollback_quietly(conn)
    return "TRUE"


def _ensure_table(conn) -> None:
    """Create source_coverage if the synced DB predates the schema entry."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS source_coverage (
            metric VARCHAR NOT NULL,
            iso3 VARCHAR NOT NULL,
            ym VARCHAR NOT NULL,
            row_count INTEGER NOT NULL DEFAULT 0,
            refreshed_at TIMESTAMP,
            PRIMARY KEY (metric, iso3, ym)
        )
        """
    )


def refresh_source_coverage(conn) -> dict[str, int]:
    """Rebuild source_coverage from the metric source tables.

    Returns per-metric row counts. Missing source tables are skipped
    (a fresh/partial DB is not an error); failed queries are logged and
    skipped so one broken source never blocks resolution of the others.
    """
    _ensure_table(conn)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    written: dict[str, int] = {}
    for metric, queries in _COVERAGE_QUERIES.items():
        rows: dict[tuple[str, str], int] = {}
        for table, sql in queries:
            if not table_exists(conn, table):
                continue
            if "{acled_complete}" in sql:
                sql = sql.replace("{acled_complete}", acled_complete_clause(conn, table))
            try:
                for iso3, ym, n in conn.execute(sql).fetchall():
                    if not iso3 or not ym:
                        continue
                    key = (str(iso3), str(ym))
                    rows[key] = rows.get(key, 0) + int(n or 0)
            except Exception as exc:
                LOGGER.warning(
                    "source_coverage: query on %s for %s failed: %r", table, metric, exc
                )
                rollback_quietly(conn)
        conn.execute("DELETE FROM source_coverage WHERE metric = ?", [metric])
        if rows:
            conn.executemany(
                "INSERT INTO source_coverage (metric, iso3, ym, row_count, refreshed_at) "
                "VALUES (?, ?, ?, ?, ?)",
                [[metric, iso3, ym, n, now] for (iso3, ym), n in rows.items()],
            )
        written[metric] = len(rows)
        LOGGER.info("source_coverage: %s -> %d (iso3, ym) cells", metric, len(rows))
    return written


def months_with_source_data(conn, metric: str) -> set[str]:
    """Months ('YYYY-MM') where the metric's sources have ANY row globally."""
    if not table_exists(conn, "source_coverage"):
        return set()
    try:
        rows = conn.execute(
            "SELECT DISTINCT ym FROM source_coverage WHERE metric = ?", [metric]
        ).fetchall()
        return {str(r[0]) for r in rows if r[0]}
    except Exception as exc:
        LOGGER.warning("source_coverage month read failed for %s: %r", metric, exc)
        rollback_quietly(conn)
        return set()


def countries_with_source_data(conn, metric: str) -> set[str]:
    """ISO3s that appear at least once (any month) in the metric's sources."""
    if not table_exists(conn, "source_coverage"):
        return set()
    try:
        rows = conn.execute(
            "SELECT DISTINCT iso3 FROM source_coverage WHERE metric = ?", [metric]
        ).fetchall()
        return {str(r[0]) for r in rows if r[0]}
    except Exception as exc:
        LOGGER.warning("source_coverage country read failed for %s: %r", metric, exc)
        rollback_quietly(conn)
        return set()
