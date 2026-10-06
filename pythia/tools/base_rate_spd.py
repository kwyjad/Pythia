# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Base-rate SPDs: the anchor distribution the forecaster was shown, as buckets.

``base_rate_spd(con, iso3, hazard_code, metric, as_of)`` returns the base-rate
probability distribution over the metric's buckets, derived from THE SAME
tables, filters and exclusion rules the prompt-time base-rate loaders query.
Routing mirrors ``forecaster.cli._build_history_summary`` exactly:

- ACE/FATALITIES  -> ``acled_monthly_fatalities`` (``_build_conflict_base_rate``)
- ACE/PA          -> IDMC flow rows in ``facts_deltas`` (same builder)
- DR/PHASE3PLUS_IN_NEED -> ``facts_resolved`` ``phase3plus_in_need``
                    (``_load_fewsnet_phase3_history``, 36-month window)
- FL/TC (+HW)/PA  -> occurrence x severity mixture: GDACS ``event_occurrence``
                    seasonal rates (``_build_gdacs_event_history``) x historical
                    PA magnitudes filtered by ``_pa_metric_in_clause()``
                    (``_build_natural_hazard_seasonal_profile``)
- */EVENT_OCCURRENCE (FL/DR/TC) -> binary ``[p, 1-p]`` from the same
                    ``event_occurrence`` seasonal rates ``build_binary_base_rate``
                    renders (``forecaster/binary_prompts.py``)
- CU, DI, anything else -> no base rate (``([], "NONE", ...)``), matching the
                    ``no_base_rate`` terminal in ``_build_history_summary``.

Why this must not drift from the prompt-time loaders: the deviation metric
(``compute_deviation``) measures how far the ensemble moved AWAY from its
anchor. If the comparison anchor differs from what the model actually saw, the
deviation measures nothing useful. The same coupling argument as
``_pa_metric_in_clause`` (a base rate must be drawn from the same source class
that resolves the question) applies here one level up.

Determinism and leakage: only months STRICTLY BEFORE ``as_of`` (the question's
window_start month) enter the distribution, so an anchor recomputed after the
window resolves is identical to the one computable at forecast time, and a
climatology reference forecast (``score_baselines``) never contains its own
outcome months.

Smoothing: empirical bucket counts get a Jeffreys pseudo-count of 0.5 per
bucket before normalising (and binary rates are computed as
``(events + 0.5) / (total + 1)``). Small samples therefore never assign an
exact zero to a bucket, without introducing a tunable parameter.

All bucket structure comes from ``pythia.buckets`` — no literal thresholds,
labels, or centroid lists here.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pythia.buckets import get_bucket_specs, n_buckets_for

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    LOGGER.addHandler(logging.NullHandler())

# Jeffreys prior pseudo-count added to every bucket before normalising.
SMOOTHING_PSEUDOCOUNT = 0.5

# Window lengths.
#
# The conflict window used to be 6, mirroring the SIX ROWS the prompt-time
# trajectory block prints. That was a mistake of kind: the prompt block is a
# trajectory (last month, trailing average, direction) and six points are
# plenty for it, while this is a DISTRIBUTION over seven buckets, where six
# observations leave the Jeffreys smoothing carrying nearly as much weight as
# the data. It also made the thin-anchor flag arithmetic rather than
# evidence: with a 6-month window and a 12-observation cutoff, no armed
# conflict anchor could ever clear it, which is most of why 138 of 185 anchors
# were marked thin.
#
# 36 months matches the Phase 3+ window, is well inside what ACLED serves, and
# leaves the SOURCE class untouched — the invariant that matters is that an
# anchor is drawn from the series that will resolve the question, not that it
# uses the same number of rows as a prose summary.
CONFLICT_WINDOW_MONTHS = 36
PHASE3_WINDOW_MONTHS = 36       # _load_fewsnet_phase3_history(months=36)

# A month in which a source recorded nothing for a country is an OBSERVED
# ZERO, not an absent observation — but only where the source was live and
# the country is inside its universe. Both gates matter: without them an
# ingestion gap becomes a run of quiet months and the anchor understates,
# which is the same defect `source_coverage` exists to prevent on the
# resolution side.
#
# Without this the conflict anchors were built only from months something was
# reported, so they put almost no weight on the "nothing recorded" bucket and
# said displacement happens every month in countries where IDMC reports twice
# a year.
COUNT_QUIET_MONTHS_AS_ZERO = True

# Hazards the seasonal-profile / GDACS loaders cover (see NATURAL_HAZARD_CODES
# in forecaster/history_loaders.py and _build_gdacs_event_history's guard).
_SEASONAL_PA_HAZARDS = ("FL", "TC", "HW", "DR")
_GDACS_HAZARDS = ("FL", "DR", "TC")

NO_BASE_RATE_SOURCE = "NONE"


def _parse_ym(value: Any) -> Optional[str]:
    """Normalise a month key to 'YYYY-MM', or None."""
    s = str(value or "").strip()
    if len(s) >= 7 and s[4] == "-":
        head = s[:7]
        try:
            y = int(head[:4])
            m = int(head[5:7])
        except ValueError:
            return None
        if 1 <= m <= 12 and 1900 <= y <= 2200:
            return head
    return None


def _as_of_ym(as_of: Any) -> str:
    """Normalise ``as_of`` (str 'YYYY-MM[-DD]' or date) to 'YYYY-MM'."""
    if isinstance(as_of, date):
        return as_of.strftime("%Y-%m")
    ym = _parse_ym(as_of)
    if ym is None:
        raise ValueError(f"as_of must be a YYYY-MM month key or date, got {as_of!r}")
    return ym


def _add_months(ym: str, n: int) -> str:
    y = int(ym[:4])
    m = int(ym[5:7]) + n
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    return f"{y:04d}-{m:02d}"


def forecast_months(as_of: Any, n: int = 6) -> List[str]:
    """The question's forecast window months: as_of .. as_of+n-1 (month 1 =
    the window_start month, per the repo-wide month-anchoring convention)."""
    ym = _as_of_ym(as_of)
    return [_add_months(ym, i) for i in range(n)]


def _bucket_index_for_value(value: float, metric: str) -> Optional[int]:
    """0-based bucket index for a value (same semantics as compute_scores)."""
    specs = get_bucket_specs(metric)
    if not specs:
        return None
    v = float(value)
    if v < 0.0 or v != v:  # negative or NaN
        return None
    for i, s in enumerate(specs):
        lower = float(s.lower) if s.lower is not None else 0.0
        upper = float(s.upper) if s.upper is not None else float("inf")
        if lower <= v < upper:
            return i
    return len(specs) - 1


def _counts_to_probs(counts: Sequence[float]) -> List[float]:
    """Jeffreys-smoothed normalisation of bucket counts."""
    smoothed = [float(c) + SMOOTHING_PSEUDOCOUNT for c in counts]
    total = sum(smoothed)
    return [c / total for c in smoothed]


def _empirical_bucket_probs(values: Sequence[float], metric: str) -> Optional[List[float]]:
    """Empirical bucket distribution of a monthly value series."""
    k = n_buckets_for(metric)
    if k == 0:
        return None
    counts = [0.0] * k
    n_used = 0
    for v in values:
        j = _bucket_index_for_value(v, metric)
        if j is None:
            continue
        counts[j] += 1.0
        n_used += 1
    if n_used == 0:
        return None
    return _counts_to_probs(counts)


def _column_exists(con, table: str, column: str) -> bool:
    try:
        rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
        return column.lower() in {str(r[1]).lower() for r in rows}
    except Exception:
        return False


def _table_exists(con, table: str) -> bool:
    try:
        con.execute(f"SELECT 1 FROM {table} LIMIT 0")
        return True
    except Exception:
        return False


def _pa_metric_in_clause(column: str = "metric") -> str:
    """Same single-sourcing as forecaster/history_loaders._pa_metric_in_clause."""
    try:
        from pythia.tools.compute_resolutions import PA_FACTS_RESOLVED_METRICS as metrics
    except Exception:  # pragma: no cover - defensive literal fallback
        metrics = ("affected", "people_affected", "pa", "displaced")
    joined = ",".join(f"'{m}'" for m in metrics)
    return f"lower({column}) IN ({joined})"


# ---------------------------------------------------------------------------
# Per-pair builders
# ---------------------------------------------------------------------------

def _window_months(before_ym: str, n: int) -> List[str]:
    """The n complete months immediately before ``before_ym``, oldest first."""
    return [_add_months(before_ym, -i) for i in range(n, 0, -1)]


def _live_months(
    con, table: str, column: str, months: List[str], *, extra_where: str = ""
) -> set[str]:
    """Which of ``months`` the source recorded ANYTHING in, for any country.

    The month gate. A month with no row for any country is a month the
    ingestion did not cover, and counting it as quiet for one country would
    manufacture a zero out of an outage — the same rule
    ``pythia/tools/source_coverage.py`` applies on the resolution side.

    ``extra_where`` narrows the gate to the SAME source the caller is
    anchoring on. Without it a month in which some other publisher wrote to a
    shared table would read as a month this source was live for.
    """
    if not months:
        return set()
    clause = f" AND ({extra_where})" if extra_where else ""
    try:
        rows = con.execute(
            f"""
            SELECT DISTINCT substr(CAST({column} AS VARCHAR), 1, 7) AS ym
            FROM {table}
            WHERE substr(CAST({column} AS VARCHAR), 1, 7) >= ?
              AND substr(CAST({column} AS VARCHAR), 1, 7) <= ?{clause}
            """,
            [months[0], months[-1]],
        ).fetchall()
    except Exception:  # noqa: BLE001 - no gate is safer than a wrong gate
        return set()
    return {str(r[0]) for r in rows if r[0]}


# IDMC's monthly new displacements, as ``facts_deltas`` holds them. Until
# Oct 2026 ``load_and_derive`` differenced this flow as if it were a stock,
# so a quiet month after a busy one printed as a NEGATIVE number of new
# displacements (Afghanistan, August 2026: -1,791). The writer is fixed and
# the rows repaired (``repair_flow_deltas``); a negative row that still
# arrives is dropped and counted here rather than printed. Kept beside the
# reader so the live-month gate and the anchor cannot drift apart into two
# ideas of what an IDMC row is.
#: The conflict displacement series every ACE/PA reader uses: the prompt's
#: trajectory block, the ACE/PA anchor, the climatology and persistence
#: references, and ``compute_resolutions``. IDMC conflict displacement only:
#: until Oct 2026 every IDMC row was hazard ``IDU`` and counted every cause,
#: so typhoon evacuations in China and the Philippines were shown to the
#: models as conflict displacement, and no ACE/PA question ever resolved.
#: Written by ``resolver.ingestion.idmc_conflict`` through the IDMC adapter;
#: the hazard, metric and publisher are pinned equal to that writer's by
#: forecaster/tests/test_base_rate_matches_resolution_source.py.
CONFLICT_DISPLACEMENT_TABLE = "facts_resolved"
CONFLICT_DISPLACEMENT_HAZARD = "ACE"
CONFLICT_DISPLACEMENT_METRIC = "new_displacements"
CONFLICT_DISPLACEMENT_PUBLISHER = "IDMC"
CONFLICT_DISPLACEMENT_WHERE = (
    "upper(hazard_code) = 'ACE' AND lower(metric) = 'new_displacements' "
    "AND lower(series_semantics) = 'new' "
    "AND upper(COALESCE(publisher, '')) = 'IDMC'"
)
#: ``resolutions.source_desc`` for a value read from the series.
CONFLICT_DISPLACEMENT_SERIES = "facts_resolved:IDMC:conflict_new_displacements"

def _conflict_displacement_table_ok(con) -> bool:
    """True when the table exists and carries every column the series is
    keyed on (a hand-built test table or a pre-publisher DB holds none)."""
    if not _table_exists(con, CONFLICT_DISPLACEMENT_TABLE):
        return False
    return all(
        _column_exists(con, CONFLICT_DISPLACEMENT_TABLE, col)
        for col in ("ym", "iso3", "hazard_code", "metric", "series_semantics", "publisher", "value")
    )


# Legacy names, kept for callers outside this module.
IDMC_FLOW_TABLE = CONFLICT_DISPLACEMENT_TABLE
_IDMC_FLOW_WHERE = CONFLICT_DISPLACEMENT_WHERE
_IDMC_DELTA_WHERE = CONFLICT_DISPLACEMENT_WHERE


def conflict_displacement_rows(
    con, iso3: str, before_ym: str, *,
    since_ym: Optional[str] = None, limit: Optional[int] = None,
) -> Tuple[List[Tuple[str, float]], int]:
    """IDMC monthly CONFLICT displacement before ``before_ym``, ascending.

    Returns ``(rows, n_negative_dropped)``. Only months IDMC reported are
    returned; a quiet month is decided by :func:`conflict_displacement_coverage`.
    A flow cannot be negative; a row that is was mislabelled upstream and is
    dropped and counted rather than printed. ``limit`` keeps the most recent.
    """
    if not _conflict_displacement_table_ok(con):
        return [], 0
    params: List[Any] = [iso3.upper(), before_ym]
    since = ""
    if since_ym:
        since = " AND substr(CAST(ym AS VARCHAR), 1, 7) >= ?"
        params.append(since_ym)
    rows = con.execute(
        f"""
        SELECT substr(CAST(ym AS VARCHAR), 1, 7) AS ym_key, SUM(value) AS flow
        FROM {CONFLICT_DISPLACEMENT_TABLE}
        WHERE upper(iso3) = ?
          AND {CONFLICT_DISPLACEMENT_WHERE}
          AND substr(CAST(ym AS VARCHAR), 1, 7) < ?{since}
          AND value IS NOT NULL
        GROUP BY ym_key
        ORDER BY ym_key DESC
        """,
        params,
    ).fetchall()
    kept = [(str(ym), float(v)) for ym, v in rows if ym is not None and v is not None and float(v) >= 0]
    dropped = sum(1 for _ym, v in rows if v is not None and float(v) < 0)
    if limit is not None:
        kept = kept[: int(limit)]
    return list(reversed(kept)), dropped


def idmc_flow_rows(
    con, iso3: str, hazard_code: str, before_ym: str, *,
    since_ym: Optional[str] = None, limit: Optional[int] = None,
) -> Tuple[List[Tuple[str, float]], int]:
    """Legacy name for :func:`conflict_displacement_rows` (``hazard_code`` is
    ignored: the series is conflict displacement whatever the caller asks)."""
    del hazard_code
    return conflict_displacement_rows(con, iso3, before_ym, since_ym=since_ym, limit=limit)


def conflict_displacement_coverage(con) -> Tuple[set[str], set[str]]:
    """``(live_months, universe)`` of the conflict displacement series.

    The two gates the ACLED series uses, applied to this one. A month is LIVE
    when IDMC reported conflict displacement for ANY country in it; a country
    is in the UNIVERSE when IDMC has reported conflict displacement for it in
    any month the table holds. Only a month that is live, for a country in
    the universe, can be read as an observed zero: a month nobody reported
    for is an ingestion gap, and a country never reported is outside the
    source's sight. Empty sets when the table is absent.
    """
    if not _conflict_displacement_table_ok(con):
        return set(), set()
    try:
        rows = con.execute(
            f"""
            SELECT DISTINCT substr(CAST(ym AS VARCHAR), 1, 7), upper(iso3)
            FROM {CONFLICT_DISPLACEMENT_TABLE}
            WHERE {CONFLICT_DISPLACEMENT_WHERE} AND value IS NOT NULL
            """
        ).fetchall()
    except Exception:  # noqa: BLE001 - no gate is safer than a wrong gate
        return set(), set()
    live = {str(ym) for ym, _iso in rows if ym}
    universe = {str(iso) for _ym, iso in rows if iso}
    return live, universe


#: How long after a month ends before IDMC conflict displacement for it is
#: read at all. IDMC's update feed reports late and irregularly: on the
#: 5 October 2026 release 14 countries had a 2026-09 row against 27 for
#: 2026-03, and 60 of the 64 ACE/PA zero-defaults were months whose first
#: report had not arrived yet. Measured from the feed's own ``created_at``
#: stamps (``tools/probe_idmc_conflict.py``, run 37429579956: recommended
#: figures for 543 country-months, 2024-01 to 2026-03), the share of a
#: month's eventual people that had arrived N days after it ended was 37% at
#: 30, 49% at 60, 63% at 75, 68% at 90, 73% at 120 and 90% at 180; the share
#: of country-months with a first report was 61%, 76%, 81%, 83%, 89% and 94%.
#: 60 days is short of half the people. 90 is the first point where most of
#: a month's total and five in six of its reports have arrived; a later
#: revision still overwrites the resolution on the next run.
CONFLICT_SETTLE_DAYS = 90
#: A country is a REGULAR reporter for a month when IDMC reported conflict
#: displacement for it in at least this many of the twelve months before.
#: Only a regular reporter's missing month can be a quiet month: for a
#: country IDMC reports twice a year a missing month says nothing.
CONFLICT_REGULAR_WINDOW_MONTHS = 12
CONFLICT_REGULAR_MIN_MONTHS = 8
#: The day of the month the forecast runs (the 13th since 2026-10-05). A
#: reference built months later reads the series as it stood that day.
CONFLICT_FORECAST_DAY = 13

#: The metric a held-out IDMC record is written under (a record spanning more
#: than a month, or a country-month above the country's population). A
#: country-month carrying one is neither a value nor a quiet month.
CONFLICT_DISPLACEMENT_HELD_METRIC = "new_displacements_held"

# How a conflict displacement month resolves. Only the first two write a
# resolution; every other is unknown and is looked at again on the next run.
CONFLICT_STATUS_REPORTED = "reported"
CONFLICT_STATUS_QUIET = "zero_default"
CONFLICT_STATUS_UNSETTLED = "unresolved_for_lag"
CONFLICT_STATUS_TRAILING = "unresolved_trailing"
CONFLICT_STATUS_IRREGULAR = "unresolved_irregular_reporter"
CONFLICT_STATUS_HELD = "unresolved_held_out"
CONFLICT_STATUS_NO_SERIES = "unresolved_no_series"


def _month_end(ym: str) -> date:
    year, month = int(ym[:4]), int(ym[5:7])
    nxt = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return nxt - timedelta(days=1)


def conflict_month_settled(ym: str, today: date) -> bool:
    """True once ``CONFLICT_SETTLE_DAYS`` have passed since ``ym`` ended."""
    return _month_end(ym) + timedelta(days=CONFLICT_SETTLE_DAYS) <= today


def conflict_forecast_date(before_ym: str) -> date:
    """The day a forecast whose window starts in ``before_ym`` was made."""
    prev = _add_months(before_ym, -1)
    return date(int(prev[:4]), int(prev[5:7]), CONFLICT_FORECAST_DAY)


def conflict_regular_reporter(reported: Mapping[str, Any], ym: str) -> bool:
    """Reported in at least ``CONFLICT_REGULAR_MIN_MONTHS`` of the twelve
    months before ``ym``."""
    months = _window_months(ym, CONFLICT_REGULAR_WINDOW_MONTHS)
    return sum(1 for m in months if m in reported) >= CONFLICT_REGULAR_MIN_MONTHS


def resolve_conflict_month(
    reported: Mapping[str, float], ym: str, today: date,
    *, held: Optional[set] = None,
) -> Tuple[Optional[float], str]:
    """``(value, status)`` for one country-month of conflict displacement.

    Pure: ``reported`` is the country's reported months and values, ``held``
    its held-out months. A month resolves only once it is settled; a
    reported figure then resolves it; a missing month is zero only for a
    regular reporter with a reported figure in a LATER month (the gap is
    bracketed). A trailing month, a held-out month and an irregular
    reporter's missing month are unknown, never quiet.
    """
    if not conflict_month_settled(ym, today):
        return None, CONFLICT_STATUS_UNSETTLED
    if held and ym in held:
        return None, CONFLICT_STATUS_HELD
    if ym in reported:
        return float(reported[ym]), CONFLICT_STATUS_REPORTED
    if not conflict_regular_reporter(reported, ym):
        return None, CONFLICT_STATUS_IRREGULAR
    if not any(m > ym for m in reported):
        return None, CONFLICT_STATUS_TRAILING
    return 0.0, CONFLICT_STATUS_QUIET


def conflict_displacement_series(
    con, *, before_ym: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, set]]:
    """``(reported, held)`` for every country: reported months and values,
    and held-out months. ``before_ym`` keeps months strictly before it."""
    reported: Dict[str, Dict[str, float]] = {}
    held: Dict[str, set] = {}
    if not _conflict_displacement_table_ok(con):
        return reported, held
    bound = " AND substr(CAST(ym AS VARCHAR), 1, 7) < ?" if before_ym else ""
    params = [before_ym] if before_ym else []
    rows = con.execute(
        f"""
        SELECT upper(iso3), substr(CAST(ym AS VARCHAR), 1, 7), SUM(value)
        FROM {CONFLICT_DISPLACEMENT_TABLE}
        WHERE {CONFLICT_DISPLACEMENT_WHERE} AND value IS NOT NULL{bound}
        GROUP BY 1, 2
        """,
        params,
    ).fetchall()
    for iso, ym, value in rows:
        if iso and ym and value is not None and float(value) >= 0:
            reported.setdefault(str(iso), {})[str(ym)] = float(value)
    held_where = CONFLICT_DISPLACEMENT_WHERE.replace(
        f"lower(metric) = '{CONFLICT_DISPLACEMENT_METRIC}'",
        f"lower(metric) = '{CONFLICT_DISPLACEMENT_HELD_METRIC}'",
    )
    try:
        for iso, ym in con.execute(
            f"""
            SELECT DISTINCT upper(iso3), substr(CAST(ym AS VARCHAR), 1, 7)
            FROM {CONFLICT_DISPLACEMENT_TABLE} WHERE {held_where}{bound}
            """,
            params,
        ).fetchall():
            if iso and ym:
                held.setdefault(str(iso), set()).add(str(ym))
    except Exception:  # noqa: BLE001 - no held rows is the pre-Oct-2026 table
        pass
    return reported, held


def conflict_displacement_status(
    con, iso3: str, ym: str, *, today: Optional[date] = None,
    series: Optional[Tuple[Dict[str, Dict[str, float]], Dict[str, set]]] = None,
) -> Tuple[Optional[float], str]:
    """:func:`resolve_conflict_month` for one country-month read from the DB."""
    if not _conflict_displacement_table_ok(con):
        return None, CONFLICT_STATUS_NO_SERIES
    reported, held = series if series is not None else conflict_displacement_series(con)
    iso = (iso3 or "").upper()
    return resolve_conflict_month(
        reported.get(iso, {}), ym, today or date.today(), held=held.get(iso, set()),
    )


def conflict_displacement_value(
    con, iso3: str, ym: str, *,
    coverage: Any = None,
    today: Optional[date] = None,
    series: Optional[Tuple[Dict[str, Dict[str, float]], Dict[str, set]]] = None,
) -> Optional[Tuple[float, str]]:
    """The resolved conflict displacement for one country-month, or None.

    ``(value, CONFLICT_DISPLACEMENT_SERIES)`` for a settled reported month,
    ``(0.0, 'zero_default')`` for a settled, bracketed quiet month of a
    regular reporter, None otherwise (see :func:`resolve_conflict_month`).
    ``coverage`` is accepted for old callers and ignored.
    """
    del coverage
    value, status = conflict_displacement_status(con, iso3, ym, today=today, series=series)
    if status == CONFLICT_STATUS_REPORTED:
        return float(value), CONFLICT_DISPLACEMENT_SERIES
    if status == CONFLICT_STATUS_QUIET:
        return 0.0, "zero_default"
    return None


def conflict_displacement_settled_rows(
    con, iso3: str, today: date, *, before_ym: Optional[str] = None,
) -> Dict[str, Any]:
    """What the prompt's displacement block shows, under the resolution rule.

    ``rows``: settled months resolved as reported or quiet, ascending (an
    irregular reporter's rows are its reported months alone);
    ``latest_settled_month``; ``regular_reporter`` (as of the month after the
    latest settled one) with ``n_reported_12m``.
    """
    reported_all, held_all = conflict_displacement_series(con, before_ym=before_ym)
    iso = (iso3 or "").upper()
    reported = reported_all.get(iso, {})
    held = held_all.get(iso, set())
    # The newest settled month: walk back from the month before today.
    probe = _add_months(today.strftime("%Y-%m"), -1)
    for _ in range(6):
        if conflict_month_settled(probe, today):
            break
        probe = _add_months(probe, -1)
    latest_settled = probe
    regular = conflict_regular_reporter(reported, _add_months(latest_settled, 1))
    n_12 = sum(1 for m in _window_months(_add_months(latest_settled, 1), 12) if m in reported)
    rows: List[Tuple[str, float]] = []
    first = min(reported) if reported else None
    if first:
        ym = first
        while ym <= latest_settled:
            value, status = resolve_conflict_month(reported, ym, today, held=held)
            if status in (CONFLICT_STATUS_REPORTED, CONFLICT_STATUS_QUIET):
                rows.append((ym, float(value)))
            ym = _add_months(ym, 1)
    return {
        "rows": rows,
        "latest_settled_month": latest_settled,
        "regular_reporter": regular,
        "n_reported_12m": n_12,
        "n_unsettled_reported": sum(1 for m in reported if m > latest_settled),
    }


def _fill_quiet_months(
    observed: Dict[str, float], months: List[str], live: set[str]
) -> Tuple[List[float], int, int]:
    """(values, n reported, n quiet) over the window.

    A month inside the window that the source was live for and did not report
    for this country is an observed zero. A month the source was dark for is
    not an observation at all and is left out.
    """
    values: List[float] = []
    n_quiet = 0
    for ym in months:
        if ym in observed:
            values.append(observed[ym])
        elif ym in live:
            values.append(0.0)
            n_quiet += 1
    return values, len(values) - n_quiet, n_quiet


#: The table the ACE/FATALITIES anchor is built from. ``compute_resolutions``
#: resolves ACE/FATALITIES from ``ACE_FATALITIES_TABLE`` and the resolver
#: debug bundle's ``ace_fatalities_resolve_from_the_base_rate_series`` check
#: holds the two equal — a question scored against one series and anchored
#: on another is the Sept 2026 battle-only fault.
CONFLICT_FATALITIES_TABLE = "acled_monthly_fatalities"


#: A month row is COMPLETE when it was written after the month ended. Until
#: Sept 2026 ``acled_to_duckdb`` also wrote the month in progress, and the
#: monthly ingest ran on the 28th while the forecast ran on the 1st, so the
#: "last month" in the 1 August 2026 prompts was a row written on 15 July
#: holding a median 28% of July's settled deaths (Afghanistan: 9 against 64),
#: and on 1 September a row written on 28 August holding 75%. The writer now
#: skips the month in progress; this predicate keeps the rows it wrote before
#: that fix out of every reader until the next ingest rewrites them.
ACLED_COMPLETE_MONTH_SQL = "updated_at >= CAST(month AS DATE) + INTERVAL 1 MONTH"

#: A month is USABLE at time t once this many days have passed since it
#: ended. The level-and-volatility reference is scored months after the
#: forecast, when every month before the window is complete, so without a
#: calendar rule it would read a month the forecaster never saw.
#:
#: Derived from the monthly cycle (since 2026-10-05): Resolver Update writes
#: the month just ended on the 11th, ten days after it ended, and can finish
#: late on the 11th (crons arrive hours late and the ingest takes up to six
#: hours); the forecast runs on the 13th, twelve days after, and its
#: ``forecasts_ensemble.created_at`` (``known_at``) can fall on the 13th to
#: the 15th while batches complete. So the bound must be at least 11 (a
#: forecast dated the 11th may have run before the ingest wrote the month)
#: and at most 12 (a forecast on the 13th DID read it). 11 is the value.
#: Under the old 28th/1st cycle the value was 14; for forecasts made on the
#: 1st or the 15th the two values pick the same month, so references scored
#: for those runs are unchanged (``test_settle_days_matches_the_11th_13th_cycle``).
ACLED_SETTLE_DAYS = 11


def acled_complete_month_clause(con, table: str = "acled_monthly_fatalities") -> str:
    """``ACLED_COMPLETE_MONTH_SQL`` when the table records ``updated_at``,
    else ``TRUE`` (a hand-built table in a test, or a pre-stamp database)."""
    return ACLED_COMPLETE_MONTH_SQL if _column_exists(con, table, "updated_at") else "TRUE"


def _conflict_fatalities(con, iso3: str, before_ym: str) -> Tuple[List[float], str, Dict[str, Any]]:
    """ACE/FATALITIES: the ACLED monthly-fatalities series the prompt anchors on."""
    if not _table_exists(con, "acled_monthly_fatalities"):
        return [], NO_BASE_RATE_SOURCE, {"reason": "acled_monthly_fatalities missing"}
    months = _window_months(before_ym, CONFLICT_WINDOW_MONTHS)
    complete = acled_complete_month_clause(con)
    rows = con.execute(
        f"""
        SELECT substr(CAST(month AS VARCHAR), 1, 7) AS ym, SUM(fatalities)
        FROM acled_monthly_fatalities
        WHERE iso3 = ?
          AND substr(CAST(month AS VARCHAR), 1, 7) < ?
          AND substr(CAST(month AS VARCHAR), 1, 7) >= ?
          AND {complete}
        GROUP BY ym
        """,
        [iso3, before_ym, months[0]],
    ).fetchall()
    observed = {str(ym): float(v or 0) for ym, v in rows if ym}
    n_quiet = 0
    if COUNT_QUIET_MONTHS_AS_ZERO and observed:
        # The country gate: a country that never appears in the table is
        # outside ACLED's universe, and its silence says nothing. `observed`
        # being non-empty is that gate, evaluated over this window.
        live = _live_months(
            con, "acled_monthly_fatalities", "month", months, extra_where=complete
        )
        values, n_reported, n_quiet = _fill_quiet_months(observed, months, live)
    else:
        values = [observed[k] for k in sorted(observed)]
        n_reported = len(values)
    probs = _empirical_bucket_probs(values, "FATALITIES")
    if probs is None:
        return [], NO_BASE_RATE_SOURCE, {"reason": "no ACLED fatalities history before window"}
    detail = {
        "score_family": "spd",
        "method": "empirical_monthly_buckets",
        "window_months": CONFLICT_WINDOW_MONTHS,
        "n_months_used": len(values),
        "n_months_reported": n_reported,
        "n_months_quiet": n_quiet,
        "values": values,
    }
    return probs, f"acled_monthly_fatalities:{len(values)}m", detail


def _conflict_displacement(
    con, iso3: str, hazard_code: str, before_ym: str, *, known_at: Optional[date] = None,
) -> Tuple[List[float], str, Dict[str, Any]]:
    """ACE/PA: IDMC monthly CONFLICT displacement, the series an ACE/PA
    question resolves on and the prompt marks THIS QUESTION'S SERIES.

    Each month of the window is read as the resolver reads it on the day the
    forecast was made (:func:`resolve_conflict_month` at ``known_at``,
    default :func:`conflict_forecast_date`): a settled reported month is its
    figure, a settled bracketed quiet month of a regular reporter is zero,
    and every other month (unsettled, trailing, held out, or a missing month
    of an irregular reporter) is unknown and left out, never counted quiet."""
    del hazard_code
    if not _table_exists(con, CONFLICT_DISPLACEMENT_TABLE):
        return [], NO_BASE_RATE_SOURCE, {"reason": f"{CONFLICT_DISPLACEMENT_TABLE} missing"}
    when = known_at or conflict_forecast_date(before_ym)
    months = _window_months(before_ym, CONFLICT_WINDOW_MONTHS)
    reported_all, held_all = conflict_displacement_series(con, before_ym=before_ym)
    iso = (iso3 or "").upper()
    reported = reported_all.get(iso, {})
    held = held_all.get(iso, set())
    values: List[float] = []
    n_reported = n_quiet = n_unknown = 0
    for ym in months:
        value, status = resolve_conflict_month(reported, ym, when, held=held)
        if status == CONFLICT_STATUS_REPORTED:
            values.append(float(value))
            n_reported += 1
        elif status == CONFLICT_STATUS_QUIET:
            values.append(0.0)
            n_quiet += 1
        else:
            n_unknown += 1
    probs = _empirical_bucket_probs(values, "PA")
    if probs is None:
        return [], NO_BASE_RATE_SOURCE, {"reason": "no settled IDMC conflict displacement before window"}
    detail = {
        "score_family": "spd",
        "method": "empirical_monthly_buckets",
        "window_months": CONFLICT_WINDOW_MONTHS,
        "n_months_used": len(values),
        "n_months_reported": n_reported,
        "n_months_quiet": n_quiet,
        "n_months_unknown": n_unknown,
        "known_at": when.isoformat(),
        "values": values,
    }
    return probs, f"idmc_conflict:{len(values)}m", detail


def _phase3_history(con, iso3: str, before_ym: str) -> Tuple[List[float], str, Dict[str, Any]]:
    """DR/PHASE3PLUS_IN_NEED: FEWS NET / IPC Phase 3+ monthly stock series."""
    if not _table_exists(con, "facts_resolved"):
        return [], NO_BASE_RATE_SOURCE, {"reason": "facts_resolved missing"}
    rows = con.execute(
        """
        SELECT ym, value
        FROM facts_resolved
        WHERE iso3 = ?
          AND hazard_code = 'DR'
          AND lower(metric) = 'phase3plus_in_need'
          AND substr(CAST(ym AS VARCHAR), 1, 7) < ?
        ORDER BY ym DESC
        LIMIT ?
        """,
        [iso3, before_ym, PHASE3_WINDOW_MONTHS],
    ).fetchall()
    values = [float(v) for _, v in rows if v is not None]
    probs = _empirical_bucket_probs(values, "PHASE3PLUS_IN_NEED")
    if probs is None:
        return [], NO_BASE_RATE_SOURCE, {"reason": "no Phase 3+ history before window"}
    detail = {
        "score_family": "spd",
        "method": "empirical_monthly_buckets",
        "window_months": PHASE3_WINDOW_MONTHS,
        "n_months_used": len(values),
    }
    return probs, f"facts_resolved:phase3plus_in_need:{len(values)}m", detail


def last_observed_value(
    con, iso3: str, hazard_code: str, metric: str, as_of: Any
) -> Optional[Tuple[float, str, str]]:
    """The last value OBSERVED strictly before the question window.

    Returns ``(value, ym, source)`` or None. It is the input of the
    persistence reference forecaster (``score_baselines``), and is drawn from
    the same series as the climatology anchor, for the same reason: a
    reference scored against one quantity and built from another measures
    the difference between them.

    * ACE/FATALITIES: ``acled_monthly_fatalities`` (all event types). The
      month before the window counts as an observed ZERO when ACLED was live
      that month and the country is in its universe but has no row — the
      quiet-month rule the climatology anchor applies.
    * ACE/PA: the IDMC conflict displacement series; the month before the
      window resolved as the resolver would (reported, quiet zero, or
      unknown), else the latest reported month.
    * DR/PHASE3PLUS_IN_NEED: the latest ``phase3plus_in_need`` row before the
      window (a stock, reported every few months).

    Other pairs have no persistence reference.
    """
    hz = (hazard_code or "").upper()
    m = (metric or "").upper()
    before = _as_of_ym(as_of)
    iso = (iso3 or "").upper()
    try:
        if hz == "ACE" and m == "FATALITIES":
            if not _table_exists(con, CONFLICT_FATALITIES_TABLE):
                return None
            prev = _add_months(before, -1)
            complete = acled_complete_month_clause(con, CONFLICT_FATALITIES_TABLE)
            row = con.execute(
                f"""
                SELECT substr(CAST(month AS VARCHAR), 1, 7) AS ym, SUM(fatalities)
                FROM {CONFLICT_FATALITIES_TABLE}
                WHERE iso3 = ? AND substr(CAST(month AS VARCHAR), 1, 7) < ?
                  AND {complete}
                GROUP BY ym ORDER BY ym DESC LIMIT 1
                """,
                [iso, before],
            ).fetchone()
            if row and str(row[0]) == prev:
                return float(row[1] or 0), prev, CONFLICT_FATALITIES_TABLE
            if row and prev in _live_months(
                con, CONFLICT_FATALITIES_TABLE, "month", [prev], extra_where=complete
            ):
                return 0.0, prev, f"{CONFLICT_FATALITIES_TABLE}:quiet_month"
            if row:
                return float(row[1] or 0), str(row[0]), CONFLICT_FATALITIES_TABLE
            return None
        if hz == "ACE" and m == "PA":
            # The newest month before the window that resolves (a settled
            # reported figure or a settled bracketed quiet month) on the day
            # the forecast was made; unsettled and trailing months are unknown.
            when = conflict_forecast_date(before)
            reported_all, held_all = conflict_displacement_series(con, before_ym=before)
            reported = reported_all.get(iso, {})
            held = held_all.get(iso, set())
            ym = _add_months(before, -1)
            for _ in range(CONFLICT_WINDOW_MONTHS):
                value, status = resolve_conflict_month(reported, ym, when, held=held)
                if status == CONFLICT_STATUS_REPORTED:
                    return float(value), ym, CONFLICT_DISPLACEMENT_SERIES
                if status == CONFLICT_STATUS_QUIET:
                    return 0.0, ym, f"{CONFLICT_DISPLACEMENT_SERIES}:quiet_month"
                ym = _add_months(ym, -1)
            return None
        if hz == "DR" and m == "PHASE3PLUS_IN_NEED":
            if not _table_exists(con, "facts_resolved"):
                return None
            row = con.execute(
                """
                SELECT substr(CAST(ym AS VARCHAR), 1, 7), value
                FROM facts_resolved
                WHERE iso3 = ? AND hazard_code = 'DR'
                  AND lower(metric) = 'phase3plus_in_need'
                  AND value IS NOT NULL
                  AND substr(CAST(ym AS VARCHAR), 1, 7) < ?
                ORDER BY ym DESC LIMIT 1
                """,
                [iso, before],
            ).fetchone()
            if row:
                return float(row[1]), str(row[0]), "facts_resolved:phase3plus_in_need"
            return None
    except Exception:  # noqa: BLE001 - no reference is better than a wrong one
        return None
    return None


# ---------------------------------------------------------------------------
# Level and volatility (ACE/FATALITIES)
# ---------------------------------------------------------------------------

#: The prompt-block version a forecast was shown when it carries this anchor.
LEVEL_VOLATILITY_VERSION = "prior_anchor_v1"
#: The same distribution with its Spread sentence read off the vector shown
#: (stay / up / down at months 1 and 6) instead of the pooled move shares,
#: which disagree with the vector at the edge buckets once mass that would
#: fall off the end is clipped onto the end bucket (Israel, November 2026
#: test run: "stayed 42%" beside 71% on zero). Selected by
#: ``PYTHIA_PRIOR_ANCHOR_BLOCK_VERSION=v2``; the distribution is unchanged.
LEVEL_VOLATILITY_VERSION_V2 = "prior_anchor_v2"
LEVEL_VOLATILITY_MODEL_SOURCE = "level_volatility:acled_monthly_fatalities"
#: Fewer bucket-move pairs than this and the country's own history is too
#: thin to say how far a count wanders, so pairs are pooled from countries in
#: the same activity band (the bucket of their median month).
LEVEL_VOLATILITY_MIN_PAIRS = 12
#: No bucket falls below this before renormalising: a reference that says a
#: bucket is impossible pays an unbounded log loss the first time it happens.
LEVEL_VOLATILITY_FLOOR = 0.005


def _ym_of(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, date):
        return value.strftime("%Y-%m")
    return _parse_ym(value)


def _month_diff(a: str, b: str) -> int:
    """Months from ``a`` to ``b`` ('YYYY-MM')."""
    return (int(b[:4]) - int(a[:4])) * 12 + int(b[5:7]) - int(a[5:7])


def _usable_at(ym: str, known_at: date) -> bool:
    """True when month ``ym`` had ended ``ACLED_SETTLE_DAYS`` before ``known_at``."""
    from datetime import timedelta

    nxt = _add_months(ym, 1)
    first_after = date(int(nxt[:4]), int(nxt[5:7]), 1)
    return first_after + timedelta(days=ACLED_SETTLE_DAYS) <= known_at


def _as_date(value: Any) -> date:
    from datetime import datetime

    if value is None:
        return date.today()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value).strip()[:10]
    return date(int(s[:4]), int(s[5:7]), int(s[8:10]) if len(s) >= 10 else 1)


def _acled_series_all(con, first_ym: str, last_ym: str) -> Tuple[Dict[str, Dict[str, float]], set]:
    """Complete-month ACLED fatalities for every country over [first, last],
    plus the months the source was live for (any complete row)."""
    complete = acled_complete_month_clause(con)
    rows = con.execute(
        f"""
        SELECT iso3, substr(CAST(month AS VARCHAR), 1, 7) AS ym, SUM(fatalities)
        FROM acled_monthly_fatalities
        WHERE substr(CAST(month AS VARCHAR), 1, 7) >= ?
          AND substr(CAST(month AS VARCHAR), 1, 7) <= ?
          AND iso3 IS NOT NULL
          AND {complete}
        GROUP BY iso3, ym
        """,
        [first_ym, last_ym],
    ).fetchall()
    by_iso: Dict[str, Dict[str, float]] = {}
    live: set = set()
    for iso, ym, v in rows:
        if not ym:
            continue
        by_iso.setdefault(str(iso).upper(), {})[str(ym)] = float(v or 0)
        live.add(str(ym))
    return by_iso, live


def _filled(observed: Dict[str, float], months: List[str], live: set) -> Dict[str, float]:
    """The window as {month: value}, quiet live months as zero, dark months absent."""
    out: Dict[str, float] = {}
    for ym in months:
        if ym in observed:
            out[ym] = observed[ym]
        elif ym in live:
            out[ym] = 0.0
    return out


def _bucket_moves(
    series: Dict[str, float], gap: int, from_bucket: Optional[int] = None
) -> List[int]:
    """Bucket index change between every pair of months ``gap`` apart.

    With ``from_bucket`` only pairs whose START month sits in that bucket
    count: the transition reference's rule, under which a country at level
    zero can never borrow a downward move from a pair that started higher.
    """
    moves: List[int] = []
    for ym, v in series.items():
        later = _add_months(ym, gap)
        if later not in series:
            continue
        a = _bucket_index_for_value(v, "FATALITIES")
        b = _bucket_index_for_value(series[later], "FATALITIES")
        if a is None or b is None:
            continue
        if from_bucket is not None and a != from_bucket:
            continue
        moves.append(b - a)
    return moves


def _median(values: Sequence[float]) -> float:
    vals = sorted(values)
    n = len(vals)
    mid = n // 2
    return vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2.0


#: The transition reference (scored only, never shown in a prompt): the
#: level-and-volatility recipe with moves counted only from pairs whose start
#: month sits in the level's bucket. The plain recipe pools every pair, so a
#: country at zero inherits the downward moves of months that started higher
#: and they pile onto bucket 0 at the edge — which is how a vector can put 71%
#: on "no deaths" beside a stated 42% chance of staying put.
LEVEL_TRANSITION_MODEL_SOURCE = "level_transition:acled_monthly_fatalities"


def level_transition_spds(
    con,
    iso3: str,
    as_of: Any,
    horizons: Sequence[int] = (1, 2, 3, 4, 5, 6),
    known_at: Any = None,
) -> Tuple[Dict[int, List[float]], str, Dict[str, Any]]:
    """:func:`level_volatility_spds` with moves conditioned on the start bucket.

    Same level, settle rule, floor, window and pooling (the country's own
    pairs first; below ``LEVEL_VOLATILITY_MIN_PAIRS`` the same-band
    countries' pairs that also start in the level's bucket). A horizon with
    no such pair at all is left out rather than borrowed from other buckets.
    """
    return _level_reference_spds(con, iso3, as_of, horizons, known_at, transition=True)


def level_volatility_spds(
    con,
    iso3: str,
    as_of: Any,
    horizons: Sequence[int] = (1, 2, 3, 4, 5, 6),
    known_at: Any = None,
) -> Tuple[Dict[int, List[float]], str, Dict[str, Any]]:
    """The ACE/FATALITIES level-and-volatility distribution for each horizon.

    * The LEVEL is the last complete month the forecaster could have read:
      before the window (``as_of``), ended at least ``ACLED_SETTLE_DAYS``
      before ``known_at`` (the forecast date; today when omitted), and held
      in the table as a complete row. A country with no row that month, in a
      month ACLED was live for, is at level zero.
    * The SPREAD is how far a monthly count moved, in buckets, over the same
      number of months as separates the level from the target month, across
      the country's last ``CONFLICT_WINDOW_MONTHS`` complete months (quiet
      months zero, dark months left out, the climatology rules). Below
      ``LEVEL_VOLATILITY_MIN_PAIRS`` pairs, pairs come from every country in
      the same activity band as well, and the detail says so.
    * That move distribution is centred on the level's bucket; mass that
      would fall off either end lands on the end bucket; every bucket is
      floored at ``LEVEL_VOLATILITY_FLOOR`` and the vector renormalised.

    Returns ``({horizon: probs}, source, detail)``; an empty dict with a
    ``reason`` when there is nothing to anchor on.
    """
    return _level_reference_spds(con, iso3, as_of, horizons, known_at, transition=False)


def _acled_level(
    con, iso3: str, as_of: Any, known_at: Any
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """The level month the forecaster could have read, and the series behind it.

    Shared by every ACE/FATALITIES reference that starts from "where the
    country is now": the last month before the window that had ended
    ``ACLED_SETTLE_DAYS`` before ``known_at`` and that ACLED was live for.
    Returns ``(state, {})`` or ``(None, {"reason": ...})``; ``state`` carries
    ``level_ym``, ``by_iso``, ``live``, ``window_ym`` and ``when``.
    """
    iso = (iso3 or "").upper()
    window_ym = _as_of_ym(as_of)
    when = _as_date(known_at)
    if not _table_exists(con, CONFLICT_FATALITIES_TABLE):
        return None, {"reason": "acled_monthly_fatalities missing"}
    candidate = _add_months(window_ym, -1)
    for _ in range(24):
        if _usable_at(candidate, when):
            break
        candidate = _add_months(candidate, -1)
    first = _add_months(candidate, -(CONFLICT_WINDOW_MONTHS + 3))
    by_iso, live = _acled_series_all(con, first, candidate)
    if iso not in by_iso:
        return None, {"reason": "country has no complete ACLED month before the window"}
    level_ym = None
    for back in range(4):
        ym = _add_months(candidate, -back)
        if ym in live:
            level_ym = ym
            break
    if level_ym is None:
        return None, {"reason": "no complete ACLED month near the window"}
    return {
        "iso": iso, "level_ym": level_ym, "by_iso": by_iso, "live": live,
        "window_ym": window_ym, "when": when,
    }, {}


#: The 12-month conflictology reference: bucket shares of the country's last
#: twelve complete monthly values ending at the level month. In a backtest
#: over 8,371 country-forecasts (Mar 2021 - Dec 2025, production timing) it
#: scored Brier 0.390 against 0.470 for level_volatility; pooled 75/25 with
#: level_transition it scored 0.384 (scripts/analysis/sibyl_reference_backtest.py).
CONFLICTOLOGY_MONTHS = 12
CONFLICTOLOGY_MODEL_SOURCE = "conflictology12:acled_monthly_fatalities"
REFERENCE_POOL_WEIGHT_CONFLICTOLOGY = 0.75
REFERENCE_POOL_MODEL_SOURCE = "ref_pool:conflictology12_0.75+level_transition_0.25"


def conflictology_spds(
    con,
    iso3: str,
    as_of: Any,
    horizons: Sequence[int] = (1, 2, 3, 4, 5, 6),
    known_at: Any = None,
) -> Tuple[Dict[int, List[float]], str, Dict[str, Any]]:
    """Bucket shares of the last 12 monthly values, the same at every horizon.

    The months end at the level month (:func:`_acled_level`, the rule the
    level references use): complete rows only, a live month with no row is a
    quiet month at zero, a dark month is left out. Each bucket is floored at
    ``LEVEL_VOLATILITY_FLOOR`` and the vector renormalised.
    """
    k = n_buckets_for("FATALITIES")
    state, why = _acled_level(con, iso3, as_of, known_at)
    if state is None or not k:
        return {}, NO_BASE_RATE_SOURCE, why or {"reason": "no FATALITIES buckets"}
    level_ym = state["level_ym"]
    months = _window_months(_add_months(level_ym, 1), CONFLICTOLOGY_MONTHS)
    series = _filled(state["by_iso"][state["iso"]], months, state["live"])
    if not series:
        return {}, NO_BASE_RATE_SOURCE, {"reason": "no complete month in the last 12"}
    counts = [0.0] * k
    for v in series.values():
        j = _bucket_index_for_value(v, "FATALITIES")
        if j is not None:
            counts[j] += 1.0
    total = sum(counts)
    if total <= 0:
        return {}, NO_BASE_RATE_SOURCE, {"reason": "no bucketable month in the last 12"}
    probs = [max(c / total, LEVEL_VOLATILITY_FLOOR) for c in counts]
    z = sum(probs)
    probs = [p / z for p in probs]
    detail = {
        "score_family": "spd",
        "method": "last_12_monthly_bucket_shares",
        "level_month": level_ym,
        "level_value": state["by_iso"][state["iso"]].get(level_ym, 0.0),
        "known_at": state["when"].isoformat(),
        "months": sorted(series),
        "values": [series[m] for m in sorted(series)],
        "n_months": len(series),
    }
    return {int(h): list(probs) for h in horizons}, CONFLICTOLOGY_MODEL_SOURCE, detail


def reference_pool_spds(
    con,
    iso3: str,
    as_of: Any,
    horizons: Sequence[int] = (1, 2, 3, 4, 5, 6),
    known_at: Any = None,
    weight: float = REFERENCE_POOL_WEIGHT_CONFLICTOLOGY,
) -> Tuple[Dict[int, List[float]], str, Dict[str, Any]]:
    """Per horizon: ``weight`` x conflictology + the rest x level_transition.

    Where the transition vector is missing for a horizon the 12-month vector
    stands alone, and the detail says which horizons that happened to.
    """
    c12, _src, c_detail = conflictology_spds(con, iso3, as_of, horizons, known_at)
    if not c12:
        return {}, NO_BASE_RATE_SOURCE, c_detail
    tr, _tsrc, t_detail = level_transition_spds(con, iso3, as_of, horizons, known_at)
    out: Dict[int, List[float]] = {}
    alone: List[int] = []
    for h in horizons:
        h = int(h)
        a = c12[h]
        b = tr.get(h)
        if not b or len(b) != len(a):
            out[h] = list(a)
            alone.append(h)
            continue
        mixed = [weight * x + (1.0 - weight) * y for x, y in zip(a, b)]
        z = sum(mixed)
        out[h] = [x / z for x in mixed]
    detail = {
        "score_family": "spd",
        "method": "pool_conflictology12_level_transition",
        "weight_conflictology": weight,
        "conflictology": c_detail,
        "transition": t_detail,
        "horizons_without_transition": alone,
    }
    return out, REFERENCE_POOL_MODEL_SOURCE, detail


def _level_reference_spds(
    con,
    iso3: str,
    as_of: Any,
    horizons: Sequence[int],
    known_at: Any,
    *,
    transition: bool,
) -> Tuple[Dict[int, List[float]], str, Dict[str, Any]]:
    k = n_buckets_for("FATALITIES")
    if not k:
        return {}, NO_BASE_RATE_SOURCE, {"reason": "acled_monthly_fatalities missing"}
    state, why = _acled_level(con, iso3, as_of, known_at)
    if state is None:
        return {}, NO_BASE_RATE_SOURCE, why
    iso = state["iso"]
    window_ym = state["window_ym"]
    when = state["when"]
    by_iso = state["by_iso"]
    live = state["live"]
    level_ym = state["level_ym"]
    level_value = by_iso[iso].get(level_ym, 0.0)
    level_bucket = _bucket_index_for_value(level_value, "FATALITIES")
    if level_bucket is None:
        return {}, NO_BASE_RATE_SOURCE, {"reason": "level value not bucketable"}

    window = _window_months(_add_months(level_ym, 1), CONFLICT_WINDOW_MONTHS)
    series = _filled(by_iso[iso], window, live)
    median_bucket = _bucket_index_for_value(_median(list(series.values())), "FATALITIES")
    band_series: Optional[List[Dict[str, float]]] = None

    out: Dict[int, List[float]] = {}
    per_horizon: Dict[str, Any] = {}
    for h in horizons:
        target = _add_months(window_ym, int(h) - 1)
        gap = _month_diff(level_ym, target)
        from_bucket = level_bucket if transition else None
        moves = _bucket_moves(series, gap, from_bucket)
        n_own = len(moves)
        pooled = False
        n_band_countries = 0
        if n_own < LEVEL_VOLATILITY_MIN_PAIRS:
            if band_series is None:
                band_series = []
                for other, obs in by_iso.items():
                    if other == iso:
                        continue
                    s2 = _filled(obs, window, live)
                    if not s2:
                        continue
                    if _bucket_index_for_value(_median(list(s2.values())), "FATALITIES") == median_bucket:
                        band_series.append(s2)
            for s2 in band_series:
                moves.extend(_bucket_moves(s2, gap, from_bucket))
            pooled = True
            n_band_countries = len(band_series)
        if not moves:
            continue
        counts: Dict[int, int] = {}
        for d in moves:
            counts[d] = counts.get(d, 0) + 1
        probs = [0.0] * k
        for d, c in counts.items():
            j = min(max(level_bucket + d, 0), k - 1)
            probs[j] += c / len(moves)
        probs = [max(p, LEVEL_VOLATILITY_FLOOR) for p in probs]
        total = sum(probs)
        out[int(h)] = [p / total for p in probs]
        n = float(len(moves))
        per_horizon[str(int(h))] = {
            "gap_months": gap,
            "n_pairs": len(moves),
            "n_own_pairs": n_own,
            "pooled": pooled,
            "n_band_countries": n_band_countries,
            "share_same": sum(c for d, c in counts.items() if d == 0) / n,
            "share_one": sum(c for d, c in counts.items() if abs(d) == 1) / n,
            "share_two_plus": sum(c for d, c in counts.items() if abs(d) >= 2) / n,
            "share_up": sum(c for d, c in counts.items() if d > 0) / n,
            "share_down": sum(c for d, c in counts.items() if d < 0) / n,
        }
    if not out:
        return {}, NO_BASE_RATE_SOURCE, {"reason": "no month pairs to measure movement from"}
    detail = {
        "score_family": "spd",
        "method": "level_plus_transition_moves" if transition else "level_plus_bucket_moves",
        "version": LEVEL_VOLATILITY_VERSION,
        "level_month": level_ym,
        "level_value": level_value,
        "level_bucket": level_bucket,
        "known_at": when.isoformat(),
        "window_months": CONFLICT_WINDOW_MONTHS,
        "n_months_in_window": len(series),
        "activity_band": median_bucket,
        "horizons": per_horizon,
    }
    if transition:
        return out, LEVEL_TRANSITION_MODEL_SOURCE, detail
    return out, LEVEL_VOLATILITY_MODEL_SOURCE, detail


def level_volatility_spd(
    con, iso3: str, as_of: Any, horizon_k: int, known_at: Any = None
) -> Tuple[List[float], str, Dict[str, Any]]:
    """One horizon of :func:`level_volatility_spds` (``[]`` when unavailable)."""
    spds, source, detail = level_volatility_spds(
        con, iso3, as_of, horizons=(int(horizon_k),), known_at=known_at
    )
    return spds.get(int(horizon_k), []), source, detail


def _event_occurrence_rates(
    con, iso3: str, hazard_code: str, before_ym: str
) -> Optional[Dict[int, Tuple[int, int]]]:
    """Per calendar month (1-12): (months observed, months with an event),
    from the GDACS event_occurrence rows in facts_resolved."""
    if not _table_exists(con, "facts_resolved"):
        return None
    try:
        rows = con.execute(
            """
            SELECT ym, value
            FROM facts_resolved
            WHERE upper(iso3) = ?
              AND upper(hazard_code) = ?
              AND lower(metric) = 'event_occurrence'
              AND substr(CAST(ym AS VARCHAR), 1, 7) < ?
            ORDER BY ym
            """,
            [iso3, hazard_code, before_ym],
        ).fetchall()
    except Exception:
        return None
    by_month: Dict[int, Tuple[int, int]] = {}
    for ym_raw, value in rows:
        ym = _parse_ym(ym_raw)
        if ym is None:
            continue
        cal = int(ym[5:7])
        total, events = by_month.get(cal, (0, 0))
        occurred = 1 if float(value or 0) >= 1.0 else 0
        by_month[cal] = (total + 1, events + occurred)
    return by_month or None


def _binary_base_rate(
    con, iso3: str, hazard_code: str, as_of_ym: str
) -> Tuple[List[float], str, Dict[str, Any]]:
    """EVENT_OCCURRENCE: [p, 1-p] over the question's six forecast months."""
    by_month = _event_occurrence_rates(con, iso3, hazard_code, as_of_ym)
    if not by_month:
        return [], NO_BASE_RATE_SOURCE, {"reason": "no GDACS event_occurrence history"}
    months = forecast_months(as_of_ym)
    probs_by_month: Dict[str, List[float]] = {}
    total_obs = 0
    total_events = 0
    for ym in months:
        cal = int(ym[5:7])
        obs, events = by_month.get(cal, (0, 0))
        total_obs += obs
        total_events += events
        p_m = (events + SMOOTHING_PSEUDOCOUNT) / (obs + 2 * SMOOTHING_PSEUDOCOUNT)
        probs_by_month[ym] = [p_m, 1.0 - p_m]
    p = (total_events + SMOOTHING_PSEUDOCOUNT) / (total_obs + 2 * SMOOTHING_PSEUDOCOUNT)
    detail = {
        "score_family": "binary",
        "method": "gdacs_seasonal_event_rate",
        "forecast_months": months,
        "probs_by_month": probs_by_month,
        "n_months_observed": total_obs,
        "n_event_months": total_events,
    }
    return [p, 1.0 - p], "facts_resolved:event_occurrence", detail


#: Below this many PA records for a calendar month, its per-month vector
#: borrows the pooled severity shares (``_seasonal_pa`` probs_by_month).
PER_MONTH_MIN_SEVERITY = 3


def _seasonal_pa(
    con, iso3: str, hazard_code: str, as_of_ym: str
) -> Tuple[List[float], str, Dict[str, Any]]:
    """FL/TC (and HW) PA: occurrence x conditional-severity mixture.

    P(bucket "0") comes from the GDACS seasonal event rate for the question's
    forecast calendar months; the remaining mass is distributed over the
    non-zero buckets in proportion to the historical monthly PA figures for
    those calendar months (the same rows the seasonal profile summarises).
    Historical PA rows only exist for months when something was reported, so
    using them alone would wildly overstate occurrence — which is exactly why
    the prompt pairs the seasonal profile with the GDACS occurrence block, and
    why this mixture does the same.
    """
    if not _table_exists(con, "facts_resolved"):
        return [], NO_BASE_RATE_SOURCE, {"reason": "facts_resolved missing"}

    months = forecast_months(as_of_ym)
    forecast_cals = {int(ym[5:7]) for ym in months}

    try:
        rows = con.execute(
            f"""
            SELECT ym, value
            FROM facts_resolved
            WHERE iso3 = ?
              AND hazard_code = ?
              AND {_pa_metric_in_clause()}
              AND substr(CAST(ym AS VARCHAR), 1, 7) < ?
            ORDER BY ym
            """,
            [iso3, hazard_code, as_of_ym],
        ).fetchall()
    except Exception as exc:
        LOGGER.warning("Seasonal PA query failed for %s/%s: %s", iso3, hazard_code, exc)
        rows = []

    severity_values: List[float] = []
    severity_by_cal: Dict[int, List[float]] = {}
    n_pa_months_all = 0
    yms_seen: set[str] = set()
    for ym_raw, value in rows:
        ym = _parse_ym(ym_raw)
        if ym is None:
            continue
        yms_seen.add(ym)
        n_pa_months_all += 1
        if int(ym[5:7]) in forecast_cals:
            try:
                v = float(value or 0)
            except (TypeError, ValueError):
                v = 0.0
            if v >= 1.0:
                severity_values.append(v)
                severity_by_cal.setdefault(int(ym[5:7]), []).append(v)

    by_month = _event_occurrence_rates(con, iso3, hazard_code, as_of_ym)

    # Occurrence: GDACS seasonal rate pooled over the forecast months when
    # available; otherwise fall back to reported-PA-month frequency (months
    # with a PA report / years of record for those calendar months).
    occurrence_method = None
    total_obs = 0
    total_events = 0
    if by_month:
        occurrence_method = "gdacs_seasonal_event_rate"
        for ym in months:
            obs, events = by_month.get(int(ym[5:7]), (0, 0))
            total_obs += obs
            total_events += events
    elif yms_seen:
        occurrence_method = "reported_pa_month_frequency"
        years = {ym[:4] for ym in yms_seen}
        # Each forecast calendar month was observable once per year of record.
        total_obs = len(years) * len(months)
        total_events = sum(
            1 for ym in yms_seen if int(ym[5:7]) in forecast_cals
        )

    if total_obs == 0 and not severity_values:
        return [], NO_BASE_RATE_SOURCE, {
            "reason": "no PA history and no GDACS occurrence history before window"
        }

    p_event = (total_events + SMOOTHING_PSEUDOCOUNT) / (total_obs + 2 * SMOOTHING_PSEUDOCOUNT)
    p_event = min(max(p_event, 0.001), 0.999)

    k = n_buckets_for("PA")
    # Conditional severity over the non-zero buckets (indices 1..K-1).
    sev_counts = [0.0] * (k - 1)
    for v in severity_values:
        j = _bucket_index_for_value(v, "PA")
        if j is None or j == 0:
            continue
        sev_counts[j - 1] += 1.0
    sev_probs = _counts_to_probs(sev_counts)  # uniform-ish when no data

    probs = [1.0 - p_event] + [p_event * s for s in sev_probs]
    total = sum(probs)
    probs = [p / total for p in probs]

    # One vector per forecast calendar month, built the same way from that
    # month's own event rate and PA records. Below PER_MONTH_MIN_SEVERITY
    # records the pooled severity shares stand in for the non-zero buckets.
    # Callers of the pooled return value are unaffected.
    years_seen = {ym[:4] for ym in yms_seen}
    probs_by_month: Dict[str, List[float]] = {}
    for ym in months:
        cal = int(ym[5:7])
        if occurrence_method == "gdacs_seasonal_event_rate":
            obs, events = by_month.get(cal, (0, 0))
        elif occurrence_method == "reported_pa_month_frequency":
            obs = len(years_seen)
            events = sum(1 for y in yms_seen if int(y[5:7]) == cal)
        else:
            obs, events = 0, 0
        p_m = (events + SMOOTHING_PSEUDOCOUNT) / (obs + 2 * SMOOTHING_PSEUDOCOUNT)
        p_m = min(max(p_m, 0.001), 0.999)
        month_vals = severity_by_cal.get(cal, [])
        if len(month_vals) >= PER_MONTH_MIN_SEVERITY:
            mc = [0.0] * (k - 1)
            for v in month_vals:
                j = _bucket_index_for_value(v, "PA")
                if j is not None and j > 0:
                    mc[j - 1] += 1.0
            sev_m = _counts_to_probs(mc)
        else:
            sev_m = sev_probs
        vec = [1.0 - p_m] + [p_m * x for x in sev_m]
        z = sum(vec)
        probs_by_month[ym] = [x / z for x in vec]

    detail = {
        "score_family": "spd",
        "method": "occurrence_x_severity",
        "probs_by_month": probs_by_month,
        "occurrence_method": occurrence_method,
        "p_event_pooled": p_event,
        "forecast_months": months,
        "n_severity_values": len(severity_values),
        "n_pa_months_all": n_pa_months_all,
        "n_occurrence_obs": total_obs,
        "n_occurrence_events": total_events,
    }
    source_bits = ["facts_resolved:pa"]
    if occurrence_method == "gdacs_seasonal_event_rate":
        source_bits.append("gdacs_occurrence")
    return probs, "+".join(source_bits), detail


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def base_rate_spd(
    con,
    iso3: str,
    hazard_code: str,
    metric: str,
    as_of: Any,
) -> Tuple[List[float], str, Dict[str, Any]]:
    """The base-rate distribution over the metric's buckets.

    Parameters
    ----------
    con : DuckDB connection (caller-owned; never opened or closed here).
    iso3, hazard_code, metric : the question identity.
    as_of : the question's window_start month ('YYYY-MM', a date, or a
        'YYYY-MM-DD' string). Only history strictly before this month is used.

    Returns
    -------
    (probs, source, detail):
      - probs: list of K probabilities over the metric's buckets (binary
        questions return the two-element ``[p, 1-p]`` form), or ``[]`` when no
        base rate exists for this pair — callers must treat that as "no
        anchor", never invent one.
      - source: provenance string ('NONE' when probs is empty).
      - detail: dict with at least ``score_family`` when probs is non-empty;
        may carry ``probs_by_month`` for month-varying anchors.
    """
    iso3_up = (iso3 or "").upper().strip()
    hz = (hazard_code or "").upper().strip()
    m = (metric or "").upper().strip()
    try:
        as_of_ym = _as_of_ym(as_of)
    except ValueError as exc:
        return [], NO_BASE_RATE_SOURCE, {"reason": str(exc)}
    if not iso3_up or not hz or not m:
        return [], NO_BASE_RATE_SOURCE, {"reason": "missing iso3/hazard/metric"}

    if m == "EVENT_OCCURRENCE":
        if hz in _GDACS_HAZARDS:
            return _binary_base_rate(con, iso3_up, hz, as_of_ym)
        return [], NO_BASE_RATE_SOURCE, {"reason": f"EVENT_OCCURRENCE not tracked for {hz}"}

    # Mirrors _build_history_summary's dispatch order.
    if hz == "DI":
        return [], NO_BASE_RATE_SOURCE, {"reason": "DI has no Resolver base rate"}
    if hz == "DR" and m == "PHASE3PLUS_IN_NEED":
        return _phase3_history(con, iso3_up, as_of_ym)
    if m == "PA" and hz in _SEASONAL_PA_HAZARDS:
        return _seasonal_pa(con, iso3_up, hz, as_of_ym)
    if hz == "ACE" and m == "FATALITIES":
        return _conflict_fatalities(con, iso3_up, as_of_ym)
    if hz == "ACE" and m == "PA":
        return _conflict_displacement(con, iso3_up, hz, as_of_ym)

    return [], NO_BASE_RATE_SOURCE, {"reason": f"no base-rate loader for {hz}/{m}"}


# ---------------------------------------------------------------------------
# Conflict trajectory (the figures the ACE base-rate block prints)
# ---------------------------------------------------------------------------


def conflict_trajectory(
    rows: list[tuple],
    source_name: str,
) -> Dict[str, Any]:
    """Trajectory stats the conflict base-rate block prints, from (ym, value) rows in
    ASCENDING month order: the last month, the trailing and prior three-month means, and
    the trend. One implementation, used by forecaster.cli._build_conflict_base_rate
    (the prompt) and by the scored bundle's base_rate_shown (the record of it)."""
    if not rows:
        return {
            "source": source_name,
            "last_month": None,
            "trailing_3m_avg": None,
            "prior_3m_avg": None,
            "trend_pct": None,
            "trend_direction": None,
            "last_6m": [],
            "note": f"No {source_name} data available for this country.",
        }

    # rows should be sorted ascending by ym
    last_6 = [{"ym": str(ym), "value": round(float(val or 0))} for ym, val in rows]
    values = [entry["value"] for entry in last_6]

    last_month_entry = last_6[-1]
    # trailing 3m = last 3 months, prior 3m = months 4-6
    trailing_3m_vals = values[-3:] if len(values) >= 3 else values
    prior_3m_vals = values[-6:-3] if len(values) >= 6 else values[:max(0, len(values) - 3)]

    trailing_3m_avg = round(sum(trailing_3m_vals) / len(trailing_3m_vals)) if trailing_3m_vals else None
    prior_3m_avg = round(sum(prior_3m_vals) / len(prior_3m_vals)) if prior_3m_vals else None

    trend_pct: Any = None
    trend_direction: Any = None
    trend_note: Any = None
    if trailing_3m_avg is not None and prior_3m_avg is not None:
        if prior_3m_avg < 0:
            # A percentage change from a negative base means nothing (DRC's
            # displacement read "-2364.8%" off differenced flows, Oct 2026).
            trend_note = "no trend: the earlier window has no positive base"
        elif prior_3m_avg == 0:
            if trailing_3m_avg > 0:
                trend_pct = "new_activity"
                trend_direction = "escalating"
            else:
                trend_pct = 0.0
                trend_direction = "stable"
        else:
            pct = ((trailing_3m_avg - prior_3m_avg) / prior_3m_avg) * 100
            trend_pct = round(pct, 1)
            if pct > 10:
                trend_direction = "escalating"
            elif pct < -10:
                trend_direction = "de-escalating"
            else:
                trend_direction = "stable"

    return {
        "source": source_name,
        "last_month": last_month_entry,
        "trailing_3m_avg": trailing_3m_avg,
        "prior_3m_avg": prior_3m_avg,
        "trend_pct": trend_pct,
        "trend_direction": trend_direction,
        "trend_note": trend_note,
        "last_6m": last_6,
    }
