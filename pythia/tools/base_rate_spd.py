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
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


# The IDMC rows inside facts_deltas, as a WHERE fragment. Kept beside the
# query that selects them so the live-month gate and the anchor cannot drift
# apart into two different ideas of what an IDMC row is.
_IDMC_DELTA_WHERE = (
    "lower(series_semantics) = 'new' AND ("
    "lower(source_id) IN ('idmc', 'idmc_idu') OR lower(metric) IN ("
    "'new_displacements', 'idp_displacement_new_dtm', "
    "'idp_displacement_flow_idmc'))"
)


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


def _conflict_fatalities(con, iso3: str, before_ym: str) -> Tuple[List[float], str, Dict[str, Any]]:
    """ACE/FATALITIES: the ACLED monthly-fatalities series the prompt anchors on."""
    if not _table_exists(con, "acled_monthly_fatalities"):
        return [], NO_BASE_RATE_SOURCE, {"reason": "acled_monthly_fatalities missing"}
    months = _window_months(before_ym, CONFLICT_WINDOW_MONTHS)
    rows = con.execute(
        """
        SELECT substr(CAST(month AS VARCHAR), 1, 7) AS ym, SUM(fatalities)
        FROM acled_monthly_fatalities
        WHERE iso3 = ?
          AND substr(CAST(month AS VARCHAR), 1, 7) < ?
          AND substr(CAST(month AS VARCHAR), 1, 7) >= ?
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
        live = _live_months(con, "acled_monthly_fatalities", "month", months)
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


def _conflict_displacement(con, iso3: str, hazard_code: str, before_ym: str) -> Tuple[List[float], str, Dict[str, Any]]:
    """ACE/PA: the IDMC displacement-flow series (facts_deltas), the series the
    prompt marks THIS QUESTION'S SERIES for ACE/PA."""
    if not _table_exists(con, "facts_deltas"):
        return [], NO_BASE_RATE_SOURCE, {"reason": "facts_deltas missing"}
    months = _window_months(before_ym, CONFLICT_WINDOW_MONTHS)
    rows = con.execute(
        """
        SELECT substr(CAST(ym AS VARCHAR), 1, 7) AS ym_key,
               SUM(COALESCE(value_new, 0)) AS flow_value
        FROM facts_deltas
        WHERE upper(iso3) = ?
          AND COALESCE(NULLIF(upper(hazard_code), ''), 'ACE') IN (?, 'IDU')
          AND lower(series_semantics) = 'new'
          AND (
              lower(source_id) IN ('idmc', 'idmc_idu')
              OR lower(metric) IN (
                  'new_displacements',
                  'idp_displacement_new_dtm',
                  'idp_displacement_flow_idmc'
              )
          )
          AND substr(CAST(ym AS VARCHAR), 1, 7) < ?
          AND substr(CAST(ym AS VARCHAR), 1, 7) >= ?
        GROUP BY ym_key
        """,
        [iso3, hazard_code, before_ym, months[0]],
    ).fetchall()
    observed = {str(ym): float(v or 0) for ym, v in rows if ym}
    n_quiet = 0
    if COUNT_QUIET_MONTHS_AS_ZERO and observed:
        # IDMC reports a country only when it records displacement, so an
        # anchor built from present rows alone said displacement happens every
        # month in a country IDMC reported twice a year. The window's quiet
        # months are observations, and leaving them out inflated the anchor
        # and therefore suppressed every excess measured against it.
        live = _live_months(
            con, "facts_deltas", "ym", months, extra_where=_IDMC_DELTA_WHERE
        )
        values, n_reported, n_quiet = _fill_quiet_months(observed, months, live)
    else:
        values = [observed[k] for k in sorted(observed)]
        n_reported = len(values)
    probs = _empirical_bucket_probs(values, "PA")
    if probs is None:
        return [], NO_BASE_RATE_SOURCE, {"reason": "no IDMC displacement history before window"}
    detail = {
        "score_family": "spd",
        "method": "empirical_monthly_buckets",
        "window_months": CONFLICT_WINDOW_MONTHS,
        "n_months_used": len(values),
        "n_months_reported": n_reported,
        "n_months_quiet": n_quiet,
        "values": values,
    }
    return probs, f"facts_deltas:idmc:{len(values)}m", detail


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

    detail = {
        "score_family": "spd",
        "method": "occurrence_x_severity",
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
