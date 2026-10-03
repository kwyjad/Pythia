# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""What a scored question was forecast WITH, recovered from the DB.

The August 2026 scored bundle could not answer the questions an analyst asks
before reading a single forecast: which ENSO reading was current, how many
months of GDACS history the model saw, how old the CrisisWatch edition was,
what the base rate rested on, which models (at which effort) made the
forecast, what series resolved it, why the calibration weights table was
empty, what the question cost. Each helper here answers one of them from
tables the travelling DB already carries. None raises: a helper that cannot
answer returns its reason, because an absent field and an unknown one are
different statements.

Two of the answers are RECONSTRUCTED rather than recorded, and say so:
the GDACS window is recomputed from ``facts_resolved`` as it stands at bundle
time, and model effort is read from today's config.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from typing import Any, Mapping

from scripts.ai_bundle.common import (
    column_exists,
    latest_run_clause,
    rows_as_dicts,
    table_exists,
)

# Mirrors pythia/tools/compute_calibration_pythia.MIN_QUESTIONS; imported
# lazily so a sparse environment still builds the bundle.
_DEFAULT_CALIBRATION_FLOOR = 20

_AGGREGATE_NAMES = frozenset(
    {"ensemble_mean_v2", "ensemble_bayesmc_v2", "track2_flash", "ensemble", "sibyl"}
)

#: How each (hazard, metric) question is resolved, in words a reader can
#: check against the question's own ``source_desc``. EVENT_OCCURRENCE and
#: PHASE3PLUS_IN_NEED are hazard-independent.
RESOLUTION_SERIES: dict[tuple[str, str], str] = {
    ("ACE", "FATALITIES"): (
        "acled_monthly_fatalities: ACLED reported deaths across ALL event "
        "types, per country-month; a live month with no row for the country "
        "resolves to 0"
    ),
    ("ACE", "PA"): (
        "facts_resolved/facts_deltas IDMC conflict displacement flow "
        "(new_displacements), per country-month; unresolved when absent"
    ),
    ("*", "PHASE3PLUS_IN_NEED"): (
        "facts_resolved phase3plus_in_need: FEWS NET or IPC Current Situation "
        "Phase 3+ population, per country-month; unresolved when absent"
    ),
    ("*", "EVENT_OCCURRENCE"): (
        "facts_resolved event_occurrence: 1 when GDACS listed an Orange or Red "
        "alert naming the country that month, else 0 for a month GDACS covered"
    ),
    ("*", "PA"): (
        "facts_resolved people affected (affected / people_affected / pa / "
        "displaced), source precedence IFRC > IDMC; unresolved when absent"
    ),
}


def resolution_series(hazard_code: str | None, metric: str | None) -> str:
    hz = str(hazard_code or "").upper()
    mt = str(metric or "").upper()
    return (
        RESOLUTION_SERIES.get((hz, mt))
        or RESOLUTION_SERIES.get(("*", mt))
        or "unknown: no resolution series is defined for this hazard/metric"
    )


# ---------------------------------------------------------------------------
# Run date
# ---------------------------------------------------------------------------


def run_date(con, hs_run_id: str | None) -> date | None:
    """The date a run was generated: hs_runs.generated_at, else the run id."""
    if not hs_run_id:
        return None
    if table_exists(con, "hs_runs"):
        try:
            row = con.execute(
                "SELECT generated_at FROM hs_runs WHERE hs_run_id = ? LIMIT 1",
                [hs_run_id],
            ).fetchone()
            if row and row[0]:
                val = row[0]
                if isinstance(val, datetime):
                    return val.date()
                if isinstance(val, date):
                    return val
                return datetime.fromisoformat(str(val)[:19]).date()
        except Exception:  # noqa: BLE001
            pass
    text = str(hs_run_id)
    digits = "".join(ch for ch in text.split("T")[0] if ch.isdigit())
    if len(digits) >= 8:
        try:
            return datetime.strptime(digits[:8], "%Y%m%d").date()
        except ValueError:
            return None
    return None


def _months_between(earlier: date, later: date) -> int:
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


# ---------------------------------------------------------------------------
# Inject status
# ---------------------------------------------------------------------------


def _enso_status(con, as_of: date | None) -> dict[str, Any]:
    if not table_exists(con, "enso_state"):
        return {"available": False, "reason": "enso_state table absent"}
    cols = [c for c in ("enso_phase", "oni", "observation_date", "status", "age_days")
            if column_exists(con, "enso_state", c)]
    where = ["1=1"]
    params: list[Any] = []
    if as_of is not None:
        where.append("fetch_date <= ?")
        params.append(as_of)
    if column_exists(con, "enso_state", "row_kind"):
        where.append("COALESCE(row_kind, 'live') IN ('live', 'repaired')")
    try:
        rows = rows_as_dicts(
            con,
            f"SELECT fetch_date, {', '.join(cols) if cols else 'NULL AS x'} "
            f"FROM enso_state WHERE {' AND '.join(where)} "
            "ORDER BY fetch_date DESC LIMIT 1",
            params,
        )
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"enso_state unreadable ({type(exc).__name__})"}
    if not rows:
        return {"available": False, "reason": "no ENSO record on or before the run date"}
    r = rows[0]
    out = {"available": True, "fetch_date": str(r.get("fetch_date"))}
    for c in cols:
        v = r.get(c)
        out[c] = str(v) if isinstance(v, (date, datetime)) else v
    return out


def _gdacs_status(con, iso3: str, hz: str, as_of: date | None) -> dict[str, Any]:
    if hz not in {"FL", "DR", "TC"}:
        return {"applicable": False}
    try:
        from forecaster.gdacs_history import gdacs_calendar_series
    except Exception as exc:  # noqa: BLE001
        return {"applicable": True, "available": False,
                "reason": f"gdacs_history unavailable ({type(exc).__name__})"}
    series = gdacs_calendar_series(con, iso3, hz, today=as_of)
    return {
        "applicable": True,
        "available": bool(series.get("history_available")),
        "window_start": series.get("window_start"),
        "window_end": series.get("window_end"),
        "total_months": series.get("total_months"),
        "event_months": series.get("event_months"),
        "reason": series.get("unavailable_reason") or None,
        "reconstructed_at_bundle_time": True,
    }


def _crisiswatch_status(con, iso3: str, hz: str, as_of: date | None) -> dict[str, Any]:
    """The CrisisWatch edition an ACE prompt could carry, with its arrow and alert.

    ACE questions ALWAYS carry this block: the edition month, its age in
    editions (months from the edition month to the run month; ICG publishes
    one a month), the country's arrow and alert, or ``available: false`` with
    the reason.
    """
    if hz != "ACE":
        return {"applicable": False}
    if not table_exists(con, "crisiswatch_entries"):
        return {"applicable": True, "available": False, "reason": "crisiswatch_entries absent"}
    where = "upper(iso3) = ?"
    params: list[Any] = [iso3.upper()]
    if as_of is not None and column_exists(con, "crisiswatch_entries", "fetched_at"):
        where += " AND CAST(fetched_at AS DATE) <= ?"
        params.append(as_of)
    extra = ", ".join(
        c if column_exists(con, "crisiswatch_entries", c) else f"NULL AS {c}"
        for c in ("arrow", "alert_type")
    )
    try:
        row = con.execute(
            f"SELECT year, month, {extra} FROM crisiswatch_entries WHERE {where} "
            "ORDER BY year DESC, month DESC LIMIT 1",
            params,
        ).fetchone()
    except Exception as exc:  # noqa: BLE001
        return {"applicable": True, "available": False,
                "reason": f"crisiswatch_entries unreadable ({type(exc).__name__})"}
    # The newest edition the system held at the run date: the prompt states
    # it, and says when this country is absent from it (Oct 2026).
    newest = None
    try:
        nwhere, nparams = "TRUE", []
        if as_of is not None and column_exists(con, "crisiswatch_entries", "fetched_at"):
            nwhere, nparams = "CAST(fetched_at AS DATE) <= ?", [as_of]
        n = con.execute(
            f"SELECT MAX(year * 100 + month) FROM crisiswatch_entries WHERE {nwhere}", nparams,
        ).fetchone()
        if n and n[0]:
            newest = f"{int(n[0]) // 100:04d}-{int(n[0]) % 100:02d}"
    except Exception:  # noqa: BLE001
        newest = None
    if not row:
        return {"applicable": True, "available": False,
                "newest_edition_held": newest,
                "coverage": "not_covered" if newest else "no_edition_held",
                "reason": "no edition for this country on or before the run date"}
    year, month = int(row[0]), int(row[1])
    edition = f"{year:04d}-{month:02d}"
    out = {
        "applicable": True, "available": True, "edition": edition,
        "arrow": row[2], "alert": row[3],
        "newest_edition_held": newest,
        "coverage": "listed" if newest in (None, edition) else "not_listed_in_newest",
    }
    if as_of is not None:
        age = _months_between(date(year, month, 1), as_of)
        out["edition_age_months"] = age
        out["edition_age_editions"] = age
        out["stale"] = age >= CRISISWATCH_STALE_EDITIONS
    return out


#: An edition this many months old is labelled stale in the prompt
#: (horizon_scanner.crisiswatch._STALE_EDITION_MONTHS).
CRISISWATCH_STALE_EDITIONS = 3

#: Conflict-forecast vintage staleness, from the issue date
#: (resolver.tools.fetch_conflict_forecasts.STALENESS_THRESHOLD_DAYS).
CONFLICT_FORECAST_STALE_DAYS = 45

#: Bundle label -> conflict_forecasts.source.
CONFLICT_FORECAST_SOURCES = {"acled_cast": "ACLED_CAST", "views": "VIEWS"}


def _conflict_forecast_vintage(con, iso3: str, source: str, as_of: date | None) -> dict[str, Any]:
    """The newest vintage of one conflict-forecast source on or before the run date."""
    if not table_exists(con, "conflict_forecasts"):
        return {"available": False, "reason": "conflict_forecasts absent"}
    params: list[Any] = [iso3.upper(), source.upper()]
    where = "upper(iso3) = ? AND upper(source) = ?"
    if as_of is not None:
        where += " AND forecast_issue_date <= ?"
        params.append(as_of)
    try:
        row = con.execute(
            f"SELECT MAX(forecast_issue_date) FROM conflict_forecasts WHERE {where}",
            params,
        ).fetchone()
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"conflict_forecasts unreadable ({type(exc).__name__})"}
    if not row or row[0] is None:
        return {"available": False,
                "reason": "no vintage for this country on or before the run date "
                          "(never ingested, or pruned by vintage retention)"}
    issued = row[0] if isinstance(row[0], date) else datetime.fromisoformat(str(row[0])[:10]).date()
    out: dict[str, Any] = {"available": True, "vintage": str(issued)}
    if as_of is not None:
        age = (as_of - issued).days
        out["age_days"] = age
        out["stale"] = age > CONFLICT_FORECAST_STALE_DAYS
    return out


def _base_rate_status(con, qid: str, run_id: str | None) -> dict[str, Any]:
    if not table_exists(con, "forecast_deviation"):
        return {"available": False, "reason": "forecast_deviation absent"}
    params: list[Any] = [qid]
    run_filter = ""
    if run_id:
        run_filter = " AND run_id = ?"
        params.append(run_id)
    try:
        rows = rows_as_dicts(
            con,
            "SELECT model_name, baserate_source FROM forecast_deviation "
            f"WHERE question_id = ?{run_filter} ORDER BY "
            "CASE model_name WHEN 'ensemble_bayesmc_v2' THEN 0 "
            "WHEN 'ensemble_mean_v2' THEN 1 WHEN 'track2_flash' THEN 2 ELSE 3 END",
            params,
        )
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"forecast_deviation unreadable ({type(exc).__name__})"}
    if not rows:
        return {"available": False,
                "reason": "no deviation row: the question had no base-rate anchor"}
    return {"available": True, "source": rows[0].get("baserate_source")}


def inject_status(con, q: Mapping[str, Any], run_id: str | None) -> dict[str, Any]:
    """Per-question: what the forecast prompt was built on."""
    iso3 = str(q.get("iso3") or "")
    hz = str(q.get("hazard_code") or "").upper()
    as_of = run_date(con, q.get("hs_run_id"))
    return {
        "run_date": str(as_of) if as_of else None,
        "enso": _enso_status(con, as_of),
        "gdacs_history": _gdacs_status(con, iso3, hz, as_of),
        "crisiswatch": _crisiswatch_status(con, iso3, hz, as_of),
        "base_rate": _base_rate_status(con, str(q.get("question_id")), run_id),
        **{
            label: (
                {"applicable": True, **_conflict_forecast_vintage(con, iso3, source, as_of)}
                if hz == "ACE" else {"applicable": False}
            )
            for label, source in CONFLICT_FORECAST_SOURCES.items()
        },
    }


# ---------------------------------------------------------------------------
# Lineup
# ---------------------------------------------------------------------------


def _config_efforts() -> dict[str, dict[str, Any]]:
    try:
        from pythia.llm_profiles import get_ensemble_resolved

        return {
            e["model_id"]: {"effort": e.get("thinking"), "shadow": bool(e.get("shadow"))}
            for e in get_ensemble_resolved()
        }
    except Exception:  # noqa: BLE001
        return {}


def lineup(con, run_id: str | None, qid: str | None = None) -> dict[str, Any]:
    """The members that forecast a question (or a run), with a stable id.

    Model ids come from the run's own llm_calls; effort from today's config
    (``effort_source`` says so), because the call log does not record it.
    """
    if not run_id or not table_exists(con, "llm_calls"):
        return {"lineup_id": None, "members": [], "reason": "no run id or no llm_calls"}
    params: list[Any] = [run_id]
    q_filter = ""
    if qid:
        q_filter = " AND question_id = ?"
        params.append(qid)
    try:
        rows = rows_as_dicts(
            con,
            "SELECT DISTINCT model_name, model_id, provider FROM llm_calls "
            f"WHERE run_id = ? AND phase IN ('spd_v2', 'binary_v2'){q_filter} "
            "AND model_id IS NOT NULL ORDER BY model_id",
            params,
        )
    except Exception as exc:  # noqa: BLE001
        return {"lineup_id": None, "members": [], "reason": f"llm_calls unreadable ({type(exc).__name__})"}
    efforts = _config_efforts()
    members = []
    for r in rows:
        mid = str(r.get("model_id") or "")
        cfg = efforts.get(mid, {})
        members.append({
            "model_name": r.get("model_name"),
            "model_id": mid,
            "provider": r.get("provider"),
            "effort": cfg.get("effort"),
            "shadow": cfg.get("shadow", False),
        })
    if not members:
        return {"lineup_id": None, "members": [], "reason": "no member calls logged"}
    return {
        "lineup_id": lineup_key(members),
        "members": members,
        "effort_source": "config_at_bundle_time",
    }


def lineup_key(members: list[Mapping[str, Any]]) -> str:
    """The lineup id: sha1 over "model_id:effort" of each member, sorted by id.

    One recipe, so a per-question record and the bulk id below agree.
    """
    ordered = sorted(members, key=lambda m: str(m.get("model_id") or ""))
    key = "|".join(f"{m.get('model_id')}:{m.get('effort') or ''}" for m in ordered)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


def lineup_ids_bulk(con) -> dict[tuple[str, str], str]:
    """{(run_id, question_id): lineup_id} for every question with member calls.

    The same recipe as :func:`lineup`, in one query, for tables that span the
    whole database (skill history) rather than one question at a time.
    """
    if not table_exists(con, "llm_calls"):
        return {}
    try:
        rows = con.execute(
            "SELECT DISTINCT run_id, question_id, model_id FROM llm_calls "
            "WHERE phase IN ('spd_v2', 'binary_v2') AND model_id IS NOT NULL "
            "AND run_id IS NOT NULL AND question_id IS NOT NULL"
        ).fetchall()
    except Exception:  # noqa: BLE001
        return {}
    efforts = _config_efforts()
    by: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run_id, qid, mid in rows:
        by.setdefault((str(run_id), str(qid)), []).append(
            {"model_id": str(mid), "effort": efforts.get(str(mid), {}).get("effort")}
        )
    return {k: lineup_key(v) for k, v in by.items()}


# ---------------------------------------------------------------------------
# Prompt, calibration, resolution counts, costs
# ---------------------------------------------------------------------------


def spd_prompt_missing_reason(con, qid: str, run_id: str | None) -> str:
    """Why a question's record carries no forecast prompt."""
    if not table_exists(con, "llm_calls"):
        return "llm_calls table absent"
    params: list[Any] = [qid]
    run_filter = ""
    if run_id:
        run_filter = " AND run_id = ?"
        params.append(run_id)
    try:
        n, n_prompt = con.execute(
            "SELECT COUNT(*), COUNT(NULLIF(prompt_text, '')) FROM llm_calls "
            f"WHERE question_id = ? AND phase IN ('spd_v2', 'binary_v2'){run_filter}",
            params,
        ).fetchone()
    except Exception as exc:  # noqa: BLE001
        return f"llm_calls unreadable ({type(exc).__name__})"
    if not n:
        return "no spd_v2/binary_v2 call logged for this question and run"
    if not n_prompt:
        return f"{n} call(s) logged with empty prompt_text"
    return "prompt present"


def calibration_status(con, qids: list[str] | None = None) -> list[dict[str, Any]]:
    """Per (hazard, metric): resolved questions with member Brier vs the floor.

    Mirrors compute_calibration_pythia: questions counted by
    (iso3, hazard, metric, target_month), member scores only, test and
    reference rows excluded. A group below the floor has no weights, and
    this is the only place the bundle says so.
    """
    try:
        from pythia.tools.compute_calibration_pythia import MIN_QUESTIONS as floor
    except Exception:  # noqa: BLE001
        floor = _DEFAULT_CALIBRATION_FLOOR
    if not (table_exists(con, "scores") and table_exists(con, "questions")):
        return []
    agg = ", ".join(f"'{n}'" for n in sorted(_AGGREGATE_NAMES))
    test = (
        " AND COALESCE(q.is_test, FALSE) = FALSE"
        if column_exists(con, "questions", "is_test") else ""
    )
    try:
        rows = rows_as_dicts(
            con,
            "SELECT UPPER(q.hazard_code) AS hazard_code, UPPER(q.metric) AS metric, "
            "COUNT(DISTINCT upper(q.iso3) || '|' || upper(q.hazard_code) || '|' || "
            "upper(q.metric) || '|' || CAST(q.target_month AS VARCHAR)) AS n_questions "
            "FROM scores s JOIN questions q ON q.question_id = s.question_id "
            "WHERE s.score_type = 'brier' AND s.model_name IS NOT NULL "
            f"AND s.model_name NOT IN ({agg}) AND s.model_name NOT LIKE '\\_\\_ext\\_%' ESCAPE '\\'"
            f"{test} GROUP BY 1, 2 ORDER BY 1, 2",
        )
    except Exception:  # noqa: BLE001
        return []
    weighted: set[tuple[str, str]] = set()
    if table_exists(con, "calibration_weights"):
        try:
            weighted = {
                (str(h).upper(), str(m).upper())
                for h, m in con.execute(
                    "SELECT DISTINCT hazard_code, metric FROM calibration_weights"
                ).fetchall()
            }
        except Exception:  # noqa: BLE001
            weighted = set()
    out = []
    for r in rows:
        n = int(r.get("n_questions") or 0)
        has = (r["hazard_code"], r["metric"]) in weighted
        out.append({
            "hazard_code": r["hazard_code"],
            "metric": r["metric"],
            "n_questions_with_member_scores": n,
            "floor": floor,
            "has_weights": has,
            "status": (
                "weighted" if has
                else f"below floor ({n} of {floor} questions)" if n < floor
                else "at or above floor but no weights stored (calibration not yet run?)"
            ),
        })
    return out


def resolution_counts(con, qids: list[str]) -> dict[str, Any]:
    """Resolved (question, horizon) pairs by horizon and by calendar month."""
    if not table_exists(con, "resolutions") or not qids:
        return {"by_horizon": {}, "by_observed_month": {}}
    try:
        by_h = con.execute(
            "SELECT horizon_m, COUNT(DISTINCT question_id) FROM resolutions "
            "WHERE question_id IN (SELECT UNNEST(?::VARCHAR[])) GROUP BY 1 ORDER BY 1",
            [qids],
        ).fetchall()
        by_m = con.execute(
            "SELECT CAST(observed_month AS VARCHAR), COUNT(DISTINCT question_id) "
            "FROM resolutions WHERE question_id IN (SELECT UNNEST(?::VARCHAR[])) "
            "GROUP BY 1 ORDER BY 1",
            [qids],
        ).fetchall()
    except Exception:  # noqa: BLE001
        return {"by_horizon": {}, "by_observed_month": {}}
    return {
        "by_horizon": {str(int(h)): int(n) for h, n in by_h if h is not None},
        "by_observed_month": {str(m)[:7]: int(n) for m, n in by_m if m is not None},
    }


def question_costs(con, qids: list[str]) -> dict[str, dict[str, float]]:
    """{question_id: {model_name: cost_usd, '__total__': cost}} for the latest run.

    Forecast-phase calls only (spd_v2 / binary_v2), because those are the
    calls a question's members made; HS and grounding spend is per country.
    """
    if not table_exists(con, "llm_calls") or not qids:
        return {}
    latest = ""
    if table_exists(con, "forecasts_ensemble"):
        latest = (
            " AND l.run_id = (SELECT MAX(fe.run_id) FROM forecasts_ensemble fe "
            "WHERE fe.question_id = l.question_id)"
        )
    try:
        rows = con.execute(
            "SELECT l.question_id, l.model_name, SUM(l.cost_usd) "
            "FROM llm_calls l WHERE l.question_id IN (SELECT UNNEST(?::VARCHAR[])) "
            f"AND l.phase IN ('spd_v2', 'binary_v2'){latest} GROUP BY 1, 2",
            [qids],
        ).fetchall()
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, dict[str, float]] = {}
    for qid, model, cost in rows:
        # A call logged with no cost is an unknown cost, not a free one.
        if cost is None:
            continue
        d = out.setdefault(str(qid), {"__total__": 0.0})
        d[str(model)] = round(float(cost or 0.0), 6)
        d["__total__"] = round(d["__total__"] + float(cost or 0.0), 6)
    return out


def reference_vectors(con, qids: list[str]) -> dict[tuple[str, str, int], list[float]]:
    """{(question_id, model_name, horizon_m): probs} for the __ext_ references."""
    if not table_exists(con, "baseline_scored_forecasts") or not qids:
        return {}
    import json

    try:
        rows = con.execute(
            "SELECT question_id, model_name, horizon_m, spd_json FROM baseline_scored_forecasts "
            "WHERE question_id IN (SELECT UNNEST(?::VARCHAR[]))",
            [qids],
        ).fetchall()
    except Exception:  # noqa: BLE001
        return {}
    out: dict[tuple[str, str, int], list[float]] = {}
    for qid, model, h, spd in rows:
        try:
            vec = [float(p) for p in json.loads(spd or "[]")]
        except Exception:  # noqa: BLE001
            continue
        out[(str(qid), str(model), int(h))] = vec
    return out


__all__ = [
    "RESOLUTION_SERIES",
    "calibration_status",
    "inject_status",
    "latest_run_clause",
    "lineup",
    "lineup_ids_bulk",
    "lineup_key",
    "question_costs",
    "reference_vectors",
    "resolution_counts",
    "resolution_series",
    "run_date",
    "spd_prompt_missing_reason",
]
