# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Build the current-run analysis bundle (the interpreter's input pack).

Sibling of build_scored_forecast_bundle.py, sharing common.py and guides.py.
Built at the end of the forecast chain (run_sibyl.yml, after compute_deviation
and after Sibyl writes), BEFORE any outcome exists: it packages what the run
forecast, how far each forecast moved from its base-rate anchor
(forecast_deviation), what the models were reacting to, and what changed
since the previous run.

Contents (zip root):
- MANIFEST.json    — run ids, window, counts, lineup, cost, pack_tokens +
                     the truncation record.
- ANALYST_GUIDE.md — generated (bucket tables from pythia.buckets, deviation
                     metric definitions, the never-blend rule, blind spots).
- attention_index.csv — one row per question with the deviation/impact
                     metrics and rank columns for the four orderings.
- questions/{question_id}.json — the full record (reuses the scored bundle's
                     build_question_record, plus base-rate anchor, deviation,
                     scenarios and bucket labels; no outcome exists yet).
- deltas.json      — matched against the previous run on
                     (iso3, hazard_code, metric): top-N entries/exits,
                     largest SPD movements, previous flagged risks tracking.
- blind_spots.json — no-base-rate questions, structural pending-too-new,
                     standing caveats.

Size budget: the model input is hard-capped (default 250k tokens,
PYTHIA_INTERPRETER_MAX_PACK_TOKENS). Question records are retained in
attention-rank order until the budget is spent; the truncated low-ranked
tail is recorded in the manifest, never silent. Top-ranked questions are
never truncated.

Usage:
    python -m scripts.ai_bundle.build_current_run_bundle \
        --db "$PYTHIA_DB_URL" --out-dir ai_bundle

Read-only; must never fail the pipeline — wire with continue-on-error.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.ai_bundle.build_scored_forecast_bundle import (
    _score_family,
    _test_clause,
    build_question_record,
)
from scripts.ai_bundle.common import (
    column_exists,
    open_db,
    resolve_db_path,
    row_count,
    rows_as_dicts,
    safe_json_loads,
    table_exists,
    write_bundle_zip,
    write_csv,
    write_json,
    write_manifest,
)
from scripts.ai_bundle.guides import build_current_run_guide

# The pack is what the model sees, so the names and the section assignment
# have to be decided HERE, not left to the prose. Both modules are pure
# (names reads one CSV; selection is arithmetic), so importing the
# interpreter package from the bundle builder costs nothing at import time.
from interpreter import config as _interp_config
from interpreter import decisions as _decisions
from interpreter import names as _names
from interpreter import gating as _gating
from interpreter import panels as _panels
from interpreter import performance as _performance
from interpreter import persistence as _persistence
from interpreter import secondopinion as _secondopinion
from interpreter import sector as _sector
from interpreter import selection as _selection

LOGGER = logging.getLogger(__name__)

# Best-available aggregate per question. Single-sourced from
# interpreter.names so the pack builder, the prompt pack, the API and the
# printed map cannot drift apart. Binary questions fall through to whichever
# aggregate carries them.
AGGREGATE_PREFERENCE = _names.AGGREGATE_PREFERENCE

LN2 = math.log(2.0)

DEFAULT_MAX_PACK_TOKENS = 250_000
DEFAULT_TOP_N = 8
DEFAULT_PER_CAPITA_FLOOR = 10_000.0

# chars-per-token estimate for the budget arithmetic (recorded in the
# manifest so the consumer knows how pack_tokens was derived).
CHARS_PER_TOKEN = 4.0


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name) or default)
    except ValueError:
        return default


def _estimate_tokens(text: str) -> int:
    return int(math.ceil(len(text) / CHARS_PER_TOKEN))


# ---------------------------------------------------------------------------
# Run resolution
# ---------------------------------------------------------------------------


def _resolve_run_id(con, include_test: bool) -> str | None:
    """Latest forecaster run with any forecast rows (run ids sort by mint
    time: fc_<epoch>)."""
    if not table_exists(con, "forecasts_raw"):
        return None
    rows = rows_as_dicts(
        con,
        "SELECT MAX(fr.run_id) AS run_id FROM forecasts_raw fr "
        "JOIN questions q ON q.question_id = fr.question_id "
        "WHERE fr.run_id IS NOT NULL AND fr.run_id <> ''"
        + _test_clause(con, "questions", "q", include_test),
    )
    return str(rows[0]["run_id"]) if rows and rows[0].get("run_id") else None


def _previous_run_id(con, run_id: str, include_test: bool) -> str | None:
    rows = rows_as_dicts(
        con,
        "SELECT MAX(fr.run_id) AS run_id FROM forecasts_raw fr "
        "JOIN questions q ON q.question_id = fr.question_id "
        "WHERE fr.run_id IS NOT NULL AND fr.run_id <> '' AND fr.run_id < ?"
        + _test_clause(con, "questions", "q", include_test),
        [run_id],
    )
    return str(rows[0]["run_id"]) if rows and rows[0].get("run_id") else None


def _questions_for_run(con, run_id: str, include_test: bool) -> list[dict[str, Any]]:
    return rows_as_dicts(
        con,
        "SELECT DISTINCT q.question_id, q.hs_run_id, q.iso3, q.hazard_code, "
        "q.metric, q.target_month, q.window_start_date, q.window_end_date, "
        "q.wording, q.status, q.track, q.pythia_metadata_json "
        "FROM questions q JOIN forecasts_raw fr ON fr.question_id = q.question_id "
        "WHERE fr.run_id = ?"
        + _test_clause(con, "questions", "q", include_test)
        + " ORDER BY q.question_id",
        [run_id],
    )


# ---------------------------------------------------------------------------
# Deviation + triage lookups
# ---------------------------------------------------------------------------


# What the gate and the planning sentence read. Selected by name and guarded
# by column_exists, because the table predates them: compute_deviation adds
# them by migration, and a DB from before that migration must still bundle.
#
# These were missing from the SELECT until 2026-08-10, so every attention row
# carried excess_nominal=None and exceedances=[]: the gate stamped nothing,
# selection categorised nothing, and the report fell back on whatever the
# earlier ordering produced. A column that is written and never read is the
# same as a column that was never written.
_DEVIATION_V3_COLUMNS = (
    "eiv_baserate", "excess_nominal", "excess_per_100k", "exceedances_json",
    "baserate_n_obs", "peak_horizon", "p50_peak", "p90_peak", "p_zero_peak",
)


def _load_deviation(con, run_id: str) -> dict[str, dict[str, dict[str, Any]]]:
    """{question_id: {model_name: deviation row}} for one run."""
    if not table_exists(con, "forecast_deviation"):
        return {}
    extra = [
        c for c in _DEVIATION_V3_COLUMNS
        if column_exists(con, "forecast_deviation", c)
    ]
    missing = [c for c in _DEVIATION_V3_COLUMNS if c not in extra]
    if missing:
        LOGGER.warning(
            "forecast_deviation is missing %s — the selection gate will have "
            "nothing to read and the report will carry no entries. Re-run "
            "compute_deviation against this database.",
            ", ".join(missing),
        )
    rows = rows_as_dicts(
        con,
        "SELECT question_id, model_name, score_family, js_vs_baserate, "
        "log_ev_ratio, eiv_nominal, eiv_per_100k, baserate_source, baserate_json"
        + ("".join(f", {c}" for c in extra))
        + " FROM forecast_deviation WHERE run_id = ?",
        [run_id],
    )
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(str(r["question_id"]), {})[str(r["model_name"])] = r
    return out


def _preferred_deviation(by_model: dict[str, dict[str, Any]] | None) -> dict[str, Any] | None:
    if not by_model:
        return None
    for pref in AGGREGATE_PREFERENCE:
        if pref in by_model:
            return by_model[pref]
    # Sibyl-only coverage would be odd but is not worth dropping.
    return next(iter(by_model.values()))


def _load_triage(con, hs_run_id: str | None) -> dict[tuple, dict[str, Any]]:
    """{(iso3, hazard_code): triage row} for the HS run."""
    if not hs_run_id or not table_exists(con, "hs_triage"):
        return {}
    rows = rows_as_dicts(
        con,
        "SELECT iso3, hazard_code, tier, triage_score, regime_change_score, "
        "regime_change_level FROM hs_triage WHERE run_id = ?",
        [hs_run_id],
    )
    return {(str(r["iso3"]), str(r["hazard_code"])): r for r in rows}


def _sibyl_covered(con, qids: list[str]) -> set[str]:
    if not table_exists(con, "sibyl_forecasts") or not qids:
        return set()
    rows = rows_as_dicts(
        con,
        "SELECT DISTINCT question_id FROM sibyl_forecasts "
        "WHERE question_id IN (SELECT UNNEST(?::VARCHAR[])) "
        "AND status NOT IN ('skipped')",
        [qids],
    )
    return {str(r["question_id"]) for r in rows}


# ---------------------------------------------------------------------------
# Attention index
# ---------------------------------------------------------------------------


_MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
)


def _month_label(window_start: Any, horizon: Any) -> str | None:
    """Horizon 3 of a window starting 2026-09 is "November 2026".

    Month index 1 IS the window_start month, the repo's anchoring convention
    everywhere (see CLAUDE.md's per-horizon architecture note).
    """
    if not window_start or not horizon:
        return None
    try:
        text = str(window_start)[:7]
        year, month = int(text[:4]), int(text[5:7])
        offset = month - 1 + (int(horizon) - 1)
        return f"{_MONTH_NAMES[offset % 12]} {year + offset // 12}"
    except (TypeError, ValueError, IndexError):
        return None


def _json_list(raw: Any) -> list[float]:
    """A JSON list column back to floats; [] when absent or unreadable."""
    if not raw:
        return []
    if isinstance(raw, list):
        return [float(v) for v in raw if v is not None]
    try:
        import json as _json

        return [float(v) for v in (_json.loads(raw) or []) if v is not None]
    except (TypeError, ValueError):
        return []


def _rank(rows: list[dict[str, Any]], key: str, rank_col: str, *, eligible=None) -> None:
    """Attach a 1-based descending rank on ``key`` (None = unranked)."""
    pool = [
        r for r in rows
        if r.get(key) is not None and (eligible is None or eligible(r))
    ]
    pool.sort(key=lambda r: float(r[key]), reverse=True)
    for i, r in enumerate(pool, start=1):
        r[rank_col] = i


def build_attention_rows(
    questions: list[dict[str, Any]],
    deviation: dict[str, dict[str, dict[str, Any]]],
    triage: dict[tuple, dict[str, Any]],
    sibyl_qids: set[str],
    *,
    per_capita_floor: float,
) -> list[dict[str, Any]]:
    """One attention row per question, with the four rank columns and the
    blended attention_rank the pack is ordered (and truncated) by."""
    rows: list[dict[str, Any]] = []
    for q in questions:
        qid = str(q["question_id"])
        metric = str(q.get("metric") or "").upper()
        t = triage.get((str(q.get("iso3")), str(q.get("hazard_code")))) or {}
        dev = _preferred_deviation(deviation.get(qid))
        js = dev.get("js_vs_baserate") if dev else None
        rc_score = t.get("regime_change_score")
        disagreement = None
        if js is not None and rc_score is not None:
            # Both on [0, 1]: rc_score = likelihood x magnitude; js normalised
            # by its ln 2 maximum. Large = HS and the ensemble disagree about
            # whether something is happening here.
            disagreement = abs(float(rc_score) - float(js) / LN2)
        log_ev = dev.get("log_ev_ratio") if dev else None
        rows.append(
            {
                "question_id": qid,
                "iso3": q.get("iso3"),
                # Names, not codes: the model must never have to write
                # "NIC — DR/EVENT_OCCURRENCE", and it can only write what the
                # pack gives it.
                "country_name": _names.country_name(q.get("iso3")),
                "hazard_code": q.get("hazard_code"),
                "hazard_name": _names.hazard_name(q.get("hazard_code")),
                "hazard_family": _names.hazard_family(q.get("hazard_code")),
                "metric": metric,
                "metric_name": _names.metric_name(metric),
                # Signed direction + the multiple a reader can check against
                # the sentence it appears in.
                "direction": _selection.direction(log_ev),
                "ev_multiple": _selection.ev_multiple(log_ev),
                "score_family": _score_family(metric),
                "track": q.get("track"),
                "tier": t.get("tier"),
                "triage_score": t.get("triage_score"),
                "rc_level": t.get("regime_change_level"),
                "rc_score": rc_score,
                "deviation_model": dev.get("model_name") if dev else None,
                "js_vs_baserate": js,
                "log_ev_ratio": dev.get("log_ev_ratio") if dev else None,
                "eiv_nominal": dev.get("eiv_nominal") if dev else None,
                "eiv_per_100k": dev.get("eiv_per_100k") if dev else None,
                # v3: the ordering. Expected EXCESS, not the ratio.
                "eiv_baserate": dev.get("eiv_baserate") if dev else None,
                "excess_nominal": dev.get("excess_nominal") if dev else None,
                "excess_per_100k": dev.get("excess_per_100k") if dev else None,
                "exceedances": _json_list(dev.get("exceedances_json") if dev else None),
                "baserate_n_obs": dev.get("baserate_n_obs") if dev else None,
                # What a planner acts on, at the month most likely to breach
                # the threshold. Named as a month, not "horizon 3".
                "p50_peak": dev.get("p50_peak") if dev else None,
                "p90_peak": dev.get("p90_peak") if dev else None,
                "p_zero_peak": dev.get("p_zero_peak") if dev else None,
                "peak_month": _month_label(
                    q.get("window_start_date"),
                    dev.get("peak_horizon") if dev else None,
                ),
                "baserate_source": dev.get("baserate_source") if dev else None,
                "sibyl_covered": qid in sibyl_qids,
                "rc_deviation_disagreement": disagreement,
                "rank_deviation": None,
                "rank_impact_nominal": None,
                "rank_impact_per_capita": None,
                "rank_rc_disagreement": None,
            }
        )

    # The window each row's horizons are counted from, kept for the decision
    # calendar (below) and dropped before the CSV is written.
    starts = {str(q["question_id"]): q.get("window_start_date") for q in questions}

    _rank(rows, "js_vs_baserate", "rank_deviation")
    _rank(rows, "eiv_nominal", "rank_impact_nominal")
    # The absolute floor keeps the per-capita ordering from returning the
    # same five small island states every cycle (plan §11.3).
    _rank(
        rows, "eiv_per_100k", "rank_impact_per_capita",
        eligible=lambda r: (r.get("eiv_nominal") or 0) >= per_capita_floor,
    )
    _rank(rows, "rc_deviation_disagreement", "rank_rc_disagreement")

    # The two-key gate (v3). Done here so the model receives a decided list
    # rather than being asked to apply thresholds in prose, which it cannot
    # be trusted to do. The counts come back so the selection panel, the
    # entry tags and the appendix table all read one number.
    gate_counts = _gating.gate_rows(
        rows,
        unusual_percentile=_interp_config.unusual_percentile(),
        min_probability=_interp_config.min_probability(),
        thin_min_obs=_interp_config.baserate_min_obs(),
    )
    LOGGER.info(
        "gate: %d considered, %d cleared both, %d heavy burden, %d watchlist, "
        "%d thin anchors (unusual cut js>=%.4f)",
        gate_counts.get("considered", 0), gate_counts.get("both", 0),
        gate_counts.get("major", 0), gate_counts.get("watchlist", 0),
        gate_counts.get("thin", 0), gate_counts.get("unusual_cut") or 0.0,
    )
    # The four-box assignment reads the gate the line above stamped: which
    # box a row belongs in is the gate's decision, and this only orders the
    # gated rows by expected excess and cuts them to length. The validator
    # then holds the model to the result.
    _selection.assign_categories(rows, max_entries=_interp_config.max_entries())

    # The decision calendar. Derived, never invented: the peak horizon the
    # materiality gate already found, less the hazard's configured lead time.
    for row in rows:
        decision = _decisions.decision_point(
            window_start=starts.get(str(row["question_id"])),
            peak_horizon=row.get("peak_horizon"),
            lead_months=_interp_config.lead_time_months(row.get("hazard_code")),
        )
        row["decision"] = decision or None
        row["decision_deadline"] = (decision or {}).get("deadline_month")

    rank_cols = (
        "rank_deviation", "rank_impact_nominal",
        "rank_impact_per_capita", "rank_rc_disagreement",
    )
    for r in rows:
        ranks = [r[c] for c in rank_cols if r.get(c) is not None]
        r["attention_rank"] = round(sum(ranks) / len(ranks), 2) if ranks else None
    # Order: rankable first (best blend first), then unrankable by qid.
    rows.sort(
        key=lambda r: (r["attention_rank"] is None, r["attention_rank"] or 0,
                       r["question_id"])
    )
    return rows


def _build_run_summary(
    con,
    hs_run_id: str | None,
    questions: list[dict[str, Any]],
    attention_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """The counts the report opens with: how wide the scan was, how much of
    it survived triage, and how the survivors split across the two tracks.

    Track 1 is the full ensemble (regime change detected); Track 2 is a
    single model on quiet-but-notable country-hazards. Counted over
    COUNTRIES, not questions, because that is the unit a reader pictures.
    """
    summary: dict[str, Any] = {
        "countries_scanned": None,
        "countries_with_questions": None,
        "countries_track1": None,
        "countries_track2": None,
        "n_questions": len(questions),
    }
    if hs_run_id and table_exists(con, "hs_triage"):
        try:
            rows = rows_as_dicts(
                con,
                # hs_triage keys the HS run as run_id, not hs_run_id (the
                # sibling loader at the top of this file already does this).
                "SELECT COUNT(DISTINCT iso3) AS n FROM hs_triage WHERE run_id = ?",
                [hs_run_id],
            )
            summary["countries_scanned"] = int(rows[0]["n"]) if rows else None
        except Exception as exc:  # noqa: BLE001 - a count must never fail a pack
            LOGGER.warning("run_summary: hs_triage count failed: %s", exc)

    by_track: dict[str, set[str]] = {"1": set(), "2": set()}
    countries: set[str] = set()
    for q in questions:
        iso3 = str(q.get("iso3") or "")
        if not iso3:
            continue
        countries.add(iso3)
        track = str(q.get("track") or "").strip()
        if track in by_track:
            by_track[track].add(iso3)
    summary["countries_with_questions"] = len(countries) or None
    summary["countries_track1"] = len(by_track["1"]) or None
    summary["countries_track2"] = len(by_track["2"]) or None
    summary["n_above_base_rate"] = sum(
        1 for r in attention_rows if r.get("direction") == "above"
    )
    summary["n_below_base_rate"] = sum(
        1 for r in attention_rows if r.get("direction") == "below"
    )
    return summary


ATTENTION_FIELDS = [
    "question_id", "iso3", "hazard_code", "metric", "score_family", "track",
    "tier", "triage_score", "rc_level", "rc_score", "deviation_model",
    "js_vs_baserate", "log_ev_ratio", "eiv_nominal", "eiv_per_100k",
    "baserate_source", "sibyl_covered", "rc_deviation_disagreement",
    "rank_deviation", "rank_impact_nominal", "rank_impact_per_capita",
    "rank_rc_disagreement", "attention_rank", "record_path",
    # Names and section assignment (the report's inclusion criteria).
    "country_name", "hazard_name", "hazard_family", "metric_name",
    "direction", "ev_multiple", "category", "category_rank",
    # v3: the ordering, the gate and what a planner acts on.
    "eiv_baserate", "excess_nominal", "excess_per_100k", "baserate_n_obs",
    "passed_unusual", "passed_material", "peak_horizon", "gate",
    "baserate_thin", "p50_peak", "p90_peak", "p_zero_peak", "peak_month",
    "window_shape",
    # v3: when the decision is due, and what the second reader made of it.
    "decision_deadline", "sibyl_tag",
]


# ---------------------------------------------------------------------------
# Deltas vs the previous run
# ---------------------------------------------------------------------------


def _mean_spd(con, run_id: str, qid: str, model_name: str) -> list[float] | None:
    """Month-mean bucket vector for one (run, question, model) from
    forecasts_raw; None when absent or degenerate."""
    rows = rows_as_dicts(
        con,
        "SELECT month_index, bucket_index, probability FROM forecasts_raw "
        "WHERE run_id = ? AND question_id = ? AND model_name = ? "
        "ORDER BY month_index, bucket_index",
        [run_id, qid, model_name],
    )
    if not rows:
        return None
    by_month: dict[int, dict[int, float]] = {}
    for r in rows:
        by_month.setdefault(int(r["month_index"] or 0), {})[
            int(r["bucket_index"] or 0)
        ] = float(r["probability"] or 0.0)
    k = max((max(b) for b in by_month.values() if b), default=0)
    if k < 2:
        return None
    acc = [0.0] * k
    n = 0
    for buckets in by_month.values():
        vec = [buckets.get(i + 1, 0.0) for i in range(k)]
        if sum(vec) <= 0:
            continue
        total = sum(vec)
        for i, p in enumerate(vec):
            acc[i] += p / total
        n += 1
    if n == 0:
        return None
    return [v / n for v in acc]


def _jsd(p: list[float], q: list[float]) -> float | None:
    """Lazy reuse of the calibration-advice JSD (the repo's single
    implementation); None when its import chain is unavailable."""
    try:
        import numpy as np

        from pythia.tools.generate_calibration_advice import (  # noqa: PLC0415
            _js_divergence,
        )
    except Exception:  # noqa: BLE001
        return None
    return float(_js_divergence(np.asarray(p, dtype=float), np.asarray(q, dtype=float)))


def _preferred_model_for(con, run_id: str, qid: str) -> str | None:
    rows = rows_as_dicts(
        con,
        "SELECT DISTINCT model_name FROM forecasts_raw "
        "WHERE run_id = ? AND question_id = ?",
        [run_id, qid],
    )
    present = {str(r["model_name"]) for r in rows}
    for pref in AGGREGATE_PREFERENCE:
        if pref in present:
            return pref
    return None


def _planning_by_month(
    con,
    run_id: str,
    qid: str,
    metric: str,
    window_start: Any,
) -> dict[str, float]:
    """{calendar month: planning figure} for one question in one run.

    Keyed by CALENDAR month, not horizon index, because consecutive runs
    share five of their six months at different horizon numbers. Comparing
    horizon 1 against horizon 1 would read the window sliding forward as the
    forecast changing.
    """
    model = _preferred_model_for(con, run_id, qid)
    if not model:
        return {}
    out: dict[str, float] = {}
    for horizon, vec in _monthly_spd(con, run_id, qid, model).items():
        month = _decisions.horizon_month(window_start, horizon)
        if not month:
            continue
        value = _gating.quantile(vec, metric, 0.5)
        if value is not None:
            out[month] = float(value)
    return out


def _build_persistence(
    con,
    *,
    run_id: str,
    previous_run_id: str | None,
    attention_rows: list[dict[str, Any]],
    previous_questions: list[dict[str, Any]],
    current_questions: list[dict[str, Any]],
    previous_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """How long each shown risk has been flagged, and how it has moved.

    Persistence counts the REPORTS that flagged it, newest first, stopping at
    the first that did not: a risk flagged in June and August but not July has
    been flagged once, and calling that two runs would be a lie about a gap.

    Movement is measured only on the calendar months the two runs share.
    """
    flag_sets = [r.get("flagged_keys") or [] for r in previous_reports]
    cur_starts = {
        str(q["question_id"]): q.get("window_start_date") for q in current_questions
    }
    prev_starts = {
        str(q["question_id"]): q.get("window_start_date") for q in previous_questions
    }
    prev_by_key: dict[tuple, dict[str, Any]] = {}
    for q in previous_questions:
        prev_by_key[_persistence.match_key(q)] = q

    shown = [r for r in attention_rows if r.get("category")]
    entries: list[dict[str, Any]] = []
    for row in shown:
        key = _persistence.match_key(row)
        qid = str(row["question_id"])
        runs = _persistence.consecutive_runs(key, flag_sets)
        move = None
        prev_q = prev_by_key.get(key)
        if previous_run_id and prev_q is not None:
            metric = str(row.get("metric") or "").upper()
            move = _persistence.movement(
                _planning_by_month(
                    con, previous_run_id, str(prev_q["question_id"]), metric,
                    prev_starts.get(str(prev_q["question_id"])),
                ),
                _planning_by_month(
                    con, run_id, qid, metric, cur_starts.get(qid),
                ),
            )
        entries.append({
            "question_id": qid,
            "iso3": row.get("iso3"),
            "country_name": row.get("country_name"),
            "hazard_code": row.get("hazard_code"),
            "hazard_name": row.get("hazard_name"),
            "metric": row.get("metric"),
            "metric_name": row.get("metric_name"),
            "consecutive_runs": runs,
            "persistence": _persistence.persistence_phrase(runs),
            "movement": move,
        })

    # Risks the last report flagged that this one does not show, and why.
    shown_keys = {_persistence.match_key(r) for r in shown}
    shown_qids = {str(r["question_id"]) for r in shown}
    cur_by_key = {_persistence.match_key(r): r for r in attention_rows}
    dropped: list[dict[str, Any]] = []
    if previous_reports:
        last = previous_reports[0]
        for entry in last.get("entries") or []:
            key = _persistence.match_key(entry)
            if key in shown_keys:
                continue
            reason = _persistence.drop_reason(
                cur_by_key.get(key), shown_question_ids=shown_qids
            )
            if not reason:
                continue
            dropped.append({
                "iso3": key[0],
                "country_name": _names.country_name(key[0]),
                "hazard_code": key[1],
                "hazard_name": _names.hazard_name(key[1]),
                "metric": key[2],
                "metric_name": _names.metric_name(key[2]),
                "previous_report_month": last.get("month_label"),
                "reason": reason,
            })
    return {
        "n_previous_reports": len(previous_reports),
        "entry_persistence": entries,
        "dropped_flags": dropped,
    }


def build_deltas(
    con,
    *,
    run_id: str,
    previous_run_id: str | None,
    attention_rows: list[dict[str, Any]],
    previous_deviation: dict[str, dict[str, dict[str, Any]]],
    previous_questions: list[dict[str, Any]],
    current_questions: list[dict[str, Any]] | None = None,
    previous_reports: list[dict[str, Any]] | None = None,
    top_n: int,
) -> dict[str, Any]:
    """Entries/exits from the top-N attention list, largest SPD movements,
    and how the previous run's flagged risks are tracking. Matched on
    (iso3, hazard_code, metric) — question ids are epoch-suffixed and never
    match across runs by design."""
    previous_reports = previous_reports or []
    _key = _persistence.match_key

    persistence_block = _build_persistence(
        con,
        run_id=run_id,
        previous_run_id=previous_run_id,
        attention_rows=attention_rows,
        previous_questions=previous_questions,
        current_questions=current_questions or [],
        previous_reports=previous_reports,
    )

    if not previous_run_id:
        return {
            "previous_run_id": None,
            "note": "no previous run in this DB — first cycle, no deltas",
            "match_key": "(iso3, hazard_code, metric)",
            **persistence_block,
        }

    # Previous run's attention ordering, rebuilt from its deviation rows
    # (same preference, same js-descending ordering).
    prev_by_key: dict[tuple, dict[str, Any]] = {}
    prev_q_by_id = {str(q["question_id"]): q for q in previous_questions}
    prev_rows: list[dict[str, Any]] = []
    for qid, by_model in previous_deviation.items():
        dev = _preferred_deviation(by_model)
        q = prev_q_by_id.get(qid)
        if not dev or not q:
            continue
        row = {
            "question_id": qid,
            "iso3": q.get("iso3"),
            "hazard_code": q.get("hazard_code"),
            "metric": str(q.get("metric") or "").upper(),
            "js_vs_baserate": dev.get("js_vs_baserate"),
            "log_ev_ratio": dev.get("log_ev_ratio"),
            "eiv_nominal": dev.get("eiv_nominal"),
        }
        prev_rows.append(row)
        prev_by_key[_key(row)] = row
    prev_rows.sort(key=lambda r: float(r.get("js_vs_baserate") or 0), reverse=True)
    for i, r in enumerate(prev_rows, start=1):
        r["rank_deviation"] = i
    prev_top = {_key(r) for r in prev_rows[:top_n]}

    cur_ranked = [r for r in attention_rows if r.get("attention_rank") is not None]
    cur_top = {_key(r) for r in cur_ranked[:top_n]}
    cur_by_key = {_key(r): r for r in attention_rows}

    def _describe(key: tuple, row: dict[str, Any] | None) -> dict[str, Any]:
        iso3, hz, metric = key
        d = {"iso3": iso3, "hazard_code": hz, "metric": metric}
        if row:
            d.update(
                {
                    "question_id": row.get("question_id"),
                    "js_vs_baserate": row.get("js_vs_baserate"),
                    "eiv_nominal": row.get("eiv_nominal"),
                }
            )
        return d

    entries = [_describe(k, cur_by_key.get(k)) for k in sorted(cur_top - prev_top)]
    exits = [_describe(k, prev_by_key.get(k)) for k in sorted(prev_top - cur_top)]

    # Largest SPD movements among matched pairs (JSD current vs previous
    # month-mean SPD of each run's preferred aggregate).
    movements: list[dict[str, Any]] = []
    jsd_unavailable = False
    for key, cur in cur_by_key.items():
        prev = prev_by_key.get(key)
        if not prev:
            continue
        cur_model = _preferred_model_for(con, run_id, str(cur["question_id"]))
        prev_model = _preferred_model_for(con, previous_run_id, str(prev["question_id"]))
        if not cur_model or not prev_model:
            continue
        cur_spd = _mean_spd(con, run_id, str(cur["question_id"]), cur_model)
        prev_spd = _mean_spd(con, previous_run_id, str(prev["question_id"]), prev_model)
        if not cur_spd or not prev_spd or len(cur_spd) != len(prev_spd):
            continue
        jsd = _jsd(cur_spd, prev_spd)
        if jsd is None:
            jsd_unavailable = True
            break
        movements.append(
            {
                "iso3": key[0],
                "hazard_code": key[1],
                "metric": key[2],
                "question_id": cur.get("question_id"),
                "previous_question_id": prev.get("question_id"),
                "js_current_vs_previous": round(jsd, 5),
                "current_model": cur_model,
                "previous_model": prev_model,
            }
        )
    movements.sort(key=lambda m: m["js_current_vs_previous"], reverse=True)

    # How the previous run's flagged (top-N) risks are tracking now.
    tracking = []
    for r in prev_rows[:top_n]:
        key = _key(r)
        cur = cur_by_key.get(key)
        tracking.append(
            {
                "iso3": key[0],
                "hazard_code": key[1],
                "metric": key[2],
                "previous_rank_deviation": r.get("rank_deviation"),
                "previous_js_vs_baserate": r.get("js_vs_baserate"),
                "previous_eiv_nominal": r.get("eiv_nominal"),
                "current_rank_deviation": cur.get("rank_deviation") if cur else None,
                "current_js_vs_baserate": cur.get("js_vs_baserate") if cur else None,
                "current_eiv_nominal": cur.get("eiv_nominal") if cur else None,
                "still_in_current_run": cur is not None,
            }
        )

    out: dict[str, Any] = {
        "previous_run_id": previous_run_id,
        "match_key": "(iso3, hazard_code, metric)",
        "top_n": top_n,
        "attention_entries": entries,
        "attention_exits": exits,
        "largest_spd_movements": movements[:20],
        "previous_flagged_tracking": tracking,
        **persistence_block,
    }
    if jsd_unavailable:
        out["largest_spd_movements"] = []
        out["movement_note"] = (
            "JSD implementation unavailable in this environment "
            "(pythia.tools.generate_calibration_advice import failed) — "
            "movements skipped rather than re-implemented"
        )
    return out


# ---------------------------------------------------------------------------
# The second reader (Sibyl)
# ---------------------------------------------------------------------------


def _monthly_spd(con, run_id: str, qid: str, model_name: str) -> dict[int, list[float]]:
    """{month_index: bucket vector} for one (run, question, model)."""
    rows = rows_as_dicts(
        con,
        "SELECT month_index, bucket_index, probability FROM forecasts_raw "
        "WHERE run_id = ? AND question_id = ? AND model_name = ? "
        "ORDER BY month_index, bucket_index",
        [run_id, qid, model_name],
    )
    by_month: dict[int, dict[int, float]] = {}
    for r in rows:
        by_month.setdefault(int(r["month_index"] or 0), {})[
            int(r["bucket_index"] or 0)
        ] = float(r["probability"] or 0.0)
    out: dict[int, list[float]] = {}
    k = max((max(b) for b in by_month.values() if b), default=0)
    for month, buckets in by_month.items():
        vec = [buckets.get(i + 1, 0.0) for i in range(k)]
        if sum(vec) > 0:
            out[month] = vec
    return out


def _expected_value(vec: list[float] | None, metric: str) -> float | None:
    if not vec:
        return None
    try:
        return float(_gating.expected_value(vec, metric))
    except Exception:  # noqa: BLE001 - a metric with no centroids has no EV
        return None


def _main_pipeline_evidence(con, run_id: str, qid: str) -> str:
    """What the ensemble actually read for this question.

    The assembled SPD prompt carries every inject the pipeline had, so it is
    the right thing to test Sibyl's findings against: anything in Sibyl's
    trials whose content is absent from here is, by construction, something
    the structured connectors did not carry.
    """
    if not table_exists(con, "llm_calls"):
        return ""
    try:
        rows = rows_as_dicts(
            con,
            "SELECT prompt_text FROM llm_calls WHERE run_id = ? AND "
            "question_id = ? AND phase IN ('spd_v2', 'binary_v2') AND "
            "prompt_text IS NOT NULL ORDER BY length(prompt_text) DESC LIMIT 1",
            [run_id, qid],
        )
    except Exception as exc:  # noqa: BLE001 - evidence is best effort
        LOGGER.debug("sibyl: prompt lookup failed for %s: %s", qid, exc)
        return ""
    return str(rows[0]["prompt_text"] or "") if rows else ""


# Below this much comparison text we do not claim anything is novel. With no
# prompt to compare against, EVERY sentence looks new, and the report would
# announce discoveries that the main system had in front of it all along.
_MIN_EVIDENCE_CHARS = 500


def build_sibyl_section(
    con,
    run_id: str,
    attention_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Sibyl as a second reader: where it disagrees, and what it found.

    Sibyl writes its pooled SPD into forecasts_raw under model_name='sibyl'
    on the same run and question as the standard track, so the two expected
    values come from one table and one aggregation.
    """
    if not table_exists(con, "sibyl_forecasts"):
        return {"available": False, "reason": "no sibyl_forecasts table in this DB"}
    by_qid = {str(r["question_id"]): r for r in attention_rows}
    try:
        forecasts = rows_as_dicts(
            con,
            "SELECT question_id, status, skip_reason, volatility_score, "
            "js_divergence_vs_standard, js_divergence_inter_trial, trials_json, "
            "cost_usd FROM sibyl_forecasts WHERE run_id = ? ORDER BY question_id",
            [run_id],
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("sibyl section unavailable: %s", exc)
        return {"available": False, "reason": str(exc)}
    if not forecasts:
        return {"available": False, "reason": f"Sibyl covered nothing in run {run_id}"}

    ratio = _interp_config.sibyl_disagreement_ratio()
    share = _interp_config.sibyl_unsettled_share()
    n_facts = _interp_config.sibyl_novel_facts()

    rows: list[dict[str, Any]] = []
    for f in forecasts:
        qid = str(f["question_id"])
        row = by_qid.get(qid) or {}
        metric = str(row.get("metric") or "").upper()
        status = str(f.get("status") or "")
        fred_model = _preferred_model_for(con, run_id, qid)
        fred_ev = _expected_value(
            _mean_spd(con, run_id, qid, fred_model) if fred_model else None, metric
        )
        sibyl_ev = _expected_value(_mean_spd(con, run_id, qid, "sibyl"), metric)
        tag = (
            _secondopinion.direction_tag(fred_ev, sibyl_ev, ratio=ratio)
            if status not in ("skipped",)
            else _secondopinion.TAG_NOT_COVERED
        )

        novel: list[str] = []
        evidence = _main_pipeline_evidence(con, run_id, qid)
        if len(evidence) >= _MIN_EVIDENCE_CHARS:
            novel = _secondopinion.novel_facts(
                _secondopinion.trial_texts(safe_json_loads(f.get("trials_json"))),
                evidence,
                limit=n_facts,
            )
        rows.append({
            "question_id": qid,
            "iso3": row.get("iso3"),
            "country_name": row.get("country_name") or _names.country_name(row.get("iso3")),
            "hazard_name": row.get("hazard_name") or _names.hazard_name(row.get("hazard_code")),
            "metric": metric,
            "metric_name": row.get("metric_name") or _names.metric_name(metric),
            "status": status,
            "skip_reason": f.get("skip_reason"),
            "fred_expected": fred_ev,
            "sibyl_expected": sibyl_ev,
            "js_divergence_vs_standard": f.get("js_divergence_vs_standard"),
            "js_divergence_inter_trial": f.get("js_divergence_inter_trial"),
            "volatility_score": f.get("volatility_score"),
            "cost_usd": f.get("cost_usd"),
            "tag": tag,
            "unsettled": _secondopinion.unsettled(
                f.get("js_divergence_inter_trial"), share=share
            ),
            "novel_evidence": novel,
            "evidence_comparable": len(evidence) >= _MIN_EVIDENCE_CHARS,
        })

    rows.sort(
        key=lambda r: (
            -(float(r.get("js_divergence_vs_standard") or 0.0)),
            str(r.get("question_id")),
        )
    )
    for row in rows:
        tagged = by_qid.get(str(row["question_id"]))
        if tagged is not None:
            tagged["sibyl_tag"] = row["tag"]
    return {
        "available": True,
        "caveat": _secondopinion.SIBYL_CAVEAT,
        "summary": _secondopinion.summarise(rows),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# The sector comparison (ACAPS)
# ---------------------------------------------------------------------------


def build_sector_comparison(con, attention_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Where Fred is more or less worried than the prevailing sector view."""
    inform: list[dict[str, Any]] = []
    risk: list[dict[str, Any]] = []
    if table_exists(con, "acaps_inform_severity"):
        try:
            # The newest snapshot only: an older one would mix two vintages
            # into one ranking and the gaps would be partly a date artefact.
            inform = rows_as_dicts(
                con,
                "SELECT iso3, severity_score FROM acaps_inform_severity "
                "WHERE snapshot_date = (SELECT MAX(snapshot_date) "
                "FROM acaps_inform_severity)",
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("sector: INFORM severity unavailable: %s", exc)
    if table_exists(con, "acaps_risk_radar"):
        try:
            risk = rows_as_dicts(
                con, "SELECT iso3, impact, risk_level FROM acaps_risk_radar"
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("sector: Risk Radar unavailable: %s", exc)

    block = _sector.build(
        attention_rows,
        inform_rows=inform,
        risk_rows=risk,
        list_size=_interp_config.sector_list_size(),
        min_rank_gap=_interp_config.sector_min_rank_gap(),
    )
    # Names, so the model never has to turn a code into a country.
    for comparison in block.get("comparisons") or []:
        for side in ("more_worried", "less_worried"):
            for entry in comparison.get(side) or []:
                entry["country_name"] = _names.country_name(entry.get("iso3"))
    block["available"] = bool(block.get("comparisons"))
    if not block["available"]:
        block["reason"] = (
            "no ACAPS severity or risk rows in this DB to compare against"
        )
    return block


# ---------------------------------------------------------------------------
# Part B outlook: what resolves, when, and the forecast diary
# ---------------------------------------------------------------------------


def _previous_reports(con, *, include_test: bool, limit: int = 12) -> list[dict[str, Any]]:
    """Stored interpretations, newest first, with their attention entries.

    The reports themselves are the only honest record of what was flagged:
    re-running today's thresholds over an old run would tell us what we WOULD
    have said, which is a different claim.
    """
    if not table_exists(con, "interpretations"):
        return []
    test_clause = "" if include_test else " AND COALESCE(is_test, FALSE) = FALSE"
    try:
        rows = rows_as_dicts(
            con,
            "SELECT run_id, hs_run_id, created_at, content_json FROM interpretations "
            f"WHERE kind IN ('current', 'combined') AND status = 'ok'{test_clause} "
            "ORDER BY created_at DESC, version DESC LIMIT ?",
            [int(limit)],
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("previous reports unavailable: %s", exc)
        return []

    out: list[dict[str, Any]] = []
    seen_runs: set[str] = set()
    for r in rows:
        run = str(r.get("run_id") or r.get("hs_run_id") or "")
        if run in seen_runs:
            continue  # one report per run: later versions supersede earlier
        seen_runs.add(run)
        content = safe_json_loads(r.get("content_json")) or {}
        entries = [e for e in (content.get("attention") or []) if isinstance(e, dict)]
        out.append({
            "run_id": r.get("run_id"),
            "month_label": str(r.get("created_at") or "")[:7],
            "entries": entries,
            "flagged_keys": [
                _persistence.match_key(e) for e in entries
            ],
        })
    return out


def _resolutions_for(con, qids: list[str]) -> dict[str, list[dict[str, Any]]]:
    if not table_exists(con, "resolutions") or not qids:
        return {}
    try:
        rows = rows_as_dicts(
            con,
            "SELECT question_id, horizon_m, observed_month, value FROM resolutions "
            "WHERE question_id IN (SELECT UNNEST(?::VARCHAR[]))",
            [qids],
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("diary: resolutions unavailable: %s", exc)
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(str(r["question_id"]), []).append(r)
    return out


def build_performance_outlook(
    con,
    questions: list[dict[str, Any]],
    *,
    include_test: bool,
    as_of: str,
) -> dict[str, Any]:
    """Part B's material: what is scored, what is due, and the diary.

    The scored run is resolved from the tables that would carry outcomes, not
    from the current run. Before this the runner fell back to the previous
    stored interpretation, of which there has never been one, so the section
    could only ever report emptiness.
    """
    n_resolutions = row_count(con, "resolutions") if table_exists(con, "resolutions") else 0
    n_scores = row_count(con, "scores") if table_exists(con, "scores") else 0
    scored_run_id = None
    if n_scores:
        try:
            rows = rows_as_dicts(
                con,
                "SELECT MAX(run_id) AS run_id FROM scores WHERE run_id IS NOT NULL "
                "AND run_id <> ''",
            )
            scored_run_id = str(rows[0]["run_id"]) if rows and rows[0].get("run_id") else None
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("scored run lookup failed: %s", exc)

    upcoming = _performance.upcoming_resolutions(questions, as_of=as_of)
    previous = _previous_reports(con, include_test=include_test)
    diary_qids = sorted({
        str(q)
        for report in previous
        for entry in report["entries"]
        for q in (entry.get("question_ids") or [])
    })
    diary = _performance.diary(previous, _resolutions_for(con, diary_qids))
    return {
        "has_scored_run": bool(n_scores and n_resolutions),
        "scored_run_id": scored_run_id,
        "n_resolution_rows": n_resolutions,
        "n_score_rows": n_scores,
        "min_resolved_for_skill": _interp_config.min_resolved_for_skill(),
        "upcoming_resolutions": upcoming,
        "dormant_sentences": (
            [] if (n_scores and n_resolutions)
            else _performance.dormant_sentences(upcoming)
        ),
        "diary": diary,
        "score_explanations": [
            {"term": term, "text": text}
            for term, text in _performance.SCORE_EXPLANATIONS
        ],
    }


# ---------------------------------------------------------------------------
# Blind spots
# ---------------------------------------------------------------------------

STANDING_CAVEATS = [
    "PA resolution coverage is thin (~4% via the legacy IFRC-GO path); the "
    "PA resolution machine that will replace it runs in shadow mode and does "
    "not feed scoring yet.",
    "ACE inputs carry narrative-salience bias: heavily reported conflicts "
    "generate more model-visible signal, independent of severity.",
    "Blocked hazards (CU, DI, HW, ACO) are fully deactivated upstream: no "
    "questions are generated for them. Their absence is policy, not a gap.",
]


def build_blind_spots(
    attention_rows: list[dict[str, Any]],
    *,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    now = as_of or datetime.now(timezone.utc)
    # Previous complete month — the resolution pipeline's calendar cutoff.
    y, m = now.year, now.month - 1
    if m == 0:
        y, m = y - 1, 12
    cutoff = f"{y:04d}-{m:02d}"

    no_baserate = [
        {
            "question_id": r["question_id"],
            "hazard_code": r["hazard_code"],
            "metric": r["metric"],
        }
        for r in attention_rows
        if r.get("js_vs_baserate") is None
    ]
    pairs: dict[tuple, int] = {}
    for r in attention_rows:
        key = (r["hazard_code"], r["metric"])
        pairs[key] = pairs.get(key, 0) + 1
    return {
        "resolution_calendar_cutoff": cutoff,
        "note": (
            "Every question in a current run is structurally unresolved until "
            "the calendar advances past its window months — that is not a "
            "failure; performance material lives in the scored bundle."
        ),
        "questions_by_pair": [
            {"hazard_code": hz, "metric": metric, "n_questions": n}
            for (hz, metric), n in sorted(pairs.items())
        ],
        "no_baserate_questions": no_baserate,
        "standing_caveats": STANDING_CAVEATS,
    }


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _run_cost(con, run_id: str, hs_run_id: str | None) -> dict[str, Any]:
    if not table_exists(con, "llm_calls") or not column_exists(con, "llm_calls", "cost_usd"):
        return {}
    out: dict[str, Any] = {}
    try:
        rows = rows_as_dicts(
            con,
            "SELECT SUM(cost_usd) AS c FROM llm_calls WHERE run_id = ?",
            [run_id],
        )
        out["forecaster_cost_usd"] = round(float(rows[0]["c"] or 0), 4)
        if hs_run_id:
            rows = rows_as_dicts(
                con,
                "SELECT SUM(cost_usd) AS c FROM llm_calls "
                "WHERE hs_run_id = ? AND (run_id IS NULL OR run_id = '' OR run_id = ?)",
                [hs_run_id, run_id],
            )
            out["hs_and_run_cost_usd"] = round(float(rows[0]["c"] or 0), 4)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("cost rollup failed: %s", exc)
    return out


def _model_lineup() -> list[str]:
    try:
        from pythia.llm_profiles import get_ensemble_resolved

        out = []
        for m in get_ensemble_resolved():
            model_id = m.get("model_id") if isinstance(m, dict) else getattr(m, "model_id", None)
            if model_id:
                out.append(str(model_id))
        return out
    except Exception:  # noqa: BLE001
        return []


# A question record built for an ANALYST carries everything: every member's
# raw response, the whole assembled prompt, every grounding snippet. On the
# August run that is 46 to 125 KB each, and the pack could hold nine of them
# against twenty-one the report was required to cover.
#
# The interpreter does not re-read the evidence. It explains what the system
# produced and, in a sentence, what the system was reacting to. So the record
# is trimmed here rather than in the shared builder: the scored bundle is read
# by a person who does want the full text.
#
# Every cut says so in place. A silently shortened prompt would let the
# interpreter describe evidence that was never there.
TRIM_CAPS = {
    "grounding.report_markdown": 4000,
    "grounding.structural_context": 1200,
    "grounding.sources": 2000,
    "grounding.recent_signals": 1500,
    "spd_prompt": 4000,
    # The parsed spd and the reasoning trace already carry what the member
    # said; the raw response text is the same thing again in prose.
    "member.response_text": 600,
    "member.human_explanation": 1000,
    "adversarial": 3000,
}


def _cut(value: Any, cap: int) -> Any:
    """Cap a string (or the JSON form of a structure) with a visible marker."""
    if value is None:
        return None
    text = value if isinstance(value, str) else _json_dumps(value)
    if len(text) <= cap:
        return value
    return text[:cap] + f"\n[trimmed for the interpreter pack: {len(text) - cap} more characters]"


def _json_dumps(value: Any) -> str:
    import json as _json

    return _json.dumps(value, ensure_ascii=False, default=str)


def _trim_for_interpreter(record: dict[str, Any]) -> None:
    """Shrink a record to what the interpreter actually reads."""
    for pack in record.get("grounding") or []:
        if not isinstance(pack, dict):
            continue
        for field in ("report_markdown", "structural_context", "sources",
                      "recent_signals"):
            if field in pack:
                pack[field] = _cut(pack[field], TRIM_CAPS[f"grounding.{field}"])

    if record.get("spd_prompt"):
        record["spd_prompt"] = _cut(record["spd_prompt"], TRIM_CAPS["spd_prompt"])

    members = record.get("members")
    member_list = (
        members if isinstance(members, list)
        else list(members.values()) if isinstance(members, dict)
        else []
    )
    for member in member_list:
        if not isinstance(member, dict):
            continue
        for field in ("response_text", "human_explanation"):
            if member.get(field):
                member[field] = _cut(member[field], TRIM_CAPS[f"member.{field}"])
        # A diagnostic score for the trace, not something the report explains.
        member.pop("trace_quality", None)

    if record.get("adversarial"):
        record["adversarial"] = _cut(record["adversarial"], TRIM_CAPS["adversarial"])


def _augment_record(con, record: dict[str, Any], q: dict[str, Any],
                    run_id: str, deviation_by_model: dict[str, dict[str, Any]] | None) -> None:
    """Current-run additions on top of the shared question record."""
    _trim_for_interpreter(record)
    metric = str(q.get("metric") or "").upper()
    try:
        from pythia.buckets import labels_for

        labels = labels_for(metric)
        if labels:
            record["bucket_labels"] = {str(i): lbl for i, lbl in enumerate(labels, start=1)}
    except Exception:  # noqa: BLE001
        pass

    if deviation_by_model:
        record["deviation"] = {
            model: {
                k: v for k, v in row.items()
                if k in ("score_family", "js_vs_baserate", "log_ev_ratio",
                         "eiv_nominal", "eiv_per_100k", "baserate_source")
            }
            for model, row in deviation_by_model.items()
        }
        dev = _preferred_deviation(deviation_by_model)
        if dev and dev.get("baserate_json"):
            record["baserate"] = safe_json_loads(dev.get("baserate_json"))
            record["baserate_source"] = dev.get("baserate_source")

    if table_exists(con, "scenarios"):
        scen = rows_as_dicts(
            con,
            "SELECT scenario_type, bucket_label, probability, text FROM scenarios "
            "WHERE run_id = ? AND iso3 = ? AND hazard_code = ? AND UPPER(metric) = ? "
            "ORDER BY scenario_type, bucket_label",
            [run_id, q.get("iso3"), q.get("hazard_code"), metric],
        )
        if scen:
            record["scenarios"] = scen

    # A current-run record has no outcome by construction; drop the empty
    # outcome/scores stubs the shared builder emits so the consumer is not
    # invited to read absence as zeros.
    record.pop("outcome", None)
    record.pop("scores", None)


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------


def build_bundle(
    db: str,
    out_dir: Path,
    *,
    run_id: str | None = None,
    top_n: int | None = None,
    max_pack_tokens: int | None = None,
    per_capita_floor: float | None = None,
    include_test: bool = False,
    keep_staging: bool = False,
) -> Path | None:
    top_n = top_n if top_n is not None else _env_int("PYTHIA_INTERPRETER_TOP_N", DEFAULT_TOP_N)
    max_pack_tokens = (
        max_pack_tokens
        if max_pack_tokens is not None
        else _env_int("PYTHIA_INTERPRETER_MAX_PACK_TOKENS", DEFAULT_MAX_PACK_TOKENS)
    )
    per_capita_floor = (
        per_capita_floor
        if per_capita_floor is not None
        else _env_float("PYTHIA_INTERPRETER_PER_CAPITA_FLOOR", DEFAULT_PER_CAPITA_FLOOR)
    )

    con = open_db(db)
    db_path = resolve_db_path(db)
    try:
        if not table_exists(con, "questions") or not table_exists(con, "forecasts_raw"):
            LOGGER.warning("questions/forecasts_raw missing — nothing to bundle")
            return None

        run_id = run_id or _resolve_run_id(con, include_test)
        if not run_id:
            LOGGER.warning("No forecaster run found — nothing to bundle")
            return None
        questions = _questions_for_run(con, run_id, include_test)
        if not questions:
            LOGGER.warning("Run %s has no questions — nothing to bundle", run_id)
            return None
        hs_run_ids = sorted({str(q.get("hs_run_id") or "") for q in questions} - {""})
        hs_run_id = hs_run_ids[0] if hs_run_ids else None
        LOGGER.info("Bundling current run %s (%d questions)", run_id, len(questions))

        deviation = _load_deviation(con, run_id)
        triage = _load_triage(con, hs_run_id)
        qids = [str(q["question_id"]) for q in questions]
        sibyl_qids = _sibyl_covered(con, qids)

        attention_rows = build_attention_rows(
            questions, deviation, triage, sibyl_qids,
            per_capita_floor=per_capita_floor,
        )

        # The second reader runs before the deltas because it stamps each
        # attention row with its tag, and the deltas read those rows.
        sibyl_section = build_sibyl_section(con, run_id, attention_rows)
        sector_block = build_sector_comparison(con, attention_rows)

        previous_run = _previous_run_id(con, run_id, include_test)
        previous_reports = _previous_reports(con, include_test=include_test)
        deltas = build_deltas(
            con,
            run_id=run_id,
            previous_run_id=previous_run,
            attention_rows=attention_rows,
            previous_deviation=_load_deviation(con, previous_run) if previous_run else {},
            previous_questions=(
                _questions_for_run(con, previous_run, include_test) if previous_run else []
            ),
            current_questions=questions,
            previous_reports=previous_reports,
            top_n=top_n,
        )
        blind_spots = build_blind_spots(attention_rows)
        outlook = build_performance_outlook(
            con, questions, include_test=include_test,
            as_of=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        )

        label = datetime.now(timezone.utc).strftime("%Y-%m")
        staging = out_dir / f"current_run_analysis__{label}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)

        write_json(staging / "deltas.json", deltas)
        write_json(staging / "blind_spots.json", blind_spots)
        write_json(staging / "sibyl.json", sibyl_section)
        write_json(staging / "sector_comparison.json", sector_block)
        write_json(staging / "performance_outlook.json", outlook)
        write_json(
            staging / "decision_calendar.json",
            {
                "note": (
                    "Deadlines are derived: the month with most expected "
                    "impact, less a lead time configured per hazard. The "
                    "report's model writes the action, never the date."
                ),
                "lead_time_months": {
                    hz: _interp_config.lead_time_months(hz)
                    for hz in sorted({
                        str(r.get("hazard_code") or "") for r in attention_rows
                    } - {""})
                },
                "rows": _decisions.calendar_rows(
                    [r for r in attention_rows if r.get("category")]
                ),
            },
        )

        guide = build_current_run_guide({"n_questions": len(questions), "run_id": run_id})
        (staging / "ANALYST_GUIDE.md").write_text(guide, encoding="utf-8")

        # ------------------------------------------------------------------
        # Question records under the token budget: attention-rank order, the
        # low-ranked tail truncated (recorded), the top never truncated.
        # ------------------------------------------------------------------
        q_by_id = {str(q["question_id"]): q for q in questions}
        fixed_texts = [
            guide,
            *(
                (staging / name).read_text(encoding="utf-8")
                for name in (
                    "deltas.json", "blind_spots.json", "sibyl.json",
                    "sector_comparison.json", "performance_outlook.json",
                    "decision_calendar.json",
                )
            ),
        ]
        budget_left = max_pack_tokens - sum(_estimate_tokens(t) for t in fixed_texts)
        # Reserve a slice for attention_index.csv + MANIFEST (written after).
        budget_left -= _estimate_tokens("x" * (len(attention_rows) * 220 + 4000))

        kept: list[str] = []
        truncated: list[str] = []
        pack_tokens = max_pack_tokens - budget_left
        # Records are written in REPORT order, not attention order: the rows
        # the report is required to cover come first, then everything else by
        # attention rank. Without this a categorised row could lose its record
        # to a row the report never mentions, and the interpreter would be
        # asked to write about a question it cannot see.
        categorised = _selection.selected_rows(attention_rows)
        seen_ids = {str(r["question_id"]) for r in categorised}
        record_order = categorised + [
            r for r in attention_rows if str(r["question_id"]) not in seen_ids
        ]
        for row in record_order:
            qid = str(row["question_id"])
            q = q_by_id.get(qid)
            if q is None:
                continue
            if truncated:
                # Once one record is dropped, everything below it drops too —
                # the tail is contiguous, never a sampling.
                truncated.append(qid)
                row["record_path"] = ""
                continue
            try:
                record = build_question_record(
                    con, q, include_test=include_test,
                    include_sibyl_trials=False, run_id=run_id,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed to build record for %s: %s", qid, exc)
                row["record_path"] = ""
                continue
            _augment_record(con, record, q, run_id, deviation.get(qid))
            import json as _json

            text = _json.dumps(record, ensure_ascii=False, indent=1, default=str)
            tokens = _estimate_tokens(text)
            if kept and tokens > budget_left:
                truncated.append(qid)
                row["record_path"] = ""
                continue
            path = staging / "questions" / f"{qid}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            budget_left -= tokens
            pack_tokens += tokens
            kept.append(qid)
            row["record_path"] = f"questions/{qid}.json"
            del record, text

        if truncated:
            LOGGER.warning(
                "Token budget %d: kept %d question records, truncated the "
                "%d lowest-ranked (recorded in MANIFEST.json)",
                max_pack_tokens, len(kept), len(truncated),
            )
        # A categorised row without a record is the one truncation that
        # actually damages the report, so it is reported on its own rather
        # than buried in a count.
        dropped_categorised = [q for q in truncated if q in seen_ids]
        if dropped_categorised:
            LOGGER.warning(
                "Token budget: %d of the %d rows the report must cover lost "
                "their question record: %s",
                len(dropped_categorised), len(seen_ids), dropped_categorised[:5],
            )

        write_csv(staging / "attention_index.csv", ATTENTION_FIELDS, attention_rows)

        table_counts = {}
        for t in ("questions", "forecasts_raw", "forecasts_ensemble",
                  "forecast_deviation", "hs_triage", "sibyl_forecasts"):
            if table_exists(con, t):
                table_counts[t] = row_count(con, t)

        window_starts = sorted(
            str(q.get("window_start_date")) for q in questions if q.get("window_start_date")
        )
        target_months = sorted(
            str(q.get("target_month")) for q in questions if q.get("target_month")
        )
        by_pair: dict[str, int] = {}
        for r in attention_rows:
            key = f"{r['hazard_code']}/{r['metric']}"
            by_pair[key] = by_pair.get(key, 0) + 1

        run_summary = _build_run_summary(con, hs_run_id, questions, attention_rows)
        section_counts: dict[str, int] = {}
        for r in attention_rows:
            if r.get("category"):
                key = f"{r['category']}/{r['hazard_family']}"
                section_counts[key] = section_counts.get(key, 0) + 1

        write_manifest(
            staging,
            bundle_kind="current_run_analysis",
            db_path=db_path,
            table_counts=table_counts,
            extra={
                "run_id": run_id,
                "hs_run_id": hs_run_id,
                "previous_run_id": previous_run,
                "window": {
                    "start": window_starts[0] if window_starts else None,
                    "target_month": target_months[-1] if target_months else None,
                },
                "n_questions": len(questions),
                "questions_by_pair": by_pair,
                # The report opens with these, so they are computed here and
                # quoted, never counted by the model.
                "run_summary": run_summary,
                "section_counts": section_counts,
                # The report's own length budget, so the model is told the
                # number rather than left to infer it from the row count.
                "max_entries": _interp_config.max_entries(),
                "n_entries": len(
                    [r for r in attention_rows if r.get("category")]
                ),
                "n_sibyl_rows": len((sibyl_section.get("rows") or [])),
                "has_scored_run": outlook.get("has_scored_run"),
                "n_with_deviation": sum(
                    1 for r in attention_rows if r.get("js_vs_baserate") is not None
                ),
                "n_sibyl_covered": len(sibyl_qids),
                "model_lineup": _model_lineup(),
                "cost": _run_cost(con, run_id, hs_run_id),
                "top_n": top_n,
                "per_capita_floor": per_capita_floor,
                "token_budget": max_pack_tokens,
                "chars_per_token_estimate": CHARS_PER_TOKEN,
                "pack_tokens": pack_tokens,
                "n_question_records_kept": len(kept),
                "truncated_question_ids": truncated,
                # The rows the report must cover that lost their record.
                # Empty is the only healthy value.
                "truncated_categorised_question_ids": dropped_categorised,
                "include_test": include_test,
            },
        )

        zip_path = out_dir / f"current_run_analysis__{label}.zip"
        write_bundle_zip(staging, zip_path)
        if not keep_staging:
            shutil.rmtree(staging, ignore_errors=True)
        LOGGER.info(
            "Bundle written: %s (%.1f MB, pack_tokens≈%d)",
            zip_path, zip_path.stat().st_size / 1e6, pack_tokens,
        )
        return zip_path
    finally:
        try:
            con.close()
        except Exception:  # noqa: BLE001
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="DuckDB URL or path")
    parser.add_argument("--out-dir", default="ai_bundle", help="Output directory")
    parser.add_argument("--run-id", default=None,
                        help="Forecaster run to bundle (default: latest)")
    parser.add_argument("--top-n", type=int, default=None,
                        help="Attention list length (default: PYTHIA_INTERPRETER_TOP_N or 8)")
    parser.add_argument("--max-pack-tokens", type=int, default=None,
                        help="Hard token cap (default: PYTHIA_INTERPRETER_MAX_PACK_TOKENS or 250000)")
    parser.add_argument("--include-test", action="store_true",
                        help="Include is_test rows (default: excluded)")
    parser.add_argument("--keep-staging", action="store_true",
                        help="Keep the unzipped staging directory")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[ai_bundle] %(message)s")
    zip_path = build_bundle(
        args.db,
        Path(args.out_dir),
        run_id=args.run_id,
        top_n=args.top_n,
        max_pack_tokens=args.max_pack_tokens,
        include_test=args.include_test,
        keep_staging=args.keep_staging,
    )
    if zip_path is None:
        print("[ai_bundle] no bundle produced (no current run)")
        return 0
    print(f"[ai_bundle] BUNDLE_PATH={zip_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
