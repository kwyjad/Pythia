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
from interpreter import names as _names
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


def _load_deviation(con, run_id: str) -> dict[str, dict[str, dict[str, Any]]]:
    """{question_id: {model_name: deviation row}} for one run."""
    if not table_exists(con, "forecast_deviation"):
        return {}
    rows = rows_as_dicts(
        con,
        "SELECT question_id, model_name, score_family, js_vs_baserate, "
        "log_ev_ratio, eiv_nominal, eiv_per_100k, baserate_source, baserate_json "
        "FROM forecast_deviation WHERE run_id = ?",
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
                "baserate_source": dev.get("baserate_source") if dev else None,
                "sibyl_covered": qid in sibyl_qids,
                "rc_deviation_disagreement": disagreement,
                "rank_deviation": None,
                "rank_impact_nominal": None,
                "rank_impact_per_capita": None,
                "rank_rc_disagreement": None,
            }
        )

    _rank(rows, "js_vs_baserate", "rank_deviation")
    _rank(rows, "eiv_nominal", "rank_impact_nominal")
    # The absolute floor keeps the per-capita ordering from returning the
    # same five small island states every cycle (plan §11.3).
    _rank(
        rows, "eiv_per_100k", "rank_impact_per_capita",
        eligible=lambda r: (r.get("eiv_nominal") or 0) >= per_capita_floor,
    )
    _rank(rows, "rc_deviation_disagreement", "rank_rc_disagreement")

    # Which of the four report sections each row belongs to (if any). Done
    # here so the model receives a decided list rather than being asked to
    # apply thresholds in prose, which it cannot be trusted to do.
    _selection.assign_categories(
        rows,
        threshold_multiple=_interp_config.worsening_multiple(),
        min_entries=_interp_config.min_per_category(),
        max_entries=_interp_config.max_per_category(),
        per_capita_floor=per_capita_floor,
    )

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


def build_deltas(
    con,
    *,
    run_id: str,
    previous_run_id: str | None,
    attention_rows: list[dict[str, Any]],
    previous_deviation: dict[str, dict[str, dict[str, Any]]],
    previous_questions: list[dict[str, Any]],
    top_n: int,
) -> dict[str, Any]:
    """Entries/exits from the top-N attention list, largest SPD movements,
    and how the previous run's flagged risks are tracking. Matched on
    (iso3, hazard_code, metric) — question ids are epoch-suffixed and never
    match across runs by design."""
    if not previous_run_id:
        return {
            "previous_run_id": None,
            "note": "no previous run in this DB — first cycle, no deltas",
        }

    def _key(row: dict[str, Any]) -> tuple:
        return (
            str(row.get("iso3") or ""),
            str(row.get("hazard_code") or ""),
            str(row.get("metric") or "").upper(),
        )

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

        previous_run = _previous_run_id(con, run_id, include_test)
        deltas = build_deltas(
            con,
            run_id=run_id,
            previous_run_id=previous_run,
            attention_rows=attention_rows,
            previous_deviation=_load_deviation(con, previous_run) if previous_run else {},
            previous_questions=(
                _questions_for_run(con, previous_run, include_test) if previous_run else []
            ),
            top_n=top_n,
        )
        blind_spots = build_blind_spots(attention_rows)

        label = datetime.now(timezone.utc).strftime("%Y-%m")
        staging = out_dir / f"current_run_analysis__{label}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)

        write_json(staging / "deltas.json", deltas)
        write_json(staging / "blind_spots.json", blind_spots)

        guide = build_current_run_guide({"n_questions": len(questions), "run_id": run_id})
        (staging / "ANALYST_GUIDE.md").write_text(guide, encoding="utf-8")

        # ------------------------------------------------------------------
        # Question records under the token budget: attention-rank order, the
        # low-ranked tail truncated (recorded), the top never truncated.
        # ------------------------------------------------------------------
        q_by_id = {str(q["question_id"]): q for q in questions}
        fixed_texts = [
            guide,
            (staging / "deltas.json").read_text(encoding="utf-8"),
            (staging / "blind_spots.json").read_text(encoding="utf-8"),
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
                "worsening_multiple": _interp_config.worsening_multiple(),
                "min_per_category": _interp_config.min_per_category(),
                "max_per_category": _interp_config.max_per_category(),
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
