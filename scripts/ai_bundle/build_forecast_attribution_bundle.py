# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Build the forecast attribution bundle: where probability mass moved, and
what each model said moved it.

Sibling of build_scored_forecast_bundle.py and build_current_run_bundle.py,
sharing common.py and guides.py. Built at the Sibyl terminus (run_sibyl.yml)
after the operational debug bundle, for one forecaster run.

The material is ``forecasts_raw.reasoning_trace_json``: for every SPD model
call a stated prior SPD, an ordered list of update signals, a per-bucket
``delta`` for each, a claimed magnitude, a running ``post_update_spd`` and an
``rc_assessment``. trace_validation.py checks the arithmetic and discards the
content; generate_calibration_advice.py reads the prior and nothing else.
This bundle turns the whole trace into a queryable record.

Everything in the signal ledger is CLAIMED attribution: what a model said
moved it, written after the fact. Measured influence needs ablation, which
this bundle does not do. ANALYST_GUIDE.md says so in its opening paragraph.

Usage:
    python -m scripts.ai_bundle.build_forecast_attribution_bundle \
        --db "$PYTHIA_DB_URL" --out-dir ai_bundle [--run-id RUN] [--hs-run-id HS]

Output: ai_bundle/forecast_attribution__<run_id>.zip (the two Interpreter
packs are ``*_analysis__*``; this is deliberately not).

Read-only with respect to the database; never fails the workflow — every
section is wrapped, every failure lands in MANIFEST.json, main() returns 0.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import shutil
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlparse

import duckdb

from scripts.ai_bundle.build_current_run_bundle import (
    _expected_value,
    _jsd,
    _load_deviation,
    _mean_spd,
    _model_lineup,
    _preferred_deviation,
    _previous_run_id,
    _questions_for_run,
    _resolve_run_id,
    _run_cost,
)
from scripts.ai_bundle.build_scored_forecast_bundle import (
    _score_family,
    _test_clause,
    build_question_record,
)
from scripts.ai_bundle.common import (
    BUILDER_VERSION,
    column_exists,
    gz_write_jsonl,
    open_db,
    resolve_db_path,
    row_count,
    rows_as_dicts,
    safe_json_loads,
    table_exists,
    write_csv,
    write_json,
)
from scripts.ai_bundle.guides import build_attribution_guide, build_linkage_md

# The matching key two runs are compared on — the repo's single
# implementation, shared with the current-run bundle's deltas.
from interpreter import persistence as _persistence

LOGGER = logging.getLogger(__name__)

ATTRIBUTION_BUILDER_VERSION = "1.0.0"
BUNDLE_KIND = "forecast_attribution"
TAXONOMY_PATH = Path(__file__).resolve().parent / "signal_taxonomy.csv"
# Bump when a class is added, removed or re-patterned in a way that changes
# what an existing signal text classifies as. Month-over-month comparison
# is only meaningful within one version.
TAXONOMY_VERSION = "1.0"

# Aggregates never carry a trace and are not "models"; they are excluded
# from the ledger. track2_flash IS the member for Track-2 questions and
# stays; sibyl stays too (no trace by design — the guide says so).
ENSEMBLE_AGGREGATES = ("ensemble_mean_v2", "ensemble_bayesmc_v2")
AGGREGATE_PREFERENCE = ("ensemble_bayesmc_v2", "ensemble_mean_v2", "track2_flash")

DEFAULT_EVIDENCE_MATCH_THRESHOLD = 0.5
DEFAULT_SPLIT_CEILING_MB = 200.0
ZIP_WARN_MB = 500.0
LEDGER_SAMPLE_ROWS = 500
CHARS_PER_TOKEN = 4.0
EVIDENCE_LINKS_PER_SIGNAL = 3

HAZARD_BRIEFS = ("ACE", "DR", "FL", "TC", "HW")

# The trace_validation thresholds, restated by name so the ledger's two
# booleans are the SAME tests that module performs (its per-update detail
# dicts only carry the failing side, so the pass side is inferred from the
# absence of the failure key).
_DELTA_SUM_TOLERANCE = 0.05
_RECONCILE_L1_TOLERANCE = 0.1

# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------


def attribution_id(run_id: str, question_id: str, model_name: str, update_index: int) -> str:
    """The join key the future resolutions bundle depends on. Do not change
    the recipe once merged: sha256 of "run|question|model|index", 16 hex."""
    payload = f"{run_id}|{question_id}|{model_name}|{int(update_index)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def evidence_id(url: str | None, title: str | None) -> str:
    payload = f"{url or ''}|{title or ''}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Signal taxonomy
# ---------------------------------------------------------------------------


def load_taxonomy(path: Path = TAXONOMY_PATH) -> list[dict[str, Any]]:
    """The taxonomy rows with compiled patterns, highest priority first."""
    entries: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            pattern = (row.get("pattern") or "").strip()
            entry = {
                "signal_class": (row.get("signal_class") or "").strip(),
                "pattern": pattern,
                "priority": int(row.get("priority") or 0),
                "notes": row.get("notes") or "",
                "regex": re.compile(pattern, re.IGNORECASE) if pattern else None,
            }
            entries.append(entry)
    entries.sort(key=lambda e: -e["priority"])
    return entries


def classify_signal(text: str | None, taxonomy: list[dict[str, Any]]) -> tuple[str, float]:
    """(signal_class, confidence). Highest-priority match wins; confidence
    rises with how much text the pattern actually matched (a six-letter hit
    is a weaker claim than a whole phrase) and with a second hit for the
    same class. Deterministic: no model call, no randomness."""
    if not text or not str(text).strip():
        return "other", 0.0
    s = str(text)
    best: dict[str, Any] | None = None
    for entry in taxonomy:
        rx = entry.get("regex")
        if rx is None or entry["signal_class"] == "no_trace":
            continue
        hits = [m.group(0) for m in rx.finditer(s)]
        if not hits:
            continue
        candidate = {
            "signal_class": entry["signal_class"],
            "priority": entry["priority"],
            "longest": max(len(h) for h in hits),
            "n_hits": len(hits),
        }
        if best is None or candidate["priority"] > best["priority"]:
            best = candidate
    if best is None:
        return "other", 0.1
    confidence = 0.5 + 0.04 * best["longest"]
    if best["n_hits"] >= 2:
        confidence += 0.05
    return best["signal_class"], round(min(confidence, 0.95), 3)


def _model_family(model_name: str) -> str:
    name = (model_name or "").lower()
    if name == "sibyl":
        return "sibyl"
    if name.startswith("track2"):
        return "track2"
    if "gpt" in name or name.startswith("o1") or name.startswith("o3"):
        return "openai"
    if "claude" in name:
        return "anthropic"
    if "gemini" in name:
        return "google"
    return "other"


def _normalise_magnitude(value: Any) -> str:
    s = str(value or "").strip().lower()
    for word in ("small", "moderate", "large"):
        if word in s:
            return word
    return "unknown"


def _normalise_rc(value: Any) -> str:
    s = str(value or "").strip().lower()
    if not s:
        return "absent"
    if "partial" in s:
        return "partial"
    if "rebut" in s or "reject" in s:
        return "rebutted"
    if "accept" in s:
        return "accepted"
    return "absent"


# ---------------------------------------------------------------------------
# Vector arithmetic
# ---------------------------------------------------------------------------


def _as_vector(value: Any) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    out: list[float] = []
    for v in value:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            try:
                v = float(v)
            except (TypeError, ValueError):
                return None
        if not math.isfinite(float(v)):
            return None
        out.append(float(v))
    return out


def expected_index(vec: list[float] | None) -> float | None:
    """Expected zero-based bucket index; None for an empty or zero-sum vector."""
    if not vec:
        return None
    total = sum(vec)
    if total <= 0:
        return None
    return sum(i * p for i, p in enumerate(vec)) / total


def _add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def _jsd_safe(p: list[float] | None, q: list[float] | None) -> float | None:
    if not p or not q or len(p) != len(q):
        return None
    if sum(p) <= 0 or sum(q) <= 0:
        return None
    try:
        return _jsd(p, q)
    except Exception:  # noqa: BLE001
        return None


def _json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Run context
# ---------------------------------------------------------------------------


def _hs_run_for(questions: list[dict[str, Any]]) -> str | None:
    counts = Counter(str(q.get("hs_run_id")) for q in questions if q.get("hs_run_id"))
    return counts.most_common(1)[0][0] if counts else None


def _n_buckets(metric: str, fallback: int | None) -> int | None:
    try:
        from pythia.buckets import n_buckets_for

        k = n_buckets_for(metric)
        if k:
            return int(k)
    except Exception:  # noqa: BLE001
        pass
    return fallback


def _created_at_by_question(con, run_id: str) -> dict[str, Any]:
    if not table_exists(con, "forecasts_ensemble"):
        return {}
    rows = rows_as_dicts(
        con,
        "SELECT question_id, MAX(created_at) AS created_at FROM forecasts_ensemble "
        "WHERE run_id = ? GROUP BY question_id",
        [run_id],
    )
    return {str(r["question_id"]): r.get("created_at") for r in rows}


def _run_timestamp(created_by_q: Mapping[str, Any]) -> datetime:
    stamps = [v for v in created_by_q.values() if isinstance(v, datetime)]
    if stamps:
        ts = max(stamps)
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _load_triage(con, hs_run_id: str | None) -> dict[tuple[str, str], dict[str, Any]]:
    if not hs_run_id or not table_exists(con, "hs_triage"):
        return {}
    rows = rows_as_dicts(
        con,
        "SELECT iso3, hazard_code, tier, triage_score, regime_change_score, "
        "regime_change_level, regime_change_direction, regime_change_likelihood, "
        "regime_change_magnitude FROM hs_triage WHERE run_id = ?",
        [hs_run_id],
    )
    return {(str(r["iso3"]), str(r["hazard_code"])): r for r in rows}


def _members_for(con, run_id: str, qid: str) -> list[dict[str, Any]]:
    """One entry per model in forecasts_raw for (run, question), with the
    trace parsed. Aggregates excluded; sibyl and track2_flash kept."""
    rows = rows_as_dicts(
        con,
        "SELECT DISTINCT model_name, reasoning_trace_json FROM forecasts_raw "
        "WHERE run_id = ? AND question_id = ? ORDER BY model_name",
        [run_id, qid],
    )
    seen: dict[str, dict[str, Any]] = {}
    for r in rows:
        name = str(r.get("model_name") or "")
        if not name or name in ENSEMBLE_AGGREGATES:
            continue
        trace = safe_json_loads(r.get("reasoning_trace_json"))
        entry = seen.setdefault(name, {"model_name": name, "trace": None})
        if isinstance(trace, dict) and entry["trace"] is None:
            entry["trace"] = trace
    return list(seen.values())


# ---------------------------------------------------------------------------
# The signal ledger
# ---------------------------------------------------------------------------

LEDGER_COLUMNS = [
    "attribution_id", "run_id", "hs_run_id", "question_id", "iso3", "hazard_code",
    "metric", "model_name", "model_family", "update_index", "signal_text",
    "signal_class", "signal_class_confidence", "delta_json", "mass_moved_l1",
    "direction", "claimed_magnitude", "pre_spd_json", "post_spd_json",
    "delta_sums_to_zero", "post_spd_reconciles", "is_prior_row", "created_at",
]

_LEDGER_DDL = """
CREATE TABLE ledger (
    attribution_id VARCHAR, run_id VARCHAR, hs_run_id VARCHAR, question_id VARCHAR,
    iso3 VARCHAR, hazard_code VARCHAR, metric VARCHAR, model_name VARCHAR,
    model_family VARCHAR, update_index INTEGER, signal_text VARCHAR,
    signal_class VARCHAR, signal_class_confidence DOUBLE, delta_json VARCHAR,
    mass_moved_l1 DOUBLE, direction DOUBLE, claimed_magnitude VARCHAR,
    pre_spd_json VARCHAR, post_spd_json VARCHAR, delta_sums_to_zero BOOLEAN,
    post_spd_reconciles BOOLEAN, is_prior_row BOOLEAN, created_at TIMESTAMP
)
"""


def _delta_details(trace: dict[str, Any], k: int | None) -> list[dict[str, Any]]:
    """Per-update detail dicts from trace_validation's own arithmetic check,
    aligned to the dict-shaped updates in order. Empty on any failure."""
    if not k:
        return []
    try:
        from forecaster.trace_validation import _check_delta_arithmetic

        result = _check_delta_arithmetic(trace, int(k))
        details = result.get("details")
        return list(details) if isinstance(details, list) else []
    except Exception:  # noqa: BLE001
        return []


def ledger_rows_for_member(
    *,
    run_id: str,
    hs_run_id: str | None,
    q: Mapping[str, Any],
    model_name: str,
    trace: dict[str, Any] | None,
    taxonomy: list[dict[str, Any]],
    created_at: Any,
    n_buckets: int | None,
) -> list[dict[str, Any]]:
    qid = str(q["question_id"])
    base = {
        "run_id": run_id,
        "hs_run_id": hs_run_id,
        "question_id": qid,
        "iso3": q.get("iso3"),
        "hazard_code": q.get("hazard_code"),
        "metric": q.get("metric"),
        "model_name": model_name,
        "model_family": _model_family(model_name),
        "created_at": created_at,
    }

    def _row(update_index: int, **fields: Any) -> dict[str, Any]:
        row = dict(base)
        row.update(
            {
                "attribution_id": attribution_id(run_id, qid, model_name, update_index),
                "update_index": update_index,
                "signal_text": None,
                "signal_class": None,
                "signal_class_confidence": None,
                "delta_json": None,
                "mass_moved_l1": None,
                "direction": None,
                "claimed_magnitude": None,
                "pre_spd_json": None,
                "post_spd_json": None,
                "delta_sums_to_zero": None,
                "post_spd_reconciles": None,
                "is_prior_row": update_index < 0,
            }
        )
        row.update(fields)
        return row

    if not isinstance(trace, dict):
        # Absence has to be visible in the table, not inferred from a
        # missing row.
        return [_row(-1, signal_class="no_trace", signal_class_confidence=1.0)]

    prior = trace.get("prior") if isinstance(trace.get("prior"), dict) else {}
    prior_spd = _as_vector(prior.get("spd"))
    rationale = prior.get("rationale") if isinstance(prior.get("rationale"), str) else None
    prior_class, prior_conf = classify_signal(rationale, taxonomy)
    rows = [
        _row(
            -1,
            signal_text=rationale,
            signal_class=prior_class,
            signal_class_confidence=prior_conf,
            post_spd_json=_json(prior_spd),
            is_prior_row=True,
        )
    ]

    updates = trace.get("updates")
    if not isinstance(updates, list):
        return rows
    k = n_buckets or (len(prior_spd) if prior_spd else None)
    details = _delta_details(trace, k)
    detail_iter = iter(details)
    prev = prior_spd
    for idx, update in enumerate(updates):
        if not isinstance(update, dict):
            rows.append(_row(idx, signal_class="other", signal_class_confidence=0.0))
            continue
        detail = next(detail_iter, {}) or {}
        signal = update.get("signal")
        signal_text = signal if isinstance(signal, str) else (_json(signal) if signal else None)
        klass, conf = classify_signal(signal_text, taxonomy)
        delta = _as_vector(update.get("delta"))
        post = _as_vector(update.get("post_update_spd"))
        if post is None and delta is not None and prev is not None and len(prev) == len(delta):
            post = _add(prev, delta)
        mass = 0.5 * sum(abs(d) for d in delta) if delta else None
        e_pre = expected_index(prev)
        e_post = expected_index(post)
        direction = (e_post - e_pre) if (e_pre is not None and e_post is not None) else None
        delta_ok: bool | None
        if not details:
            delta_ok = (abs(sum(delta)) < _DELTA_SUM_TOLERANCE) if delta and (not k or len(delta) == k) else (False if delta is not None or update.get("delta") is not None else None)
        else:
            delta_ok = not ("issue" in detail or "delta_sum" in detail)
        reconciles: bool | None
        if "l1_norm" in detail:
            reconciles = False
        elif delta_ok and prev is not None and post is not None and delta is not None and len(prev) == len(post) == len(delta):
            l1 = sum(abs(post[i] - (prev[i] + delta[i])) for i in range(len(delta)))
            reconciles = l1 < _RECONCILE_L1_TOLERANCE
        else:
            reconciles = None
        rows.append(
            _row(
                idx,
                signal_text=signal_text,
                signal_class=klass,
                signal_class_confidence=conf,
                delta_json=_json(delta if delta is not None else update.get("delta")),
                mass_moved_l1=round(mass, 6) if mass is not None else None,
                direction=round(direction, 6) if direction is not None else None,
                claimed_magnitude=_normalise_magnitude(update.get("magnitude")),
                pre_spd_json=_json(prev),
                post_spd_json=_json(post),
                delta_sums_to_zero=delta_ok,
                post_spd_reconciles=reconciles,
            )
        )
        if post is not None:
            prev = post
    return rows


def _trace_quality_for(members: list[dict[str, Any]], hz: str, metric: str) -> dict[str, dict[str, Any]]:
    """{model_name: trace_validation result}. The prior check runs without
    the original base-rate summary and returns its neutral 0.7 — the guide
    says which components to trust."""
    try:
        from forecaster.trace_validation import validate_reasoning_traces
    except Exception:  # noqa: BLE001
        return {}
    raw_calls = [
        {"model_spec": SimpleNamespace(name=m["model_name"]), "reasoning_trace": m.get("trace")}
        for m in members
    ]
    try:
        results = validate_reasoning_traces(raw_calls, {}, hz, metric)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("trace validation failed: %s", exc)
        return {}
    return {str(r.get("model_name")): r for r in results}


def build_ledger(
    con,
    *,
    run_id: str,
    hs_run_id: str | None,
    questions: list[dict[str, Any]],
    taxonomy: list[dict[str, Any]],
) -> dict[str, Any]:
    """The ledger, the per-model trace-quality rows and the members per
    question, for one run."""
    created_by_q = _created_at_by_question(con, run_id)
    ledger: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    members_by_q: dict[str, list[dict[str, Any]]] = {}
    for q in questions:
        qid = str(q["question_id"])
        metric = str(q.get("metric") or "")
        hz = str(q.get("hazard_code") or "")
        members = _members_for(con, run_id, qid)
        members_by_q[qid] = members
        created_at = created_by_q.get(qid)
        first_prior = next(
            (
                _as_vector((m["trace"].get("prior") or {}).get("spd"))
                for m in members
                if isinstance(m.get("trace"), dict) and isinstance(m["trace"].get("prior"), dict)
            ),
            None,
        )
        k = _n_buckets(metric, len(first_prior) if first_prior else None)
        tq = _trace_quality_for(members, hz, metric)
        for m in members:
            ledger.extend(
                ledger_rows_for_member(
                    run_id=run_id,
                    hs_run_id=hs_run_id,
                    q=q,
                    model_name=m["model_name"],
                    trace=m.get("trace"),
                    taxonomy=taxonomy,
                    created_at=created_at,
                    n_buckets=k,
                )
            )
            trace = m.get("trace")
            updates = trace.get("updates") if isinstance(trace, dict) else None
            result = tq.get(m["model_name"]) or {}
            quality.append(
                {
                    "question_id": qid,
                    "iso3": q.get("iso3"),
                    "hazard_code": hz,
                    "metric": metric,
                    "model_name": m["model_name"],
                    "model_family": _model_family(m["model_name"]),
                    "has_trace": bool(isinstance(trace, dict)),
                    "n_updates": len(updates) if isinstance(updates, list) else 0,
                    "trace_quality_score": result.get("trace_quality_score"),
                    "prior_quality_score": (result.get("prior_quality") or {}).get("score"),
                    "delta_arithmetic_score": (result.get("delta_arithmetic") or {}).get("score"),
                    "magnitude_consistency_score": (result.get("magnitude_consistency") or {}).get("score"),
                }
            )
    return {"ledger": ledger, "trace_quality": quality, "members_by_q": members_by_q}


def write_ledger_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    """Parquet via DuckDB's own writer — no new dependency."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.execute(_LEDGER_DDL)
        placeholders = ", ".join("?" for _ in LEDGER_COLUMNS)
        con.executemany(
            f"INSERT INTO ledger VALUES ({placeholders})",
            [[_parquet_value(r.get(c)) for c in LEDGER_COLUMNS] for r in rows],
        )
        con.execute(f"COPY ledger TO '{path.as_posix()}' (FORMAT PARQUET)")
    finally:
        con.close()


def _parquet_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    return value


# ---------------------------------------------------------------------------
# Prior anchoring and RC assessment
# ---------------------------------------------------------------------------


def _anchor_for(deviation_by_model: dict[str, dict[str, Any]] | None) -> tuple[list[float] | None, str | None, dict[str, Any]]:
    dev = _preferred_deviation(deviation_by_model)
    if not dev:
        return None, None, {}
    payload = safe_json_loads(dev.get("baserate_json")) or {}
    probs = _as_vector(payload.get("probs")) if isinstance(payload, dict) else None
    detail = payload.get("detail") if isinstance(payload, dict) and isinstance(payload.get("detail"), dict) else {}
    return probs, dev.get("baserate_source"), detail


def build_prior_anchoring(
    questions: list[dict[str, Any]],
    members_by_q: Mapping[str, list[dict[str, Any]]],
    deviation: Mapping[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for q in questions:
        qid = str(q["question_id"])
        anchor, source, _detail = _anchor_for(deviation.get(qid))
        for m in members_by_q.get(qid, []):
            trace = m.get("trace")
            prior = _as_vector(((trace or {}).get("prior") or {}).get("spd")) if isinstance(trace, dict) else None
            jsd = _jsd_safe(prior, anchor)
            e_prior = expected_index(prior)
            e_anchor = expected_index(anchor)
            out.append(
                {
                    "question_id": qid,
                    "iso3": q.get("iso3"),
                    "hazard_code": q.get("hazard_code"),
                    "metric": q.get("metric"),
                    "model_name": m["model_name"],
                    "model_family": _model_family(m["model_name"]),
                    "has_trace": isinstance(trace, dict),
                    "prior_spd_json": _json(prior),
                    "anchor_present": anchor is not None,
                    "anchor_source": source,
                    "anchor_spd_json": _json(anchor),
                    "js_divergence": round(jsd, 6) if jsd is not None else None,
                    "js_distance": round(math.sqrt(jsd), 6) if jsd is not None and jsd >= 0 else None,
                    "expected_index_prior": round(e_prior, 4) if e_prior is not None else None,
                    "expected_index_anchor": round(e_anchor, 4) if e_anchor is not None else None,
                    "delta_expected_index": (
                        round(e_prior - e_anchor, 4)
                        if e_prior is not None and e_anchor is not None
                        else None
                    ),
                    "length_mismatch": bool(prior and anchor and len(prior) != len(anchor)),
                }
            )
    return out


def build_rc_assessment(
    questions: list[dict[str, Any]],
    members_by_q: Mapping[str, list[dict[str, Any]]],
    ledger: list[dict[str, Any]],
    triage: Mapping[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    mass_by_key: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"rc": 0.0, "total": 0.0, "n_rc": 0})
    for r in ledger:
        if r.get("is_prior_row") or r.get("mass_moved_l1") is None:
            continue
        key = (str(r["question_id"]), str(r["model_name"]))
        mass_by_key[key]["total"] += float(r["mass_moved_l1"])
        if r.get("signal_class") == "rc_flag":
            mass_by_key[key]["rc"] += float(r["mass_moved_l1"])
            mass_by_key[key]["n_rc"] += 1
    out: list[dict[str, Any]] = []
    for q in questions:
        qid = str(q["question_id"])
        t = triage.get((str(q.get("iso3")), str(q.get("hazard_code")))) or {}
        for m in members_by_q.get(qid, []):
            trace = m.get("trace")
            raw = trace.get("rc_assessment") if isinstance(trace, dict) else None
            masses = mass_by_key.get((qid, m["model_name"])) or {"rc": 0.0, "total": 0.0, "n_rc": 0}
            out.append(
                {
                    "question_id": qid,
                    "iso3": q.get("iso3"),
                    "hazard_code": q.get("hazard_code"),
                    "metric": q.get("metric"),
                    "model_name": m["model_name"],
                    "model_family": _model_family(m["model_name"]),
                    "has_trace": isinstance(trace, dict),
                    "hs_rc_level": t.get("regime_change_level"),
                    "hs_rc_score": t.get("regime_change_score"),
                    "hs_rc_direction": t.get("regime_change_direction"),
                    "hs_rc_likelihood": t.get("regime_change_likelihood"),
                    "hs_rc_magnitude": t.get("regime_change_magnitude"),
                    "rc_assessment": _normalise_rc(raw) if isinstance(trace, dict) else "absent",
                    "rc_assessment_raw": raw if isinstance(raw, str) else (_json(raw) if raw else None),
                    "n_rc_flag_signals": int(masses["n_rc"]),
                    "rc_flag_mass_moved_l1": round(masses["rc"], 6),
                    "total_mass_moved_l1": round(masses["total"], 6),
                }
            )
    return out


# ---------------------------------------------------------------------------
# Inputs: what was on the table at forecast time
# ---------------------------------------------------------------------------

_PA_METRICS = {
    "PA": ("affected", "people_affected", "pa", "displaced"),
    "FATALITIES": ("fatalities",),
    "PHASE3PLUS_IN_NEED": ("phase3plus_in_need",),
    "EVENT_OCCURRENCE": ("event_occurrence",),
}


def _ym(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y-%m")
    s = str(value)
    return s[:7] if len(s) >= 7 else None


def _scalar(con, sql: str, params: list[Any]) -> Any:
    try:
        row = con.execute(sql, params).fetchone()
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("inventory query failed: %s", exc)
        return None
    return row[0] if row else None


def _parse_date(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    s = str(value).strip()
    if not s:
        return None
    for candidate in (s, s[:19], s[:10]):
        try:
            parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _n_years_from_detail(detail: Mapping[str, Any], n_obs: int | None) -> float | None:
    for key in ("n_years", "years_assessed", "n_years_assessed"):
        v = detail.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    if n_obs:
        return round(n_obs / 12.0, 2)
    return None


def _baserate_n_obs(dev: Mapping[str, Any] | None, detail: Mapping[str, Any]) -> int | None:
    if dev and dev.get("baserate_n_obs") is not None:
        try:
            return int(dev["baserate_n_obs"])
        except (TypeError, ValueError):
            pass
    try:
        from pythia.tools.compute_deviation import _baserate_n_obs as _n_obs

        return _n_obs(dict(detail))
    except Exception:  # noqa: BLE001
        return None


def build_input_inventory(
    con,
    *,
    run_id: str,
    questions: list[dict[str, Any]],
    deviation: Mapping[str, dict[str, dict[str, Any]]],
    triage: Mapping[tuple[str, str], dict[str, Any]],
    evidence_by_q: Mapping[str, list[dict[str, Any]]],
    run_ts: datetime,
    include_test: bool,
) -> list[dict[str, Any]]:
    has_facts = table_exists(con, "facts_resolved")
    has_acled_pol = table_exists(con, "acled_political_events")
    has_acled_fat = table_exists(con, "acled_monthly_fatalities")
    has_enso = table_exists(con, "enso_state")
    has_gdelt = table_exists(con, "gdelt_conflict_indicators")
    has_tc = table_exists(con, "seasonal_tc_context_cache")
    has_cw = table_exists(con, "crisiswatch_entries")
    has_research = table_exists(con, "question_research")
    has_dev_n_obs = table_exists(con, "forecast_deviation") and column_exists(con, "forecast_deviation", "baserate_n_obs")
    run_date = run_ts.date().isoformat()
    out: list[dict[str, Any]] = []
    for q in questions:
        qid = str(q["question_id"])
        iso3 = str(q.get("iso3") or "")
        hz = str(q.get("hazard_code") or "")
        metric = str(q.get("metric") or "").upper()
        window_ym = _ym(q.get("window_start_date"))
        row: dict[str, Any] = {
            "question_id": qid,
            "iso3": iso3,
            "hazard_code": hz,
            "metric": metric,
            "track": q.get("track"),
            "window_start_month": window_ym,
        }

        # Resolver history depth (months with a row before the window).
        depth = None
        if has_facts and window_ym:
            metrics = _PA_METRICS.get(metric, (metric.lower(),))
            depth = _scalar(
                con,
                "SELECT COUNT(DISTINCT ym) FROM facts_resolved WHERE iso3 = ? "
                "AND hazard_code = ? AND LOWER(metric) IN (SELECT UNNEST(?::VARCHAR[])) AND ym < ?",
                [iso3, hz, list(metrics), window_ym],
            )
        row["resolver_history_months"] = int(depth) if depth is not None else None
        row["resolver_history_empty"] = (depth == 0) if depth is not None else None

        # Base-rate anchor.
        dev_by_model = deviation.get(qid) or {}
        dev = _preferred_deviation(dev_by_model)
        anchor, source, detail = _anchor_for(dev_by_model)
        n_obs = _baserate_n_obs(dev if has_dev_n_obs else None, detail)
        row["baserate_anchor_present"] = anchor is not None
        row["baserate_source"] = source
        row["baserate_n_obs"] = n_obs
        row["baserate_years"] = _n_years_from_detail(detail, n_obs)

        # Structured injects: counts of rows present for the country.
        row["acled_political_rows"] = (
            _scalar(con, "SELECT COUNT(*) FROM acled_political_events WHERE iso3 = ?", [iso3])
            if has_acled_pol else None
        )
        row["acled_fatality_months"] = (
            _scalar(con, "SELECT COUNT(*) FROM acled_monthly_fatalities WHERE iso3 = ?", [iso3])
            if has_acled_fat else None
        )
        row["ipc_phase_rows"] = (
            _scalar(
                con,
                "SELECT COUNT(*) FROM facts_resolved WHERE iso3 = ? AND LOWER(metric) = 'phase3plus_in_need'",
                [iso3],
            )
            if has_facts else None
        )
        row["enso_state_rows"] = (
            _scalar(con, "SELECT COUNT(*) FROM enso_state WHERE fetch_date <= ?", [run_date])
            if has_enso else None
        )
        row["gdelt_rows"] = (
            _scalar(con, "SELECT COUNT(*) FROM gdelt_conflict_indicators WHERE iso3 = ?", [iso3])
            if has_gdelt else None
        )
        row["seasonal_tc_outlook_present"] = (
            bool(_scalar(con, "SELECT COUNT(*) FROM seasonal_tc_context_cache WHERE iso3 = ?", [iso3]))
            if has_tc else None
        )
        if has_cw:
            edition = _scalar(
                con,
                "SELECT MAX(year * 100 + month) FROM crisiswatch_entries WHERE iso3 = ?",
                [iso3],
            )
            row["crisiswatch_present"] = bool(edition)
            row["crisiswatch_edition"] = f"{int(edition) // 100:04d}-{int(edition) % 100:02d}" if edition else None
        else:
            row["crisiswatch_present"] = None
            row["crisiswatch_edition"] = None

        pm_present = None
        if has_research:
            research = _scalar(
                con,
                "SELECT research_json FROM question_research WHERE run_id = ? AND question_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                [run_id, qid],
            )
            payload = safe_json_loads(research)
            pm = payload.get("prediction_market_signals") if isinstance(payload, dict) else None
            pm_present = bool(isinstance(pm, dict) and pm.get("questions"))
        row["prediction_market_present"] = pm_present

        # Evidence.
        items = evidence_by_q.get(qid) or []
        ages = sorted(
            (run_ts - d).days for d in (_parse_date(i.get("published")) for i in items) if d is not None
        )
        row["evidence_items"] = len(items)
        row["evidence_from_hs_pack"] = sum(1 for i in items if i.get("pack") in ("hs_country_pack", "hs_grounding_pack"))
        row["evidence_from_question_research"] = sum(1 for i in items if i.get("pack") == "question_research")
        row["evidence_newest_age_days"] = ages[0] if ages else None
        row["evidence_median_age_days"] = int(statistics.median(ages)) if ages else None

        # HS view.
        t = triage.get((iso3, hz)) or {}
        row["hs_tier"] = t.get("tier")
        row["hs_triage_score"] = t.get("triage_score")
        row["hs_rc_level"] = t.get("regime_change_level")

        mean_dev = dev_by_model.get("ensemble_mean_v2") or dev
        row["js_vs_baserate_mean_ensemble"] = mean_dev.get("js_vs_baserate") if mean_dev else None
        row["log_ev_ratio_mean_ensemble"] = mean_dev.get("log_ev_ratio") if mean_dev else None
        row["deviation_model"] = mean_dev.get("model_name") if mean_dev else None
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Evidence items and the evidence-to-signal linkage
# ---------------------------------------------------------------------------

_URL_KEYS = ("url", "link", "source_url", "href")
_TITLE_KEYS = ("title", "name", "headline")
_TEXT_KEYS = ("snippet", "text", "summary", "description", "content", "excerpt")
_DATE_KEYS = ("published", "published_at", "published_date", "date", "page_age", "age")
_SOURCE_KEYS = ("source", "publisher", "domain", "site")


def _first(d: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        v = d.get(k)
        if v not in (None, "", [], {}):
            return v
    return None


def extract_evidence_items(payload: Any) -> list[dict[str, Any]]:
    """Every dict in a JSON payload that carries a url or a title, wherever
    it sits. Packs differ in shape across producers and vintages; walking
    the tree is the only extractor that survives all of them."""
    found: list[dict[str, Any]] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            url = _first(node, _URL_KEYS)
            title = _first(node, _TITLE_KEYS)
            if isinstance(url, str) or isinstance(title, str):
                text = _first(node, _TEXT_KEYS)
                found.append(
                    {
                        "title": str(title) if title is not None else None,
                        "url": str(url) if url is not None else None,
                        "source": _first(node, _SOURCE_KEYS),
                        "published": _first(node, _DATE_KEYS),
                        "text": str(text) if text is not None else None,
                    }
                )
            for v in node.values():
                if isinstance(v, (dict, list)):
                    _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    _walk(payload)
    return found


def build_evidence(
    con,
    *,
    run_id: str,
    hs_run_id: str | None,
    questions: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """{question_id: [evidence item]} across question_research's three packs
    and the HS grounding packs (hs_hazard_tail_packs) the SPD prompt carried."""
    out: dict[str, list[dict[str, Any]]] = {}
    has_research = table_exists(con, "question_research")
    has_packs = table_exists(con, "hs_hazard_tail_packs")
    for q in questions:
        qid = str(q["question_id"])
        items: dict[str, dict[str, Any]] = {}

        def _add(pack: str, payload: Any) -> None:
            for it in extract_evidence_items(payload):
                eid = evidence_id(it.get("url"), it.get("title"))
                if eid in items:
                    continue
                host = ""
                if it.get("url"):
                    try:
                        host = urlparse(str(it["url"])).netloc
                    except Exception:  # noqa: BLE001
                        host = ""
                items[eid] = {
                    "evidence_id": eid,
                    "question_id": qid,
                    "iso3": q.get("iso3"),
                    "hazard_code": q.get("hazard_code"),
                    "pack": pack,
                    "title": it.get("title"),
                    "url": it.get("url"),
                    "host": host,
                    "source": it.get("source") if isinstance(it.get("source"), str) else (_json(it.get("source")) if it.get("source") else None),
                    "published": str(it["published"]) if it.get("published") is not None else None,
                    "text": it.get("text"),
                }

        if has_research:
            rows = rows_as_dicts(
                con,
                "SELECT hs_evidence_json, question_evidence_json, merged_evidence_json "
                "FROM question_research WHERE run_id = ? AND question_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                [run_id, qid],
            )
            if rows:
                r = rows[0]
                _add("hs_country_pack", safe_json_loads(r.get("hs_evidence_json")))
                _add("question_research", safe_json_loads(r.get("question_evidence_json")))
                _add("merged", safe_json_loads(r.get("merged_evidence_json")))
        if has_packs and hs_run_id:
            rows = rows_as_dicts(
                con,
                "SELECT sources_json, recent_signals_json FROM hs_hazard_tail_packs "
                "WHERE hs_run_id = ? AND iso3 = ? AND hazard_code = ?",
                [hs_run_id, q.get("iso3"), q.get("hazard_code")],
            )
            for r in rows:
                _add("hs_grounding_pack", safe_json_loads(r.get("sources_json")))
                _add("hs_grounding_pack", safe_json_loads(r.get("recent_signals_json")))
        out[qid] = list(items.values())
    return out


_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "over", "than",
    "have", "has", "been", "are", "was", "were", "will", "would", "could",
    "their", "there", "these", "those", "about", "after", "before", "since",
    "which", "while", "where", "when", "what", "more", "most", "some", "such",
    "also", "into", "onto", "upon", "very", "much", "many", "each", "other",
    "signal", "evidence", "report", "reports", "reported", "data", "month",
    "months", "year", "years", "recent", "ongoing", "continued", "continues",
}


def _tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return {
        t for t in re.findall(r"[a-z0-9][a-z0-9'-]{3,}", str(text).lower())
        if t not in _STOPWORDS
    }


def link_evidence_to_signals(
    ledger: list[dict[str, Any]],
    evidence_by_q: Mapping[str, list[dict[str, Any]]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    """Token containment between a signal's text and an evidence item's
    title plus text: |shared| / |signal tokens|. A heuristic, and the guide
    says so in the paragraph that introduces it."""
    token_cache: dict[str, set[str]] = {}
    links: list[dict[str, Any]] = []
    for r in ledger:
        if r.get("is_prior_row") or not r.get("signal_text"):
            continue
        sig = _tokens(r["signal_text"])
        if len(sig) < 2:
            continue
        candidates: list[tuple[float, str]] = []
        for item in evidence_by_q.get(str(r["question_id"])) or []:
            eid = item["evidence_id"]
            if eid not in token_cache:
                token_cache[eid] = _tokens(f"{item.get('title') or ''} {item.get('text') or ''}")
            shared = sig & token_cache[eid]
            if len(shared) < 2:
                continue
            score = len(shared) / len(sig)
            if score >= threshold:
                candidates.append((round(score, 4), eid))
        candidates.sort(key=lambda c: (-c[0], c[1]))
        for score, eid in candidates[:EVIDENCE_LINKS_PER_SIGNAL]:
            links.append(
                {
                    "attribution_id": r["attribution_id"],
                    "evidence_id": eid,
                    "question_id": r["question_id"],
                    "model_name": r["model_name"],
                    "match_score": score,
                    "match_method": "token_containment",
                }
            )
    return links


# ---------------------------------------------------------------------------
# Prompts: sections, hashes, token share
# ---------------------------------------------------------------------------

_STEP_HEADING = re.compile(r"^STEP \d+[A-Za-z]?\b")
_UPPER_HEADING = re.compile(r"^[A-Z][A-Z0-9 ,/()&'’.—–-]{2,80}:?$")
_COLON_HEADING = re.compile(r"^[A-Z][A-Za-z0-9 ,/()&'’—–-]{2,80}:$")


def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 90 or s.startswith(("-", "*", "{", "}", '"', "`", "|")):
        return False
    if _STEP_HEADING.match(s):
        return True
    if _UPPER_HEADING.match(s) and sum(c.isalpha() for c in s) >= 3:
        return True
    return bool(_COLON_HEADING.match(s))


def _section_name(heading: str) -> str:
    s = re.sub(r"\s+", " ", heading.strip()).rstrip(":").strip()
    return s[:80]


def parse_prompt_sections(prompt: str) -> list[tuple[str, str]]:
    """[(section_name, section_text)] split on the headed blocks the prompt
    builders emit. Text before the first heading is ``unclassified`` so
    nothing goes missing; the heading line stays inside its section."""
    sections: list[tuple[str, str]] = []
    name = "unclassified"
    buf: list[str] = []
    for line in (prompt or "").splitlines(keepends=True):
        if _is_heading(line):
            if buf:
                sections.append((name, "".join(buf)))
            name = _section_name(line)
            buf = [line]
        else:
            buf.append(line)
    if buf:
        sections.append((name, "".join(buf)))
    return [(n, t) for n, t in sections if t.strip()]


def _prompts_for_run(con, run_id: str, include_test: bool) -> dict[str, dict[str, Any]]:
    """{question_id: {phase, prompt_text}} — the longest logged prompt per
    question (the sent prompt, evidence appendices included)."""
    if not table_exists(con, "llm_calls"):
        return {}
    rows = rows_as_dicts(
        con,
        "SELECT question_id, phase, prompt_text FROM llm_calls WHERE run_id = ? "
        "AND phase IN ('spd_v2', 'binary_v2') AND prompt_text IS NOT NULL"
        + _test_clause(con, "llm_calls", "llm_calls", include_test)
        + " ORDER BY question_id, length(prompt_text) DESC",
        [run_id],
    )
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        qid = str(r.get("question_id") or "")
        if qid and qid not in out:
            out[qid] = {"phase": r.get("phase"), "prompt_text": r.get("prompt_text") or ""}
    return out


def build_prompt_tables(
    questions: list[dict[str, Any]],
    prompts: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    templates: dict[str, dict[str, Any]] = {}
    token_share: list[dict[str, Any]] = []
    fingerprints: dict[str, dict[str, str]] = {}
    for q in questions:
        qid = str(q["question_id"])
        entry = prompts.get(qid)
        if not entry or not entry.get("prompt_text"):
            continue
        template_id = f"{entry.get('phase')}|{q.get('hazard_code')}|{q.get('metric')}|t{q.get('track')}"
        sections = parse_prompt_sections(entry["prompt_text"])
        total = sum(len(t) for _, t in sections) or 1
        fp: dict[str, str] = {}
        for name, text in sections:
            sha = _sha256_text(text)
            fp[name] = sha
            token_share.append(
                {
                    "question_id": qid,
                    "template_id": template_id,
                    "section_name": name,
                    "char_count": len(text),
                    "est_tokens": int(math.ceil(len(text) / CHARS_PER_TOKEN)),
                    "share": round(len(text) / total, 4),
                }
            )
        fingerprints[qid] = fp
        if template_id not in templates:
            templates[template_id] = {
                "example_question_id": qid,
                "sections": [
                    {
                        "template_id": template_id,
                        "section_index": i,
                        "section_name": name,
                        "char_count": len(text),
                        "est_tokens": int(math.ceil(len(text) / CHARS_PER_TOKEN)),
                        "sha256": _sha256_text(text),
                        "example_question_id": qid,
                    }
                    for i, (name, text) in enumerate(sections)
                ],
            }
    section_rows = [s for t in templates.values() for s in t["sections"]]
    section_hashes = {
        tid: {s["section_name"]: s["sha256"] for s in t["sections"]} for tid, t in templates.items()
    }
    return {
        "prompt_sections": section_rows,
        "section_hashes": section_hashes,
        "token_share": token_share,
        "fingerprints": fingerprints,
    }


# ---------------------------------------------------------------------------
# Contrasts
# ---------------------------------------------------------------------------


def _class_mix(ledger_rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Share of total mass moved by signal class, over the given rows."""
    mass: Counter[str] = Counter()
    for r in ledger_rows:
        if r.get("is_prior_row") or r.get("mass_moved_l1") is None:
            continue
        mass[str(r.get("signal_class") or "other")] += float(r["mass_moved_l1"])
    total = sum(mass.values())
    if total <= 0:
        return {}
    return {k: round(v / total, 4) for k, v in sorted(mass.items(), key=lambda kv: -kv[1])}


def _classes_str(mix: Mapping[str, float], n: int = 3) -> str:
    return ";".join(f"{k}:{v}" for k, v in list(mix.items())[:n])


def build_model_disagreement(
    con, run_id: str, questions: list[dict[str, Any]], members_by_q: Mapping[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for q in questions:
        qid = str(q["question_id"])
        members = [m for m in members_by_q.get(qid, []) if m["model_name"] != "sibyl"]
        finals: dict[str, list[float] | None] = {}
        priors: dict[str, list[float] | None] = {}
        for m in members:
            finals[m["model_name"]] = _mean_spd(con, run_id, qid, m["model_name"])
            trace = m.get("trace")
            priors[m["model_name"]] = (
                _as_vector(((trace or {}).get("prior") or {}).get("spd")) if isinstance(trace, dict) else None
            )
        names = sorted(finals)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                jf = _jsd_safe(finals[a], finals[b])
                jp = _jsd_safe(priors[a], priors[b])
                out.append(
                    {
                        "question_id": qid,
                        "iso3": q.get("iso3"),
                        "hazard_code": q.get("hazard_code"),
                        "metric": q.get("metric"),
                        "model_a": a,
                        "model_b": b,
                        "jsd_final": round(jf, 6) if jf is not None else None,
                        "jsd_prior": round(jp, 6) if jp is not None else None,
                        "jsd_from_updates": round(jf - jp, 6) if jf is not None and jp is not None else None,
                    }
                )
    return out


def _sibyl_texts(trials: Any) -> list[str]:
    """Every string over 40 characters in Sibyl's trial traces — the closest
    thing to a signal list a trace-less track has."""
    found: list[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, str):
            if len(node) > 40:
                found.append(node)
        elif isinstance(node, dict):
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    _walk(trials)
    return found


def build_fred_vs_sibyl(
    con,
    run_id: str,
    questions: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
    taxonomy: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    has_sibyl_table = table_exists(con, "sibyl_forecasts")
    by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in ledger:
        if r.get("model_name") != "sibyl":
            by_q[str(r["question_id"])].append(r)
    out: list[dict[str, Any]] = []
    for q in questions:
        qid = str(q["question_id"])
        metric = str(q.get("metric") or "")
        sib = _mean_spd(con, run_id, qid, "sibyl")
        if sib is None:
            continue
        fred_model = next(
            (m for m in AGGREGATE_PREFERENCE if _mean_spd(con, run_id, qid, m) is not None), None
        )
        fred = _mean_spd(con, run_id, qid, fred_model) if fred_model else None
        jsd = _jsd_safe(fred, sib)
        sibyl_classes: Counter[str] = Counter()
        sibyl_status = None
        if has_sibyl_table:
            rows = rows_as_dicts(
                con,
                "SELECT status, trials_json FROM sibyl_forecasts WHERE question_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                [qid],
            )
            if rows:
                sibyl_status = rows[0].get("status")
                for text in _sibyl_texts(safe_json_loads(rows[0].get("trials_json"))):
                    klass, _ = classify_signal(text, taxonomy)
                    if klass != "other":
                        sibyl_classes[klass] += 1
        out.append(
            {
                "question_id": qid,
                "iso3": q.get("iso3"),
                "hazard_code": q.get("hazard_code"),
                "metric": metric,
                "fred_model": fred_model,
                "sibyl_status": sibyl_status,
                "jsd_fred_vs_sibyl": round(jsd, 6) if jsd is not None else None,
                "ev_fred": _expected_value(fred, metric),
                "ev_sibyl": _expected_value(sib, metric),
                "fred_signal_classes": _classes_str(_class_mix(by_q.get(qid, [])), 5),
                "sibyl_signal_classes": ";".join(f"{k}:{v}" for k, v in sibyl_classes.most_common(5)),
            }
        )
    return out


def _prior_ev(members: list[dict[str, Any]], metric: str) -> float | None:
    evs = []
    for m in members:
        trace = m.get("trace")
        prior = _as_vector(((trace or {}).get("prior") or {}).get("spd")) if isinstance(trace, dict) else None
        ev = _expected_value(prior, metric)
        if ev is not None:
            evs.append(ev)
    return round(sum(evs) / len(evs), 4) if evs else None


def _aggregate_ev(con, run_id: str, qid: str, metric: str) -> float | None:
    for m in AGGREGATE_PREFERENCE:
        vec = _mean_spd(con, run_id, qid, m)
        if vec is not None:
            return _expected_value(vec, metric)
    return None


def build_run_over_run(
    con,
    *,
    run_id: str,
    previous_run_id: str | None,
    questions: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
    members_by_q: Mapping[str, list[dict[str, Any]]],
    previous: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Matched on (iso3, hazard_code, metric) — persistence.match_key, the
    same rule the current-run bundle's deltas use."""
    if not previous_run_id or not previous:
        return []
    prev_q_by_key = {_persistence.match_key(q): q for q in previous["questions"]}
    prev_ledger_by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in previous["ledger"]:
        prev_ledger_by_q[str(r["question_id"])].append(r)
    cur_ledger_by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in ledger:
        cur_ledger_by_q[str(r["question_id"])].append(r)
    out: list[dict[str, Any]] = []
    for q in questions:
        key = _persistence.match_key(q)
        pq = prev_q_by_key.get(key)
        if not pq:
            continue
        qid, pqid = str(q["question_id"]), str(pq["question_id"])
        metric = str(q.get("metric") or "")
        ev_cur = _aggregate_ev(con, run_id, qid, metric)
        ev_prev = _aggregate_ev(con, previous_run_id, pqid, metric)
        pev_cur = _prior_ev(members_by_q.get(qid, []), metric)
        pev_prev = _prior_ev(previous["members_by_q"].get(pqid, []), metric)
        mix_cur = _class_mix(cur_ledger_by_q.get(qid, []))
        mix_prev = _class_mix(prev_ledger_by_q.get(pqid, []))
        classes = set(mix_cur) | set(mix_prev)
        mix_l1 = round(0.5 * sum(abs(mix_cur.get(c, 0.0) - mix_prev.get(c, 0.0)) for c in classes), 4) if classes else None
        out.append(
            {
                "iso3": key[0],
                "hazard_code": key[1],
                "metric": key[2],
                "question_id": qid,
                "previous_question_id": pqid,
                "previous_run_id": previous_run_id,
                "ev_current": ev_cur,
                "ev_previous": ev_prev,
                "ev_change": round(ev_cur - ev_prev, 4) if ev_cur is not None and ev_prev is not None else None,
                "prior_ev_current": pev_cur,
                "prior_ev_previous": pev_prev,
                "prior_ev_change": round(pev_cur - pev_prev, 4) if pev_cur is not None and pev_prev is not None else None,
                "top_class_current": next(iter(mix_cur), None),
                "top_class_previous": next(iter(mix_prev), None),
                "class_mix_current_json": _json(mix_cur),
                "class_mix_previous_json": _json(mix_prev),
                "class_mix_l1_change": mix_l1,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Hazard briefs
# ---------------------------------------------------------------------------


def build_hazard_brief(
    hazard: str,
    *,
    questions: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
    prior_rows: list[dict[str, Any]],
    rc_rows: list[dict[str, Any]],
    evidence_by_q: Mapping[str, list[dict[str, Any]]],
) -> str:
    qs = [q for q in questions if str(q.get("hazard_code")) == hazard]
    qids = {str(q["question_id"]) for q in qs}
    lines = [f"# {hazard} — attribution brief", ""]
    if not qs:
        lines.append(f"No {hazard} questions in this run.")
        return "\n".join(lines) + "\n"
    rows = [r for r in ledger if str(r["question_id"]) in qids and not r.get("is_prior_row")]
    mass: Counter[str] = Counter()
    freq: Counter[str] = Counter()
    per_q_mass: Counter[str] = Counter()
    per_q_models: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        klass = str(r.get("signal_class") or "other")
        freq[klass] += 1
        if r.get("mass_moved_l1") is not None:
            mass[klass] += float(r["mass_moved_l1"])
            per_q_mass[str(r["question_id"])] += float(r["mass_moved_l1"])
            per_q_models[str(r["question_id"])].add(str(r["model_name"]))
    n_no_trace = sum(1 for r in ledger if str(r["question_id"]) in qids and r.get("signal_class") == "no_trace")
    lines.append(f"Questions: {len(qs)}. Update signals in the ledger: {len(rows)}. "
                 f"Model calls with no parseable trace: {n_no_trace}.")
    lines.append("")
    lines.append("Every figure below is CLAIMED attribution — what the models said moved them.")
    lines.append("")
    lines.append("## Signal classes by total mass moved")
    lines.append("")
    lines.append("| class | mass moved (L1/2, summed) | share |")
    lines.append("|---|---|---|")
    total_mass = sum(mass.values()) or 1.0
    for klass, m in mass.most_common(8):
        lines.append(f"| {klass} | {m:.3f} | {m / total_mass:.1%} |")
    lines.append("")
    lines.append("## Signal classes by frequency")
    lines.append("")
    lines.append("| class | signals |")
    lines.append("|---|---|")
    for klass, n in freq.most_common(8):
        lines.append(f"| {klass} | {n} |")
    lines.append("")
    dists = [r["js_distance"] for r in prior_rows if str(r["question_id"]) in qids and r.get("js_distance") is not None]
    mean_dist = f"{statistics.mean(dists):.4f} over {len(dists)} model priors" if dists else "no comparable priors"
    lines.append(f"Mean prior-anchoring JS distance (stated prior vs base-rate anchor): {mean_dist}.")
    rc = [r for r in rc_rows if str(r["question_id"]) in qids and r.get("has_trace")]
    verdicts = Counter(r.get("rc_assessment") for r in rc)
    judged = sum(v for k, v in verdicts.items() if k in ("accepted", "partial", "rebutted"))
    if judged:
        lines.append(
            f"RC assessment across {judged} traced model calls: accepted {verdicts.get('accepted', 0)}, "
            f"partial {verdicts.get('partial', 0)}, rebutted {verdicts.get('rebutted', 0)} "
            f"(acceptance rate {verdicts.get('accepted', 0) / judged:.0%})."
        )
    else:
        lines.append("RC assessment: no traced model call stated one.")
    n_ev = sum(len(evidence_by_q.get(qid) or []) for qid in qids)
    lines.append(f"Evidence items across these questions: {n_ev}.")
    lines.append("")
    lines.append("## Largest movement (mean mass moved per model, summed over signals)")
    lines.append("")
    lines.append("| question_id | mass moved per model | models |")
    lines.append("|---|---|---|")
    ranked = sorted(per_q_mass.items(), key=lambda kv: -(kv[1] / max(1, len(per_q_models[kv[0]]))))
    for qid, m in ranked[:5]:
        n_models = max(1, len(per_q_models[qid]))
        lines.append(f"| {qid} | {m / n_models:.3f} | {n_models} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------


def _zip_subset(staging: Path, zip_path: Path, *, include: Callable[[str], bool]) -> int:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(staging.rglob("*")):
            if not file.is_file() or file.suffix.lower() in {".duckdb", ".wal", ".pyc"}:
                continue
            rel = file.relative_to(staging).as_posix()
            if include(rel):
                zf.write(file, rel)
                n += 1
    return n


def _window(questions: list[dict[str, Any]]) -> dict[str, Any]:
    starts = sorted(_ym(q.get("window_start_date")) for q in questions if q.get("window_start_date"))
    return {"window_start_min": starts[0] if starts else None, "window_start_max": starts[-1] if starts else None}


class _Sections:
    """Runs each collector, records what failed, never raises."""

    def __init__(self) -> None:
        self.failures: list[dict[str, str]] = []

    def run(self, name: str, fn: Callable[[], Any], default: Any = None) -> Any:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("attribution bundle: %s failed: %s", name, exc)
            self.failures.append({"collector": name, "error": f"{type(exc).__name__}: {exc}"})
            return default


def build_bundle(
    db: str,
    out_dir: Path,
    *,
    run_id: str | None = None,
    hs_run_id: str | None = None,
    previous_run_id: str | None = None,
    include_test: bool = False,
    evidence_match_threshold: float = DEFAULT_EVIDENCE_MATCH_THRESHOLD,
    split_ceiling_mb: float = DEFAULT_SPLIT_CEILING_MB,
    keep_staging: bool = False,
) -> Path | None:
    con = open_db(db)
    db_path = resolve_db_path(db)
    sections = _Sections()
    try:
        if not table_exists(con, "forecasts_raw") or not table_exists(con, "questions"):
            LOGGER.warning("forecasts_raw/questions missing — nothing to bundle")
            return None
        run_id = run_id or _resolve_run_id(con, include_test)
        if not run_id:
            LOGGER.warning("No forecaster run found — nothing to bundle")
            return None
        questions = _questions_for_run(con, run_id, include_test)
        if not questions:
            LOGGER.warning("Run %s has no questions — nothing to bundle", run_id)
            return None
        hs_run_id = hs_run_id or _hs_run_for(questions)
        previous_run_id = previous_run_id or sections.run(
            "previous_run", lambda: _previous_run_id(con, run_id, include_test)
        )
        LOGGER.info("Attribution bundle for run %s (%d questions, hs %s, previous %s)",
                    run_id, len(questions), hs_run_id, previous_run_id)

        staging = out_dir / f"{BUNDLE_KIND}__{run_id}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        counts: dict[str, int] = {}

        taxonomy = load_taxonomy()
        triage = sections.run("triage", lambda: _load_triage(con, hs_run_id), {})
        deviation = sections.run("deviation", lambda: _load_deviation(con, run_id), {})
        created_by_q = sections.run("created_at", lambda: _created_at_by_question(con, run_id), {})
        run_ts = _run_timestamp(created_by_q)

        # --- attribution -------------------------------------------------
        built = sections.run(
            "ledger",
            lambda: build_ledger(con, run_id=run_id, hs_run_id=hs_run_id, questions=questions, taxonomy=taxonomy),
            {"ledger": [], "trace_quality": [], "members_by_q": {}},
        )
        ledger: list[dict[str, Any]] = built["ledger"]
        members_by_q = built["members_by_q"]
        att_dir = staging / "attribution"
        parquet_path = att_dir / "signal_ledger.parquet"
        parquet_ok = sections.run("ledger_parquet", lambda: (write_ledger_parquet(ledger, parquet_path), True)[1], False)
        if not parquet_ok:
            write_csv(att_dir / "signal_ledger.csv", LEDGER_COLUMNS, ledger)
        counts["signal_ledger"] = len(ledger)
        counts["signal_ledger_sample"] = write_csv(
            att_dir / "signal_ledger_sample.csv", LEDGER_COLUMNS, ledger[:LEDGER_SAMPLE_ROWS]
        )
        prior_rows = sections.run(
            "prior_anchoring", lambda: build_prior_anchoring(questions, members_by_q, deviation), []
        )
        counts["prior_anchoring"] = write_csv(
            att_dir / "prior_anchoring.csv",
            ["question_id", "iso3", "hazard_code", "metric", "model_name", "model_family",
             "has_trace", "prior_spd_json", "anchor_present", "anchor_source", "anchor_spd_json",
             "js_divergence", "js_distance", "expected_index_prior", "expected_index_anchor",
             "delta_expected_index", "length_mismatch"],
            prior_rows,
        )
        rc_rows = sections.run(
            "rc_assessment", lambda: build_rc_assessment(questions, members_by_q, ledger, triage), []
        )
        counts["rc_assessment"] = write_csv(
            att_dir / "rc_assessment.csv",
            ["question_id", "iso3", "hazard_code", "metric", "model_name", "model_family",
             "has_trace", "hs_rc_level", "hs_rc_score", "hs_rc_direction", "hs_rc_likelihood",
             "hs_rc_magnitude", "rc_assessment", "rc_assessment_raw", "n_rc_flag_signals",
             "rc_flag_mass_moved_l1", "total_mass_moved_l1"],
            rc_rows,
        )
        quality_rows = built["trace_quality"]
        counts["trace_quality"] = write_csv(
            att_dir / "trace_quality.csv",
            ["question_id", "iso3", "hazard_code", "metric", "model_name", "model_family",
             "has_trace", "n_updates", "trace_quality_score", "prior_quality_score",
             "delta_arithmetic_score", "magnitude_consistency_score"],
            quality_rows,
        )
        trace_rollup = _trace_quality_rollup(quality_rows)

        # --- inputs ------------------------------------------------------
        evidence_by_q = sections.run(
            "evidence", lambda: build_evidence(con, run_id=run_id, hs_run_id=hs_run_id, questions=questions), {}
        )
        in_dir = staging / "inputs"
        inventory = sections.run(
            "input_inventory",
            lambda: build_input_inventory(
                con, run_id=run_id, questions=questions, deviation=deviation, triage=triage,
                evidence_by_q=evidence_by_q, run_ts=run_ts, include_test=include_test,
            ),
            [],
        )
        counts["input_inventory"] = write_csv(in_dir / "input_inventory.csv", INVENTORY_COLUMNS, inventory)
        all_items = [it for qid in (str(q["question_id"]) for q in questions) for it in evidence_by_q.get(qid, [])]
        counts["evidence_items"] = gz_write_jsonl(in_dir / "evidence_items.jsonl.gz", all_items)
        links = sections.run(
            "evidence_to_signal",
            lambda: link_evidence_to_signals(ledger, evidence_by_q, threshold=evidence_match_threshold),
            [],
        )
        counts["evidence_to_signal"] = write_csv(
            in_dir / "evidence_to_signal.csv",
            ["attribution_id", "evidence_id", "question_id", "model_name", "match_score", "match_method"],
            links,
        )
        base_rows = []
        for q in questions:
            qid = str(q["question_id"])
            anchor, source, detail = _anchor_for(deviation.get(qid))
            dev = _preferred_deviation(deviation.get(qid))
            base_rows.append(
                {
                    "question_id": qid, "iso3": q.get("iso3"), "hazard_code": q.get("hazard_code"),
                    "metric": q.get("metric"), "anchor_present": anchor is not None,
                    "baserate_source": source, "baserate_spd_json": _json(anchor),
                    "baserate_n_obs": _baserate_n_obs(dev, detail),
                    "score_family": _score_family(str(q.get("metric") or "")),
                    "detail_json": _json(detail) if detail else None,
                }
            )
        counts["base_rates"] = write_csv(
            in_dir / "base_rates.csv",
            ["question_id", "iso3", "hazard_code", "metric", "anchor_present", "baserate_source",
             "baserate_spd_json", "baserate_n_obs", "score_family", "detail_json"],
            base_rows,
        )

        # --- prompts -----------------------------------------------------
        prompts = sections.run("prompts", lambda: _prompts_for_run(con, run_id, include_test), {})
        prompt_tables = sections.run(
            "prompt_tables", lambda: build_prompt_tables(questions, prompts),
            {"prompt_sections": [], "section_hashes": {}, "token_share": [], "fingerprints": {}},
        )
        pr_dir = staging / "prompts"
        counts["prompt_sections"] = write_csv(
            pr_dir / "prompt_sections.csv",
            ["template_id", "section_index", "section_name", "char_count", "est_tokens", "sha256",
             "example_question_id"],
            prompt_tables["prompt_sections"],
        )
        write_json(pr_dir / "section_hashes.json", prompt_tables["section_hashes"])
        counts["token_share"] = write_csv(
            pr_dir / "token_share.csv",
            ["question_id", "template_id", "section_name", "char_count", "est_tokens", "share"],
            prompt_tables["token_share"],
        )

        # --- contrasts ---------------------------------------------------
        ct_dir = staging / "contrasts"
        disagreement = sections.run(
            "model_disagreement", lambda: build_model_disagreement(con, run_id, questions, members_by_q), []
        )
        counts["model_disagreement"] = write_csv(
            ct_dir / "model_disagreement.csv",
            ["question_id", "iso3", "hazard_code", "metric", "model_a", "model_b", "jsd_final",
             "jsd_prior", "jsd_from_updates"],
            disagreement,
        )
        fvs = sections.run(
            "fred_vs_sibyl", lambda: build_fred_vs_sibyl(con, run_id, questions, ledger, taxonomy), []
        )
        counts["fred_vs_sibyl"] = write_csv(
            ct_dir / "fred_vs_sibyl.csv",
            ["question_id", "iso3", "hazard_code", "metric", "fred_model", "sibyl_status",
             "jsd_fred_vs_sibyl", "ev_fred", "ev_sibyl", "fred_signal_classes", "sibyl_signal_classes"],
            fvs,
        )
        previous = None
        if previous_run_id:
            def _prev() -> dict[str, Any]:
                pq = _questions_for_run(con, previous_run_id, include_test)
                built_prev = build_ledger(
                    con, run_id=previous_run_id, hs_run_id=_hs_run_for(pq), questions=pq, taxonomy=taxonomy
                )
                built_prev["questions"] = pq
                return built_prev

            previous = sections.run("previous_run_ledger", _prev)
        rr = sections.run(
            "run_over_run",
            lambda: build_run_over_run(
                con, run_id=run_id, previous_run_id=previous_run_id, questions=questions,
                ledger=ledger, members_by_q=members_by_q, previous=previous,
            ),
            [],
        )
        counts["run_over_run"] = write_csv(
            ct_dir / "run_over_run.csv",
            ["iso3", "hazard_code", "metric", "question_id", "previous_question_id", "previous_run_id",
             "ev_current", "ev_previous", "ev_change", "prior_ev_current", "prior_ev_previous",
             "prior_ev_change", "top_class_current", "top_class_previous", "class_mix_current_json",
             "class_mix_previous_json", "class_mix_l1_change"],
            rr,
        )

        # --- hazard briefs -----------------------------------------------
        hz_dir = staging / "hazard"
        hz_dir.mkdir(parents=True, exist_ok=True)
        for hz in HAZARD_BRIEFS:
            text = sections.run(
                f"hazard_brief_{hz}",
                lambda hz=hz: build_hazard_brief(
                    hz, questions=questions, ledger=ledger, prior_rows=prior_rows, rc_rows=rc_rows,
                    evidence_by_q=evidence_by_q,
                ),
                f"# {hz}\n\nBrief unavailable — see MANIFEST.json collector_failures.\n",
            )
            (hz_dir / f"{hz}.md").write_text(text, encoding="utf-8")

        # --- per-question records ----------------------------------------
        ledger_by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in ledger:
            ledger_by_q[str(r["question_id"])].append(r)
        inventory_by_q = {str(r["question_id"]): r for r in inventory}
        n_records = 0
        for q in questions:
            qid = str(q["question_id"])
            extras = {
                "attribution": {
                    "ledger": ledger_by_q.get(qid, []),
                    "input_inventory": inventory_by_q.get(qid),
                    "prompt_section_fingerprint": prompt_tables["fingerprints"].get(qid),
                    "evidence_ids": [it["evidence_id"] for it in evidence_by_q.get(qid, [])],
                }
            }
            try:
                record = build_question_record(
                    con, q, include_test=include_test, include_sibyl_trials=False, run_id=run_id,
                    extras=extras,
                )
            except Exception as exc:  # noqa: BLE001
                sections.failures.append({"collector": f"question_record:{qid}", "error": str(exc)})
                record = {"question": q, "forecast_run_id": run_id, **extras, "record_error": str(exc)}
            write_json(staging / "questions" / f"{qid}.json", record)
            n_records += 1
            del record
        counts["question_records"] = n_records

        # --- split decision, manifest, guide, zips -----------------------
        ceiling_bytes = int(split_ceiling_mb * 1e6)
        split_candidates = ["attribution/signal_ledger.parquet", "inputs/evidence_items.jsonl.gz"]
        moved = [
            rel for rel in split_candidates
            if (staging / rel).exists() and (staging / rel).stat().st_size > ceiling_bytes
        ]
        main_zip = out_dir / f"{BUNDLE_KIND}__{run_id}.zip"
        part2_zip = out_dir / f"{BUNDLE_KIND}__{run_id}__part2.zip"
        split_record = {
            "ceiling_mb": split_ceiling_mb,
            "moved_to_part2": moved,
            "part2_zip": part2_zip.name if moved else None,
            "note": (
                "Files over the ceiling were moved into the part2 zip rather than dropped; "
                "MANIFEST.json is present in both."
                if moved else "Nothing exceeded the ceiling; one zip."
            ),
        }
        models_in_run = sorted({m["model_name"] for ms in members_by_q.values() for m in ms})
        manifest = {
            "bundle_kind": BUNDLE_KIND,
            "builder_version": ATTRIBUTION_BUILDER_VERSION,
            "common_builder_version": BUILDER_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_sha": os.getenv("GITHUB_SHA") or None,
            "workflow": os.getenv("GITHUB_WORKFLOW") or None,
            "workflow_run_id": os.getenv("GITHUB_RUN_ID") or None,
            "db_path": str(db_path),
            "run_id": run_id,
            "hs_run_id": hs_run_id,
            "previous_run_id": previous_run_id,
            "run_timestamp": run_ts.isoformat(),
            "include_test": include_test,
            "window": _window(questions),
            "counts": {"questions": len(questions), "models_in_run": len(models_in_run), **counts},
            "file_row_counts": counts,
            "model_lineup": _model_lineup(),
            "models_in_run": models_in_run,
            "taxonomy_version": TAXONOMY_VERSION,
            "taxonomy_sha256": _sha256_text(TAXONOMY_PATH.read_text(encoding="utf-8")),
            "taxonomy_classes": [e["signal_class"] for e in taxonomy],
            "thresholds": {
                "evidence_match_threshold": evidence_match_threshold,
                "evidence_links_per_signal": EVIDENCE_LINKS_PER_SIGNAL,
                "split_ceiling_mb": split_ceiling_mb,
                "zip_warn_mb": ZIP_WARN_MB,
                "delta_sum_tolerance": _DELTA_SUM_TOLERANCE,
                "reconcile_l1_tolerance": _RECONCILE_L1_TOLERANCE,
                "chars_per_token": CHARS_PER_TOKEN,
            },
            "attribution_id_recipe": 'sha256("{run_id}|{question_id}|{model_name}|{update_index}")[:16]',
            "evidence_id_recipe": 'sha256("{url}|{title}")[:16]',
            "trace_quality_rollup": trace_rollup,
            "ledger_parquet_written": bool(parquet_ok),
            "collector_failures": sections.failures,
            "cost": sections.run("cost", lambda: _run_cost(con, run_id, hs_run_id), {}),
            "linked_bundles": {
                "operational_debug_bundle": {
                    "artifact": "pythia-debug-bundle",
                    "produced_under_workflow_run_id": os.getenv("GITHUB_RUN_ID") or None,
                    "join_keys": ["run_id", "hs_run_id", "question_id", "model_name"],
                },
                "current_run_bundle": {"artifact": "pythia-current-run-bundle", "join_keys": ["run_id", "question_id"]},
                "scored_forecast_bundle": {"artifact": "pythia-ai-analysis-bundle", "join_keys": ["question_id", "run_id"]},
                "resolutions_bundle": {"status": "not yet built", "will_attach_outcomes_to": "attribution_id"},
            },
            "split": split_record,
        }
        write_json(staging / "MANIFEST.json", manifest)
        guide_ctx = {
            "run_id": run_id, "hs_run_id": hs_run_id, "n_questions": len(questions),
            "n_ledger_rows": len(ledger), "taxonomy_version": TAXONOMY_VERSION,
            "evidence_match_threshold": evidence_match_threshold,
            "trace_quality_rollup": trace_rollup, "models_in_run": models_in_run,
            "taxonomy": taxonomy,
        }
        (staging / "ANALYST_GUIDE.md").write_text(build_attribution_guide(guide_ctx), encoding="utf-8")
        (staging / "LINKAGE.md").write_text(build_linkage_md(guide_ctx), encoding="utf-8")

        moved_set = set(moved)
        _zip_subset(staging, main_zip, include=lambda rel: rel not in moved_set)
        if moved:
            _zip_subset(staging, part2_zip, include=lambda rel: rel in moved_set or rel == "MANIFEST.json")
        if not keep_staging:
            shutil.rmtree(staging, ignore_errors=True)
        zip_mb = main_zip.stat().st_size / 1e6
        LOGGER.info("Bundle written: %s (%.1f MB)", main_zip, zip_mb)
        if zip_mb > ZIP_WARN_MB:
            LOGGER.warning("Bundle zip is %.0f MB (warn threshold %.0f MB)", zip_mb, ZIP_WARN_MB)
        return main_zip
    finally:
        try:
            con.close()
        except Exception:  # noqa: BLE001
            pass


INVENTORY_COLUMNS = [
    "question_id", "iso3", "hazard_code", "metric", "track", "window_start_month",
    "resolver_history_months", "resolver_history_empty", "baserate_anchor_present",
    "baserate_source", "baserate_n_obs", "baserate_years", "acled_political_rows",
    "acled_fatality_months", "ipc_phase_rows", "enso_state_rows", "gdelt_rows",
    "seasonal_tc_outlook_present", "crisiswatch_present", "crisiswatch_edition",
    "prediction_market_present", "evidence_items", "evidence_from_hs_pack",
    "evidence_from_question_research", "evidence_newest_age_days",
    "evidence_median_age_days", "hs_tier", "hs_triage_score", "hs_rc_level",
    "js_vs_baserate_mean_ensemble", "log_ev_ratio_mean_ensemble", "deviation_model",
]


def _trace_quality_rollup(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _roll(key: str) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            groups[str(r.get(key))].append(r)
        out: dict[str, Any] = {}
        for g, rs in sorted(groups.items()):
            scores = [float(r["trace_quality_score"]) for r in rs if r.get("trace_quality_score") is not None]
            out[g] = {
                "n": len(rs),
                "share_with_trace": round(sum(1 for r in rs if r.get("has_trace")) / len(rs), 4) if rs else None,
                "mean_trace_quality": round(sum(scores) / len(scores), 4) if scores else None,
                "mean_updates": round(sum(int(r.get("n_updates") or 0) for r in rs) / len(rs), 2) if rs else None,
            }
        return out

    return {"by_model": _roll("model_name"), "by_hazard": _roll("hazard_code")}


def _write_failure_stub(out_dir: Path, run_id: str | None, error: str) -> Path:
    """The bundle that says why there is no bundle."""
    out_dir.mkdir(parents=True, exist_ok=True)
    label = run_id or "unknown_run"
    zip_path = out_dir / f"{BUNDLE_KIND}__{label}.zip"
    manifest = {
        "bundle_kind": BUNDLE_KIND,
        "builder_version": ATTRIBUTION_BUILDER_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "status": "failed",
        "collector_failures": [{"collector": "build_bundle", "error": error}],
    }
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("MANIFEST.json", json.dumps(manifest, indent=1, default=str))
    return zip_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="DuckDB URL or path")
    parser.add_argument("--out-dir", default="ai_bundle", help="Output directory")
    parser.add_argument("--run-id", default=None, help="Forecaster run (default: latest non-test)")
    parser.add_argument("--hs-run-id", default=None, help="HS run (default: the run's questions' hs_run_id)")
    parser.add_argument("--previous-run-id", default=None, help="Previous run for run_over_run (default: discovered)")
    parser.add_argument("--include-test", action="store_true", help="Include is_test rows (default: excluded)")
    parser.add_argument("--evidence-match-threshold", type=float, default=DEFAULT_EVIDENCE_MATCH_THRESHOLD,
                        help="Minimum token-containment score for an evidence_to_signal link")
    parser.add_argument("--split-ceiling-mb", type=float, default=DEFAULT_SPLIT_CEILING_MB,
                        help="Move the ledger/evidence file into a second zip above this size")
    parser.add_argument("--keep-staging", action="store_true", help="Keep the unzipped staging directory")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[ai_bundle] %(message)s")
    try:
        zip_path = build_bundle(
            args.db,
            Path(args.out_dir),
            run_id=args.run_id,
            hs_run_id=args.hs_run_id,
            previous_run_id=args.previous_run_id,
            include_test=args.include_test,
            evidence_match_threshold=args.evidence_match_threshold,
            split_ceiling_mb=args.split_ceiling_mb,
            keep_staging=args.keep_staging,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("attribution bundle failed: %s", exc)
        zip_path = _write_failure_stub(Path(args.out_dir), args.run_id, f"{type(exc).__name__}: {exc}")
        print(f"[ai_bundle] BUNDLE_PATH={zip_path} (failure stub)")
        return 0
    if zip_path is None:
        print("[ai_bundle] no bundle produced (no forecaster run)")
        return 0
    print(f"[ai_bundle] BUNDLE_PATH={zip_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
