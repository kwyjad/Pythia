# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Per-family recalibration of ensemble members: fit, store, apply.

Advice asks a model to correct itself and nothing checks that it did. A
recalibration factor corrects the forecast arithmetically, after the model
has answered, so it is measurable and reversible. Factors are fitted per
(model FAMILY, hazard, metric), so a version bump inherits its predecessor's
bias rather than starting from none, and per PROMPT VERSION
(``forecasts_raw.base_rate_block_version`` and ``rc_guidance``), because a
correction learned on forecasts made under one prompt says nothing about
forecasts made under another. The one exception is a pair of prompt
versions that hand the member the SAME distribution in different words:
``BLOCK_VERSION_EQUIVALENCE`` groups them for fitting and for applying, while
``forecasts_raw`` keeps the version each member actually saw and the factor
row names every version that contributed (``contributing_versions_json``).

Fitting (``fit_family_recalibration``) reads scored MEMBER forecasts only:
no aggregate, no Sibyl, no ``__ext_`` reference, no ``__raw``/``__recal``
row; non-test questions, the latest run per question, resolved horizons.
Where a question carries a ``<model>__raw`` row (the forecast before an
applied correction), that row is fitted rather than the corrected one, so a
correction is never fitted on its own output.

* SPD: per bucket, the observed share over the mean assigned probability,
  shrunk toward 1 with a prior worth ``PRIOR_PSEUDO_QUESTIONS`` questions and
  clipped to ``[SPD_FACTOR_MIN, SPD_FACTOR_MAX]``.
* Binary: a logit intercept shift with a Gaussian prior of sd
  ``BINARY_PRIOR_SD``, clipped to ``[-BINARY_SHIFT_MAX, BINARY_SHIFT_MAX]``,
  per (family, hazard).
* No group with fewer than ``MIN_FIT_QUESTIONS`` distinct questions is
  fitted; the log says which.

Applying (``recalibrate_spd``/``recalibrate_binary``) is governed by
``PYTHIA_FAMILY_RECALIBRATION_MODE`` (``off`` | ``shadow`` | ``apply``,
default ``off``). A group is applied only with factors fitted under the SAME
prompt versions as the forecast; factors for other versions put the group in
``auto_shadow``. A failure falls back to the raw forecast and never costs a
forecast.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    LOGGER.addHandler(logging.NullHandler())

TABLE = "family_recalibration"

#: Suffix of the uncorrected member row stored when corrections are APPLIED.
RAW_SUFFIX = "__raw"
#: Suffix of the corrected member row stored in SHADOW mode.
RECAL_SUFFIX = "__recal"
DERIVED_SUFFIXES = (RAW_SUFFIX, RECAL_SUFFIX)

PRIOR_PSEUDO_QUESTIONS = 20.0
SPD_FACTOR_MIN = 0.5
SPD_FACTOR_MAX = 2.0
SPD_FLOOR = 0.001
BINARY_PRIOR_SD = 0.5
BINARY_SHIFT_MAX = 1.0
BINARY_CLAMP = (0.001, 0.999)
MIN_FIT_QUESTIONS = 10

MODES = ("off", "shadow", "apply")

#: Prompt versions that are one group for fitting and applying. Both
#: prior-anchor wordings print the SAME level-and-volatility vector and tell
#: the member to copy it; v2 only replaces v1's pooled move shares (which
#: disagree with the printed vector at the edge buckets) with stay / up /
#: down read off that vector (owner decision 2026-10-06). No other pair is
#: equivalent, and a NULL version (no block shown) is always its own group.
BLOCK_VERSION_EQUIVALENCE: Dict[str, str] = {
    "prior_anchor_v1": "prior_anchor_v1|prior_anchor_v2",
    "prior_anchor_v2": "prior_anchor_v1|prior_anchor_v2",
    # A stored group label maps to itself, so rows written by this module
    # and rows written before the equivalence both land in the same group.
    "prior_anchor_v1|prior_anchor_v2": "prior_anchor_v1|prior_anchor_v2",
}


def block_version_group(version: Optional[str]) -> Optional[str]:
    """The fitting/applying group of a ``base_rate_block_version``.

    None stays None; a version with no equivalence is its own group.
    """
    if version is None:
        return None
    v = str(version)
    return BLOCK_VERSION_EQUIVALENCE.get(v, v)

#: Aggregates and tracks that are never fitted or corrected. ``sibyl`` is a
#: separate research track whose output is its own; ``track2_flash`` is a
#: stored aggregate name.
_NEVER_MEMBERS = {
    "ensemble", "ensemble_mean_v2", "ensemble_bayesmc_v2", "track2_flash", "sibyl",
}


def recalibration_mode() -> str:
    """``PYTHIA_FAMILY_RECALIBRATION_MODE``; anything unknown reads ``off``."""
    mode = (os.getenv("PYTHIA_FAMILY_RECALIBRATION_MODE", "off") or "off").strip().lower()
    return mode if mode in MODES else "off"


def is_derived_name(model_name: Optional[str]) -> bool:
    """True for ``<model>__raw`` / ``<model>__recal``: scored, never voting,
    never weighted, never advised, never fitted as themselves."""
    name = str(model_name or "")
    return any(name.endswith(s) for s in DERIVED_SUFFIXES)


def base_model_name(model_name: Optional[str]) -> str:
    name = str(model_name or "")
    for s in DERIVED_SUFFIXES:
        if name.endswith(s):
            return name[: -len(s)]
    return name


def _family(model_name: str) -> Optional[str]:
    try:
        from pythia.llm_profiles import model_family

        return model_family(model_name)
    except Exception:  # noqa: BLE001
        return None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def ensure_table(con) -> None:
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            family TEXT,
            hazard_code TEXT,
            metric TEXT,
            score_family TEXT,
            bucket_index INTEGER,
            factor DOUBLE,
            n_questions INTEGER,
            base_rate_block_version TEXT,
            rc_guidance TEXT,
            as_of_month TEXT,
            fitted_at TIMESTAMP,
            is_test BOOLEAN DEFAULT FALSE,
            contributing_versions_json TEXT
        )
        """
    )
    # Added after the table shipped: which forecasts_raw versions fed the
    # group, with the distinct questions each gave.
    con.execute(
        f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS contributing_versions_json TEXT"
    )


_INSERT_COLS = (
    "family, hazard_code, metric, score_family, bucket_index, factor, n_questions, "
    "base_rate_block_version, rc_guidance, as_of_month, fitted_at, is_test, "
    "contributing_versions_json"
)


# ---------------------------------------------------------------------------
# Pure arithmetic
# ---------------------------------------------------------------------------

def shrunk_spd_factors(
    mean_assigned: Sequence[float],
    observed_share: Sequence[float],
    n_questions: int,
    prior: float = PRIOR_PSEUDO_QUESTIONS,
) -> List[float]:
    """Per-bucket factor: observed / assigned, shrunk toward 1, clipped."""
    out: List[float] = []
    for a, o in zip(mean_assigned, observed_share):
        a = float(a)
        raw = (float(o) / a) if a > 1e-9 else 1.0
        f = (n_questions * raw + prior * 1.0) / (n_questions + prior)
        out.append(min(max(f, SPD_FACTOR_MIN), SPD_FACTOR_MAX))
    return out


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def binary_logit_shift(
    probs: Sequence[float], outcomes: Sequence[int], prior_sd: float = BINARY_PRIOR_SD
) -> float:
    """MAP intercept shift: maximise the Bernoulli log-likelihood of the
    shifted forecasts plus a Gaussian log-prior on the shift; clipped."""
    logits = [_logit(p) for p in probs]
    ys = [1 if int(y) else 0 for y in outcomes]
    inv_var = 1.0 / (prior_sd * prior_sd)
    delta = 0.0
    # Damped Newton: the objective is concave, but a full step from zero on
    # saturated forecasts overshoots and can oscillate.
    for _ in range(200):
        grad = -delta * inv_var
        hess = -inv_var
        for l, y in zip(logits, ys):
            s = _sigmoid(l + delta)
            grad += y - s
            hess -= s * (1 - s)
        step = max(min(grad / hess, 0.25), -0.25)
        delta -= step
        if abs(step) < 1e-9:
            break
    return min(max(delta, -BINARY_SHIFT_MAX), BINARY_SHIFT_MAX)


def apply_spd_factors(probs: Sequence[float], factors: Sequence[float]) -> List[float]:
    """Multiply, floor at ``SPD_FLOOR``, renormalise."""
    if len(probs) != len(factors):
        raise ValueError(f"{len(probs)} probabilities against {len(factors)} factors")
    vec = [max(float(p) * float(f), SPD_FLOOR) for p, f in zip(probs, factors)]
    total = sum(vec)
    return [v / total for v in vec]


def apply_binary_shift(p: float, delta: float) -> float:
    lo, hi = BINARY_CLAMP
    return min(max(_sigmoid(_logit(p) + float(delta)), lo), hi)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _has_col(con, table: str, col: str) -> bool:
    try:
        rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
        return col.lower() in {str(r[1]).lower() for r in rows}
    except Exception:  # noqa: BLE001
        return False


def _member_rows(con) -> List[Tuple]:
    """(question_id, model_name, month_index, bucket_index, probability,
    hazard, metric, brbv, rcg, resolved_value) for fittable member rows."""
    brbv = "fr.base_rate_block_version" if _has_col(con, "forecasts_raw", "base_rate_block_version") else "NULL"
    rcg = "fr.rc_guidance" if _has_col(con, "forecasts_raw", "rc_guidance") else "NULL"
    latest = (
        " AND fr.run_id = (SELECT MAX(_lr.run_id) FROM forecasts_ensemble _lr "
        "WHERE _lr.question_id = fr.question_id)"
        if _has_col(con, "forecasts_ensemble", "run_id") else ""
    )
    sql = f"""
        SELECT fr.question_id, fr.model_name, fr.month_index, fr.bucket_index,
               fr.probability, upper(q.hazard_code), upper(q.metric),
               {brbv}, {rcg}, r.value, fr.run_id
        FROM forecasts_raw fr
        JOIN questions q ON q.question_id = fr.question_id
        JOIN resolutions r ON r.question_id = fr.question_id AND r.horizon_m = fr.month_index
        WHERE COALESCE(q.is_test, FALSE) = FALSE
          AND fr.probability IS NOT NULL
          AND fr.bucket_index IS NOT NULL
          AND fr.model_name NOT LIKE '\\_\\_ext\\_%' ESCAPE '\\'
          AND fr.model_name NOT LIKE '%\\_\\_recal' ESCAPE '\\'{latest}
    """
    return con.execute(sql).fetchall()


def fit_family_recalibration(con, as_of_month: Optional[str] = None) -> Dict[str, Any]:
    """Fit and store factors; returns a summary of what was and was not fitted."""
    from pythia.buckets import n_buckets_for
    from pythia.tools.base_rate_spd import _bucket_index_for_value

    ensure_table(con)
    as_of = as_of_month or _utcnow().strftime("%Y-%m")
    summary: Dict[str, Any] = {"as_of_month": as_of, "fitted": [], "skipped": []}
    for table in ("forecasts_raw", "questions", "resolutions"):
        try:
            con.execute(f"SELECT 1 FROM {table} LIMIT 0")
        except Exception:  # noqa: BLE001
            summary["skipped"].append({"reason": f"{table} missing"})
            return summary
    rows = _member_rows(con)

    # Which (question, run, model) carry a raw row: fit the raw one instead.
    has_raw = set()
    for r in rows:
        if str(r[1]).endswith(RAW_SUFFIX):
            has_raw.add((r[0], r[10], base_model_name(r[1])))

    # samples[(fam, hz, metric, family_kind, brbv, rcg)][(qid, model, month)] = {bucket: p}
    spd: Dict[tuple, Dict[tuple, Dict[int, float]]] = {}
    # contributors[key][version_seen] = {question_id, ...}
    contributors: Dict[tuple, Dict[str, set]] = {}
    resolved: Dict[tuple, float] = {}
    for qid, model, month, bucket, p, hz, metric, brbv, rcg, value, run in rows:
        name = str(model)
        if name.endswith(RAW_SUFFIX):
            base = base_model_name(name)
        else:
            base = name
            if (qid, run, base) in has_raw:
                continue
        if base in _NEVER_MEMBERS:
            continue
        fam = _family(base)
        if not fam:
            continue
        kind = "binary" if metric == "EVENT_OCCURRENCE" else "spd"
        met = metric
        key = (fam, hz, met, kind, block_version_group(brbv), rcg)
        contributors.setdefault(key, {}).setdefault(
            "null" if brbv is None else str(brbv), set()
        ).add(qid)
        spd.setdefault(key, {}).setdefault((qid, base, int(month)), {})[int(bucket)] = float(p)
        resolved[(qid, int(month))] = float(value) if value is not None else float("nan")

    now = _utcnow()
    con.execute(f"DELETE FROM {TABLE} WHERE as_of_month = ?", [as_of])
    for key, samples in sorted(spd.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        fam, hz, metric, kind, brbv, rcg = key
        n_q = len({s[0] for s in samples})
        contrib = json.dumps(
            {v: len(q) for v, q in sorted(contributors.get(key, {}).items())},
            sort_keys=True,
        )
        label = f"{fam} {hz}/{metric} (block={brbv}, rc={rcg})"
        if n_q < MIN_FIT_QUESTIONS:
            LOGGER.info(
                "family_recalibration: no fit for %s: %d distinct question(s), need %d",
                label, n_q, MIN_FIT_QUESTIONS,
            )
            summary["skipped"].append({"group": label, "n_questions": n_q,
                                       "reason": f"fewer than {MIN_FIT_QUESTIONS} questions"})
            continue
        if kind == "spd":
            k = n_buckets_for(metric)
            if not k:
                continue
            sum_p = [0.0] * k
            hits = [0.0] * k
            n = 0
            for (qid, _m, month), probs in samples.items():
                v = resolved.get((qid, month))
                if v is None or v != v:
                    continue
                j = _bucket_index_for_value(v, metric)
                if j is None or set(probs) != set(range(1, k + 1)):
                    continue
                n += 1
                for b in range(1, k + 1):
                    sum_p[b - 1] += probs[b]
                hits[j] += 1.0
            if not n:
                continue
            factors = shrunk_spd_factors(
                [s / n for s in sum_p], [h / n for h in hits], n_q
            )
            for b, f in enumerate(factors, start=1):
                con.execute(
                    f"INSERT INTO {TABLE} ({_INSERT_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,FALSE,?)",
                    [fam, hz, metric, kind, b, f, n_q, brbv, rcg, as_of, now, contrib],
                )
            summary["fitted"].append({"group": label, "n_questions": n_q,
                                      "contributing_versions": json.loads(contrib),
                                      "factors": [round(f, 4) for f in factors]})
        else:
            ps: List[float] = []
            ys: List[int] = []
            for (qid, _m, month), probs in samples.items():
                v = resolved.get((qid, month))
                if v is None or v != v or 1 not in probs:
                    continue
                ps.append(probs[1])
                ys.append(1 if v >= 1.0 else 0)
            if not ps:
                continue
            delta = binary_logit_shift(ps, ys)
            con.execute(
                f"INSERT INTO {TABLE} ({_INSERT_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,FALSE,?)",
                [fam, hz, metric, kind, 0, delta, n_q, brbv, rcg, as_of, now, contrib],
            )
            summary["fitted"].append({"group": label, "n_questions": n_q,
                                      "contributing_versions": json.loads(contrib),
                                      "shift": round(delta, 4)})
    LOGGER.info(
        "family_recalibration: %d group(s) fitted, %d skipped for %s",
        len(summary["fitted"]), len(summary["skipped"]), as_of,
    )
    return summary


# ---------------------------------------------------------------------------
# Lookup and application
# ---------------------------------------------------------------------------

_FACTOR_CACHE: Dict[str, Any] = {}
_FACTOR_LOCK = threading.Lock()


def reset_factor_cache() -> None:
    with _FACTOR_LOCK:
        _FACTOR_CACHE.clear()


def _load_factors(db_url: Optional[str] = None) -> Tuple[Optional[str], Dict[tuple, Dict[int, float]]]:
    """(as_of_month, {(fam, hz, metric, kind, brbv, rcg): {bucket: factor}}) for
    the newest fitted month; empty when nothing was fitted."""
    with _FACTOR_LOCK:
        if "factors" in _FACTOR_CACHE:
            return _FACTOR_CACHE["as_of"], _FACTOR_CACHE["factors"]
    as_of: Optional[str] = None
    factors: Dict[tuple, Dict[int, float]] = {}
    try:
        from resolver.db import duckdb_io

        url = db_url or os.getenv("PYTHIA_DB_URL", "").strip() or os.getenv("RESOLVER_DB_URL", "").strip()
        if not url:
            try:
                from forecaster.prompts import _pythia_db_url_from_config

                url = _pythia_db_url_from_config() or ""
            except Exception:  # noqa: BLE001
                url = ""
        con = duckdb_io.get_db(url or duckdb_io.DEFAULT_DB_URL)
        try:
            row = con.execute(
                f"SELECT MAX(as_of_month) FROM {TABLE} WHERE NOT COALESCE(is_test, FALSE)"
            ).fetchone()
            as_of = row[0] if row else None
            if as_of:
                for fam, hz, metric, kind, b, f, brbv, rcg in con.execute(
                    f"SELECT family, hazard_code, metric, score_family, bucket_index, factor, "
                    f"base_rate_block_version, rc_guidance FROM {TABLE} "
                    f"WHERE as_of_month = ? AND NOT COALESCE(is_test, FALSE)",
                    [as_of],
                ).fetchall():
                    key = (fam, hz, metric, kind, block_version_group(brbv), rcg)
                    factors.setdefault(key, {})[int(b)] = float(f)
        finally:
            duckdb_io.close_db(con)
    except Exception as exc:  # noqa: BLE001
        LOGGER.info("family_recalibration: no factors loaded (%s)", exc)
    with _FACTOR_LOCK:
        _FACTOR_CACHE["as_of"] = as_of
        _FACTOR_CACHE["factors"] = factors
    return as_of, factors


def lookup(
    model_name: str,
    hazard_code: str,
    metric: str,
    *,
    base_rate_block_version: Optional[str],
    rc_guidance: Optional[str],
    db_url: Optional[str] = None,
) -> Dict[str, Any]:
    """What the configured mode does for this member and group.

    ``{"mode": "apply"|"shadow"|"auto_shadow"|"none"|"off", "family", "factors",
    "as_of_month", "reason"}`` — ``factors`` present only when there is
    something to compute.
    """
    mode = recalibration_mode()
    out: Dict[str, Any] = {"mode": "off"}
    if mode == "off":
        return out
    name = base_model_name(model_name)
    if name in _NEVER_MEMBERS or is_derived_name(model_name):
        return {"mode": "none", "reason": "not a correctable member"}
    fam = _family(name)
    if not fam:
        return {"mode": "none", "reason": f"{name} belongs to no model family"}
    hz = (hazard_code or "").upper()
    met = (metric or "").upper()
    kind = "binary" if met == "EVENT_OCCURRENCE" else "spd"
    as_of, factors = _load_factors(db_url)
    group = block_version_group(base_rate_block_version)
    exact = factors.get((fam, hz, met, kind, group, rc_guidance))
    out = {"family": fam, "as_of_month": as_of}
    if exact:
        out.update({"mode": mode, "factors": exact})
        if group != base_rate_block_version:
            out["version_group"] = group
        return out
    other = [k for k in factors if k[:4] == (fam, hz, met, kind)]
    if other:
        # Factors exist, but for another prompt version: compute and store,
        # never let them vote.
        k = other[0]
        LOGGER.info(
            "family_recalibration: %s %s/%s has factors only for block=%s rc=%s, "
            "not block=%s rc=%s; shadowing",
            fam, hz, met, k[4], k[5], base_rate_block_version, rc_guidance,
        )
        out.update({
            "mode": "auto_shadow", "factors": factors[k],
            "reason": f"factors fitted under block={k[4]} rc={k[5]}",
            "fitted_versions": {"base_rate_block_version": k[4], "rc_guidance": k[5]},
        })
        return out
    out.update({"mode": "none", "reason": "no factors for this family and group"})
    return out


def recalibrate_spd(
    spd: Dict[str, List[float]], factors: Dict[int, float]
) -> Optional[Dict[str, List[float]]]:
    """Corrected month -> probs, or None when anything does not fit."""
    try:
        k = len(factors)
        vec_f = [factors[b] for b in range(1, k + 1)]
        out: Dict[str, List[float]] = {}
        for month, probs in spd.items():
            if len(probs) != k:
                return None
            out[month] = apply_spd_factors(probs, vec_f)
        return out
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("family_recalibration: SPD correction failed (%s); raw kept", exc)
        return None


def recalibrate_binary(p: float, factors: Dict[int, float]) -> Optional[float]:
    try:
        return apply_binary_shift(p, factors[0])
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("family_recalibration: binary correction failed (%s); raw kept", exc)
        return None


def meta_json(info: Dict[str, Any], applied: bool, role: str) -> str:
    """The ``forecasts_raw.recalibration_json`` payload."""
    payload = {k: v for k, v in info.items() if k != "factors"}
    if "factors" in info:
        payload["factors"] = {str(b): round(f, 6) for b, f in sorted(info["factors"].items())}
    payload["applied"] = applied
    payload["row"] = role
    return json.dumps(payload, sort_keys=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fit per-family recalibration factors.")
    parser.add_argument("--db-url", default=None)
    parser.add_argument("--as-of-month", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    from resolver.db import duckdb_io

    con = duckdb_io.get_db(args.db_url or duckdb_io.DEFAULT_DB_URL)
    try:
        summary = fit_family_recalibration(con, args.as_of_month)
    finally:
        duckdb_io.close_db(con)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
