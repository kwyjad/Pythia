# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl calibration advice: what Sibyl's own scored record says it gets wrong.

``python -m sibyl.advice --db-url ...`` runs monthly in
``compute_calibration_pythia.yml``. It measures, per (hazard, metric) class
and pooled across the four classes, where Sibyl's resolved forecasts missed,
and writes one ``sibyl_calibration_advice`` row per class. Sibyl's prompt
shows the advice for the question's class on the next run
(``sibyl.calibration.load_advice``), to half the questions; the other half
is the comparison arm.

What counts as Sibyl's record
-----------------------------
``sibyl_forecasts`` rows with ``status = 'ok'`` from production runs that
rested on evidence (``evidence_ok`` not FALSE, see ``sibyl/evidence.py``), the
LATEST Sibyl run of each question, joined to ``resolutions``. Never ``scores``
alone: until October 2026 compute_scores stamped a score's ``is_test`` from
the question, so a production question carrying a same-epoch test run's
Sibyl forecast read as a production score.

The unit is the question
------------------------
A question's six resolved months share ONE forecast (Sibyl writes the same
bucket vector for every window month), so they are not six pieces of
evidence. Counts are of distinct questions, every interval comes from a
bootstrap that resamples whole questions (2,000 draws, fixed seed, numpy),
and a share "of resolved months" is a ratio of sums over the questions drawn.

Gating
------
A class gets its own advice at ``SIBYL_ADVICE_MIN_QUESTIONS`` (20) distinct
scored questions; below that the pooled row (all four classes, also at 20)
stands in; below that there is no advice and the prompt section is left out.
Coverage and log bias carry no units, so pooling across metrics is sound. A
finding becomes an instruction only when its 90% interval excludes the
calibrated value; otherwise it is stored in ``findings_json`` and kept out of
the text. Perspective bias and paired skill are findings only.

Independence
------------
The text names no country, no other model, no ensemble figure and no
standard-track advice. Paired skill against ``ensemble_mean_v2`` and
``__ext_climatology`` is computed for the dashboard and never reaches a
prompt (``build_advice_text`` does not read it).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from sibyl.config import (
    ADVICE_MIN_QUESTIONS,
    ELIGIBLE_HAZARD_METRICS,
    SIBYL_MODEL_NAME,
)

logger = logging.getLogger(__name__)

ADVICE_VERSION = "sibyl_advice_v1"
ADVICE_MAX_CHARS = 1200
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261002
INTERVAL = (0.05, 0.95)  # a 90% interval
ARM_MIN_QUESTIONS = 10
#: Hazards whose zero bucket is "zero or no record": no zero-gap instruction.
ZERO_GAP_EXCLUDED = frozenset({"FL", "TC"})
POOLED = "*"

#: Calibrated values each diagnostic is tested against.
CALIBRATED = {
    "coverage_10_90": 0.80,
    "below_q10": 0.10,
    "above_q90": 0.10,
    "above_q99": 0.01,
    "centre_bias_log": 0.0,
    "zero_gap": 0.0,
    "anchor_departure": 0.5,
}


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class SibylRecord:
    """One question's standing Sibyl forecast and its resolved months."""

    question_id: str
    hazard_code: str
    metric: str
    quantiles: Dict[float, float]
    outcomes: List[float]
    zero_mass: Optional[float] = None
    base_median: Optional[float] = None
    trial_medians: Dict[str, float] = field(default_factory=dict)
    advice_arm: Optional[str] = None
    sibyl_run_id: Optional[str] = None
    forecast_run_id: Optional[str] = None
    # Oct 2026: the raw pooled series by window month (BEFORE the reference
    # pool), and the horizon of each outcome. A resolved month is compared
    # with its own month's quantiles where they exist; the advice speaks to
    # the agent about its own distribution, never the published pool.
    month_quantiles: Dict[int, Dict[float, float]] = field(default_factory=dict)
    month_zero_mass: Dict[int, float] = field(default_factory=dict)
    outcome_horizons: List[Optional[int]] = field(default_factory=list)
    # 'floor' | 'fill' | 'control' (Oct 2026); controls are reported apart.
    selection_pass: Optional[str] = None

    def per_outcome(self) -> List[Tuple[float, Dict[float, float], Optional[float]]]:
        """(outcome, quantiles for its month, zero mass for its month)."""
        out = []
        hs = self.outcome_horizons or [None] * len(self.outcomes)
        for y, h in zip(self.outcomes, hs):
            q = self.month_quantiles.get(h) if h is not None else None
            z = self.month_zero_mass.get(h) if h is not None else None
            out.append((y, q or self.quantiles, z if z is not None else self.zero_mass))
        return out


def _q(quantiles: Dict[float, float], level: float) -> Optional[float]:
    for k, v in quantiles.items():
        if abs(float(k) - level) < 1e-9 and v is not None:
            return float(v)
    return None


def _quantile_dict(raw: Any) -> Dict[float, float]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[float, float] = {}
    for k, v in raw.items():
        try:
            out[float(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _perspective_key(text: Any) -> Optional[str]:
    s = str(text or "").strip()
    if not s:
        return None
    return s.split(":", 1)[0].strip() or None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class Stat:
    """A bootstrap estimate over questions."""

    value: Optional[float]
    lo: Optional[float]
    hi: Optional[float]
    n_questions: int
    numerator: Optional[float] = None
    denominator: Optional[float] = None

    def excludes(self, target: float) -> bool:
        if self.lo is None or self.hi is None:
            return False
        return target < self.lo or target > self.hi

    def to_dict(self) -> Dict[str, Any]:
        r = lambda x: None if x is None else round(float(x), 4)  # noqa: E731
        return {
            "value": r(self.value), "lo": r(self.lo), "hi": r(self.hi),
            "n_questions": self.n_questions,
            "numerator": r(self.numerator), "denominator": r(self.denominator),
        }


def bootstrap_ratio(
    pairs: Sequence[Tuple[float, float]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> Stat:
    """Ratio of sums over questions, with a question-resampling interval.

    *pairs* holds one (numerator, denominator) per QUESTION. A plain mean of
    per-question values is the case denominator == 1.
    """
    pairs = [(float(a), float(b)) for a, b in pairs if b and b > 0 and math.isfinite(a)]
    n = len(pairs)
    if n == 0:
        return Stat(None, None, None, 0)
    num = np.array([p[0] for p in pairs])
    den = np.array([p[1] for p in pairs])
    value = float(num.sum() / den.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(draws, n))
    boot = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    lo, hi = np.quantile(boot, INTERVAL)
    return Stat(value, float(lo), float(hi), n, float(num.sum()), float(den.sum()))


def diagnose(records: Sequence[SibylRecord]) -> Dict[str, Stat]:
    """Every diagnostic over *records* (one record per distinct question)."""
    cov, below, above, above99, bias, zero_gap, anchor = [], [], [], [], [], [], []
    zero_share, zero_mass = [], []
    for r in records:
        rows = [
            (float(y), q, z) for y, q, z in r.per_outcome()
            if y is not None and math.isfinite(float(y))
        ]
        if not rows:
            continue
        m = float(len(rows))
        qs = [(y, _q(q, 0.1), _q(q, 0.5), _q(q, 0.9), _q(q, 0.99), z) for y, q, z in rows]
        if all(a[1] is not None and a[3] is not None for a in qs):
            cov.append((sum(q10 <= y <= q90 for y, q10, _, q90, _, _ in qs), m))
            below.append((sum(y < q10 for y, q10, _, _, _, _ in qs), m))
            above.append((sum(y > q90 for y, _, _, q90, _, _ in qs), m))
        if all(a[4] is not None for a in qs):
            above99.append((sum(y > q99 for y, _, _, _, q99, _ in qs), m))
        if all(a[2] is not None for a in qs):
            bias.append((sum(math.log1p(max(y, 0.0)) - math.log1p(max(q50, 0.0))
                             for y, _, q50, _, _, _ in qs), m))
        # FL/TC: the zero bucket means "zero or no record", and PA leaves a
        # month without a record unresolved, so the zero share of resolved
        # months says nothing about it. Their two-part scores
        # (sibyl/score_variants.py) measure it instead; no zero line here.
        if r.hazard_code not in ZERO_GAP_EXCLUDED and all(a[5] is not None for a in qs):
            n0 = sum(y == 0 for y, *_ in qs)
            mass = sum(float(z) for *_, z in qs)
            zero_gap.append((n0 - mass, m))
            zero_share.append((n0, m))
            zero_mass.append((mass, m))
        q50s = [a[2] for a in qs]
        if all(v is not None for v in q50s) and r.base_median is not None:
            err_s = np.mean([abs(math.log1p(max(y, 0.0)) - math.log1p(max(q50, 0.0)))
                             for y, _, q50, _, _, _ in qs])
            err_b = np.mean([abs(math.log1p(max(y, 0.0)) - math.log1p(max(r.base_median, 0.0)))
                             for y, *_ in qs])
            anchor.append((1.0 if err_s < err_b else 0.0, 1.0))
    return {
        "coverage_10_90": bootstrap_ratio(cov),
        "below_q10": bootstrap_ratio(below),
        "above_q90": bootstrap_ratio(above),
        "above_q99": bootstrap_ratio(above99),
        "centre_bias_log": bootstrap_ratio(bias),
        "zero_gap": bootstrap_ratio(zero_gap),
        "anchor_departure": bootstrap_ratio(anchor),
        # Context for the zero line; never tested on their own.
        "zero_share": bootstrap_ratio(zero_share),
        "zero_mass": bootstrap_ratio(zero_mass),
    }


def perspective_bias(records: Sequence[SibylRecord]) -> Dict[str, Dict[str, Any]]:
    """Centre bias per trial lane (the perspective seed before Oct 2026). Findings only, never advice."""
    by: Dict[str, List[Tuple[float, float]]] = {}
    for r in records:
        ys = [float(y) for y in r.outcomes if y is not None]
        if not ys:
            continue
        for key, med in r.trial_medians.items():
            s = sum(math.log1p(max(y, 0.0)) - math.log1p(max(med, 0.0)) for y in ys)
            by.setdefault(key, []).append((s, float(len(ys))))
    return {k: bootstrap_ratio(v).to_dict() for k, v in sorted(by.items())}


# ---------------------------------------------------------------------------
# Advice text: fixed templates, no model call
# ---------------------------------------------------------------------------


def _months(stat: Stat) -> int:
    return int(round(stat.denominator or 0))


def _count(stat: Stat) -> int:
    return int(round(stat.numerator or 0))


def _expected(stat: Stat, rate: float) -> str:
    e = (stat.denominator or 0) * rate
    lo, hi = math.floor(e), math.ceil(e)
    return f"{lo}" if lo == hi else f"{lo} or {hi}"


def build_advice_text(diag: Dict[str, Stat], n_questions: int) -> str:
    """The instructions the findings support, or '' when none clears its interval.

    Each line gives the evidence with its count, then one proportionate
    action. Reads only Sibyl's own diagnostics: no country, no model name,
    no ensemble figure.
    """
    lines: List[str] = []
    q = f"In {n_questions} resolved questions"

    s = diag.get("below_q10")
    if s and s.excludes(CALIBRATED["below_q10"]):
        if s.value > CALIBRATED["below_q10"]:
            lines.append(
                f"{q} the outcome fell below your 10% quantile in {_count(s)} of "
                f"{_months(s)} resolved months, against {_expected(s, 0.10)} expected. "
                "Your lower tail starts too high. Bring q0.1 and q0.25 down."
            )
        else:
            lines.append(
                f"{q} the outcome fell below your 10% quantile in {_count(s)} of "
                f"{_months(s)} resolved months, against {_expected(s, 0.10)} expected. "
                "Your lower tail reaches too low. Raise q0.1 and q0.25."
            )

    s = diag.get("above_q90")
    if s and s.excludes(CALIBRATED["above_q90"]):
        if s.value > CALIBRATED["above_q90"]:
            lines.append(
                f"{q} the outcome rose above your 90% quantile in {_count(s)} of "
                f"{_months(s)} resolved months, against {_expected(s, 0.10)} expected. "
                "Your upper tail stops too low. Raise q0.9 and q0.95."
            )
        else:
            lines.append(
                f"{q} the outcome rose above your 90% quantile in {_count(s)} of "
                f"{_months(s)} resolved months, against {_expected(s, 0.10)} expected. "
                "Your upper tail reaches too high. Bring q0.9 and q0.95 down."
            )

    s = diag.get("above_q99")
    if s and s.excludes(CALIBRATED["above_q99"]) and s.value > CALIBRATED["above_q99"]:
        lines.append(
            f"The outcome exceeded your 99% quantile in {_count(s)} of {_months(s)} "
            "resolved months, where about one in a hundred is expected. Raise q0.99."
        )

    s = diag.get("coverage_10_90")
    if s and s.excludes(CALIBRATED["coverage_10_90"]) and s.value > CALIBRATED["coverage_10_90"]:
        lines.append(
            f"{q} {_count(s)} of {_months(s)} resolved months fell between your 10% "
            f"and 90% quantiles, against {_expected(s, 0.80)} expected. Your ranges "
            "are wider than the outcomes needed. Draw q0.1 and q0.9 closer to the median."
        )

    s = diag.get("centre_bias_log")
    if s and s.excludes(CALIBRATED["centre_bias_log"]):
        factor = math.exp(abs(s.value))
        if s.value > 0:
            lines.append(
                f"{q} outcomes ran above your median, by a factor of about "
                f"{factor:.1f} on average. Raise q0.5."
            )
        else:
            lines.append(
                f"{q} outcomes ran below your median, by a factor of about "
                f"{factor:.1f} on average. Lower q0.5."
            )

    s = diag.get("zero_gap")
    if s and s.excludes(CALIBRATED["zero_gap"]):
        share = diag.get("zero_share")
        mass = diag.get("zero_mass")
        said = (
            f"the outcome was exactly zero in {100 * (share.value or 0):.0f}% of "
            f"resolved months, against the {100 * (mass.value or 0):.0f}% you put on zero"
            if share and mass and share.value is not None and mass.value is not None
            else "the outcome was exactly zero at a rate that differed from your zero bucket"
        )
        if s.value > 0:
            lines.append(f"{q} {said}. Put more probability on zero, by about that gap.")
        else:
            lines.append(f"{q} {said}. Put less probability on zero, by about that gap.")

    s = diag.get("anchor_departure")
    if s and s.excludes(CALIBRATED["anchor_departure"]) and s.value < CALIBRATED["anchor_departure"]:
        lines.append(
            f"In {s.n_questions} resolved questions your median sat closer to the "
            f"outcome than the base-rate median in only {_count(s)}. Moving away from "
            "the outside view has cost accuracy more often than it helped. Depart from "
            "it only on strong, specific evidence."
        )

    text = "\n".join(f"- {line}" for line in lines)
    if len(text) > ADVICE_MAX_CHARS:
        kept: List[str] = []
        for line in lines:
            candidate = "\n".join(f"- {x}" for x in kept + [line])
            if len(candidate) > ADVICE_MAX_CHARS:
                break
            kept.append(line)
        text = "\n".join(f"- {x}" for x in kept)
    return text


# ---------------------------------------------------------------------------
# Experiment arm
# ---------------------------------------------------------------------------


def advice_arm(question_id: Optional[str], share: float) -> str:
    """``"no_advice"`` or ``"advice"`` for a question.

    Hashes ``"sibyl:" + question_id`` (as ``forecaster.prompts.advice_arm``
    hashes the bare id), so a rerun keeps its arm and the split is
    independent of the standard track's.
    """
    import hashlib

    if share <= 0 or not question_id:
        return "advice"
    key = f"sibyl:{question_id}".encode("utf-8")
    frac = int(hashlib.sha1(key).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "no_advice" if frac < share else "advice"


def arm_comparison(
    per_question_scores: Dict[str, Dict[str, float]],
    arms: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    """Mean Brier and CRPS by arm, or 'not yet' below ARM_MIN_QUESTIONS per arm."""
    out: Dict[str, Any] = {"min_questions_per_arm": ARM_MIN_QUESTIONS}
    groups: Dict[str, List[Dict[str, float]]] = {"advice": [], "no_advice": []}
    for qid, sc in per_question_scores.items():
        arm = arms.get(qid)
        if arm in groups:
            groups[arm].append(sc)
    counts = {a: len(v) for a, v in groups.items()}
    out["n_questions"] = counts
    if min(counts.values()) < ARM_MIN_QUESTIONS:
        out["status"] = "not yet"
        return out
    out["status"] = "ok"
    for score_type in ("brier", "crps"):
        out[score_type] = {}
        for arm, rows in groups.items():
            vals = [(r[score_type], 1.0) for r in rows if r.get(score_type) is not None]
            out[score_type][arm] = bootstrap_ratio(vals).to_dict()
    return out


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


def _cols(con, table: str) -> set:
    try:
        return {str(r[1]).lower() for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()}
    except Exception:
        return set()


def _has_table(con, table: str) -> bool:
    return bool(_cols(con, table))


def load_records(con, as_of_month: Optional[str] = None) -> List[SibylRecord]:
    """Sibyl's scored record: latest production ok forecast per question."""
    if not (_has_table(con, "sibyl_forecasts") and _has_table(con, "resolutions")):
        return []
    f_cols = _cols(con, "sibyl_forecasts")
    run_join = (
        "JOIN sibyl_runs sr ON sr.sibyl_run_id = f.sibyl_run_id "
        "AND NOT COALESCE(sr.is_test, FALSE)"
        if _has_table(con, "sibyl_runs") else ""
    )
    run_order = "sr.created_at DESC NULLS LAST, " if run_join else ""
    q_join = (
        "JOIN questions q ON q.question_id = f.question_id AND NOT COALESCE(q.is_test, FALSE)"
        if _has_table(con, "questions") else ""
    )
    opt = lambda c: f"f.{c}" if c in f_cols else f"CAST(NULL AS TEXT) AS {c}"  # noqa: E731
    # A forecast that rested on no evidence (the July 2026 run) is kept and
    # scored, but it is not Sibyl's record.
    evidence = " AND COALESCE(f.evidence_ok, TRUE)" if "evidence_ok" in f_cols else ""
    rows = con.execute(
        f"""
        SELECT question_id, hazard_code, metric, pooled_quantiles_json,
               bucket_probs_json, trials_json, base_rate_json, advice_arm,
               sibyl_run_id, run_id, raw_by_month_json, selection_pass
        FROM (
            SELECT f.question_id, upper(f.hazard_code) AS hazard_code,
                   upper(f.metric) AS metric, f.pooled_quantiles_json,
                   f.bucket_probs_json, f.trials_json, {opt('base_rate_json')},
                   {opt('advice_arm')}, f.sibyl_run_id, f.run_id,
                   {opt('raw_by_month_json')}, {opt('selection_pass')},
                   ROW_NUMBER() OVER (
                       PARTITION BY f.question_id
                       ORDER BY {run_order}f.created_at DESC NULLS LAST, f.sibyl_run_id DESC
                   ) AS rn
            FROM sibyl_forecasts f
            {run_join}
            {q_join}
            WHERE f.status = 'ok' AND NOT COALESCE(f.is_test, FALSE){evidence}
        )
        WHERE rn = 1
        """
    ).fetchall()
    if not rows:
        return []

    r_cols = _cols(con, "resolutions")
    where = ["value IS NOT NULL"]
    params: List[Any] = []
    if "is_test" in r_cols:
        where.append("NOT COALESCE(is_test, FALSE)")
    if as_of_month and "observed_month" in r_cols:
        where.append("observed_month <= ?")
        params.append(as_of_month)
    # An indicative ACE/PA month (pythia/tools/scoring_class.py) is a
    # selected sample and never part of Sibyl's record.
    from pythia.tools.scoring_class import has_scoring_class, scored_only_sql  # noqa: PLC0415

    if has_scoring_class(con):
        where.append(scored_only_sql("resolutions"))
    outcomes: Dict[str, List[float]] = {}
    horizons: Dict[str, List[Optional[int]]] = {}
    h_col = "horizon_m" if "horizon_m" in r_cols else "CAST(NULL AS INTEGER)"
    for qid, val, h in con.execute(
        f"SELECT question_id, value, {h_col} FROM resolutions WHERE {' AND '.join(where)}", params
    ).fetchall():
        outcomes.setdefault(str(qid), []).append(float(val))
        horizons.setdefault(str(qid), []).append(int(h) if h is not None else None)

    records: List[SibylRecord] = []
    for (qid, hz, metric, pq, bp, trials, base, arm, srid, rid, raw_bm, sel) in rows:
        ys = outcomes.get(str(qid))
        if not ys:
            continue
        quantiles = _quantile_dict(pq)
        if not quantiles:
            continue
        zero_mass = None
        try:
            probs = json.loads(bp) if isinstance(bp, str) else bp
            if isinstance(probs, list) and probs:
                zero_mass = float(probs[0])
        except (TypeError, ValueError):
            pass
        base_median = None
        try:
            bd = json.loads(base) if isinstance(base, str) else base
            if isinstance(bd, dict):
                base_median = _q(_quantile_dict(bd.get("anchor_quantiles")), 0.5)
        except (TypeError, ValueError):
            pass
        trial_medians: Dict[str, float] = {}
        try:
            tl = json.loads(trials) if isinstance(trials, str) else trials
            for t in tl or []:
                key = _perspective_key((t or {}).get("perspective"))
                med = _q(_quantile_dict((t or {}).get("quantiles")), 0.5)
                if key and med is not None:
                    trial_medians[key] = med
        except (TypeError, ValueError):
            pass
        month_q: Dict[int, Dict[float, float]] = {}
        month_z: Dict[int, float] = {}
        try:
            bm = json.loads(raw_bm) if isinstance(raw_bm, str) else raw_bm
            if isinstance(bm, dict):
                for mk, qd in (bm.get("quantiles") or {}).items():
                    month_q[int(mk)] = _quantile_dict(qd)
                for mk, vec in (bm.get("vectors") or {}).items():
                    if isinstance(vec, list) and vec:
                        month_z[int(mk)] = float(vec[0])
        except (TypeError, ValueError):
            pass
        records.append(SibylRecord(
            question_id=str(qid), hazard_code=str(hz), metric=str(metric),
            quantiles=quantiles, outcomes=ys, zero_mass=zero_mass,
            base_median=base_median, trial_medians=trial_medians,
            advice_arm=arm, sibyl_run_id=srid, forecast_run_id=rid,
            month_quantiles=month_q, month_zero_mass=month_z,
            outcome_horizons=horizons.get(str(qid), []),
            selection_pass=sel,
        ))
    return records


def load_question_scores(con, records: Sequence[SibylRecord]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """{question_id: {model: {score_type: mean over horizons}}} for paired skill.

    Sibyl and ``ensemble_mean_v2`` are read under the question's standing
    forecast run (Sibyl writes under the standard track's run id);
    ``__ext_climatology`` has no run id. Test score rows are excluded.
    """
    if not records or not _has_table(con, "scores"):
        return {}
    s_cols = _cols(con, "scores")
    test = " AND NOT COALESCE(is_test, FALSE)" if "is_test" in s_cols else ""
    from pythia.tools.scoring_class import scored_only_clause  # noqa: PLC0415

    test += scored_only_clause(con, "scores")
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for r in records:
        rows = con.execute(
            f"""
            SELECT model_name, score_type, AVG(value)
            FROM scores
            WHERE question_id = ? AND score_type IN ('brier', 'crps')
              AND (
                (model_name IN (?, 'ensemble_mean_v2') AND run_id = ?)
                OR (model_name = '__ext_climatology' AND run_id IS NULL)
              ){test}
            GROUP BY 1, 2
            """,
            [r.question_id, SIBYL_MODEL_NAME, r.forecast_run_id],
        ).fetchall()
        for model, st, v in rows:
            if v is not None:
                out.setdefault(r.question_id, {}).setdefault(str(model), {})[str(st)] = float(v)
    return out


def paired_skill(scores: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
    """Sibyl minus reference on the same questions (negative = Sibyl better)."""
    out: Dict[str, Any] = {}
    for ref in ("ensemble_mean_v2", "__ext_climatology"):
        out[ref] = {}
        for st in ("brier", "crps"):
            diffs = [
                (m[SIBYL_MODEL_NAME][st] - m[ref][st], 1.0)
                for m in scores.values()
                if st in m.get(SIBYL_MODEL_NAME, {}) and st in m.get(ref, {})
            ]
            out[ref][st] = bootstrap_ratio(diffs).to_dict()
    return out


def by_selection(records: Sequence[SibylRecord]) -> Dict[str, Any]:
    """Diagnostics for selected questions and for controls, each with its count."""
    out: Dict[str, Any] = {}
    for name, keep in (("selected", lambda r: r.selection_pass != "control"),
                       ("control", lambda r: r.selection_pass == "control")):
        sub = [r for r in records if keep(r)]
        out[name] = {
            "n_questions": len(sub),
            "diagnostics": {k: v.to_dict() for k, v in diagnose(sub).items()} if sub else {},
        }
    return out


def _findings(records: Sequence[SibylRecord], scores) -> Tuple[Dict[str, Stat], Dict[str, Any]]:
    diag = diagnose(records)
    sub = {r.question_id: scores[r.question_id] for r in records if r.question_id in scores}
    findings = {
        "diagnostics": {k: v.to_dict() for k, v in diag.items()},
        "calibrated_values": dict(CALIBRATED),
        "instructions_from": sorted(
            k for k, v in diag.items() if k in CALIBRATED and v.excludes(CALIBRATED[k])
        ),
        "perspective_bias": perspective_bias(records),
        "paired_skill": paired_skill(sub),
        "n_resolved_months": int(sum(len(r.outcomes) for r in records)),
        "n_with_base_rate": int(sum(r.base_median is not None for r in records)),
        # Selected questions and the no-flag controls, measured apart: the
        # controls are drawn to say whether selection by RC flag picks the
        # questions where research helps, so they never blur into the rest.
        "by_selection": by_selection(records),
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "interval": list(INTERVAL)},
    }
    return diag, findings


def build_rows(
    records: Sequence[SibylRecord],
    scores: Dict[str, Dict[str, Dict[str, float]]],
    *,
    as_of_month: str,
    min_questions: int = ADVICE_MIN_QUESTIONS,
    blocked: Iterable[Tuple[str, str]] = (),
) -> List[Dict[str, Any]]:
    """One row per class with any scored question, plus the pooled row.

    Pure (no DB). ``advice`` is '' when the class is under *min_questions*,
    blocked, or no finding clears its interval; the findings are written
    regardless, so the dashboard can say how far a class has to go.
    """
    blocked = {(h.upper(), m.upper()) for h, m in blocked}
    eligible = [r for r in records if (r.hazard_code, r.metric) in ELIGIBLE_HAZARD_METRICS]
    rows: List[Dict[str, Any]] = []

    groups: Dict[Tuple[str, str], List[SibylRecord]] = {}
    for r in eligible:
        groups.setdefault((r.hazard_code, r.metric), []).append(r)
    for (hz, m), recs in sorted(groups.items()):
        diag, findings = _findings(recs, scores)
        n = len(recs)
        if (hz, m) in blocked:
            advice, reason = "", "blocked by PYTHIA_ADVICE_BLOCK_GROUPS"
        elif n < min_questions:
            advice, reason = "", f"{n} of {min_questions} resolved questions"
        else:
            advice = build_advice_text(diag, n)
            reason = None if advice else "no finding clears its 90% interval"
        findings["gate"] = reason
        rows.append({
            "as_of_month": as_of_month, "hazard_code": hz, "metric": m,
            "scope": "group", "n_questions": n, "advice": advice,
            "findings": findings,
        })

    pool = [r for r in eligible if (r.hazard_code, r.metric) not in blocked]
    diag, findings = _findings(pool, scores)
    n = len(pool)
    if n < min_questions:
        advice, reason = "", f"{n} of {min_questions} resolved questions"
    else:
        advice = build_advice_text(diag, n)
        reason = None if advice else "no finding clears its 90% interval"
    findings["gate"] = reason
    per_q = {
        qid: {st: v for st, v in (s.get(SIBYL_MODEL_NAME) or {}).items()}
        for qid, s in scores.items()
    }
    findings["arm_comparison"] = arm_comparison(
        per_q, {r.question_id: r.advice_arm for r in eligible}
    )
    findings["n_questions_by_class"] = {f"{h}/{m}": len(v) for (h, m), v in sorted(groups.items())}
    rows.append({
        "as_of_month": as_of_month, "hazard_code": POOLED, "metric": POOLED,
        "scope": "pooled", "n_questions": n, "advice": advice, "findings": findings,
    })
    return rows


def write_rows(con, rows: Sequence[Dict[str, Any]]) -> None:
    from pythia.db.schema import ensure_sibyl_calibration_advice_table  # noqa: PLC0415

    ensure_sibyl_calibration_advice_table(con)
    for row in rows:
        con.execute(
            "DELETE FROM sibyl_calibration_advice WHERE as_of_month = ? "
            "AND hazard_code = ? AND metric = ?",
            [row["as_of_month"], row["hazard_code"], row["metric"]],
        )
        con.execute(
            """
            INSERT INTO sibyl_calibration_advice (
                as_of_month, hazard_code, metric, scope, n_questions, advice,
                findings_json, advice_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [
                row["as_of_month"], row["hazard_code"], row["metric"], row["scope"],
                int(row["n_questions"]), row["advice"],
                json.dumps(row["findings"], default=str), ADVICE_VERSION,
            ],
        )


def generate(con, *, as_of_month: Optional[str] = None) -> List[Dict[str, Any]]:
    """Measure, build and write this month's rows. Returns them."""
    from pythia.tools.generate_calibration_advice import advice_blocked_groups  # noqa: PLC0415

    from sibyl.evidence import backfill_evidence_ok  # noqa: PLC0415

    as_of_month = as_of_month or date.today().strftime("%Y-%m")
    backfill_evidence_ok(con)
    records = load_records(con, as_of_month)
    scores = load_question_scores(con, records)
    rows = build_rows(records, scores, as_of_month=as_of_month, blocked=advice_blocked_groups())
    # The shadow arm (sibyl/shadow.py): shadow minus Sibyl over every scored
    # question, "not yet" below SIBYL_SHADOW_MIN_QUESTIONS. A finding on the
    # pooled row only; it never reaches the advice text.
    try:
        from sibyl.shadow import shadow_comparison  # noqa: PLC0415

        for row in rows:
            if row["scope"] == "pooled":
                row["findings"]["shadow"] = shadow_comparison(con)
    except Exception as exc:  # noqa: BLE001
        logger.warning("sibyl.advice: shadow comparison failed: %s", exc)
    # Failure types from the post-mortems (sibyl/postmortem.py): pooled row
    # only, a finding for the dashboard; never in the advice text.
    try:
        from sibyl.postmortem import failure_rates  # noqa: PLC0415

        for row in rows:
            if row["scope"] == "pooled":
                row["findings"]["failure_types"] = failure_rates(con)
    except Exception as exc:  # noqa: BLE001
        logger.warning("sibyl.advice: failure rates failed: %s", exc)
    write_rows(con, rows)
    for row in rows:
        logger.info(
            "sibyl.advice %s %s/%s: %d scored question(s); %s",
            row["as_of_month"], row["hazard_code"], row["metric"], row["n_questions"],
            f"advice written ({len(row['advice'])} chars)" if row["advice"]
            else f"no advice ({row['findings'].get('gate')})",
        )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Sibyl calibration advice")
    parser.add_argument("--db-url", default=None, help="DuckDB URL (default: config)")
    parser.add_argument("--as-of-month", default=None, help="YYYY-MM (default: this month)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import os

    if args.db_url:
        os.environ["PYTHIA_DB_URL"] = args.db_url
    from pythia.db.schema import connect, ensure_schema  # noqa: PLC0415

    ensure_schema()
    con = connect(read_only=False)
    try:
        rows = generate(con, as_of_month=args.as_of_month)
    finally:
        con.close()
    written = sum(1 for r in rows if r["advice"])
    print(f"sibyl_advice rows={len(rows)} with_advice={written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
