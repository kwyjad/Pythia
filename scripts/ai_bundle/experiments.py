# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""Two measured experiments in the scored bundle.

* ``advice_experiment.csv`` — the no-advice arm against the advice arm
  (``PYTHIA_ADVICE_EXPERIMENT_SHARE``), per (hazard, metric, score family,
  track): the primary aggregate's mean Brier per arm over distinct questions
  (the latest run), and a seeded bootstrap 90% interval on the difference.
* ``recalibration_effect.csv`` — family recalibration
  (``PYTHIA_FAMILY_RECALIBRATION_MODE``): the corrected forecast against the
  uncorrected one on the SAME (question, horizon), per member (``<model>``
  against ``<model>__raw`` where it was applied, ``<model>__recal`` against
  ``<model>`` where it was shadowed) and for an unweighted member mean
  rebuilt both ways, with a paired bootstrap 90% interval.

Both are empty, with a header, until the flags have run and resolved.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scripts.ai_bundle.common import write_csv

BOOTSTRAP = 2000
SEED = 20260930


def _bootstrap_diff(a: Sequence[float], b: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    """90% interval of mean(a) - mean(b), resampling each side."""
    if not a or not b:
        return None, None
    rng = random.Random(SEED)
    diffs = []
    for _ in range(BOOTSTRAP):
        sa = [a[rng.randrange(len(a))] for _ in a]
        sb = [b[rng.randrange(len(b))] for _ in b]
        diffs.append(sum(sa) / len(sa) - sum(sb) / len(sb))
    diffs.sort()
    return diffs[int(0.05 * BOOTSTRAP)], diffs[int(0.95 * BOOTSTRAP) - 1]


def _bootstrap_paired(d: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    """90% interval of the mean paired difference."""
    if not d:
        return None, None
    rng = random.Random(SEED)
    means = []
    for _ in range(BOOTSTRAP):
        s = [d[rng.randrange(len(d))] for _ in d]
        means.append(sum(s) / len(s))
    means.sort()
    return means[int(0.05 * BOOTSTRAP)], means[int(0.95 * BOOTSTRAP) - 1]


def _cols(con, table: str) -> set[str]:
    try:
        return {str(r[1]).lower() for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()}
    except Exception:  # noqa: BLE001
        return set()


def _r(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(v, 4)


ADVICE_COLUMNS = [
    "hazard_code", "metric", "score_family", "track",
    "n_advice", "n_no_advice", "mean_brier_advice", "mean_brier_no_advice",
    "difference", "ci90_low", "ci90_high",
]


def emit_advice_experiment(con, out_dir: Path, qids: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if "advice_arm" not in _cols(con, "forecasts_ensemble") or not qids:
        write_csv(out_dir / "advice_experiment.csv", ADVICE_COLUMNS, rows)
        return rows
    track = "q.track" if "track" in _cols(con, "questions") else "NULL"
    data = con.execute(
        f"""
        WITH arm AS (
            SELECT fe.question_id, ANY_VALUE(fe.advice_arm) AS arm
            FROM forecasts_ensemble fe
            WHERE fe.advice_arm IS NOT NULL
              AND fe.run_id = (SELECT MAX(_lr.run_id) FROM forecasts_ensemble _lr
                               WHERE _lr.question_id = fe.question_id)
            GROUP BY fe.question_id
        )
        SELECT upper(q.hazard_code), upper(q.metric), {track}, arm.arm, s.question_id, AVG(s.value)
        FROM scores s
        JOIN arm ON arm.question_id = s.question_id
        JOIN questions q ON q.question_id = s.question_id
        WHERE s.score_type = 'brier'
          AND s.model_name IN ('ensemble_mean_v2', 'track2_flash')
          AND s.run_id = (SELECT MAX(_lr.run_id) FROM forecasts_ensemble _lr
                          WHERE _lr.question_id = s.question_id)
          AND s.question_id IN (SELECT UNNEST(?::VARCHAR[]))
        GROUP BY 1, 2, 3, 4, 5
        """,
        [qids],
    ).fetchall()
    groups: Dict[tuple, Dict[str, List[float]]] = {}
    for hz, metric, tr, arm, _qid, v in data:
        groups.setdefault((hz, metric, tr), {}).setdefault(str(arm), []).append(float(v))
    for (hz, metric, tr), arms in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        a = arms.get("advice", [])
        n = arms.get("no_advice", [])
        lo, hi = _bootstrap_diff(a, n)
        rows.append({
            "hazard_code": hz, "metric": metric,
            "score_family": "binary" if metric == "EVENT_OCCURRENCE" else "spd",
            "track": tr, "n_advice": len(a), "n_no_advice": len(n),
            "mean_brier_advice": _r(sum(a) / len(a)) if a else None,
            "mean_brier_no_advice": _r(sum(n) / len(n)) if n else None,
            "difference": _r(sum(a) / len(a) - sum(n) / len(n)) if a and n else None,
            "ci90_low": _r(lo), "ci90_high": _r(hi),
        })
    write_csv(out_dir / "advice_experiment.csv", ADVICE_COLUMNS, rows)
    return rows


RECAL_COLUMNS = [
    "hazard_code", "metric", "score_family", "model", "comparison", "score_type",
    "n_paired", "mean_corrected", "mean_uncorrected", "difference", "ci90_low", "ci90_high",
]


def emit_recalibration_effect(con, out_dir: Path, qids: List[str]) -> List[Dict[str, Any]]:
    """Paired corrected-versus-uncorrected scores, per member and for the mean."""
    rows: List[Dict[str, Any]] = []
    if not qids:
        write_csv(out_dir / "recalibration_effect.csv", RECAL_COLUMNS, rows)
        return rows
    data = con.execute(
        """
        SELECT s.question_id, s.horizon_m, upper(q.hazard_code), upper(q.metric),
               s.score_type, s.model_name, s.value, s.run_id
        FROM scores s JOIN questions q ON q.question_id = s.question_id
        WHERE s.run_id IS NOT NULL
          AND s.run_id = (SELECT MAX(_lr.run_id) FROM forecasts_ensemble _lr
                          WHERE _lr.question_id = s.question_id)
          AND s.question_id IN (SELECT UNNEST(?::VARCHAR[]))
        """,
        [qids],
    ).fetchall()
    by: Dict[tuple, float] = {}
    meta: Dict[str, tuple] = {}
    for qid, h, hz, metric, st, model, v, _run in data:
        by[(qid, int(h), st, str(model))] = float(v)
        meta[qid] = (hz, metric)
    derived = {k[3] for k in by if k[3].endswith(("__raw", "__recal"))}
    pairs: Dict[tuple, List[float]] = {}
    for name in sorted(derived):
        if name.endswith("__raw"):
            base, corrected, raw, comparison = name[:-5], name[:-5], name, "applied"
        else:
            base, corrected, raw, comparison = name[:-7], name, name[:-7], "shadow"
        for (qid, h, st, model), v in by.items():
            if model != raw:
                continue
            c = by.get((qid, h, st, corrected))
            if c is None:
                continue
            hz, metric = meta[qid]
            pairs.setdefault((hz, metric, base, comparison, st), []).append((c, v))
    for (hz, metric, base, comparison, st), vals in sorted(pairs.items()):
        d = [c - u for c, u in vals]
        lo, hi = _bootstrap_paired(d)
        rows.append({
            "hazard_code": hz, "metric": metric,
            "score_family": "binary" if metric == "EVENT_OCCURRENCE" else "spd",
            "model": base, "comparison": comparison, "score_type": st,
            "n_paired": len(vals),
            "mean_corrected": _r(sum(c for c, _ in vals) / len(vals)),
            "mean_uncorrected": _r(sum(u for _, u in vals) / len(vals)),
            "difference": _r(sum(d) / len(d)), "ci90_low": _r(lo), "ci90_high": _r(hi),
        })
    rows.extend(_member_mean_effect(con, qids))
    write_csv(out_dir / "recalibration_effect.csv", RECAL_COLUMNS, rows)
    return rows


def _member_mean_effect(con, qids: List[str]) -> List[Dict[str, Any]]:
    """The unweighted mean of the members, rebuilt corrected and uncorrected,
    Brier-scored on the same (question, horizon). Only where corrections were
    applied, so both versions of every member exist."""
    try:
        from pythia.tools.base_rate_spd import _bucket_index_for_value
        from pythia.tools.compute_scores import _brier
    except Exception:  # noqa: BLE001
        return []
    rows = con.execute(
        """
        SELECT fr.question_id, fr.month_index, fr.model_name, fr.bucket_index, fr.probability,
               upper(q.hazard_code), upper(q.metric), r.value
        FROM forecasts_raw fr
        JOIN questions q ON q.question_id = fr.question_id
        JOIN resolutions r ON r.question_id = fr.question_id AND r.horizon_m = fr.month_index
        WHERE fr.probability IS NOT NULL AND fr.bucket_index IS NOT NULL
          AND upper(q.metric) <> 'EVENT_OCCURRENCE'
          AND fr.run_id = (SELECT MAX(_lr.run_id) FROM forecasts_ensemble _lr
                           WHERE _lr.question_id = fr.question_id)
          AND fr.question_id IN (SELECT UNNEST(?::VARCHAR[]))
        """,
        [qids],
    ).fetchall()
    vec: Dict[tuple, Dict[int, float]] = {}
    info: Dict[tuple, tuple] = {}
    for qid, h, model, b, p, hz, metric, v in rows:
        vec.setdefault((qid, int(h), str(model)), {})[int(b)] = float(p)
        info[(qid, int(h))] = (hz, metric, v)
    raw_names = {k[2] for k in vec if k[2].endswith("__raw")}
    groups: Dict[tuple, List[float]] = {}
    for (qid, h), (hz, metric, value) in info.items():
        bases = [n[:-5] for n in raw_names if (qid, h, n) in vec and (qid, h, n[:-5]) in vec]
        if len(bases) < 2 or value is None:
            continue
        j = _bucket_index_for_value(float(value), metric)
        if j is None:
            continue
        k = max(max(vec[(qid, h, b)]) for b in bases)

        def _mean(names: List[str]) -> List[float]:
            return [sum(vec[(qid, h, n)].get(i, 0.0) for n in names) / len(names) for i in range(1, k + 1)]

        corrected = _mean(bases)
        raw = _mean([b + "__raw" for b in bases])
        groups.setdefault((hz, metric), []).append((_brier(corrected, j), _brier(raw, j)))
    out: List[Dict[str, Any]] = []
    for (hz, metric), vals in sorted(groups.items()):
        d = [c - u for c, u in vals]
        lo, hi = _bootstrap_paired(d)
        out.append({
            "hazard_code": hz, "metric": metric, "score_family": "spd",
            "model": "member_mean (unweighted)", "comparison": "applied", "score_type": "brier",
            "n_paired": len(vals),
            "mean_corrected": _r(sum(c for c, _ in vals) / len(vals)),
            "mean_uncorrected": _r(sum(u for _, u in vals) / len(vals)),
            "difference": _r(sum(d) / len(d)), "ci90_low": _r(lo), "ci90_high": _r(hi),
        })
    return out
