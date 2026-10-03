# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Score Sibyl's two ingredients and fit the weight that mixes them (Oct 2026).

``python -m sibyl.score_variants --db-url ...`` runs in
``compute_calibration_pythia.yml`` after scoring and before ``sibyl.advice``.

What Sibyl publishes (``sibyl``) is ``w`` x its reference + ``1 - w`` x the
raw pool of its trials. The two parts are scored on their own, against the
same resolutions, under external names so they never reach a forecast table
or the calibration softmax:

* ``__ext_sibyl_raw``: the raw pool by window month, floored as the
  published vector is (``SIBYL_BUCKET_FLOOR``), so a log score stays finite.
* ``__ext_sibyl_ref``: Sibyl's reference by window month.

Rows follow ``pythia.tools.score_baselines``' write path: ``scores`` with
``run_id IS NULL`` (delete-then-insert), and the vector scored in
``baseline_scored_forecasts``. The forecast scored is the latest Sibyl
forecast of each question that rested on evidence.

FL/TC two-part scores (``sibyl_variant_scores``)
------------------------------------------------
Sibyl's zero bucket for flood and cyclone people-affected means "zero, or no
record", so a single bucket score mixes two questions. They are split:

* ``occurrence_brier``: the Brier of ``1 - p_zero`` against whether a record
  with people affected exists for the month (resolved value above zero);
* where a record exists, ``conditional_{brier,log,crps}``: the forecast's
  non-zero buckets renormalised, scored on the record's bucket.

PA leaves a month with no record unresolved, so "no record" is read as: the
month has no resolution for this question while the resolver has resolved
that calendar month for another question of the same class. That is a
proxy, stated as one; a month the resolver has not reached is not scored.
Scored for ``sibyl``, ``sibyl_raw`` and ``sibyl_ref``.

Pool weight
-----------
Once ``SIBYL_POOL_WEIGHT_MIN_QUESTIONS`` (20) distinct questions carry both a
raw and a reference log score, each ``w`` in {0.25, 0.5, 0.75} is scored as
the mean over questions of the mean log score of the floored mixture; the
best is shrunk toward 0.5 with a prior worth
``SIBYL_POOL_WEIGHT_PRIOR_QUESTIONS`` (20) questions:
``(n * w_best + 20 * 0.5) / (n + 20)``, and the nearest grid value is kept
(a tie keeps 0.5). One weight for every class. Below the threshold a row is
written with no weight, so the run uses ``SIBYL_REFERENCE_WEIGHT``.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sibyl import config as _cfg

logger = logging.getLogger(__name__)

RAW_MODEL_NAME = "__ext_sibyl_raw"
REF_MODEL_NAME = "__ext_sibyl_ref"
TWO_PART_HAZARDS = frozenset({"FL", "TC"})


# ---------------------------------------------------------------------------
# Pure scoring
# ---------------------------------------------------------------------------


def _scorers():
    from pythia.tools.compute_scores import _brier, _crps_like, _log_score  # noqa: PLC0415

    return _brier, _log_score, _crps_like


def score_vector(probs: Sequence[float], j: int) -> Dict[str, float]:
    """Brier, log and RPS (stored as 'crps') of *probs* on bucket *j*."""
    brier, log, crps = _scorers()
    p = [float(x) for x in probs]
    return {"brier": brier(p, j), "log": log(p, j), "crps": crps(p, j)}


def floored(vec: Sequence[float]) -> List[float]:
    from sibyl.spd import apply_bucket_floor  # noqa: PLC0415

    return apply_bucket_floor([float(x) for x in vec])


def two_part_scores(probs: Sequence[float], record_exists: bool, j: Optional[int]) -> Dict[str, float]:
    """The FL/TC two-part scores.

    ``occurrence_brier`` = ((1 - p_zero) - y)^2 with y = 1 when a record
    exists. When it does, the non-zero buckets are renormalised and scored on
    the record's bucket (*j* is the 0-based bucket of the full vector, so the
    conditional bucket is j - 1).
    """
    p = [max(0.0, float(x)) for x in probs]
    total = sum(p) or 1.0
    p = [x / total for x in p]
    y = 1.0 if record_exists else 0.0
    out = {"occurrence_brier": ((1.0 - p[0]) - y) ** 2}
    if record_exists and j is not None and j >= 1 and len(p) > 2:
        rest = p[1:]
        s = sum(rest)
        if s > 0:
            cond = [x / s for x in rest]
            sc = score_vector(cond, j - 1)
            out.update({f"conditional_{k}": v for k, v in sc.items()})
    return out


def mix(ref: Sequence[float], raw: Sequence[float], w: float) -> List[float]:
    """w x reference + (1 - w) x raw, floored as the published vector is."""
    mixed = [w * float(r) + (1.0 - w) * float(x) for r, x in zip(ref, raw)]
    z = sum(mixed) or 1.0
    return floored([x / z for x in mixed])


def fit_pool_weight(
    cases: Dict[str, List[Tuple[Sequence[float], Sequence[float], int]]],
    *,
    grid: Sequence[float] = _cfg.POOL_WEIGHT_GRID,
    min_questions: Optional[int] = None,
    prior_questions: Optional[int] = None,
    prior_weight: float = 0.5,
) -> Dict[str, Any]:
    """Choose the reference weight from *cases* {question_id: [(ref, raw, j), ...]}.

    Returns ``{"status", "n_questions", "losses", "fitted_weight", "weight"}``.
    ``weight`` is None when fewer than *min_questions* questions have cases.
    """
    min_q = _cfg.POOL_WEIGHT_MIN_QUESTIONS if min_questions is None else min_questions
    prior_q = _cfg.POOL_WEIGHT_PRIOR_QUESTIONS if prior_questions is None else prior_questions
    _, log, _ = _scorers()
    usable = {q: c for q, c in cases.items() if c}
    n = len(usable)
    out: Dict[str, Any] = {
        "n_questions": n, "min_questions": min_q, "prior_questions": prior_q,
        "losses": {}, "fitted_weight": None, "weight": None,
    }
    if n < min_q:
        out["status"] = "too_few"
        return out
    losses: Dict[float, float] = {}
    for w in grid:
        per_q = []
        for rows in usable.values():
            per_q.append(sum(log(mix(ref, raw, w), j) for ref, raw, j in rows) / len(rows))
        losses[float(w)] = sum(per_q) / len(per_q)
    w_best = min(sorted(losses), key=lambda w: (losses[w], abs(w - prior_weight)))
    shrunk = (n * w_best + prior_q * prior_weight) / (n + prior_q)
    chosen = min(sorted(grid), key=lambda w: (round(abs(w - shrunk), 9), abs(w - prior_weight)))
    out.update({
        "status": "fitted",
        "losses": {str(k): v for k, v in sorted(losses.items())},
        "best_weight": w_best,
        "fitted_weight": shrunk,
        "weight": float(chosen),
    })
    return out


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


def _cols(con, table: str) -> set:
    try:
        return {str(r[1]).lower() for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()}
    except Exception:
        return set()


def _json(raw: Any) -> Any:
    if raw is None or raw == "":
        return None
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def _by_month(obj: Any) -> Dict[int, List[float]]:
    out: Dict[int, List[float]] = {}
    for k, v in (obj or {}).items():
        try:
            if isinstance(v, list) and v:
                out[int(k)] = [float(x) for x in v]
        except (TypeError, ValueError):
            continue
    return out


def _month_index(window_start: Any, horizon: int) -> Optional[str]:
    if window_start is None:
        return None
    try:
        d = window_start if isinstance(window_start, date) else date.fromisoformat(str(window_start)[:10])
    except ValueError:
        return None
    m = d.month - 1 + (int(horizon) - 1)
    return f"{d.year + m // 12:04d}-{m % 12 + 1:02d}"


def load_forecasts(con) -> List[Dict[str, Any]]:
    """The latest evidence-backed ok Sibyl forecast per question, with its question."""
    f_cols = _cols(con, "sibyl_forecasts")
    need = {"raw_by_month_json", "reference_json", "final_by_month_json"}
    if not need <= f_cols or not _cols(con, "questions"):
        return []
    evidence = " AND COALESCE(f.evidence_ok, TRUE)" if "evidence_ok" in f_cols else ""
    sel = "f.selection_pass" if "selection_pass" in f_cols else "CAST(NULL AS TEXT)"
    rows = con.execute(
        f"""
        SELECT question_id, sibyl_run_id, hazard_code, metric, raw_by_month_json,
               reference_json, final_by_month_json, window_start_date, is_test, selection_pass
        FROM (
            SELECT f.question_id, f.sibyl_run_id, upper(q.hazard_code) AS hazard_code,
                   upper(q.metric) AS metric, f.raw_by_month_json, f.reference_json,
                   f.final_by_month_json, q.window_start_date,
                   (COALESCE(q.is_test, FALSE) OR COALESCE(f.is_test, FALSE)) AS is_test,
                   {sel} AS selection_pass,
                   ROW_NUMBER() OVER (
                       PARTITION BY f.question_id
                       ORDER BY COALESCE(f.is_test, FALSE), f.created_at DESC NULLS LAST,
                                f.sibyl_run_id DESC
                   ) AS rn
            FROM sibyl_forecasts f
            JOIN questions q ON q.question_id = f.question_id
            WHERE f.status = 'ok'{evidence}
        )
        WHERE rn = 1
        """
    ).fetchall()
    out = []
    for r in rows:
        out.append({
            "question_id": str(r[0]), "sibyl_run_id": r[1], "hazard_code": str(r[2]),
            "metric": str(r[3]),
            "raw": _by_month((_json(r[4]) or {}).get("vectors")),
            "ref": _by_month((_json(r[5]) or {}).get("by_month")),
            "final": _by_month(_json(r[6])),
            "window_start_date": r[7], "is_test": bool(r[8]), "selection_pass": r[9],
        })
    return out


def load_resolutions(con) -> Dict[str, Dict[int, float]]:
    r_cols = _cols(con, "resolutions")
    if not r_cols:
        return {}
    test = " AND NOT COALESCE(is_test, FALSE)" if "is_test" in r_cols else ""
    out: Dict[str, Dict[int, float]] = {}
    for qid, h, v in con.execute(
        f"SELECT question_id, horizon_m, value FROM resolutions WHERE value IS NOT NULL{test}"
    ).fetchall():
        out.setdefault(str(qid), {})[int(h)] = float(v)
    return out


def covered_months(con) -> Dict[Tuple[str, str], set]:
    """{(hazard, metric): calendar months the resolver resolved for that class}."""
    if not _cols(con, "resolutions") or not _cols(con, "questions"):
        return {}
    out: Dict[Tuple[str, str], set] = {}
    for hz, m, ws, h in con.execute(
        """
        SELECT upper(q.hazard_code), upper(q.metric), q.window_start_date, r.horizon_m
        FROM resolutions r JOIN questions q ON q.question_id = r.question_id
        WHERE r.value IS NOT NULL
        """
    ).fetchall():
        ym = _month_index(ws, int(h))
        if ym:
            out.setdefault((str(hz), str(m)), set()).add(ym)
    return out


def _bucket(value: float, metric: str) -> Optional[int]:
    from pythia.tools.compute_scores import _bucket_index  # noqa: PLC0415

    return _bucket_index(value, metric)


def score_variants(con, *, as_of_month: Optional[str] = None) -> Dict[str, Any]:
    """Score the variants, write the two-part rows and the pool weight. Returns counters."""
    from pythia.db.schema import ensure_sibyl_measurement_tables  # noqa: PLC0415
    from pythia.tools.score_baselines import (  # noqa: PLC0415
        _audit,
        _write_scores,
        ensure_baseline_tables,
    )

    ensure_sibyl_measurement_tables(con)
    ensure_baseline_tables(con)
    as_of_month = as_of_month or date.today().strftime("%Y-%m")
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    counters = {"scored_raw": 0, "scored_ref": 0, "two_part_rows": 0, "skipped_no_vector": 0}
    forecasts = load_forecasts(con)
    if not forecasts:
        counters["pool_weight"] = _write_weight(con, as_of_month, fit_pool_weight({}))
        return counters
    resolved = load_resolutions(con)
    covered = covered_months(con)
    cases: Dict[str, List[Tuple[List[float], List[float], int]]] = {}

    for f in forecasts:
        qid, metric = f["question_id"], f["metric"]
        res = resolved.get(qid, {})
        con.execute("DELETE FROM sibyl_variant_scores WHERE question_id = ? "
                    "AND series IN ('sibyl', 'sibyl_raw', 'sibyl_ref')", [qid])
        for h, value in sorted(res.items()):
            j = _bucket(value, metric)
            if j is None:
                continue
            raw = f["raw"].get(h)
            ref = f["ref"].get(h)
            if raw is not None:
                raw_f = floored(raw)
                if j < len(raw_f):
                    _write_scores(con, question_id=qid, horizon_m=h, metric=metric,
                                  model_name=RAW_MODEL_NAME,
                                  score_rows=sorted(score_vector(raw_f, j).items()),
                                  is_test=f["is_test"], now=now)
                    _audit(con, qid, h, RAW_MODEL_NAME, metric, raw_f, "sibyl_raw_pool",
                           value, j, now)
                    counters["scored_raw"] += 1
            else:
                counters["skipped_no_vector"] += 1
            if ref is not None and j < len(ref):
                _write_scores(con, question_id=qid, horizon_m=h, metric=metric,
                              model_name=REF_MODEL_NAME,
                              score_rows=sorted(score_vector(ref, j).items()),
                              is_test=f["is_test"], now=now)
                _audit(con, qid, h, REF_MODEL_NAME, metric, ref, "sibyl_reference",
                       value, j, now)
                counters["scored_ref"] += 1
            if raw is not None and ref is not None and len(raw) == len(ref) and not f["is_test"]:
                cases.setdefault(qid, []).append((ref, raw, j))

        if f["hazard_code"] in TWO_PART_HAZARDS:
            counters["two_part_rows"] += _write_two_part(con, f, res, covered, now)

    fit = fit_pool_weight(cases)
    counters["pool_weight"] = _write_weight(con, as_of_month, fit)
    logger.info("sibyl.score_variants: %s", counters)
    return counters


def _write_two_part(con, f: Dict[str, Any], res: Dict[int, float], covered, now) -> int:
    months = covered.get((f["hazard_code"], f["metric"]), set())
    series = {"sibyl": f["final"], "sibyl_raw": {m: floored(v) for m, v in f["raw"].items()},
              "sibyl_ref": f["ref"]}
    n = 0
    for h in range(1, 7):
        value = res.get(h)
        if value is None:
            ym = _month_index(f["window_start_date"], h)
            if not ym or ym not in months:
                continue  # the resolver has not reached this month
            record_exists, j = False, None
        else:
            j = _bucket(value, f["metric"])
            record_exists = value > 0
        for name, vecs in series.items():
            vec = vecs.get(h)
            if not vec:
                continue
            for st, v in two_part_scores(vec, record_exists, j).items():
                con.execute(
                    """
                    INSERT INTO sibyl_variant_scores
                        (question_id, horizon_m, series, score_type, value, sibyl_run_id,
                         detail_json, is_test, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [f["question_id"], h, name, st, float(v), f["sibyl_run_id"],
                     json.dumps({"record_exists": record_exists,
                                 "resolved": value is not None}),
                     f["is_test"], now],
                )
                n += 1
    return n


def _write_weight(con, as_of_month: str, fit: Dict[str, Any]) -> Dict[str, Any]:
    try:
        con.execute("DELETE FROM sibyl_pool_weights WHERE as_of_month = ?", [as_of_month])
        con.execute(
            """
            INSERT INTO sibyl_pool_weights
                (as_of_month, weight, n_questions, fitted_weight, status, losses_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [as_of_month, fit.get("weight"), int(fit.get("n_questions") or 0),
             fit.get("fitted_weight"), fit.get("status"), json.dumps(fit, default=str)],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("sibyl.score_variants: pool weight write failed: %s", exc)
    return {k: fit.get(k) for k in ("status", "n_questions", "weight")}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Score Sibyl's raw pool and reference")
    parser.add_argument("--db-url", default=None)
    parser.add_argument("--as-of-month", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import os

    if args.db_url:
        os.environ["PYTHIA_DB_URL"] = args.db_url
    from pythia.db.schema import connect, ensure_schema  # noqa: PLC0415

    ensure_schema()
    con = connect(read_only=False)
    try:
        counters = score_variants(con, as_of_month=args.as_of_month)
    except Exception as exc:  # noqa: BLE001 - never fails the calibration chain
        logger.exception("sibyl.score_variants failed: %s", exc)
        return 0
    finally:
        con.close()
    print(f"sibyl_score_variants {json.dumps(counters, default=str)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
