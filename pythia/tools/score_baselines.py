# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Score reference forecasters against Pythia resolutions.

Modeled directly on ``pythia/tools/score_views.py`` (the ``__ext_`` external
benchmark convention). Two reference forecasters:

- ``__ext_climatology`` — the base-rate SPD from ``pythia.tools.base_rate_spd``,
  i.e. "what you would have said with no model". The anchor is the SAME base
  rate the forecaster was shown at prompt time, built from history strictly
  before the question window.
- ``__ext_uniform`` — flat across buckets (0.5 for binary questions). The floor.

Both are scored with the same Brier/log/CRPS functions ``compute_scores.py``
uses, written to ``scores`` with ``run_id IS NULL``. Binary EVENT_OCCURRENCE
questions get a single binary Brier row (``(p - outcome)^2``), same as the
live binary scoring path — binary and SPD scores are NEVER blended.

Skill, computed downstream and never stored ambiguously:

    skill = 1 - (fred_score / climatology_score)

per (score_family, score_type), never across them.

The ``__ext_`` prefix keeps these rows out of the calibration weight softmax
(``compute_calibration_pythia`` filters ``model_name NOT LIKE '__ext_%'``) and
routes them into the Performance page's external-benchmark rendering.

Two more references score ACE/FATALITIES (and persistence DR Phase 3+):
``__ext_persistence`` (the last observed value, smoothed at a fixed 0.1) and
``__ext_level_volatility`` (the last complete month the forecaster could have
read, spread by how far counts like it move; the distribution the prompt
shows under PYTHIA_PRIOR_ANCHOR_SPD).
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pythia.buckets import n_buckets_for
from pythia.config import load as load_cfg
from pythia.tools.base_rate_spd import (
    base_rate_spd,
    forecast_months,
    last_observed_value,
    level_volatility_spds,
)
from pythia.tools.compute_deviation import _anchor_ym
from pythia.tools.compute_scores import (
    _brier,
    _bucket_index,
    _crps_like,
    _log_score,
)
from resolver.db import duckdb_io

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    LOGGER.addHandler(logging.NullHandler())

CLIMATOLOGY_MODEL_NAME = "__ext_climatology"
UNIFORM_MODEL_NAME = "__ext_uniform"

SPD_METRICS = ("PA", "FATALITIES", "PHASE3PLUS_IN_NEED")

#: "Next month looks like last month": the last observed value before the
#: window, placed in its bucket. The hardest simple reference for a
#: persistent quantity (a Phase 3+ caseload, a war's monthly death toll),
#: where climatology over three years is easy to beat by standing still.
PERSISTENCE_MODEL_NAME = "__ext_persistence"
#: (hazard, metric) pairs with a persistence reference.
PERSISTENCE_PAIRS = frozenset({("ACE", "FATALITIES"), ("DR", "PHASE3PLUS_IN_NEED")})
#: Share of the mass spread evenly over every bucket. A pure one-hot vector
#: gives an infinite log loss whenever the outcome leaves the bucket, which
#: says nothing useful; 0.1 keeps the reference sharp (90% + 0.1/k on the
#: persisted bucket) and every log score finite. It is a fixed constant on
#: purpose: tuning it against outcomes would make the reference a model.
PERSISTENCE_SMOOTHING = 0.1

#: Where the level is now and how far counts like it move: the last complete
#: month the forecaster could have read, spread by the bucket moves observed
#: over the same number of months (``base_rate_spd.level_volatility_spds``).
#: It is the distribution the prompt shows under PYTHIA_PRIOR_ANCHOR_SPD, so
#: it is also the score of a member that copied its prior and changed nothing.
LEVEL_VOLATILITY_MODEL_NAME = "__ext_level_volatility"
LEVEL_VOLATILITY_PAIRS = frozenset({("ACE", "FATALITIES")})


def _forecast_time(conn, question_id: str):
    """When the question's latest production forecast was made (None if unknown).

    The level-and-volatility reference must read the level the forecaster
    could have read, not the one the table holds at scoring time.
    """
    for test_clause in ("AND NOT COALESCE(is_test, FALSE)", ""):
        try:
            row = conn.execute(
                f"SELECT MAX(created_at) FROM forecasts_ensemble WHERE question_id = ? {test_clause}",
                [question_id],
            ).fetchone()
        except Exception:  # noqa: BLE001
            return None
        if row and row[0] is not None:
            return row[0]
    return None


def persistence_spd(value: float, metric: str) -> Optional[List[float]]:
    """The smoothed one-hot vector for ``value``, or None when unbucketable."""
    k = n_buckets_for(metric)
    j = _bucket_index(float(value), metric)
    if not k or j is None:
        return None
    eps = PERSISTENCE_SMOOTHING
    vec = [eps / k] * k
    vec[j] += 1.0 - eps
    return vec


def _table_exists(conn, name: str) -> bool:
    try:
        conn.execute(f"SELECT 1 FROM {name} LIMIT 0")
        return True
    except Exception:
        return False


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _load_resolution_pairs(conn) -> List[Dict[str, Any]]:
    """(question, horizon) pairs with resolutions — the same join
    compute_scores runs, plus the window anchor columns."""
    sql = """
      SELECT
        q.question_id,
        q.iso3,
        q.hazard_code,
        upper(q.metric) AS metric,
        r.horizon_m,
        r.value AS resolved_value,
        q.window_start_date,
        q.target_month,
        COALESCE(q.is_test, FALSE) AS is_test
      FROM questions q
      JOIN resolutions r ON q.question_id = r.question_id
      JOIN hs_runs h ON q.hs_run_id = h.hs_run_id
      WHERE upper(q.metric) IN ('PA','FATALITIES','EVENT_OCCURRENCE','PHASE3PLUS_IN_NEED')
      ORDER BY q.question_id, r.horizon_m
    """
    rows = conn.execute(sql).fetchall()
    return [
        {
            "question_id": r[0],
            "iso3": r[1],
            "hazard_code": r[2],
            "metric": r[3],
            "horizon_m": int(r[4]),
            "resolved_value": r[5],
            "window_start_date": r[6],
            "target_month": r[7],
            "is_test": bool(r[8]),
        }
        for r in rows
    ]


def _write_scores(
    conn,
    *,
    question_id: str,
    horizon_m: int,
    metric: str,
    model_name: str,
    score_rows: List[Tuple[str, float]],
    is_test: bool,
    now: datetime,
) -> None:
    """Idempotent delete-then-insert, run_id IS NULL (external convention)."""
    conn.execute(
        """
        DELETE FROM scores
        WHERE question_id = ? AND horizon_m = ? AND metric = ?
          AND model_name = ? AND run_id IS NULL
        """,
        [question_id, horizon_m, metric, model_name],
    )
    conn.executemany(
        """
        INSERT INTO scores
            (question_id, horizon_m, metric, score_type, model_name, value,
             run_id, created_at, is_test)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (question_id, horizon_m, metric, score_type, model_name, value,
             None, now, is_test)
            for score_type, value in score_rows
        ],
    )


def score_baselines(db_url: str) -> Dict[str, int]:
    """Score __ext_climatology and __ext_uniform for every resolved horizon.

    Returns counters (for tests and the workflow log).
    """
    if not duckdb_io.DUCKDB_AVAILABLE:
        raise RuntimeError(duckdb_io.duckdb_unavailable_reason())
    conn = duckdb_io.get_db(db_url or duckdb_io.DEFAULT_DB_URL)
    counters = {
        "scored_climatology": 0,
        "scored_uniform": 0,
        "scored_persistence": 0,
        "skipped_no_persistence": 0,
        "scored_level_volatility": 0,
        "skipped_no_level_volatility": 0,
        "skipped_no_baserate": 0,
        "skipped_bad_resolution": 0,
    }
    try:
        # Defensive — normally created by compute_scores in the same run.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scores (
              question_id TEXT,
              horizon_m INTEGER,
              metric TEXT,
              score_type TEXT,
              model_name TEXT,
              value DOUBLE,
              run_id TEXT,
              created_at TIMESTAMP DEFAULT now(),
              is_test BOOLEAN DEFAULT FALSE
            )
            """
        )
        # Audit trail: the reference SPD actually scored, per horizon.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS baseline_scored_forecasts (
                question_id TEXT,
                horizon_m INTEGER,
                model_name TEXT,
                metric TEXT,
                spd_json TEXT,
                baserate_source TEXT,
                resolved_value DOUBLE,
                resolved_bucket INTEGER,
                created_at TIMESTAMP DEFAULT now(),
                PRIMARY KEY (question_id, horizon_m, model_name)
            )
            """
        )

        for table in ("questions", "resolutions", "hs_runs"):
            if not _table_exists(conn, table):
                LOGGER.info("score_baselines: table %s not found; nothing to do.", table)
                return counters

        pairs = _load_resolution_pairs(conn)
        LOGGER.info("score_baselines: %d (question, horizon) pairs with resolutions.", len(pairs))
        if not pairs:
            return counters

        base_cache: Dict[str, Tuple[list, str, dict]] = {}
        persist_cache: Dict[str, Optional[Tuple[float, str, str]]] = {}
        lv_cache: Dict[str, Tuple[dict, str, dict]] = {}
        now = _utcnow_naive()

        for pair in pairs:
            qid = pair["question_id"]
            metric = pair["metric"]
            hm = pair["horizon_m"]
            resolved = pair["resolved_value"]
            if resolved is None:
                counters["skipped_bad_resolution"] += 1
                continue

            anchor = _anchor_ym(pair["window_start_date"], pair["target_month"])
            if anchor is None:
                counters["skipped_bad_resolution"] += 1
                LOGGER.warning("score_baselines: %s has no window anchor; skipping.", qid)
                continue

            if qid not in base_cache:
                base_cache[qid] = base_rate_spd(
                    conn, pair["iso3"], pair["hazard_code"], metric, anchor
                )
            base_probs, base_source, base_detail = base_cache[qid]

            is_binary = metric == "EVENT_OCCURRENCE"

            if is_binary:
                outcome = float(resolved)
                horizon_ym = forecast_months(anchor)[hm - 1] if 1 <= hm <= 6 else None
                # Uniform floor: a 50/50 guess.
                brier_u = (0.5 - outcome) ** 2
                _write_scores(
                    conn, question_id=qid, horizon_m=hm, metric=metric,
                    model_name=UNIFORM_MODEL_NAME,
                    score_rows=[("brier", brier_u)],
                    is_test=pair["is_test"], now=now,
                )
                _audit(conn, qid, hm, UNIFORM_MODEL_NAME, metric, [0.5, 0.5],
                       "uniform", resolved, None, now)
                counters["scored_uniform"] += 1

                if not base_probs:
                    counters["skipped_no_baserate"] += 1
                    LOGGER.info(
                        "score_baselines: no climatology for %s (%s/%s): %s",
                        qid, pair["hazard_code"], metric,
                        base_detail.get("reason", "unknown"),
                    )
                    continue
                probs_by_month = base_detail.get("probs_by_month") or {}
                if horizon_ym and horizon_ym in probs_by_month:
                    p_b = float(probs_by_month[horizon_ym][0])
                else:
                    p_b = float(base_probs[0])
                brier_c = (p_b - outcome) ** 2
                _write_scores(
                    conn, question_id=qid, horizon_m=hm, metric=metric,
                    model_name=CLIMATOLOGY_MODEL_NAME,
                    score_rows=[("brier", brier_c)],
                    is_test=pair["is_test"], now=now,
                )
                _audit(conn, qid, hm, CLIMATOLOGY_MODEL_NAME, metric,
                       [p_b, 1.0 - p_b], base_source, resolved, None, now)
                counters["scored_climatology"] += 1
                continue

            if metric not in SPD_METRICS:
                continue
            k = n_buckets_for(metric)
            if k == 0:
                continue
            j = _bucket_index(float(resolved), metric)
            if j is None:
                counters["skipped_bad_resolution"] += 1
                continue

            uniform = [1.0 / k] * k
            _write_scores(
                conn, question_id=qid, horizon_m=hm, metric=metric,
                model_name=UNIFORM_MODEL_NAME,
                score_rows=[
                    ("brier", _brier(uniform, j)),
                    ("log", _log_score(uniform, j)),
                    ("crps", _crps_like(uniform, j)),
                ],
                is_test=pair["is_test"], now=now,
            )
            _audit(conn, qid, hm, UNIFORM_MODEL_NAME, metric, uniform,
                   "uniform", resolved, j, now)
            counters["scored_uniform"] += 1

            if (str(pair["hazard_code"] or "").upper(), metric) in PERSISTENCE_PAIRS:
                if qid not in persist_cache:
                    persist_cache[qid] = last_observed_value(
                        conn, pair["iso3"], pair["hazard_code"], metric, anchor
                    )
                last = persist_cache[qid]
                vec = persistence_spd(last[0], metric) if last else None
                if vec is None:
                    counters["skipped_no_persistence"] += 1
                else:
                    _write_scores(
                        conn, question_id=qid, horizon_m=hm, metric=metric,
                        model_name=PERSISTENCE_MODEL_NAME,
                        score_rows=[
                            ("brier", _brier(vec, j)),
                            ("log", _log_score(vec, j)),
                            ("crps", _crps_like(vec, j)),
                        ],
                        is_test=pair["is_test"], now=now,
                    )
                    _audit(conn, qid, hm, PERSISTENCE_MODEL_NAME, metric, vec,
                           f"persistence:{last[2]}:{last[1]}={last[0]:g}",
                           resolved, j, now)
                    counters["scored_persistence"] += 1

            if (str(pair["hazard_code"] or "").upper(), metric) in LEVEL_VOLATILITY_PAIRS:
                if qid not in lv_cache:
                    known_at = _forecast_time(conn, qid) or anchor + "-01"
                    lv_cache[qid] = level_volatility_spds(
                        conn, pair["iso3"], anchor, known_at=known_at
                    )
                lv_spds, _lv_source, lv_detail = lv_cache[qid]
                lv_vec = lv_spds.get(hm)
                if not lv_vec or len(lv_vec) != k:
                    counters["skipped_no_level_volatility"] += 1
                else:
                    hd = (lv_detail.get("horizons") or {}).get(str(hm)) or {}
                    _write_scores(
                        conn, question_id=qid, horizon_m=hm, metric=metric,
                        model_name=LEVEL_VOLATILITY_MODEL_NAME,
                        score_rows=[
                            ("brier", _brier(lv_vec, j)),
                            ("log", _log_score(lv_vec, j)),
                            ("crps", _crps_like(lv_vec, j)),
                        ],
                        is_test=pair["is_test"], now=now,
                    )
                    _audit(
                        conn, qid, hm, LEVEL_VOLATILITY_MODEL_NAME, metric, lv_vec,
                        (
                            f"level_volatility:{lv_detail.get('level_month')}="
                            f"{float(lv_detail.get('level_value') or 0):g}"
                            f":gap{hd.get('gap_months')}:pairs{hd.get('n_pairs')}"
                            f"{':pooled' if hd.get('pooled') else ''}"
                            f":known_at={lv_detail.get('known_at')}"
                        ),
                        resolved, j, now,
                    )
                    counters["scored_level_volatility"] += 1

            if not base_probs:
                counters["skipped_no_baserate"] += 1
                LOGGER.info(
                    "score_baselines: no climatology for %s (%s/%s): %s",
                    qid, pair["hazard_code"], metric,
                    base_detail.get("reason", "unknown"),
                )
                continue
            if len(base_probs) != k:
                counters["skipped_no_baserate"] += 1
                LOGGER.warning(
                    "score_baselines: climatology bucket count %d != %d for %s; skipping.",
                    len(base_probs), k, qid,
                )
                continue
            _write_scores(
                conn, question_id=qid, horizon_m=hm, metric=metric,
                model_name=CLIMATOLOGY_MODEL_NAME,
                score_rows=[
                    ("brier", _brier(base_probs, j)),
                    ("log", _log_score(base_probs, j)),
                    ("crps", _crps_like(base_probs, j)),
                ],
                is_test=pair["is_test"], now=now,
            )
            _audit(conn, qid, hm, CLIMATOLOGY_MODEL_NAME, metric, base_probs,
                   base_source, resolved, j, now)
            counters["scored_climatology"] += 1

        LOGGER.info(
            "score_baselines: climatology=%d uniform=%d persistence=%d level_volatility=%d "
            "no_baserate=%d bad_resolution=%d",
            counters["scored_climatology"], counters["scored_uniform"],
            counters["scored_persistence"], counters["scored_level_volatility"],
            counters["skipped_no_baserate"], counters["skipped_bad_resolution"],
        )
        return counters
    finally:
        duckdb_io.close_db(conn)


def _audit(
    conn,
    question_id: str,
    horizon_m: int,
    model_name: str,
    metric: str,
    spd: List[float],
    source: str,
    resolved_value: Any,
    resolved_bucket: Optional[int],
    now: datetime,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO baseline_scored_forecasts
            (question_id, horizon_m, model_name, metric, spd_json,
             baserate_source, resolved_value, resolved_bucket, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            question_id, horizon_m, model_name, metric,
            json.dumps([round(float(p), 6) for p in spd]),
            source,
            float(resolved_value) if resolved_value is not None else None,
            resolved_bucket, now,
        ],
    )


def _get_db_url_from_config() -> str:
    cfg = load_cfg()
    app_cfg = cfg.get("app", {}) if isinstance(cfg, dict) else {}
    db_url = str(app_cfg.get("db_url", "")).strip()
    return db_url or duckdb_io.DEFAULT_DB_URL


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score climatology/uniform reference forecasts against resolutions.",
    )
    parser.add_argument("--db-url", default=None, help="DuckDB URL (default: from config).")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    score_baselines(db_url=args.db_url or _get_db_url_from_config())


if __name__ == "__main__":
    main()
