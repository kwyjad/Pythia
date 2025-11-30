from __future__ import annotations

import argparse
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

from pythia.config import load as load_cfg
from resolver.db import duckdb_io


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    LOGGER.addHandler(logging.NullHandler())


def _table_exists(conn, name: str) -> bool:
    try:
        conn.execute(f"PRAGMA table_info('{name}')").fetchall()
        return True
    except Exception:
        return False


def _row_count(conn, name: str) -> int:
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0] or 0
    except Exception:
        return 0


MIN_QUESTIONS = 20
HALF_LIFE_MONTHS = 12.0
TEMP_SOFTMAX = 0.1


@dataclass
class Sample:
    question_key: Tuple[str, str, str, str]
    hazard_code: str
    metric: str
    model_name: Optional[str]
    score_type: str
    value: float
    observed_month: str


def _get_db_url_from_config() -> str:
    cfg = load_cfg()
    app_cfg = cfg.get("app", {}) if isinstance(cfg, dict) else {}
    db_url = str(app_cfg.get("db_url", "")).strip()
    if not db_url:
        db_url = duckdb_io.DEFAULT_DB_URL
        LOGGER.warning("app.db_url missing; falling back to %s", db_url)
    else:
        LOGGER.info("Using app.db_url from config: %s", db_url)
    return db_url


def _months_diff(as_of_month: str, obs_month: str) -> int:
    ay, am = map(int, as_of_month.split("-"))
    oy, om = map(int, obs_month.split("-"))
    return max(0, (ay - oy) * 12 + (am - om))


def _load_samples(conn, as_of_month: str) -> List[Sample]:
    sql = """
      SELECT
        s.question_id,
        s.horizon_m,
        s.metric,
        s.score_type,
        s.model_name,
        s.value,
        q.iso3,
        q.hazard_code,
        q.target_month,
        r.observed_month
      FROM scores s
      JOIN questions q ON q.question_id = s.question_id
      JOIN resolutions r
        ON r.question_id = s.question_id
       AND r.observed_month = q.target_month
      JOIN hs_runs h ON q.hs_run_id = h.hs_run_id
      WHERE q.target_month <= ?
    """
    rows = conn.execute(sql, [as_of_month]).fetchall()

    samples: List[Sample] = []
    for (
        question_id,
        horizon_m,
        metric,
        score_type,
        model_name,
        value,
        iso3,
        hazard_code,
        target_month,
        observed_month,
    ) in rows:
        hk = (str(iso3 or "").upper(), str(hazard_code or "").upper(), str(metric or "").upper(), str(target_month))
        samples.append(
            Sample(
                question_key=hk,
                hazard_code=str(hazard_code or "").upper(),
                metric=str(metric or "").upper(),
                model_name=model_name,
                score_type=str(score_type or "").lower(),
                value=float(value),
                observed_month=str(observed_month or target_month),
            )
        )
    LOGGER.info("Loaded %d scoring samples from scores.", len(samples))
    return samples


def _group_by_hazard_metric(samples: List[Sample]) -> Dict[Tuple[str, str], List[Sample]]:
    groups: Dict[Tuple[str, str], List[Sample]] = defaultdict(list)
    for s in samples:
        groups[(s.hazard_code, s.metric)].append(s)
    return groups


def _compute_weights_for_group(as_of_month: str, samples: List[Sample]) -> Tuple[List[Dict], str]:
    brier_samples = [s for s in samples if s.score_type == "brier"]
    if not brier_samples:
        return [], "No Brier samples available for calibration."

    question_keys = {s.question_key for s in brier_samples}
    n_questions = len(question_keys)
    if n_questions < MIN_QUESTIONS:
        return [], f"Insufficient resolved questions for calibration (got {n_questions}, need {MIN_QUESTIONS})."

    lambda_ = math.log(2.0) / HALF_LIFE_MONTHS

    agg: Dict[str, Dict[Optional[str], Tuple[float, float]]] = defaultdict(lambda: defaultdict(lambda: (0.0, 0.0)))
    counts_samples: Dict[Optional[str], int] = defaultdict(int)

    for s in samples:
        age_m = _months_diff(as_of_month, s.observed_month)
        w = math.exp(-lambda_ * age_m)
        cur_sum, cur_w = agg[s.score_type][s.model_name]
        agg[s.score_type][s.model_name] = (cur_sum + w * s.value, cur_w + w)
        if s.score_type == "brier":
            counts_samples[s.model_name] += 1

    avg_scores: Dict[str, Dict[Optional[str], float]] = defaultdict(dict)
    for score_type, per_model in agg.items():
        for model_name, (wsum, wtot) in per_model.items():
            if wtot > 0.0:
                avg_scores[score_type][model_name] = wsum / wtot

    brier_avgs = avg_scores.get("brier", {})
    if not brier_avgs:
        return [], "No Brier scores after aggregation."

    model_names = list(brier_avgs.keys())
    raw_vals: Dict[Optional[str], float] = {m: -brier_avgs[m] for m in model_names}
    max_raw = max(raw_vals.values())

    weights: Dict[Optional[str], float] = {}
    denom = 0.0
    for m, rv in raw_vals.items():
        x = math.exp((rv - max_raw) / TEMP_SOFTMAX)
        weights[m] = x
        denom += x

    if denom <= 0.0:
        k = len(model_names)
        return (
            [
                {
                    "model_name": m,
                    "weight": 1.0 / float(k),
                    "avg_brier": brier_avgs[m],
                    "avg_log": avg_scores.get("log", {}).get(m),
                    "avg_crps": avg_scores.get("crps", {}).get(m),
                    "n_samples": counts_samples.get(m, 0),
                    "n_questions": n_questions,
                }
                for m in model_names
            ],
            "Weights defaulted to uniform due to numerical instability.",
        )

    for m in model_names:
        weights[m] = weights[m] / denom

    weights_rows: List[Dict] = []
    for m in model_names:
        weights_rows.append(
            {
                "model_name": m,
                "weight": weights[m],
                "avg_brier": brier_avgs[m],
                "avg_log": avg_scores.get("log", {}).get(m),
                "avg_crps": avg_scores.get("crps", {}).get(m),
                "n_samples": counts_samples.get(m, 0),
                "n_questions": n_questions,
            }
        )

    sorted_models = sorted(model_names, key=lambda mm: brier_avgs[mm])
    best = sorted_models[0]
    worst = sorted_models[-1]

    advice_lines = [
        f"As of {as_of_month}, model '{best}' has the lowest (best) Brier score in this hazard/metric.",
    ]
    if len(sorted_models) > 1:
        advice_lines.append(
            f"Model '{worst}' has the highest (worst) Brier score; its forecasts should be down-weighted."
        )
    advice_lines.append(
        "Weights are computed via a time-decayed average of scores (half-life "
        f"{HALF_LIFE_MONTHS:.0f} months) and a softmax over negative Brier scores."
    )

    advice_text = " ".join(advice_lines)
    return weights_rows, advice_text


def compute_calibration_pythia(db_url: str, as_of: Optional[date] = None) -> None:
    if as_of is None:
        as_of = date.today()
    as_of_month = as_of.strftime("%Y-%m")

    if not duckdb_io.DUCKDB_AVAILABLE:
        raise RuntimeError(duckdb_io.duckdb_unavailable_reason())

    conn = duckdb_io.get_db(db_url or duckdb_io.DEFAULT_DB_URL)

    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS calibration_weights (
              as_of_month TEXT,
              hazard_code TEXT,
              metric TEXT,
              model_name TEXT,
              weight DOUBLE,
              n_questions INTEGER,
              n_samples INTEGER,
              avg_brier DOUBLE,
              avg_log DOUBLE,
              avg_crps DOUBLE,
              created_at TIMESTAMP DEFAULT now(),
              PRIMARY KEY (as_of_month, hazard_code, metric, model_name)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS calibration_advice (
              as_of_month TEXT,
              hazard_code TEXT,
              metric TEXT,
              advice TEXT,
              created_at TIMESTAMP DEFAULT now(),
              PRIMARY KEY (as_of_month, hazard_code, metric)
            )
            """
        )

        # Early exit if scores table doesn't exist or is empty
        if not _table_exists(conn, "scores"):
            LOGGER.info("compute_calibration_pythia: scores table not found; nothing to do.")
            return

        s_count = _row_count(conn, "scores")
        if s_count == 0:
            LOGGER.info("compute_calibration_pythia: scores table is empty; nothing to do.")
            return

        samples = _load_samples(conn, as_of_month)
        groups = _group_by_hazard_metric(samples)

        total_weight_rows = 0
        for (hazard_code, metric), group_samples in groups.items():
            LOGGER.info(
                "Computing calibration for hazard=%s metric=%s with %d samples.",
                hazard_code,
                metric,
                len(group_samples),
            )
            weight_rows, advice_text = _compute_weights_for_group(as_of_month, group_samples)
            if not weight_rows:
                LOGGER.info(
                    "Skipping calibration for hazard=%s metric=%s: %s",
                    hazard_code,
                    metric,
                    advice_text,
                )
                continue

            conn.execute(
                """
                DELETE FROM calibration_weights
                WHERE as_of_month = ? AND hazard_code = ? AND metric = ?
                """,
                [as_of_month, hazard_code, metric],
            )
            conn.execute(
                """
                DELETE FROM calibration_advice
                WHERE as_of_month = ? AND hazard_code = ? AND metric = ?
                """,
                [as_of_month, hazard_code, metric],
            )

            now = datetime.utcnow()
            conn.executemany(
                """
                INSERT INTO calibration_weights (
                  as_of_month, hazard_code, metric, model_name, weight,
                  n_questions, n_samples, avg_brier, avg_log, avg_crps, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        as_of_month,
                        hazard_code,
                        metric,
                        row["model_name"],
                        row["weight"],
                        row["n_questions"],
                        row["n_samples"],
                        row["avg_brier"],
                        row["avg_log"],
                        row["avg_crps"],
                        now,
                    )
                    for row in weight_rows
                ],
            )
            conn.execute(
                """
                INSERT INTO calibration_advice (
                  as_of_month, hazard_code, metric, advice, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [as_of_month, hazard_code, metric, advice_text, now],
            )

            total_weight_rows += len(weight_rows)

        LOGGER.info(
            "compute_calibration_pythia: wrote %d calibration_weights rows for as_of_month=%s.",
            total_weight_rows,
            as_of_month,
        )
    finally:
        duckdb_io.close_db(conn)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Pythia calibration weights/advice from scores.")
    parser.add_argument(
        "--db-url",
        default=None,
        help=(
            "DuckDB URL (default: app.db_url from pythia.config, or duckdb:///"
            "/data/resolver.duckdb)"
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    db_url = args.db_url or _get_db_url_from_config()
    compute_calibration_pythia(db_url=db_url)


if __name__ == "__main__":
    main()
