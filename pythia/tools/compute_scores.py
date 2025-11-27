from __future__ import annotations

import argparse
import logging
import math
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from pythia.config import load as load_cfg
from resolver.db import duckdb_io


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    LOGGER.addHandler(logging.NullHandler())

PA_THRESHOLDS = [0.0, 10_000.0, 50_000.0, 250_000.0, 500_000.0, float("inf")]
FATAL_THRESHOLDS = [0.0, 5.0, 25.0, 100.0, 500.0, float("inf")]

SPD_CLASS_BINS_PA = ["<10k", "10k-<50k", "50k-<250k", "250k-<500k", ">=500k"]
SPD_CLASS_BINS_FATALITIES = ["<5", "5-<25", "25-<100", "100-<500", ">=500"]

EPS = 1e-9


def _get_db_url_from_config() -> str:
    cfg = load_cfg()
    app_cfg = cfg.get("app", {}) if isinstance(cfg, dict) else {}
    db_url = str(app_cfg.get("db_url", "")).strip()
    if not db_url:
        db_url = duckdb_io.DEFAULT_DB_URL
        LOGGER.warning("app.db_url missing in config; falling back to %s", db_url)
    else:
        LOGGER.info("Using app.db_url from config: %s", db_url)
    return db_url


def _open_db(db_url: str | None):
    if not duckdb_io.DUCKDB_AVAILABLE:
        raise RuntimeError(duckdb_io.duckdb_unavailable_reason())
    return duckdb_io.get_db(db_url or duckdb_io.DEFAULT_DB_URL)


def _close_db(conn) -> None:
    try:
        duckdb_io.close_db(conn)
    except Exception:
        pass


def _bucket_index(value: float, metric: str) -> Optional[int]:
    """Map a resolved value to a bucket index [0..4] based on metric."""

    v = float(value)
    metric_up = (metric or "").upper()
    if metric_up == "PA":
        thr = PA_THRESHOLDS
    elif metric_up == "FATALITIES":
        thr = FATAL_THRESHOLDS
    else:
        return None

    for i in range(len(thr) - 1):
        if thr[i] <= v < thr[i + 1]:
            return i
    return len(thr) - 2


def _class_bins(metric: str, hazard_code: Optional[str]) -> Optional[Sequence[str]]:
    metric_up = (metric or "").upper()
    if metric_up == "FATALITIES":
        return SPD_CLASS_BINS_FATALITIES
    if metric_up == "PA":
        return SPD_CLASS_BINS_PA
    return None


def _brier(p: List[float], j: int) -> float:
    b = 0.0
    for k, pk in enumerate(p):
        target = 1.0 if k == j else 0.0
        d = pk - target
        b += d * d
    return b


def _log_score(p: List[float], j: int) -> float:
    pj = max(float(p[j]), EPS)
    return -math.log(pj)


def _crps_like(p: List[float], j: int) -> float:
    K = len(p)
    F = []
    s = 0.0
    for pk in p:
        s += pk
        F.append(s)
    sq = 0.0
    for k in range(K):
        Hk = 0.0 if k < j else 1.0
        d = F[k] - Hk
        sq += d * d
    return sq / float(K)


def _load_spd(
    conn,
    *,
    question_id: str,
    horizon_m: int,
    class_bins: Sequence[str],
    table: str,
    model_name: Optional[str] = None,
) -> Optional[List[float]]:
    if table == "ensemble":
        sql = """
          SELECT class_bin, p
          FROM forecasts_ensemble
          WHERE question_id = ? AND horizon_m = ?
        """
        params: List[object] = [question_id, horizon_m]
    else:
        sql = """
          SELECT class_bin, p
          FROM forecasts_raw
          WHERE question_id = ? AND horizon_m = ? AND model_name = ?
        """
        params = [question_id, horizon_m, model_name]

    try:
        rows = conn.execute(sql, params).fetchall()
    except Exception as exc:
        LOGGER.error("SPD query failed for %s horizon %s: %r", question_id, horizon_m, exc)
        return None

    if not rows:
        return None

    by_bin: Dict[str, float] = {cb: float(p) for cb, p in rows}
    vec = [by_bin.get(cb, 0.0) for cb in class_bins]
    total = float(sum(vec))
    if total <= 0.0:
        return [1.0 / float(len(class_bins)) for _ in class_bins]
    return [float(x) / total for x in vec]


def compute_scores(db_url: str) -> None:
    conn = _open_db(db_url)

    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scores (
              question_id TEXT,
              horizon_m INTEGER,
              metric TEXT,
              score_type TEXT,
              model_name TEXT,
              value DOUBLE,
              created_at TIMESTAMP DEFAULT now()
            )
            """
        )

        q_sql = """
          SELECT
            q.question_id,
            q.iso3,
            q.hazard_code,
            upper(q.metric) AS metric,
            q.target_month,
            r.value AS resolved_value
          FROM questions q
          JOIN resolutions r
            ON q.question_id = r.question_id
           AND q.target_month = r.observed_month
          WHERE upper(q.metric) IN ('PA','FATALITIES')
          ORDER BY q.question_id
        """
        qrows = conn.execute(q_sql).fetchall()
        LOGGER.info("Found %d question_ids with resolutions for scoring.", len(qrows))

        n_written = 0

        for question_id, iso3, hazard_code, metric, target_month, resolved_value in qrows:
            class_bins = _class_bins(metric, hazard_code)
            if not class_bins:
                LOGGER.debug("Unsupported metric %s for question %s; skipping.", metric, question_id)
                continue
            j = _bucket_index(resolved_value, metric)
            if j is None:
                continue

            for horizon_m in range(1, 7):
                spd_e = _load_spd(
                    conn,
                    question_id=question_id,
                    horizon_m=horizon_m,
                    class_bins=class_bins,
                    table="ensemble",
                )
                if spd_e:
                    brier_e = _brier(spd_e, j)
                    log_e = _log_score(spd_e, j)
                    crps_e = _crps_like(spd_e, j)

                    conn.execute(
                        """
                        DELETE FROM scores
                        WHERE question_id = ? AND horizon_m = ? AND metric = ? AND model_name IS NULL
                        """,
                        [question_id, horizon_m, metric],
                    )
                    now = datetime.utcnow()
                    conn.executemany(
                        """
                        INSERT INTO scores (question_id, horizon_m, metric, score_type, model_name, value, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (question_id, horizon_m, metric, "brier", None, brier_e, now),
                            (question_id, horizon_m, metric, "log", None, log_e, now),
                            (question_id, horizon_m, metric, "crps", None, crps_e, now),
                        ],
                    )
                    n_written += 3

                model_rows = conn.execute(
                    """
                      SELECT DISTINCT model_name
                      FROM forecasts_raw
                      WHERE question_id = ? AND horizon_m = ?
                      ORDER BY model_name
                    """,
                    [question_id, horizon_m],
                ).fetchall()
                for (model_name,) in model_rows:
                    spd_m = _load_spd(
                        conn,
                        question_id=question_id,
                        horizon_m=horizon_m,
                        class_bins=class_bins,
                        table="raw",
                        model_name=model_name,
                    )
                    if not spd_m:
                        continue

                    brier_m = _brier(spd_m, j)
                    log_m = _log_score(spd_m, j)
                    crps_m = _crps_like(spd_m, j)

                    conn.execute(
                        """
                        DELETE FROM scores
                        WHERE question_id = ? AND horizon_m = ? AND metric = ? AND model_name = ?
                        """,
                        [question_id, horizon_m, metric, model_name],
                    )
                    now = datetime.utcnow()
                    conn.executemany(
                        """
                        INSERT INTO scores (question_id, horizon_m, metric, score_type, model_name, value, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (question_id, horizon_m, metric, "brier", model_name, brier_m, now),
                            (question_id, horizon_m, metric, "log", model_name, log_m, now),
                            (question_id, horizon_m, metric, "crps", model_name, crps_m, now),
                        ],
                    )
                    n_written += 3

        LOGGER.info("compute_scores: wrote %d score rows.", n_written)
    finally:
        _close_db(conn)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute SPD scores for PA/FATALITIES forecasts.")
    parser.add_argument(
        "--db-url",
        default=None,
        help="DuckDB URL (default: app.db_url from pythia.config, or resolver default)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    db_url = args.db_url or _get_db_url_from_config()
    compute_scores(db_url=db_url)


if __name__ == "__main__":
    main()
