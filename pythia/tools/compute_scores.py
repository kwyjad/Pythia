# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import argparse
import logging
import math
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from pythia.buckets import get_bucket_specs
from pythia.config import load as load_cfg
from resolver.db import duckdb_io


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    LOGGER.addHandler(logging.NullHandler())

PA_THRESHOLDS = [0.0, 10_000.0, 50_000.0, 250_000.0, 500_000.0, float("inf")]
FATAL_THRESHOLDS = [0.0, 5.0, 25.0, 100.0, 500.0, float("inf")]

SPD_CLASS_BINS_PA = ["<10k", "10k-<50k", "50k-<250k", "250k-<500k", ">=500k"]
SPD_CLASS_BINS_FATALITIES = ["<5", "5-<25", "25-<100", "100-<500", ">=500"]
PHASE3_THRESHOLDS = [0.0, 100_000.0, 1_000_000.0, 5_000_000.0, 15_000_000.0, float("inf")]
SPD_CLASS_BINS_PHASE3 = ["<100k", "100k-<1M", "1M-<5M", "5M-<15M", ">=15M"]

EPS = 1e-9


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
    elif metric_up == "PHASE3PLUS_IN_NEED":
        thr = PHASE3_THRESHOLDS
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
    if metric_up == "PHASE3PLUS_IN_NEED":
        return SPD_CLASS_BINS_PHASE3
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


# ---------------------------------------------------------------------------
# run_id helpers and migrations
# ---------------------------------------------------------------------------

def _run_id_clause(run_id: Optional[str]) -> Tuple[str, List[object]]:
    """Return (SQL fragment, params) for run_id filtering."""
    if run_id is not None:
        return "AND run_id = ?", [run_id]
    return "AND run_id IS NULL", []


def _table_columns(conn, name: str) -> set:
    """Return the set of column names for a table."""
    try:
        rows = conn.execute(f"PRAGMA table_info('{name}')").fetchall()
        return {r[1] for r in rows}
    except Exception:
        return set()


def _forecast_tables_have_run_id(conn) -> bool:
    """Check if forecasts_ensemble has a run_id column (backward compat for old DBs)."""
    return "run_id" in _table_columns(conn, "forecasts_ensemble")


def _migrate_scores_add_run_id(conn) -> None:
    """Add run_id column to scores table if missing."""
    cols = _table_columns(conn, "scores")
    if "run_id" not in cols:
        try:
            conn.execute("ALTER TABLE scores ADD COLUMN run_id TEXT")
            LOGGER.info("scores: added run_id column.")
        except Exception as exc:
            LOGGER.warning("scores: failed to add run_id column: %r", exc)


def _migrate_eiv_scores_add_run_id(conn) -> None:
    """Add run_id column to eiv_scores, removing PK if necessary (DuckDB limitation)."""
    if not _table_exists(conn, "eiv_scores"):
        return
    cols = _table_columns(conn, "eiv_scores")
    if "run_id" in cols:
        # run_id already exists, but PK may still be present from an
        # earlier ALTER TABLE migration that didn't use CTAS.
        _migrate_eiv_scores_remove_pk(conn)
        return

    # Check if a PK constraint exists (DuckDB cannot DROP CONSTRAINT for PK)
    has_pk = False
    try:
        pk_rows = conn.execute(
            "SELECT constraint_name FROM information_schema.table_constraints "
            "WHERE table_name = 'eiv_scores' AND constraint_type = 'PRIMARY KEY'"
        ).fetchall()
        has_pk = len(pk_rows) > 0
    except Exception:
        pass

    if has_pk:
        # CTAS migration: copy data -> drop original (cascades PK) -> recreate -> insert back
        LOGGER.info("eiv_scores: migrating via CTAS to remove PK and add run_id.")
        try:
            conn.execute(
                "CREATE TABLE _eiv_scores_tmp AS "
                "SELECT question_id, horizon_m, metric, model_name, "
                "       eiv_forecast, actual_value, log_ratio_err, "
                "       within_20pct, centroid_version, created_at "
                "FROM eiv_scores"
            )
            conn.execute("DROP TABLE eiv_scores")
            conn.execute("""
                CREATE TABLE eiv_scores (
                    question_id TEXT,
                    horizon_m INTEGER,
                    metric TEXT,
                    model_name TEXT,
                    eiv_forecast DOUBLE,
                    actual_value DOUBLE,
                    log_ratio_err DOUBLE,
                    within_20pct BOOLEAN,
                    centroid_version TEXT,
                    created_at TIMESTAMP DEFAULT now(),
                    run_id TEXT
                )
            """)
            conn.execute(
                "INSERT INTO eiv_scores "
                "(question_id, horizon_m, metric, model_name, eiv_forecast, "
                " actual_value, log_ratio_err, within_20pct, centroid_version, created_at) "
                "SELECT question_id, horizon_m, metric, model_name, eiv_forecast, "
                "       actual_value, log_ratio_err, within_20pct, centroid_version, created_at "
                "FROM _eiv_scores_tmp"
            )
            conn.execute("DROP TABLE _eiv_scores_tmp")
            LOGGER.info("eiv_scores: CTAS migration complete.")
        except Exception as exc:
            LOGGER.warning("eiv_scores: CTAS migration failed: %r", exc)
            # Try to recover temp table if it exists
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS eiv_scores AS SELECT * FROM _eiv_scores_tmp"
                )
                conn.execute("DROP TABLE IF EXISTS _eiv_scores_tmp")
            except Exception:
                pass
    else:
        # No PK — simple ALTER TABLE
        try:
            conn.execute("ALTER TABLE eiv_scores ADD COLUMN run_id TEXT")
            LOGGER.info("eiv_scores: added run_id column.")
        except Exception as exc:
            LOGGER.warning("eiv_scores: failed to add run_id column: %r", exc)


def _migrate_eiv_scores_remove_pk(conn) -> None:
    """Remove PK from eiv_scores via CTAS if one exists (DuckDB limitation)."""
    has_pk = False
    try:
        pk_rows = conn.execute(
            "SELECT constraint_name FROM information_schema.table_constraints "
            "WHERE table_name = 'eiv_scores' AND constraint_type = 'PRIMARY KEY'"
        ).fetchall()
        has_pk = len(pk_rows) > 0
    except Exception as exc:
        LOGGER.debug("eiv_scores: PK check failed: %r", exc)
        return

    if not has_pk:
        return

    LOGGER.info("eiv_scores: removing stale PK via CTAS migration.")
    try:
        conn.execute("CREATE TABLE _eiv_scores_tmp AS SELECT * FROM eiv_scores")
        conn.execute("DROP TABLE eiv_scores")
        conn.execute("""
            CREATE TABLE eiv_scores (
                question_id TEXT,
                horizon_m INTEGER,
                metric TEXT,
                model_name TEXT,
                eiv_forecast DOUBLE,
                actual_value DOUBLE,
                log_ratio_err DOUBLE,
                within_20pct BOOLEAN,
                centroid_version TEXT,
                created_at TIMESTAMP DEFAULT now(),
                run_id TEXT
            )
        """)
        # Use column-safe insert (handle missing columns in old data)
        tmp_cols = _table_columns(conn, "_eiv_scores_tmp")
        base_cols = [
            "question_id", "horizon_m", "metric", "model_name",
            "eiv_forecast", "actual_value", "log_ratio_err",
            "within_20pct", "centroid_version", "created_at",
        ]
        select_parts = []
        insert_cols = list(base_cols)
        for c in base_cols:
            select_parts.append(c if c in tmp_cols else f"NULL AS {c}")
        if "run_id" in tmp_cols:
            insert_cols.append("run_id")
            select_parts.append("run_id")
        conn.execute(
            f"INSERT INTO eiv_scores ({', '.join(insert_cols)}) "
            f"SELECT {', '.join(select_parts)} FROM _eiv_scores_tmp"
        )
        conn.execute("DROP TABLE _eiv_scores_tmp")
        LOGGER.info("eiv_scores: PK removal CTAS migration complete.")
    except Exception as exc:
        LOGGER.warning("eiv_scores: PK removal CTAS migration failed: %r", exc)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS eiv_scores AS SELECT * FROM _eiv_scores_tmp"
            )
            conn.execute("DROP TABLE IF EXISTS _eiv_scores_tmp")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# SPD loading
# ---------------------------------------------------------------------------

def _load_spd(
    conn,
    *,
    question_id: str,
    horizon_m: int,
    class_bins: Sequence[str],
    table: str,
    model_name: Optional[str] = None,
    run_id: Optional[str] = None,
    _has_run_id: bool = True,
) -> Optional[List[float]]:
    # Only apply run_id filtering if the forecast table actually has the column.
    if _has_run_id:
        rid_clause, rid_params = _run_id_clause(run_id)
    else:
        rid_clause, rid_params = "", []

    if table == "ensemble":
        sql = f"""
          SELECT class_bin, p
          FROM forecasts_ensemble
          WHERE question_id = ? AND horizon_m = ? {rid_clause}
        """
        params: List[object] = [question_id, horizon_m] + rid_params

        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception as exc:
            LOGGER.error("SPD query failed for %s horizon %s: %r", question_id, horizon_m, exc)
            return None

        if not rows:
            return None

        by_bin: Dict[str, float] = {cb: float(p) for cb, p in rows}
        vec = [by_bin.get(cb, 0.0) for cb in class_bins]
    else:
        # forecasts_raw stores data using month_index / bucket_index / probability.
        # Map bucket_index (1-based) back to class_bins positions.
        sql = f"""
          SELECT COALESCE(bucket_index, 0), COALESCE(probability, 0.0)
          FROM forecasts_raw
          WHERE question_id = ? AND month_index = ? AND model_name = ? {rid_clause}
        """
        params = [question_id, horizon_m, model_name] + rid_params

        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception as exc:
            LOGGER.error("SPD query failed for %s horizon %s: %r", question_id, horizon_m, exc)
            return None

        if not rows:
            return None

        vec = [0.0] * len(class_bins)
        for bucket_idx, prob in rows:
            idx = int(bucket_idx) - 1  # bucket_index is 1-based
            if 0 <= idx < len(class_bins):
                vec[idx] = float(prob)

    total = float(sum(vec))
    if total <= 0.0:
        return [1.0 / float(len(class_bins)) for _ in class_bins]
    return [float(x) / total for x in vec]


# ---------------------------------------------------------------------------
# Centroids
# ---------------------------------------------------------------------------

def _load_centroids(conn, hazard_code: str, metric: str, n_buckets: int) -> list:
    """Load centroids: hazard-specific -> wildcard -> BucketSpec defaults.

    Prefers the newest as_of_month (EMA-updated), falling back to seed/default
    rows (as_of_month IS NULL), then to BucketSpec hard-coded defaults.
    """
    for hc in [hazard_code.upper(), "*"]:
        rows = conn.execute(
            """
            SELECT bucket_index, centroid, as_of_month
            FROM bucket_centroids
            WHERE upper(hazard_code) = ? AND upper(metric) = ?
            ORDER BY as_of_month DESC NULLS LAST, bucket_index
            """,
            [hc, metric.upper()],
        ).fetchall()
        if not rows:
            continue
        # Pick the newest as_of_month (first row determines it)
        best_aom = rows[0][2]
        centroids = [0.0] * n_buckets
        found = 0
        for bi, c, aom in rows:
            if aom != best_aom:
                continue
            idx = int(bi) - 1
            if 0 <= idx < n_buckets:
                centroids[idx] = float(c)
                found += 1
        if found >= n_buckets:
            return centroids

    # Fall back to BucketSpec defaults
    specs = list(get_bucket_specs(metric))
    if len(specs) >= n_buckets:
        return [float(s.centroid) for s in specs[:n_buckets]]
    return [0.0] * n_buckets


def _get_centroid_version(conn, hazard_code: str, metric: str) -> str:
    """Return the as_of_month of the newest centroid row, or 'default'."""
    try:
        row = conn.execute(
            "SELECT MAX(as_of_month) FROM bucket_centroids "
            "WHERE upper(metric) = ? AND as_of_month IS NOT NULL",
            [metric.upper()],
        ).fetchone()
        if row and row[0]:
            return str(row[0])
    except Exception:
        pass
    return "default"


# ---------------------------------------------------------------------------
# EIV scoring
# ---------------------------------------------------------------------------

def _compute_eiv_for_question(
    conn,
    question_id: str,
    horizon_m: int,
    metric: str,
    hazard_code: str,
    resolved_value: float,
    class_bins: Sequence[str],
    run_id: Optional[str] = None,
    _has_run_id: bool = True,
) -> list:
    """Compute EIV scores for ensemble + per-model for one (question, horizon, run_id).

    Returns list of tuples ready for INSERT (includes run_id as last element).
    """
    rows_to_insert = []
    n_buckets = len(class_bins)

    centroids = _load_centroids(conn, hazard_code, metric, n_buckets)
    centroid_version = _get_centroid_version(conn, hazard_code, metric)

    def _eiv_and_error(spd_vec):
        eiv = sum(p * c for p, c in zip(spd_vec, centroids))
        actual = max(resolved_value, 1.0)   # floor at 1.0 to avoid ln(0)
        eiv_safe = max(eiv, 1.0)             # same floor
        log_ratio = (math.log(eiv_safe) - math.log(actual)) ** 2
        within_20 = abs(eiv - resolved_value) / actual <= 0.20
        return eiv, log_ratio, within_20

    if _has_run_id:
        rid_clause, rid_params = _run_id_clause(run_id)
    else:
        rid_clause, rid_params = "", []

    # Ensemble
    spd_e = _load_spd(
        conn, question_id=question_id, horizon_m=horizon_m,
        class_bins=class_bins, table="ensemble", run_id=run_id,
        _has_run_id=_has_run_id,
    )
    if spd_e:
        eiv, log_err, w20 = _eiv_and_error(spd_e)
        rows_to_insert.append((
            question_id, horizon_m, metric, "__ensemble__",
            eiv, resolved_value, log_err, w20, centroid_version, run_id,
        ))

    # Per-model
    model_rows = conn.execute(
        f"SELECT DISTINCT model_name FROM forecasts_raw "
        f"WHERE question_id = ? AND month_index = ? {rid_clause}",
        [question_id, horizon_m] + rid_params,
    ).fetchall()
    for (model_name,) in model_rows:
        spd_m = _load_spd(
            conn, question_id=question_id, horizon_m=horizon_m,
            class_bins=class_bins, table="raw", model_name=model_name,
            run_id=run_id, _has_run_id=_has_run_id,
        )
        if spd_m:
            eiv, log_err, w20 = _eiv_and_error(spd_m)
            rows_to_insert.append((
                question_id, horizon_m, metric, model_name,
                eiv, resolved_value, log_err, w20, centroid_version, run_id,
            ))

    return rows_to_insert


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

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
              run_id TEXT,
              created_at TIMESTAMP DEFAULT now(),
              is_test BOOLEAN DEFAULT FALSE
            )
            """
        )
        # Migration: add run_id to existing scores tables that lack it.
        _migrate_scores_add_run_id(conn)
        # Migration: add run_id to eiv_scores (CTAS if PK exists).
        _migrate_eiv_scores_add_run_id(conn)

        # Migration: add is_test column to scores and eiv_scores if missing.
        for _tbl in ("scores", "eiv_scores"):
            if _table_exists(conn, _tbl) and "is_test" not in _table_columns(conn, _tbl):
                try:
                    conn.execute(f"ALTER TABLE {_tbl} ADD COLUMN is_test BOOLEAN DEFAULT FALSE")
                    LOGGER.info("%s: added is_test column.", _tbl)
                except Exception as _exc:
                    LOGGER.warning("%s: failed to add is_test column: %r", _tbl, _exc)

        # Detect if forecast tables have run_id (backward compat for old DBs/tests).
        has_run_id = _forecast_tables_have_run_id(conn)

        # Early exit if resolutions table doesn't exist or is empty
        if not _table_exists(conn, "resolutions"):
            LOGGER.info("compute_scores: resolutions table not found; nothing to do.")
            return

        r_count = _row_count(conn, "resolutions")
        if r_count == 0:
            LOGGER.info("compute_scores: resolutions table is empty; nothing to do.")
            return

        # Each resolution row now has its own horizon_m, so each horizon is
        # scored against its own calendar month's ground truth.
        q_sql = """
          SELECT
            q.question_id,
            q.iso3,
            q.hazard_code,
            upper(q.metric) AS metric,
            r.horizon_m,
            r.value AS resolved_value
          FROM questions q
          JOIN resolutions r
            ON q.question_id = r.question_id
          JOIN hs_runs h ON q.hs_run_id = h.hs_run_id
          WHERE upper(q.metric) IN ('PA','FATALITIES','EVENT_OCCURRENCE','PHASE3PLUS_IN_NEED')
          ORDER BY q.question_id, r.horizon_m
        """
        qrows = conn.execute(q_sql).fetchall()
        LOGGER.info("Found %d (question, horizon) pairs with resolutions for scoring.", len(qrows))

        # Discover all distinct run_ids per (question_id, horizon_m) from
        # both forecasts_ensemble AND forecasts_raw.  Track 2 questions have
        # forecasts_raw rows but NO forecasts_ensemble rows, so we must UNION
        # both tables to avoid silently dropping them from scoring.
        run_ids_map: Dict[Tuple[str, int], List[Optional[str]]] = defaultdict(list)
        if has_run_id:
            try:
                rid_rows = conn.execute(
                    """
                    SELECT DISTINCT question_id, horizon_m, run_id
                    FROM (
                        SELECT question_id, horizon_m, run_id
                        FROM forecasts_ensemble
                        UNION
                        SELECT question_id, month_index AS horizon_m, run_id
                        FROM forecasts_raw
                    ) combined
                    WHERE (question_id, horizon_m) IN (
                        SELECT q.question_id, r.horizon_m
                        FROM questions q
                        JOIN resolutions r ON q.question_id = r.question_id
                        JOIN hs_runs h ON q.hs_run_id = h.hs_run_id
                        WHERE upper(q.metric) IN ('PA','FATALITIES','EVENT_OCCURRENCE','PHASE3PLUS_IN_NEED')
                    )
                    """
                ).fetchall()
                for qid, hm, rid in rid_rows:
                    run_ids_map[(qid, hm)].append(rid)
            except Exception as exc:
                LOGGER.warning("run_id discovery failed: %r; scoring without run_id.", exc)

        n_written = 0

        # Pre-fetch is_test for all questions to avoid repeated lookups.
        _is_test_cache: Dict[str, bool] = {}

        for question_id, iso3, hazard_code, metric, horizon_m, resolved_value in qrows:
            # Guard against NULL resolution values (should not happen with
            # source-aware null handling, but defend against edge cases).
            if resolved_value is None:
                continue

            if question_id not in _is_test_cache:
                try:
                    _qt = conn.execute(
                        "SELECT COALESCE(is_test, FALSE) FROM questions WHERE question_id = ?",
                        [question_id],
                    ).fetchone()
                    _is_test_cache[question_id] = _qt[0] if _qt else False
                except Exception:
                    _is_test_cache[question_id] = False
            is_test_val = _is_test_cache[question_id]

            # Binary EVENT_OCCURRENCE questions use Brier score directly
            is_binary = (metric or "").upper() == "EVENT_OCCURRENCE"
            if is_binary:
                outcome = float(resolved_value)
                rid_list = run_ids_map.get((question_id, horizon_m)) or [None]
                for run_id in rid_list:
                    rid_clause, rid_params = _run_id_clause(run_id)
                    if has_run_id:
                        fe_rid_clause, fe_rid_params = rid_clause, rid_params
                    else:
                        fe_rid_clause, fe_rid_params = "", []

                    # Score ensemble (bucket_1 = P(yes) in binary convention)
                    try:
                        ens_row = conn.execute(
                            f"""
                            SELECT probability FROM forecasts_ensemble
                            WHERE question_id = ? AND month_index = ? AND bucket_index = 1
                            {fe_rid_clause}
                            ORDER BY created_at DESC LIMIT 1
                            """,
                            [question_id, horizon_m] + fe_rid_params,
                        ).fetchone()
                    except Exception:
                        ens_row = None

                    if ens_row and ens_row[0] is not None:
                        p_yes = float(ens_row[0])
                        brier_val = (p_yes - outcome) ** 2
                        conn.execute(
                            f"""
                            DELETE FROM scores
                            WHERE question_id = ? AND horizon_m = ? AND metric = ?
                              AND model_name IS NULL {rid_clause}
                            """,
                            [question_id, horizon_m, metric] + rid_params,
                        )
                        now = datetime.utcnow()
                        conn.execute(
                            """
                            INSERT INTO scores (question_id, horizon_m, metric, score_type,
                                                model_name, value, run_id, created_at, is_test)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            [question_id, horizon_m, metric, "brier", None, brier_val, run_id, now, is_test_val],
                        )
                        n_written += 1

                    # Score individual models
                    try:
                        model_rows = conn.execute(
                            f"""
                            SELECT DISTINCT model_name FROM forecasts_raw
                            WHERE question_id = ? AND month_index = ? {fe_rid_clause}
                            ORDER BY model_name
                            """,
                            [question_id, horizon_m] + fe_rid_params,
                        ).fetchall()
                    except Exception:
                        model_rows = []

                    for (model_name,) in model_rows:
                        try:
                            raw_row = conn.execute(
                                f"""
                                SELECT probability FROM forecasts_raw
                                WHERE question_id = ? AND month_index = ? AND bucket_index = 1
                                  AND model_name = ? {fe_rid_clause}
                                ORDER BY rowid DESC LIMIT 1
                                """,
                                [question_id, horizon_m, model_name] + fe_rid_params,
                            ).fetchone()
                        except Exception:
                            raw_row = None

                        if raw_row and raw_row[0] is not None:
                            p_yes_m = float(raw_row[0])
                            brier_m = (p_yes_m - outcome) ** 2
                            conn.execute(
                                f"""
                                DELETE FROM scores
                                WHERE question_id = ? AND horizon_m = ? AND metric = ?
                                  AND model_name = ? {rid_clause}
                                """,
                                [question_id, horizon_m, metric, model_name] + rid_params,
                            )
                            now = datetime.utcnow()
                            conn.execute(
                                """
                                INSERT INTO scores (question_id, horizon_m, metric, score_type,
                                                    model_name, value, run_id, created_at, is_test)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                [question_id, horizon_m, metric, "brier", model_name, brier_m, run_id, now, is_test_val],
                            )
                            n_written += 1
                continue  # skip SPD scoring for binary questions

            class_bins = _class_bins(metric, hazard_code)
            if not class_bins:
                LOGGER.debug("Unsupported metric %s for question %s; skipping.", metric, question_id)
                continue
            j = _bucket_index(resolved_value, metric)
            if j is None:
                continue

            # Score each run_id independently; default to [None] for unknown/old data.
            rid_list = run_ids_map.get((question_id, horizon_m)) or [None]

            for run_id in rid_list:
                rid_clause, rid_params = _run_id_clause(run_id)
                # For forecast table queries, only apply run_id filter if columns exist.
                if has_run_id:
                    fe_rid_clause, fe_rid_params = rid_clause, rid_params
                else:
                    fe_rid_clause, fe_rid_params = "", []

                # Score ensemble for this specific (horizon, run_id)
                spd_e = _load_spd(
                    conn,
                    question_id=question_id,
                    horizon_m=horizon_m,
                    class_bins=class_bins,
                    table="ensemble",
                    run_id=run_id,
                    _has_run_id=has_run_id,
                )
                if spd_e:
                    brier_e = _brier(spd_e, j)
                    log_e = _log_score(spd_e, j)
                    crps_e = _crps_like(spd_e, j)

                    conn.execute(
                        f"""
                        DELETE FROM scores
                        WHERE question_id = ? AND horizon_m = ? AND metric = ?
                          AND model_name IS NULL {rid_clause}
                        """,
                        [question_id, horizon_m, metric] + rid_params,
                    )
                    now = datetime.utcnow()
                    conn.executemany(
                        """
                        INSERT INTO scores (question_id, horizon_m, metric, score_type,
                                            model_name, value, run_id, created_at, is_test)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (question_id, horizon_m, metric, "brier", None, brier_e, run_id, now, is_test_val),
                            (question_id, horizon_m, metric, "log", None, log_e, run_id, now, is_test_val),
                            (question_id, horizon_m, metric, "crps", None, crps_e, run_id, now, is_test_val),
                        ],
                    )
                    n_written += 3

                # Score individual models for this specific (horizon, run_id)
                model_rows = conn.execute(
                    f"""
                      SELECT DISTINCT model_name
                      FROM forecasts_raw
                      WHERE question_id = ? AND month_index = ? {fe_rid_clause}
                      ORDER BY model_name
                    """,
                    [question_id, horizon_m] + fe_rid_params,
                ).fetchall()
                for (model_name,) in model_rows:
                    spd_m = _load_spd(
                        conn,
                        question_id=question_id,
                        horizon_m=horizon_m,
                        class_bins=class_bins,
                        table="raw",
                        model_name=model_name,
                        run_id=run_id,
                        _has_run_id=has_run_id,
                    )
                    if not spd_m:
                        continue

                    brier_m = _brier(spd_m, j)
                    log_m = _log_score(spd_m, j)
                    crps_m = _crps_like(spd_m, j)

                    conn.execute(
                        f"""
                        DELETE FROM scores
                        WHERE question_id = ? AND horizon_m = ? AND metric = ?
                          AND model_name = ? {rid_clause}
                        """,
                        [question_id, horizon_m, metric, model_name] + rid_params,
                    )
                    now = datetime.utcnow()
                    conn.executemany(
                        """
                        INSERT INTO scores (question_id, horizon_m, metric, score_type,
                                            model_name, value, run_id, created_at, is_test)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (question_id, horizon_m, metric, "brier", model_name, brier_m, run_id, now, is_test_val),
                            (question_id, horizon_m, metric, "log", model_name, log_m, run_id, now, is_test_val),
                            (question_id, horizon_m, metric, "crps", model_name, crps_m, run_id, now, is_test_val),
                        ],
                    )
                    n_written += 3

        LOGGER.info("compute_scores: wrote %d score rows.", n_written)

        # --- EIV scoring (Phase 1) ---
        eiv_rows = []
        for question_id, iso3, hazard_code, metric, horizon_m, resolved_value in qrows:
            class_bins = _class_bins(metric, hazard_code)
            if not class_bins:
                continue
            rid_list = run_ids_map.get((question_id, horizon_m)) or [None]
            for run_id in rid_list:
                eiv_rows.extend(_compute_eiv_for_question(
                    conn, question_id, horizon_m, metric, hazard_code,
                    resolved_value, class_bins, run_id=run_id,
                    _has_run_id=has_run_id,
                ))

        if eiv_rows:
            # Clear stale EIV rows for all resolved (question, run_id) combos.
            deleted_qids: set[str] = set()
            for row in eiv_rows:
                # row tuple: (qid, horizon_m, metric, model, eiv, actual, log_err, w20, cv, run_id)
                qid = row[0]
                if qid not in deleted_qids:
                    conn.execute(
                        "DELETE FROM eiv_scores WHERE question_id = ?",
                        [qid],
                    )
                    deleted_qids.add(qid)
            now = datetime.utcnow()
            conn.executemany(
                """
                INSERT INTO eiv_scores (question_id, horizon_m, metric, model_name,
                                        eiv_forecast, actual_value, log_ratio_err,
                                        within_20pct, centroid_version, run_id, created_at,
                                        is_test)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [(*row, now, _is_test_cache.get(row[0], False)) for row in eiv_rows],
            )
            LOGGER.info("compute_scores: wrote %d EIV score rows.", len(eiv_rows))
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
