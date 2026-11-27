# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import argparse
import logging
import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from pythia.buckets import (
    bucket_schema_version,
    get_bucket_specs,
    labels_for,
    thresholds_for,
)
from pythia.config import load as load_cfg
from resolver.db import duckdb_io


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    LOGGER.addHandler(logging.NullHandler())

# Bucket boundaries and labels are single-sourced from pythia.buckets
# BUCKET_SPECS. The module-level names are preserved because
# generate_calibration_advice.py imports them.
PA_THRESHOLDS = thresholds_for("PA")
FATAL_THRESHOLDS = thresholds_for("FATALITIES")
PHASE3_THRESHOLDS = thresholds_for("PHASE3PLUS_IN_NEED")

SPD_CLASS_BINS_PA = labels_for("PA")
SPD_CLASS_BINS_FATALITIES = labels_for("FATALITIES")
SPD_CLASS_BINS_PHASE3 = labels_for("PHASE3PLUS_IN_NEED")

EPS = 1e-9


def _utcnow_naive() -> datetime:
    """Naive-UTC now (datetime.utcnow is deprecated; DB columns are naive)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


# Shared rollback-safe DuckDB helpers (see pythia/tools/_db_utils.py).
from pythia.tools._db_utils import (
    apply_compute_memory_guard,
    row_count as _row_count,
    table_exists as _table_exists,
)


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
    conn = duckdb_io.get_db(db_url or duckdb_io.DEFAULT_DB_URL)
    apply_compute_memory_guard(conn)
    return conn


def _close_db(conn) -> None:
    try:
        duckdb_io.close_db(conn)
    except Exception:
        pass


def _bucket_index(value: float, metric: str) -> Optional[int]:
    """Map a resolved value to a 0-based bucket index for the metric.

    Returns None (skip scoring) for negative or non-finite values — the
    old fallthrough sent them to the TOP bucket, scoring garbage input as
    a maximal-impact outcome.
    """

    v = float(value)
    if not math.isfinite(v) or v < 0.0:
        LOGGER.warning(
            "Resolved value %r for metric %s is negative/non-finite; skipping.",
            value, metric,
        )
        return None
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
    # v >= the last finite threshold (only reachable if thresholds were
    # redefined without a trailing +inf): top bucket.
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


def _rps(p: List[float], j: int) -> float:
    """Normalized Ranked Probability Score (the discrete form of CRPS).

    RPS = sum_{k=1..K-1} (F_k - H_k)^2 / (K - 1), where F is the forecast
    CDF over the ordered buckets and H is the outcome step CDF (0 below the
    observed bucket j, 1 at and above it). The K-th term is omitted — it is
    identically 0 when the probabilities sum to 1 (Epstein 1969; Murphy
    1971). Range 0 (perfect) to 1 (all mass in the bucket farthest from the
    outcome). Stored under score_type='crps' for continuity.
    """
    K = len(p)
    if K < 2:
        # A <2-bucket SPD is a misconfigured question/metric; a silent 0.0
        # would be indistinguishable from a perfect score.
        LOGGER.warning("RPS called with %d bucket(s); returning 0.0", K)
        return 0.0
    F = []
    s = 0.0
    for pk in p:
        s += pk
        F.append(s)
    sq = 0.0
    for k in range(K - 1):
        Hk = 0.0 if k < j else 1.0
        d = F[k] - Hk
        sq += d * d
    return sq / float(K - 1)


# Backward-compat alias (pythia/tools/score_views.py imports _crps_like).
_crps_like = _rps


# ---------------------------------------------------------------------------
# Censored scores: the outcome is a LOWER BOUND (Nov 2026)
# ---------------------------------------------------------------------------
#
# A lower-bound resolution in bucket k says the outcome is in bucket k or
# above. Scoring it as exactly bucket k rewards forecasts that put their
# mass on k and penalises those that put it above, which is the direction
# the truth lies. These three scores use only what the bound says, and are
# stored under their own score types (brier_censored, log_censored,
# crps_censored), never pooled with brier/log/crps. A bound in bucket 0
# says nothing (every outcome is at least 0) and is not scored.


def _tail(p: List[float], k: int) -> float:
    """P(outcome bucket >= k)."""
    return float(sum(p[k:]))


def _log_score_censored(p: List[float], k: int) -> float:
    """-log P(bucket >= k): the log score of the event the bound states."""
    return -math.log(max(_tail(p, k), EPS))


def _brier_censored(p: List[float], k: int) -> float:
    """Binary Brier of the event "outcome is in bucket k or above", which
    the bound says happened: (P(>= k) - 1)^2. Range 0..1, the binary scale,
    which is why it is never pooled with the multiclass SPD Brier (0..2)."""
    return (_tail(p, k) - 1.0) ** 2


def _rps_censored(p: List[float], k: int) -> float:
    """RPS over the thresholds the bound decides (interval form).

    The exact RPS sums (F_t - H_t)^2 over thresholds t = 0..K-2, with F_t
    the forecast CDF up to bucket t and H_t the outcome CDF (0 below the
    outcome bucket, 1 from it). A bound in bucket k fixes H_t = 0 for every
    t < k and leaves H_t unknown for t >= k, so only the first k thresholds
    are scored:

        RPS_cens = sum_{t=0..k-1} F_t^2 / (K - 1)

    Divided by K - 1 like the exact RPS, so it is the part of the exact RPS
    the bound decides, and never more than the exact RPS of any outcome at
    or above the bound.

    Worked example: PA has K = 6 buckets; forecast p = (0.1, 0.2, 0.3, 0.2,
    0.1, 0.1); the bound lies in bucket k = 2 (10,000 to 49,999). F_0 = 0.1,
    F_1 = 0.3, so RPS_cens = (0.01 + 0.09) / 5 = 0.02. Scored as exactly
    bucket 2 it would also add (1 - F_2)^2 + ... = (0.4^2 + 0.2^2 + 0.1^2) / 5
    = 0.042, a penalty for mass above the bound that the bound cannot
    support.
    """
    K = len(p)
    if K < 2:
        return 0.0
    F = 0.0
    sq = 0.0
    for t in range(min(k, K - 1)):
        F += p[t]
        sq += F * F
    return sq / float(K - 1)


#: Why a lower-bound row wrote no score.
CENSORED_SKIP_BUCKET_ZERO = "lower_bound_in_bucket_0_states_nothing"


def _migrate_resolutions_add_is_lower_bound(conn) -> int:
    """Add ``resolutions.is_lower_bound`` if missing and backfill it from
    ``source_desc`` (pythia/tools/scoring_class.py), in the style of
    :func:`_migrate_scores_add_run_id`. Returns the rows marked."""
    from pythia.tools.scoring_class import migrate_resolutions_is_lower_bound  # noqa: PLC0415

    n = migrate_resolutions_is_lower_bound(conn)
    LOGGER.info("resolutions: %d row(s) carry a lower-bound value.", n)
    return n


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


def _forecast_is_test_map(conn, has_run_id: bool) -> Dict[Tuple[str, Optional[str], str], bool]:
    """(question_id, run_id, model_name) -> whether any of its forecast rows is test.

    Empty when forecasts_raw lacks ``is_test`` (old DBs); callers then fall
    back to the question's own flag.
    """
    cols = _table_columns(conn, "forecasts_raw") if _table_exists(conn, "forecasts_raw") else set()
    if "is_test" not in cols:
        return {}
    rid = "run_id" if has_run_id and "run_id" in cols else "CAST(NULL AS TEXT)"
    try:
        rows = conn.execute(
            f"""
            SELECT question_id, {rid}, model_name, bool_or(COALESCE(is_test, FALSE))
            FROM forecasts_raw
            GROUP BY 1, 2, 3
            """
        ).fetchall()
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("forecast is_test lookup failed: %r; using question flags only.", exc)
        return {}
    return {(q, r, m): bool(t) for q, r, m, t in rows if t}


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
    model_name: str,
    run_id: Optional[str] = None,
    _has_run_id: bool = True,
) -> Optional[List[float]]:
    """Load one model's SPD vector from ``forecasts_raw``.

    Every scored entity — individual members AND the named aggregates
    (``ensemble_mean_v2`` / ``ensemble_bayesmc_v2`` / ``track2_flash``) —
    lives in forecasts_raw keyed by ``model_name``. The legacy path that
    read NULL-model rows from ``forecasts_ensemble`` via
    ``horizon_m/class_bin/p`` was retired (no live writer populated those
    columns; the NULL-model score convention with it).
    """
    # Only apply run_id filtering if the forecast table actually has the column.
    if _has_run_id:
        rid_clause, rid_params = _run_id_clause(run_id)
    else:
        rid_clause, rid_params = "", []

    # forecasts_raw stores data using month_index / bucket_index / probability.
    # Map bucket_index (1-based) back to class_bins positions.
    sql = f"""
      SELECT COALESCE(bucket_index, 0), COALESCE(probability, 0.0)
      FROM forecasts_raw
      WHERE question_id = ? AND month_index = ? AND model_name = ? {rid_clause}
    """
    params: List[object] = [question_id, horizon_m, model_name] + rid_params

    try:
        rows = conn.execute(sql, params).fetchall()
    except Exception as exc:
        LOGGER.error("SPD query failed for %s horizon %s: %r", question_id, horizon_m, exc)
        return None

    if not rows:
        return None

    # Stale-data guard: every live writer emits a full K-vector per month,
    # so a stored bucket set that doesn't match {1..K} means the forecast
    # was written under a different bucket scheme (e.g. a pre-restructure
    # 5-bucket DB resurfacing via canonical-DB discovery). Skip it loudly
    # rather than mis-score it against the wrong buckets.
    stored = {int(b) for b, _ in rows}
    expected = set(range(1, len(class_bins) + 1))
    if stored != expected:
        LOGGER.warning(
            "[warn] bucket-count mismatch for %s horizon %s (model=%s): "
            "stored bucket indices %s, expected 1..%d; skipping stale forecast.",
            question_id, horizon_m, model_name, sorted(stored), len(class_bins),
        )
        return None

    vec = [0.0] * len(class_bins)
    for bucket_idx, prob in rows:
        idx = int(bucket_idx) - 1  # bucket_index is 1-based
        if 0 <= idx < len(class_bins):
            vec[idx] = float(prob)

    total = float(sum(vec))
    if total <= 0.0:
        # A forecast that was never made (or stored as all-zero) must not
        # be scored as if the model held uniform beliefs.
        LOGGER.warning(
            "Zero-sum SPD for %s horizon %s (model=%s); skipping.",
            question_id, horizon_m, model_name,
        )
        return None
    return [float(x) / total for x in vec]


# ---------------------------------------------------------------------------
# Centroids
# ---------------------------------------------------------------------------

def _load_centroids(conn, hazard_code: str, metric: str, n_buckets: int) -> list:
    """Load centroids: hazard-specific -> wildcard -> BucketSpec defaults.

    Prefers the newest as_of_month (EMA-updated), falling back to seed/default
    rows (as_of_month IS NULL), then to BucketSpec hard-coded defaults.

    Rows stamped with a bucket-boundary ``schema_version`` different from the
    current one are refused — EMA centroids learned under an old boundary
    layout are meaningless for the new buckets. Unstamped (NULL) rows are
    grandfathered: they predate the stamp and get versioned on the next
    calibration cycle.
    """
    current_version = bucket_schema_version(metric)
    has_version_col = False
    try:
        cols = {
            str(r[1]).lower()
            for r in conn.execute("PRAGMA table_info('bucket_centroids')").fetchall()
        }
        has_version_col = "schema_version" in cols
    except Exception:
        pass
    version_filter = (
        " AND (schema_version IS NULL OR schema_version = ?)" if has_version_col else ""
    )
    for hc in [hazard_code.upper(), "*"]:
        params = [hc, metric.upper()]
        if has_version_col:
            params.append(current_version)
        rows = conn.execute(
            f"""
            SELECT bucket_index, centroid, as_of_month
            FROM bucket_centroids
            WHERE upper(hazard_code) = ? AND upper(metric) = ?{version_filter}
            ORDER BY as_of_month DESC NULLS LAST, bucket_index
            """,
            params,
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
    """Return the as_of_month of the newest centroid row, or 'default'.

    Mirrors :func:`_load_centroids` lookup order (hazard-specific, then
    wildcard) so the stamped version reflects the centroids actually used —
    a metric-only MAX would blame the wrong hazard's EMA update.
    """
    try:
        for hc in [hazard_code.upper(), "*"]:
            row = conn.execute(
                "SELECT MAX(as_of_month) FROM bucket_centroids "
                "WHERE upper(hazard_code) = ? AND upper(metric) = ? "
                "AND as_of_month IS NOT NULL",
                [hc, metric.upper()],
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

    # Per-model — includes the named aggregates (ensemble_mean_v2 etc.),
    # which are written to forecasts_raw alongside the members. The legacy
    # '__ensemble__' rows (from NULL-model forecasts_ensemble reads) were
    # retired with the NULL-model score convention.
    model_rows = conn.execute(
        f"SELECT DISTINCT model_name FROM forecasts_raw "
        f"WHERE question_id = ? AND month_index = ? {rid_clause}",
        [question_id, horizon_m] + rid_params,
    ).fetchall()
    for (model_name,) in model_rows:
        spd_m = _load_spd(
            conn, question_id=question_id, horizon_m=horizon_m,
            class_bins=class_bins, model_name=model_name,
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

def purge_orphan_scores(conn) -> dict:
    """Delete ``scores`` and ``eiv_scores`` rows whose (question, horizon)
    has no resolution row. Idempotent; never raises."""
    counts: dict = {}
    for table in ("scores", "eiv_scores"):
        try:
            if not _table_exists(conn, table) or not _table_exists(conn, "resolutions"):
                continue
            where = (
                f"NOT EXISTS (SELECT 1 FROM resolutions r WHERE r.question_id = {table}.question_id "
                f"AND r.horizon_m = {table}.horizon_m)"
            )
            n = int(conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}").fetchone()[0])
            if n:
                conn.execute(f"DELETE FROM {table} WHERE {where}")
            counts[table] = n
        except Exception as exc:  # noqa: BLE001 - a clean-up never stops scoring
            LOGGER.warning("orphan score purge on %s skipped: %r", table, exc)
    if any(counts.values()):
        LOGGER.info("compute_scores: removed scores with no resolution behind them: %s", counts)
    return counts


def compute_scores(db_url: str) -> dict:
    """Score every resolved (question, horizon). Returns the run's counts
    (orphans purged, score and EIV rows written) for the chain diagnostics."""
    report: dict = {"orphans_purged": {}, "score_rows_written": 0, "eiv_rows_written": 0}
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
            return report

        # A score whose resolution was withdrawn (compute_resolutions now
        # deletes an ACE/PA row a later rule no longer supports) is a score
        # against an outcome nobody holds. Remove it before anything reads it.
        report["orphans_purged"] = purge_orphan_scores(conn)

        r_count = _row_count(conn, "resolutions")
        if r_count == 0:
            LOGGER.info("compute_scores: resolutions table is empty; nothing to do.")
            return report

        report["lower_bound_resolutions"] = _migrate_resolutions_add_is_lower_bound(conn)
        report["censored"] = {"scored_rows": 0, "skipped": {}, "exact_rows_replaced": 0}
        from pythia.tools.scoring_class import CENSORED_HAZARD_METRICS  # noqa: PLC0415

        try:
            # Exact scores an earlier run wrote against a lower bound; this
            # run replaces them with censored ones.
            report["censored"]["exact_rows_replaced"] = int(conn.execute(
                """
                SELECT COUNT(*) FROM scores s
                JOIN resolutions r ON r.question_id = s.question_id AND r.horizon_m = s.horizon_m
                JOIN questions q ON q.question_id = s.question_id
                WHERE COALESCE(r.is_lower_bound, FALSE)
                  AND (upper(q.hazard_code), upper(q.metric)) IN ({pairs})
                  AND s.score_type IN ('brier', 'log', 'crps')
                """.format(pairs=", ".join(f"('{h}', '{m}')" for h, m in sorted(CENSORED_HAZARD_METRICS)))
            ).fetchone()[0])
        except Exception:  # noqa: BLE001 - a count only
            pass

        # Each resolution row now has its own horizon_m, so each horizon is
        # scored against its own calendar month's ground truth.
        q_sql = """
          SELECT
            q.question_id,
            q.iso3,
            q.hazard_code,
            upper(q.metric) AS metric,
            r.horizon_m,
            r.value AS resolved_value,
            COALESCE(r.is_lower_bound, FALSE) AS is_lower_bound
          FROM questions q
          JOIN resolutions r
            ON q.question_id = r.question_id
          JOIN hs_runs h ON q.hs_run_id = h.hs_run_id
          WHERE upper(q.metric) IN ('PA','FATALITIES','EVENT_OCCURRENCE','PHASE3PLUS_IN_NEED')
          ORDER BY q.question_id, r.horizon_m
        """
        rows_lb = conn.execute(q_sql).fetchall()
        lower_bound = {(r[0], r[4]): bool(r[6]) for r in rows_lb}
        qrows = [r[:6] for r in rows_lb]
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

        # A score is test data when its QUESTION is a test question OR the
        # FORECAST it scores came from a test run. Question ids are
        # epoch-keyed, so a production question can also carry a test run's
        # forecasts (same-epoch reuse); stamping from the question alone
        # wrote those scores as production. On the 2026-10-02 release that
        # was 473 score rows on 6 questions across 12 models, Sibyl included.
        forecast_is_test = _forecast_is_test_map(conn, has_run_id)

        def _row_is_test(qid: str, run_id: Optional[str], model_name: Optional[str]) -> bool:
            return bool(_is_test_cache.get(qid, False)) or forecast_is_test.get(
                (qid, run_id, model_name), False
            )

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

            # Binary EVENT_OCCURRENCE questions use Brier score directly.
            # Every scored entity is keyed by an explicit model_name — the
            # ensemble aggregations appear as 'ensemble_mean_v2' /
            # 'ensemble_bayesmc_v2' / 'track2_flash' rows in forecasts_raw
            # (the NULL-model score convention is retired: it duplicated the
            # named mean row and its value depended on write order).
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

                    # Score each model (members + named aggregates)
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
                            now = _utcnow_naive()
                            conn.execute(
                                """
                                INSERT INTO scores (question_id, horizon_m, metric, score_type,
                                                    model_name, value, run_id, created_at, is_test)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                [question_id, horizon_m, metric, "brier", model_name, brier_m, run_id, now,
                                 _row_is_test(question_id, run_id, model_name)],
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
            from pythia.tools.scoring_class import censored_applies  # noqa: PLC0415

            censored = censored_applies(hazard_code, metric,
                                        lower_bound.get((question_id, horizon_m)))
            if censored and j == 0:
                # Every outcome is at least bucket 0: the bound says nothing.
                # Clear any exact score an earlier run wrote, score nothing.
                skips = report["censored"]["skipped"]
                skips[CENSORED_SKIP_BUCKET_ZERO] = skips.get(CENSORED_SKIP_BUCKET_ZERO, 0) + 1
                conn.execute(
                    "DELETE FROM scores WHERE question_id = ? AND horizon_m = ? AND metric = ?",
                    [question_id, horizon_m, metric],
                )
                LOGGER.info("%s h%d: lower bound in bucket 0, not scored (%s).",
                            question_id, horizon_m, CENSORED_SKIP_BUCKET_ZERO)
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

                # Score each model (members + named aggregates) for this
                # specific (horizon, run_id). Aggregates are scored via
                # their explicit forecasts_raw rows (ensemble_mean_v2 etc.);
                # the NULL-model score convention is retired.
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
                        model_name=model_name,
                        run_id=run_id,
                        _has_run_id=has_run_id,
                    )
                    if not spd_m:
                        continue

                    if censored:
                        triples = (("brier_censored", _brier_censored(spd_m, j)),
                                   ("log_censored", _log_score_censored(spd_m, j)),
                                   ("crps_censored", _rps_censored(spd_m, j)))
                    else:
                        triples = (("brier", _brier(spd_m, j)), ("log", _log_score(spd_m, j)),
                                   ("crps", _rps(spd_m, j)))

                    conn.execute(
                        f"""
                        DELETE FROM scores
                        WHERE question_id = ? AND horizon_m = ? AND metric = ?
                          AND model_name = ? {rid_clause}
                        """,
                        [question_id, horizon_m, metric, model_name] + rid_params,
                    )
                    now = _utcnow_naive()
                    is_test_val = _row_is_test(question_id, run_id, model_name)
                    conn.executemany(
                        """
                        INSERT INTO scores (question_id, horizon_m, metric, score_type,
                                            model_name, value, run_id, created_at, is_test)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (question_id, horizon_m, metric, st, model_name, val, run_id, now, is_test_val)
                            for st, val in triples
                        ],
                    )
                    n_written += 3
                    if censored:
                        report["censored"]["scored_rows"] += 3

        LOGGER.info("compute_scores: wrote %d score rows.", n_written)
        report["score_rows_written"] = n_written

        # --- EIV scoring (Phase 1) ---
        eiv_rows = []
        from pythia.tools.scoring_class import censored_applies  # noqa: PLC0415

        for question_id, iso3, hazard_code, metric, horizon_m, resolved_value in qrows:
            class_bins = _class_bins(metric, hazard_code)
            if not class_bins:
                continue
            if censored_applies(hazard_code, metric, lower_bound.get((question_id, horizon_m))):
                # A lower bound is not the outcome an EIV error is measured
                # against; it would read every forecast above it as wrong.
                if _table_exists(conn, "eiv_scores"):
                    conn.execute("DELETE FROM eiv_scores WHERE question_id = ? AND horizon_m = ?",
                                 [question_id, horizon_m])
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
            now = _utcnow_naive()
            conn.executemany(
                """
                INSERT INTO eiv_scores (question_id, horizon_m, metric, model_name,
                                        eiv_forecast, actual_value, log_ratio_err,
                                        within_20pct, centroid_version, run_id, created_at,
                                        is_test)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                # row: (qid, horizon_m, metric, model, ..., run_id)
                [(*row, now, _row_is_test(row[0], row[9], row[3])) for row in eiv_rows],
            )
            LOGGER.info("compute_scores: wrote %d EIV score rows.", len(eiv_rows))
        report["eiv_rows_written"] = len(eiv_rows)
        return report
    finally:
        _close_db(conn)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute SPD scores for PA/FATALITIES forecasts.")
    parser.add_argument(
        "--db-url",
        default=None,
        help="DuckDB URL (default: app.db_url from pythia.config, or resolver default)",
    )
    parser.add_argument("--report-json", default=None,
                        help="Write the run's counts here (scripts/ci/chain_diagnostics.py)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    db_url = args.db_url or _get_db_url_from_config()
    report = compute_scores(db_url=db_url)
    if args.report_json:
        from pythia.tools.compute_resolutions import write_report_json

        write_report_json(args.report_json, report or {})


if __name__ == "__main__":
    main()
