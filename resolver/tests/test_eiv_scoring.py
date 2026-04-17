"""Tests for EIV scoring (Phase 1), EMA centroid update (Phase 2),
and centroid version tracking."""

from __future__ import annotations

import math
from datetime import date

import pytest

duckdb = pytest.importorskip("duckdb")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db():
    """Create an in-memory DuckDB with required tables for EIV scoring."""
    conn = duckdb.connect(":memory:")

    conn.execute("""
        CREATE TABLE questions (
            question_id TEXT,
            hs_run_id TEXT,
            iso3 TEXT,
            hazard_code TEXT,
            metric TEXT,
            is_test BOOLEAN DEFAULT FALSE
        )
    """)
    conn.execute("""
        CREATE TABLE hs_runs (
            hs_run_id TEXT,
            is_test BOOLEAN DEFAULT FALSE
        )
    """)
    conn.execute("""
        CREATE TABLE resolutions (
            question_id TEXT,
            horizon_m INTEGER,
            value DOUBLE,
            is_test BOOLEAN DEFAULT FALSE
        )
    """)
    conn.execute("""
        CREATE TABLE forecasts_ensemble (
            question_id TEXT,
            horizon_m INTEGER,
            class_bin TEXT,
            p DOUBLE,
            is_test BOOLEAN DEFAULT FALSE
        )
    """)
    conn.execute("""
        CREATE TABLE forecasts_raw (
            question_id TEXT,
            month_index INTEGER,
            model_name TEXT,
            bucket_index INTEGER,
            probability DOUBLE,
            is_test BOOLEAN DEFAULT FALSE
        )
    """)
    conn.execute("""
        CREATE TABLE scores (
            question_id TEXT,
            horizon_m INTEGER,
            metric TEXT,
            score_type TEXT,
            model_name TEXT,
            value DOUBLE,
            created_at TIMESTAMP DEFAULT now(),
            is_test BOOLEAN DEFAULT FALSE
        )
    """)
    conn.execute("""
        CREATE TABLE bucket_centroids (
            hazard_code TEXT,
            metric TEXT,
            bucket_index INTEGER,
            centroid DOUBLE,
            as_of_month TEXT
        )
    """)
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
            is_test BOOLEAN DEFAULT FALSE,
            PRIMARY KEY (question_id, horizon_m, model_name)
        )
    """)
    return conn


# ---------------------------------------------------------------------------
# Test 1: test_eiv_basic
# ---------------------------------------------------------------------------

def test_eiv_basic():
    """Known SPD + centroids -> verify EIV, log_ratio_err, within_20pct."""
    from pythia.tools.compute_scores import _compute_eiv_for_question

    conn = _make_db()

    # Seed centroids: PA buckets [0, 30000, 150000, 375000, 700000]
    for idx, c in enumerate([0.0, 30_000.0, 150_000.0, 375_000.0, 700_000.0], start=1):
        conn.execute(
            "INSERT INTO bucket_centroids (hazard_code, metric, bucket_index, centroid, as_of_month) "
            "VALUES ('*', 'PA', ?, ?, NULL)",
            [idx, c],
        )

    # Ensemble SPD: [0.1, 0.2, 0.4, 0.2, 0.1]
    class_bins = ["<10k", "10k-<50k", "50k-<250k", "250k-<500k", ">=500k"]
    spd = [0.1, 0.2, 0.4, 0.2, 0.1]

    conn.execute(
        "INSERT INTO questions (question_id, hs_run_id, iso3, hazard_code, metric) VALUES ('q1', 'run1', 'SOM', 'ACE', 'PA')"
    )
    conn.execute("INSERT INTO hs_runs (hs_run_id) VALUES ('run1')")
    for cb, p in zip(class_bins, spd):
        conn.execute(
            "INSERT INTO forecasts_ensemble (question_id, horizon_m, class_bin, p) VALUES ('q1', 1, ?, ?)", [cb, p]
        )

    rows = _compute_eiv_for_question(
        conn,
        question_id="q1",
        horizon_m=1,
        metric="PA",
        hazard_code="ACE",
        resolved_value=180_000.0,
        class_bins=class_bins,
        _has_run_id=False,
    )

    assert len(rows) >= 1  # at least ensemble row

    # Ensemble row
    ensemble = [r for r in rows if r[3] == "__ensemble__"]
    assert len(ensemble) == 1
    row = ensemble[0]
    eiv = row[4]
    actual = row[5]
    log_ratio_err = row[6]
    within_20 = row[7]

    # EIV = 0*0.1 + 30000*0.2 + 150000*0.4 + 375000*0.2 + 700000*0.1 = 211000
    expected_eiv = (
        0.0 * 0.1 + 30_000.0 * 0.2 + 150_000.0 * 0.4
        + 375_000.0 * 0.2 + 700_000.0 * 0.1
    )
    assert eiv == pytest.approx(expected_eiv, rel=1e-6)
    assert expected_eiv == pytest.approx(211_000.0)

    # log_ratio_err = (ln(211000) - ln(180000))^2
    expected_lre = (math.log(211_000.0) - math.log(180_000.0)) ** 2
    assert log_ratio_err == pytest.approx(expected_lre, rel=1e-4)
    assert expected_lre == pytest.approx(0.025, abs=0.005)

    # within_20pct: |211000 - 180000| / 180000 = 0.172 -> True
    assert within_20 is True
    assert actual == 180_000.0

    conn.close()


# ---------------------------------------------------------------------------
# Test 2: test_eiv_floor
# ---------------------------------------------------------------------------

def test_eiv_floor():
    """resolved_value=0, near-zero SPD -> floors to 1.0, log_ratio_err finite."""
    from pythia.tools.compute_scores import _compute_eiv_for_question

    conn = _make_db()

    for idx, c in enumerate([0.0, 30_000.0, 150_000.0, 375_000.0, 700_000.0], start=1):
        conn.execute(
            "INSERT INTO bucket_centroids (hazard_code, metric, bucket_index, centroid, as_of_month) "
            "VALUES ('*', 'PA', ?, ?, NULL)",
            [idx, c],
        )

    class_bins = ["<10k", "10k-<50k", "50k-<250k", "250k-<500k", ">=500k"]
    spd = [0.9, 0.05, 0.03, 0.01, 0.01]

    conn.execute(
        "INSERT INTO questions (question_id, hs_run_id, iso3, hazard_code, metric) VALUES ('q2', 'run1', 'SOM', 'ACE', 'PA')"
    )
    conn.execute("INSERT INTO hs_runs (hs_run_id) VALUES ('run1')")
    for cb, p in zip(class_bins, spd):
        conn.execute(
            "INSERT INTO forecasts_ensemble (question_id, horizon_m, class_bin, p) VALUES ('q2', 1, ?, ?)", [cb, p]
        )

    rows = _compute_eiv_for_question(
        conn,
        question_id="q2",
        horizon_m=1,
        metric="PA",
        hazard_code="ACE",
        resolved_value=0.0,
        class_bins=class_bins,
        _has_run_id=False,
    )

    ensemble = [r for r in rows if r[3] == "__ensemble__"]
    assert len(ensemble) == 1
    row = ensemble[0]
    log_ratio_err = row[6]

    # log_ratio_err should be finite (no inf/nan from ln(0))
    assert math.isfinite(log_ratio_err)

    conn.close()


# ---------------------------------------------------------------------------
# Test 3: test_ema_no_activation
# ---------------------------------------------------------------------------

def test_ema_no_activation(caplog):
    """Only 5 resolutions per bucket -> EMA should NOT activate."""
    from pythia.tools.compute_bucket_centroids import (
        update_bucket_centroids_ema,
        MIN_RESOLUTIONS_PER_BUCKET,
    )

    conn = _make_db()

    # Seed 5 resolutions (below threshold of 10)
    for i in range(5):
        qid = f"q_ema_{i}"
        conn.execute(
            "INSERT INTO questions (question_id, hs_run_id, iso3, hazard_code, metric) VALUES (?, 'run1', 'SOM', 'ACE', 'PA')",
            [qid],
        )
        conn.execute(
            "INSERT INTO resolutions (question_id, horizon_m, value) VALUES (?, 1, ?)",
            [qid, 3000.0],  # bucket 1 for PA (<10k)
        )

    # We need to test the function via its db_url interface.
    # Since we can't pass a conn directly, test the SQL logic directly.
    from pythia.buckets import get_bucket_specs

    specs = list(get_bucket_specs("PA"))
    case_lines = []
    for spec in specs:
        if spec.upper is None:
            case_lines.append(f"WHEN v >= {float(spec.lower)} THEN {int(spec.idx)}")
        else:
            case_lines.append(
                f"WHEN v >= {float(spec.lower)} AND v < {float(spec.upper)} THEN {int(spec.idx)}"
            )
    case_sql = "\n                ".join(case_lines)

    rows = conn.execute(
        f"""
        WITH res_with_bucket AS (
            SELECT
                q.hazard_code,
                r.value AS v,
                CASE {case_sql} ELSE NULL END AS bucket_index
            FROM resolutions r
            JOIN questions q ON q.question_id = r.question_id
            WHERE upper(q.metric) = 'PA'
              AND r.value IS NOT NULL
        )
        SELECT
            COALESCE(UPPER(hazard_code), '*') AS hazard_code,
            bucket_index,
            COUNT(*) AS n_resolutions,
            AVG(v) AS empirical_mean
        FROM res_with_bucket
        WHERE bucket_index IS NOT NULL
        GROUP BY hazard_code, bucket_index
        HAVING COUNT(*) >= {MIN_RESOLUTIONS_PER_BUCKET}
        """,
    ).fetchall()

    # Should return 0 rows (5 < 10 threshold)
    assert len(rows) == 0

    conn.close()


# ---------------------------------------------------------------------------
# Test 4: test_ema_activation
# ---------------------------------------------------------------------------

def test_ema_activation():
    """15 resolutions in bucket 1 -> EMA produces new_centroid = 0.9*0.0 + 0.1*3000 = 300.0."""
    from pythia.tools.compute_bucket_centroids import (
        EMA_LEARNING_RATE,
        MIN_RESOLUTIONS_PER_BUCKET,
    )
    from pythia.buckets import get_bucket_specs

    conn = _make_db()

    # Seed default centroid for bucket 1 (PA): centroid=0.0
    conn.execute(
        "INSERT INTO bucket_centroids (hazard_code, metric, bucket_index, centroid, as_of_month) "
        "VALUES ('*', 'PA', 1, 0.0, NULL)"
    )

    # Seed 15 resolutions all in bucket 1 (value ~3000)
    for i in range(15):
        qid = f"q_ema_{i}"
        conn.execute(
            "INSERT INTO questions (question_id, hs_run_id, iso3, hazard_code, metric) VALUES (?, 'run1', 'SOM', 'ACE', 'PA')",
            [qid],
        )
        conn.execute(
            "INSERT INTO resolutions (question_id, horizon_m, value) VALUES (?, 1, ?)",
            [qid, 3000.0],
        )

    # Compute empirical mean for activated buckets
    specs = list(get_bucket_specs("PA"))
    case_lines = []
    for spec in specs:
        if spec.upper is None:
            case_lines.append(f"WHEN v >= {float(spec.lower)} THEN {int(spec.idx)}")
        else:
            case_lines.append(
                f"WHEN v >= {float(spec.lower)} AND v < {float(spec.upper)} THEN {int(spec.idx)}"
            )
    case_sql = "\n                ".join(case_lines)

    rows = conn.execute(
        f"""
        WITH res_with_bucket AS (
            SELECT
                q.hazard_code,
                r.value AS v,
                CASE {case_sql} ELSE NULL END AS bucket_index
            FROM resolutions r
            JOIN questions q ON q.question_id = r.question_id
            WHERE upper(q.metric) = 'PA'
              AND r.value IS NOT NULL
        )
        SELECT
            COALESCE(UPPER(hazard_code), '*') AS hazard_code,
            bucket_index,
            COUNT(*) AS n_resolutions,
            AVG(v) AS empirical_mean
        FROM res_with_bucket
        WHERE bucket_index IS NOT NULL
        GROUP BY hazard_code, bucket_index
        HAVING COUNT(*) >= {MIN_RESOLUTIONS_PER_BUCKET}
        """,
    ).fetchall()

    assert len(rows) == 1
    hazard_code, bucket_index, n_res, empirical_mean = rows[0]
    assert n_res == 15
    assert empirical_mean == pytest.approx(3000.0)

    # Simulate EMA update
    current_centroid = 0.0  # default for PA bucket 1
    new_centroid = (1.0 - EMA_LEARNING_RATE) * current_centroid + EMA_LEARNING_RATE * empirical_mean
    assert new_centroid == pytest.approx(300.0)

    # Write the new centroid
    as_of_month = "2026-03"
    conn.execute(
        "INSERT INTO bucket_centroids (hazard_code, metric, bucket_index, centroid, as_of_month) "
        "VALUES (?, 'PA', ?, ?, ?)",
        [hazard_code, int(bucket_index), new_centroid, as_of_month],
    )

    # Verify it was written with as_of_month set
    result = conn.execute(
        "SELECT centroid, as_of_month FROM bucket_centroids "
        "WHERE hazard_code = ? AND metric = 'PA' AND bucket_index = ? AND as_of_month IS NOT NULL",
        [hazard_code, int(bucket_index)],
    ).fetchone()
    assert result is not None
    assert result[0] == pytest.approx(300.0)
    assert result[1] == "2026-03"

    conn.close()


# ---------------------------------------------------------------------------
# Test 5: test_centroid_version_tracking
# ---------------------------------------------------------------------------

def test_centroid_version_tracking():
    """Write EMA centroids for 2026-01 then 2026-02. _load_centroids returns 2026-02."""
    from pythia.tools.compute_scores import _load_centroids

    conn = _make_db()

    # Seed all 5 default centroids (as_of_month=NULL)
    defaults = [0.0, 30_000.0, 150_000.0, 375_000.0, 700_000.0]
    for idx, c in enumerate(defaults, start=1):
        conn.execute(
            "INSERT INTO bucket_centroids (hazard_code, metric, bucket_index, centroid, as_of_month) "
            "VALUES ('*', 'PA', ?, ?, NULL)",
            [idx, c],
        )

    # Write 2026-01 EMA centroids (slightly different)
    jan_centroids = [100.0, 31_000.0, 152_000.0, 376_000.0, 710_000.0]
    for idx, c in enumerate(jan_centroids, start=1):
        conn.execute(
            "INSERT INTO bucket_centroids (hazard_code, metric, bucket_index, centroid, as_of_month) "
            "VALUES ('*', 'PA', ?, ?, '2026-01')",
            [idx, c],
        )

    # Write 2026-02 EMA centroids (slightly different again)
    feb_centroids = [200.0, 32_000.0, 155_000.0, 378_000.0, 720_000.0]
    for idx, c in enumerate(feb_centroids, start=1):
        conn.execute(
            "INSERT INTO bucket_centroids (hazard_code, metric, bucket_index, centroid, as_of_month) "
            "VALUES ('*', 'PA', ?, ?, '2026-02')",
            [idx, c],
        )

    # _load_centroids should return 2026-02 values (newest as_of_month)
    result = _load_centroids(conn, "ACE", "PA", 5)
    assert result == pytest.approx(feb_centroids)

    conn.close()


# ---------------------------------------------------------------------------
# Test 6: test_ema_repeat_run_under_unique_index
# ---------------------------------------------------------------------------

def test_ema_repeat_run_under_unique_index(tmp_path, monkeypatch):
    """Regression: two consecutive EMA runs (different months) must not violate
    the ux_bucket_centroids unique constraint on (hazard_code, metric,
    bucket_index).

    The production schema enforces this 3-column unique index. On the second
    EMA run, the existing row (from the first run) has a different as_of_month,
    and an INSERT of a new row for the same (hazard, metric, bucket) conflicts.
    """
    from datetime import date as _date

    from pythia.tools.compute_bucket_centroids import update_bucket_centroids_ema

    db_path = tmp_path / "ema_repeat.duckdb"
    db_url = f"duckdb:///{db_path}"

    # Build the DB with the same structure as _make_db() plus the production
    # unique index on bucket_centroids.
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("""
            CREATE TABLE questions (
                question_id TEXT,
                hs_run_id TEXT,
                iso3 TEXT,
                hazard_code TEXT,
                metric TEXT,
                is_test BOOLEAN DEFAULT FALSE
            )
        """)
        conn.execute("""
            CREATE TABLE resolutions (
                question_id TEXT,
                horizon_m INTEGER,
                value DOUBLE,
                is_test BOOLEAN DEFAULT FALSE
            )
        """)
        conn.execute("""
            CREATE TABLE bucket_centroids (
                hazard_code TEXT,
                metric TEXT,
                bucket_index INTEGER,
                centroid DOUBLE,
                as_of_month TEXT
            )
        """)
        conn.execute(
            "CREATE UNIQUE INDEX ux_bucket_centroids "
            "ON bucket_centroids (hazard_code, metric, bucket_index)"
        )

        # Seed default PA bucket centroids (wildcard hazard).
        defaults = [0.0, 30_000.0, 150_000.0, 375_000.0, 700_000.0]
        for idx, c in enumerate(defaults, start=1):
            conn.execute(
                "INSERT INTO bucket_centroids "
                "(hazard_code, metric, bucket_index, centroid, as_of_month) "
                "VALUES ('*', 'PA', ?, ?, NULL)",
                [idx, c],
            )

        # Seed >=10 resolutions for TC/PA bucket 1 (<10k) so EMA activates.
        for i in range(12):
            qid = f"q_tc_{i}"
            conn.execute(
                "INSERT INTO questions "
                "(question_id, hs_run_id, iso3, hazard_code, metric) "
                "VALUES (?, 'run1', 'PHL', 'TC', 'PA')",
                [qid],
            )
            conn.execute(
                "INSERT INTO resolutions (question_id, horizon_m, value) "
                "VALUES (?, 1, ?)",
                [qid, 3000.0],
            )
    finally:
        conn.close()

    # First EMA run (March) — should succeed and insert TC/PA/1 row.
    update_bucket_centroids_ema(db_url=db_url, metric="PA", as_of=_date(2026, 3, 1))

    # Second EMA run (April) — must not raise ConstraintException even though
    # a TC/PA/1 row already exists with as_of_month='2026-03'.
    update_bucket_centroids_ema(db_url=db_url, metric="PA", as_of=_date(2026, 4, 1))

    # Verify: exactly one TC/PA/1 row exists, stamped with the April as_of_month.
    conn2 = duckdb.connect(str(db_path))
    try:
        rows = conn2.execute(
            "SELECT as_of_month, centroid FROM bucket_centroids "
            "WHERE hazard_code = 'TC' AND metric = 'PA' AND bucket_index = 1"
        ).fetchall()
    finally:
        conn2.close()
    assert len(rows) == 1
    assert rows[0][0] == "2026-04"
