# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Generate concrete, hazard-specific calibration advice from scoring data.

Reads the scores, resolutions, forecasts_ensemble and forecasts_raw tables,
computes calibration findings per hazard/metric, and writes prompt-ready
advice text to the calibration_advice table.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pythia.config import load as load_cfg
from pythia.tools.compute_scores import (
    PA_THRESHOLDS,
    FATAL_THRESHOLDS,
    SPD_CLASS_BINS_PA,
    SPD_CLASS_BINS_FATALITIES,
    _bucket_index,
)
from resolver.db import duckdb_io

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    LOGGER.addHandler(logging.NullHandler())

MIN_QUESTIONS = 20
MAX_ADVICE_CHARS = 3800

HAZARD_LABELS = {
    "ACE": "ARMED CONFLICT",
    "FL": "FLOOD",
    "DR": "DROUGHT",
    "TC": "TROPICAL CYCLONE",
    "HW": "HEATWAVE",
    "DI": "DISPLACEMENT INFLOW",
}

METRIC_LABELS = {
    "FATALITIES": "FATALITIES",
    "PA": "PEOPLE AFFECTED",
}

# Hand-written seed advice for bootstrapping before enough scoring data exists.
SEED_ADVICE: Dict[Tuple[str, str], str] = {
    ("ACE", "FATALITIES"): (
        "TAIL COVERAGE:\n"
        "- Historical pattern: conflict fatality spikes (bucket 4-5, >=100 fatalities) "
        "occur ~15-20% of months in active-conflict countries.\n"
        "- Known bias: models assign <5% to top buckets even when RC is elevated.\n"
        "- ACTION: For active conflict zones, ensure bucket 4+5 combined >= 10%. "
        "When RC >= L2, bucket 4+5 combined should be >= 15%.\n\n"
        "BUCKET CALIBRATION:\n"
        "- Bucket 1 (<5 fatalities) is typically overweighted by 10-15pp in "
        "active conflict zones.\n"
        "- ACTION: In countries with recent conflict history, do not assign >60% "
        "to bucket 1 unless there is strong de-escalation evidence.\n\n"
        "HORIZON DIFFERENTIATION:\n"
        "- Conflict fatalities show meaningful month-to-month variation. Later "
        "horizons (months 4-6) should carry wider distributions.\n"
        "- ACTION: Avoid identical SPDs across all 6 months. Widen uncertainty "
        "for months 4-6 relative to months 1-2."
    ),
    ("ACE", "PA"): (
        "TAIL COVERAGE:\n"
        "- Displacement events reaching 250k+ (buckets 4-5) occur in ~5-10% "
        "of country-months for ACE hazard.\n"
        "- ACTION: Ensure non-trivial tail mass (>=3% combined for buckets 4-5) "
        "unless the country has no recent displacement.\n\n"
        "BUCKET CALIBRATION:\n"
        "- The 10k-50k range (bucket 2) is systematically under-predicted in "
        "countries with ongoing low-level displacement.\n"
        "- ACTION: In countries with established IDP populations, give bucket 2 "
        "at least 15% probability."
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _table_exists(conn: Any, name: str) -> bool:
    try:
        conn.execute(f"PRAGMA table_info('{name}')").fetchall()
        return True
    except Exception:
        return False


def _row_count(conn: Any, name: str) -> int:
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
        LOGGER.warning("app.db_url missing; falling back to %s", db_url)
    else:
        LOGGER.info("Using app.db_url from config: %s", db_url)
    return db_url


def _ensure_findings_json_column(conn: Any) -> None:
    """Add findings_json column if it doesn't exist yet."""
    existing = set()
    try:
        for row in conn.execute("PRAGMA table_info('calibration_advice')").fetchall():
            existing.add(str(row[1]).lower())
    except Exception:
        return
    if "findings_json" not in existing:
        try:
            conn.execute(
                "ALTER TABLE calibration_advice ADD COLUMN findings_json TEXT"
            )
        except Exception:
            pass


def _class_bins_for_metric(metric: str) -> List[str]:
    """Return ordered class_bin labels for the given metric."""
    m = (metric or "").upper()
    if m == "FATALITIES":
        return list(SPD_CLASS_BINS_FATALITIES)
    return list(SPD_CLASS_BINS_PA)


def _tail_bins(metric: str) -> Tuple[str, str]:
    """Return the two class_bin labels corresponding to buckets 4 and 5."""
    m = (metric or "").upper()
    if m == "FATALITIES":
        return ("100-<500", ">=500")
    return ("250k-<500k", ">=500k")


def _bucket_case_sql(metric: str) -> str:
    """Generate a SQL CASE expression to bucketify resolutions.value."""
    m = (metric or "").upper()
    if m == "FATALITIES":
        return """
            CASE
                WHEN r.value < 5 THEN 1
                WHEN r.value < 25 THEN 2
                WHEN r.value < 100 THEN 3
                WHEN r.value < 500 THEN 4
                ELSE 5
            END
        """
    return """
        CASE
            WHEN r.value < 10000 THEN 1
            WHEN r.value < 50000 THEN 2
            WHEN r.value < 250000 THEN 3
            WHEN r.value < 500000 THEN 4
            ELSE 5
        END
    """


def _bin_to_bucket_num(class_bin: str, metric: str) -> Optional[int]:
    """Map a class_bin label to bucket number (1-based)."""
    bins = _class_bins_for_metric(metric)
    try:
        return bins.index(class_bin) + 1
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def _discover_hazard_metric_pairs(
    conn: Any,
) -> List[Tuple[str, str]]:
    """Find all (hazard_code, metric) pairs with resolved questions."""
    sql = """
        SELECT DISTINCT upper(q.hazard_code) AS hc, upper(q.metric) AS m
        FROM questions q
        JOIN resolutions r ON r.question_id = q.question_id
        WHERE upper(q.metric) IN ('PA', 'FATALITIES')
        ORDER BY hc, m
    """
    rows = conn.execute(sql).fetchall()
    return [(r[0], r[1]) for r in rows]


def _count_resolved(conn: Any, hazard_code: str, metric: str) -> int:
    """Count distinct resolved questions for a hazard/metric pair."""
    sql = """
        SELECT COUNT(DISTINCT r.question_id)
        FROM resolutions r
        JOIN questions q ON q.question_id = r.question_id
        WHERE upper(q.hazard_code) = ? AND upper(q.metric) = ?
    """
    row = conn.execute(sql, [hazard_code.upper(), metric.upper()]).fetchone()
    return row[0] if row else 0


def _compute_tail_coverage(
    conn: Any, hazard_code: str, metric: str,
) -> Optional[Dict[str, Any]]:
    """Compare ensemble tail probabilities vs actual resolution rates."""
    hz = hazard_code.upper()
    m = metric.upper()
    tail_bin_1, tail_bin_2 = _tail_bins(m)
    bucket_case = _bucket_case_sql(m)

    sql = f"""
        WITH resolved_with_bucket AS (
            SELECT
                r.question_id,
                r.horizon_m,
                {bucket_case} AS resolved_bucket
            FROM resolutions r
            JOIN questions q ON q.question_id = r.question_id
            WHERE upper(q.hazard_code) = ?
              AND upper(q.metric) = ?
        ),
        ensemble_tail_probs AS (
            SELECT
                fe.question_id,
                fe.horizon_m,
                SUM(CASE WHEN fe.class_bin IN (?, ?) THEN fe.p ELSE 0 END) AS tail_prob
            FROM forecasts_ensemble fe
            JOIN questions q ON q.question_id = fe.question_id
            WHERE upper(q.hazard_code) = ?
              AND upper(q.metric) = ?
              AND fe.class_bin IS NOT NULL
              AND fe.p IS NOT NULL
            GROUP BY fe.question_id, fe.horizon_m
        )
        SELECT
            COUNT(*) AS n_forecasts,
            AVG(et.tail_prob) AS avg_assigned_tail,
            SUM(CASE WHEN et.tail_prob < 0.05 THEN 1 ELSE 0 END)::DOUBLE
                / GREATEST(COUNT(*), 1) AS frac_starved,
            SUM(CASE WHEN rb.resolved_bucket >= 4 THEN 1 ELSE 0 END)::DOUBLE
                / GREATEST(COUNT(*), 1) AS actual_tail_rate
        FROM ensemble_tail_probs et
        JOIN resolved_with_bucket rb
            ON rb.question_id = et.question_id AND rb.horizon_m = et.horizon_m
    """
    try:
        row = conn.execute(
            sql, [hz, m, tail_bin_1, tail_bin_2, hz, m]
        ).fetchone()
    except Exception as exc:
        LOGGER.warning("Tail coverage query failed for %s/%s: %s", hz, m, exc)
        return None

    if not row or row[0] == 0:
        return None

    return {
        "n_forecasts": row[0],
        "avg_assigned_tail": float(row[1] or 0),
        "frac_starved": float(row[2] or 0),
        "actual_tail_rate": float(row[3] or 0),
    }


def _compute_bucket_calibration(
    conn: Any, hazard_code: str, metric: str,
) -> Optional[List[Dict[str, Any]]]:
    """Per-bucket reliability: mean assigned probability vs resolution rate."""
    hz = hazard_code.upper()
    m = metric.upper()
    bins = _class_bins_for_metric(m)
    bucket_case = _bucket_case_sql(m)

    sql = f"""
        WITH resolved_with_bucket AS (
            SELECT
                r.question_id,
                r.horizon_m,
                {bucket_case} AS resolved_bucket
            FROM resolutions r
            JOIN questions q ON q.question_id = r.question_id
            WHERE upper(q.hazard_code) = ?
              AND upper(q.metric) = ?
        )
        SELECT
            fe.class_bin,
            AVG(fe.p) AS mean_assigned,
            AVG(CASE WHEN rb.resolved_bucket = (
                CASE fe.class_bin
                    {' '.join(f"WHEN '{b}' THEN {i+1}" for i, b in enumerate(bins))}
                END
            ) THEN 1.0 ELSE 0.0 END) AS actual_rate,
            COUNT(*) AS n_samples
        FROM forecasts_ensemble fe
        JOIN resolved_with_bucket rb
            ON rb.question_id = fe.question_id AND rb.horizon_m = fe.horizon_m
        JOIN questions q ON q.question_id = fe.question_id
        WHERE upper(q.hazard_code) = ?
          AND upper(q.metric) = ?
          AND fe.class_bin IS NOT NULL
          AND fe.p IS NOT NULL
        GROUP BY fe.class_bin
        ORDER BY fe.class_bin
    """
    try:
        rows = conn.execute(sql, [hz, m, hz, m]).fetchall()
    except Exception as exc:
        LOGGER.warning("Bucket calibration query failed for %s/%s: %s", hz, m, exc)
        return None

    if not rows:
        return None

    result = []
    for row in rows:
        cb = str(row[0])
        if cb not in bins:
            continue
        result.append({
            "class_bin": cb,
            "mean_assigned": float(row[1] or 0),
            "actual_rate": float(row[2] or 0),
            "n_samples": int(row[3] or 0),
        })

    # Sort by bin order
    bin_order = {b: i for i, b in enumerate(bins)}
    result.sort(key=lambda x: bin_order.get(x["class_bin"], 99))
    return result if result else None


def _compute_per_model_brier(
    conn: Any, hazard_code: str, metric: str,
) -> Optional[Dict[str, Any]]:
    """Per-model and ensemble Brier scores from the scores table."""
    hz = hazard_code.upper()
    m = metric.upper()

    sql = """
        SELECT
            COALESCE(s.model_name, '__ensemble__') AS mn,
            AVG(s.value) AS avg_brier,
            COUNT(*) AS n_scores
        FROM scores s
        JOIN questions q ON q.question_id = s.question_id
        WHERE s.score_type = 'brier'
          AND upper(q.hazard_code) = ?
          AND upper(q.metric) = ?
        GROUP BY mn
        ORDER BY avg_brier ASC
    """
    try:
        rows = conn.execute(sql, [hz, m]).fetchall()
    except Exception as exc:
        LOGGER.warning("Per-model Brier query failed for %s/%s: %s", hz, m, exc)
        return None

    if not rows:
        return None

    models = []
    ensemble_brier = None
    for row in rows:
        name = str(row[0])
        brier = float(row[1] or 0)
        n = int(row[2] or 0)
        if name == "__ensemble__":
            ensemble_brier = brier
        else:
            models.append({"name": name, "brier": brier, "n": n})

    if not models:
        return None

    models.sort(key=lambda x: x["brier"])
    return {
        "ensemble_brier": ensemble_brier,
        "best": models[0],
        "worst": models[-1],
        "all_models": models,
    }


def _compute_month_position_bias(
    conn: Any, hazard_code: str, metric: str,
) -> Optional[Dict[str, Any]]:
    """Check if models differentiate SPDs across horizons 1-6."""
    hz = hazard_code.upper()
    m = metric.upper()

    # Use forecasts_raw with legacy columns for per-model data
    if not _table_exists(conn, "forecasts_raw"):
        return None

    sql = """
        SELECT
            fr.month_index,
            AVG(fr.probability) AS avg_prob_top_bucket
        FROM forecasts_raw fr
        JOIN questions q ON q.question_id = fr.question_id
        WHERE upper(q.hazard_code) = ?
          AND upper(q.metric) = ?
          AND fr.bucket_index = 5
          AND fr.month_index BETWEEN 1 AND 6
        GROUP BY fr.month_index
        ORDER BY fr.month_index
    """
    try:
        rows = conn.execute(sql, [hz, m]).fetchall()
    except Exception as exc:
        LOGGER.warning("Month position query failed for %s/%s: %s", hz, m, exc)
        return None

    if not rows or len(rows) < 2:
        return None

    probs = [float(r[1] or 0) for r in rows]
    mean_prob = sum(probs) / len(probs)
    variance = sum((p - mean_prob) ** 2 for p in probs) / len(probs)
    stdev = variance ** 0.5

    by_month = {int(r[0]): float(r[1] or 0) for r in rows}

    return {
        "by_month": by_month,
        "stdev": stdev,
        "mean_top_bucket_prob": mean_prob,
        "flat": stdev < 0.02,
    }


def _compute_rc_conditional(
    conn: Any, hazard_code: str, metric: str,
) -> Optional[Dict[str, Any]]:
    """Compare tail mass at RC L0 vs RC L2+."""
    hz = hazard_code.upper()
    m = metric.upper()

    if not _table_exists(conn, "hs_triage"):
        return None

    tail_bin_1, tail_bin_2 = _tail_bins(m)

    sql = """
        WITH forecast_tail AS (
            SELECT
                fe.question_id,
                fe.horizon_m,
                SUM(CASE WHEN fe.class_bin IN (?, ?) THEN fe.p ELSE 0 END) AS tail_prob
            FROM forecasts_ensemble fe
            JOIN questions q ON q.question_id = fe.question_id
            WHERE upper(q.hazard_code) = ?
              AND upper(q.metric) = ?
              AND fe.class_bin IS NOT NULL
              AND fe.p IS NOT NULL
            GROUP BY fe.question_id, fe.horizon_m
        )
        SELECT
            CASE WHEN ht.regime_change_level >= 2 THEN 'rc_elevated'
                 ELSE 'rc_normal' END AS rc_group,
            AVG(ft.tail_prob) AS avg_tail,
            COUNT(*) AS n
        FROM forecast_tail ft
        JOIN questions q ON q.question_id = ft.question_id
        JOIN hs_triage ht ON ht.run_id = q.hs_run_id
            AND ht.iso3 = q.iso3
            AND ht.hazard_code = q.hazard_code
        WHERE ht.regime_change_level IS NOT NULL
        GROUP BY rc_group
    """
    try:
        rows = conn.execute(sql, [tail_bin_1, tail_bin_2, hz, m]).fetchall()
    except Exception as exc:
        LOGGER.warning("RC-conditional query failed for %s/%s: %s", hz, m, exc)
        return None

    if not rows:
        return None

    result: Dict[str, Any] = {}
    for row in rows:
        group = str(row[0])
        result[group] = {"avg_tail": float(row[1] or 0), "n": int(row[2] or 0)}

    if not result:
        return None
    return result


# ---------------------------------------------------------------------------
# Advice formatting
# ---------------------------------------------------------------------------

def _format_advice(
    findings: Dict[str, Any],
    hazard_code: str,
    metric: str,
    as_of_month: str,
    n_resolved: int,
) -> str:
    """Format calibration findings into prompt-ready advice text."""
    hz_label = HAZARD_LABELS.get(hazard_code, hazard_code)
    m_label = METRIC_LABELS.get(metric, metric)
    key = f"{hazard_code}/{metric}"
    lines: List[str] = []

    header = f"{hz_label} \u2014 {m_label} ({key}, {as_of_month}, n={n_resolved} resolved):"
    if hazard_code == "*":
        header = f"OVERALL ENSEMBLE ({as_of_month}):"
    lines.append(header)
    lines.append("")

    # Tail coverage
    tc = findings.get("tail_coverage")
    if tc:
        avg_tail = tc["avg_assigned_tail"] * 100
        actual = tc["actual_tail_rate"] * 100
        frac_starved = tc["frac_starved"] * 100

        lines.append(
            f"TAIL: Ensemble assigns avg {avg_tail:.1f}% to buckets 4-5; "
            f"actual resolution rate is {actual:.1f}%."
        )
        if actual > avg_tail + 3:
            gap = actual - avg_tail
            target = max(avg_tail + gap * 0.6, actual * 0.7)
            lines.append(
                f"ACTION: Increase bucket 4+5 combined probability to at least "
                f"{target:.0f}% unless you have strong evidence against tail outcomes."
            )
        elif avg_tail > actual + 5:
            lines.append(
                "ACTION: Tail probabilities may be slightly overweighted. "
                "Verify with reference class before assigning high tail mass."
            )
        else:
            lines.append("Tail calibration is reasonable. No correction needed.")
        lines.append("")

    # Bucket calibration
    bc = findings.get("bucket_calibration")
    if bc:
        lines.append("CALIBRATION BY BUCKET:")
        worst_gap = 0.0
        worst_bucket = ""
        for entry in bc:
            assigned = entry["mean_assigned"] * 100
            actual = entry["actual_rate"] * 100
            gap = assigned - actual
            direction = "overconfident" if gap > 2 else ("underconfident" if gap < -2 else "well calibrated")
            suffix = f"by {abs(gap):.0f}pp" if abs(gap) > 2 else ""
            lines.append(
                f"  {entry['class_bin']:>12s}: assigned {assigned:.0f}%, "
                f"actual {actual:.0f}% -> {direction} {suffix}".rstrip()
            )
            if abs(gap) > abs(worst_gap):
                worst_gap = gap
                worst_bucket = entry["class_bin"]
        if abs(worst_gap) > 3:
            if worst_gap > 0:
                lines.append(
                    f"ACTION: Reduce probability on bucket {worst_bucket} by ~{abs(worst_gap):.0f}pp "
                    "and redistribute to under-predicted buckets."
                )
            else:
                lines.append(
                    f"ACTION: Increase probability on bucket {worst_bucket} by ~{abs(worst_gap):.0f}pp."
                )
        lines.append("")

    # Per-model Brier
    mb = findings.get("per_model_brier")
    if mb:
        parts = []
        if mb.get("best"):
            parts.append(f"{mb['best']['name']} (Brier {mb['best']['brier']:.3f}, best)")
        if mb.get("worst"):
            parts.append(f"{mb['worst']['name']} (Brier {mb['worst']['brier']:.3f}, worst)")
        if parts:
            lines.append(f"MODELS: {', '.join(parts)}.")
        if mb.get("ensemble_brier") is not None:
            lines.append(
                f"Ensemble mean Brier: {mb['ensemble_brier']:.3f}."
            )
        lines.append("")

    # Month position bias
    mpb = findings.get("month_position_bias")
    if mpb:
        stdev = mpb["stdev"]
        if mpb["flat"]:
            lines.append(
                f"HORIZONS: Models show <{stdev*100:.1f}pp variation in top-bucket "
                f"probability across months 1-6 (nearly flat)."
            )
            lines.append(
                "ACTION: Differentiate across horizons. Widen uncertainty for "
                "later months (4-6); do not copy month 1 SPD to all months."
            )
        else:
            lines.append(
                f"HORIZONS: Models differentiate across months (stdev {stdev*100:.1f}pp). "
                "Good horizon differentiation."
            )
        lines.append("")

    # RC-conditional
    rc = findings.get("rc_conditional")
    if rc:
        normal = rc.get("rc_normal", {})
        elevated = rc.get("rc_elevated", {})
        if normal and elevated:
            norm_tail = normal["avg_tail"] * 100
            elev_tail = elevated["avg_tail"] * 100
            lines.append(
                f"RC-CONDITIONAL: At RC L0-1, avg tail mass = {norm_tail:.1f}%. "
                f"At RC L2+, avg tail mass = {elev_tail:.1f}%."
            )
            if elev_tail < norm_tail + 5:
                lines.append(
                    "ACTION: When RC >= L2, tail mass should be meaningfully "
                    "higher than at RC L0. Ensure at least 5pp increase in "
                    "bucket 4+5 combined probability."
                )
            else:
                lines.append("RC-conditional adjustment looks appropriate.")
            lines.append("")

    text = "\n".join(lines).strip()
    if len(text) > MAX_ADVICE_CHARS:
        text = text[:MAX_ADVICE_CHARS - 20] + "\n...[truncated]"
    return text


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def _upsert_advice(
    conn: Any,
    as_of_month: str,
    hazard_code: str,
    metric: str,
    advice_text: str,
    findings: Dict[str, Any],
) -> None:
    """Write (or replace) a calibration_advice row."""
    conn.execute(
        "DELETE FROM calibration_advice WHERE as_of_month = ? AND hazard_code = ? AND metric = ?",
        [as_of_month, hazard_code, metric],
    )

    findings_safe = {}
    for k, v in findings.items():
        try:
            json.dumps(v)
            findings_safe[k] = v
        except (TypeError, ValueError):
            findings_safe[k] = str(v)

    conn.execute(
        """
        INSERT INTO calibration_advice
            (as_of_month, hazard_code, metric, advice, findings_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            as_of_month,
            hazard_code,
            metric,
            advice_text,
            json.dumps(findings_safe),
            datetime.utcnow(),
        ],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_calibration_advice(
    db_url: str,
    as_of: Optional[date] = None,
    seed_defaults: bool = False,
) -> None:
    """Generate and write calibration advice for all hazard/metric pairs."""
    if as_of is None:
        as_of = date.today()
    as_of_month = as_of.strftime("%Y-%m")

    if not duckdb_io.DUCKDB_AVAILABLE:
        raise RuntimeError(duckdb_io.duckdb_unavailable_reason())

    conn = duckdb_io.get_db(db_url or duckdb_io.DEFAULT_DB_URL)

    try:
        _ensure_findings_json_column(conn)

        # Seed defaults if requested (uses sentinel date so real data supersedes)
        if seed_defaults:
            _seed_default_advice(conn)

        # Check required tables
        for table in ("scores", "resolutions", "questions", "forecasts_ensemble"):
            if not _table_exists(conn, table):
                LOGGER.info(
                    "generate_calibration_advice: table %s not found; skipping.",
                    table,
                )
                return

        if _row_count(conn, "scores") == 0:
            LOGGER.info("generate_calibration_advice: scores table is empty; skipping.")
            return

        pairs = _discover_hazard_metric_pairs(conn)
        if not pairs:
            LOGGER.info("generate_calibration_advice: no resolved hazard/metric pairs found.")
            return

        total_written = 0
        all_model_briers: List[Dict[str, Any]] = []

        for hazard_code, metric in pairs:
            n_resolved = _count_resolved(conn, hazard_code, metric)
            if n_resolved < MIN_QUESTIONS:
                LOGGER.info(
                    "Skipping %s/%s: only %d resolved questions (need %d).",
                    hazard_code, metric, n_resolved, MIN_QUESTIONS,
                )
                continue

            findings: Dict[str, Any] = {}
            findings["tail_coverage"] = _compute_tail_coverage(conn, hazard_code, metric)
            findings["bucket_calibration"] = _compute_bucket_calibration(
                conn, hazard_code, metric,
            )
            findings["per_model_brier"] = _compute_per_model_brier(
                conn, hazard_code, metric,
            )
            findings["month_position_bias"] = _compute_month_position_bias(
                conn, hazard_code, metric,
            )
            findings["rc_conditional"] = _compute_rc_conditional(
                conn, hazard_code, metric,
            )

            advice_text = _format_advice(
                findings, hazard_code, metric, as_of_month, n_resolved,
            )
            _upsert_advice(conn, as_of_month, hazard_code, metric, advice_text, findings)
            total_written += 1

            if findings.get("per_model_brier"):
                all_model_briers.append(findings["per_model_brier"])

            LOGGER.info(
                "Wrote calibration advice for %s/%s (%d chars).",
                hazard_code, metric, len(advice_text),
            )

        # Write global row
        global_findings = _compute_global_findings(all_model_briers)
        if global_findings:
            global_advice = _format_advice(
                global_findings, "*", "*", as_of_month, 0,
            )
            _upsert_advice(conn, as_of_month, "*", "*", global_advice, global_findings)
            total_written += 1

        LOGGER.info(
            "generate_calibration_advice: wrote %d advice rows for as_of_month=%s.",
            total_written, as_of_month,
        )

    finally:
        duckdb_io.close_db(conn)


def _compute_global_findings(
    all_model_briers: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate across all hazard/metrics for global stats."""
    if not all_model_briers:
        return {}

    # Aggregate per-model scores across hazard/metrics
    model_scores: Dict[str, List[float]] = {}
    ensemble_scores: List[float] = []

    for mb in all_model_briers:
        if mb.get("ensemble_brier") is not None:
            ensemble_scores.append(mb["ensemble_brier"])
        for model in mb.get("all_models", []):
            model_scores.setdefault(model["name"], []).append(model["brier"])

    if not model_scores:
        return {}

    model_avgs = {
        name: sum(scores) / len(scores) for name, scores in model_scores.items()
    }
    sorted_models = sorted(model_avgs.items(), key=lambda x: x[1])

    result: Dict[str, Any] = {
        "per_model_brier": {
            "best": {"name": sorted_models[0][0], "brier": sorted_models[0][1]},
            "worst": {"name": sorted_models[-1][0], "brier": sorted_models[-1][1]},
            "ensemble_brier": (
                sum(ensemble_scores) / len(ensemble_scores)
                if ensemble_scores else None
            ),
            "all_models": [
                {"name": n, "brier": b} for n, b in sorted_models
            ],
        }
    }
    return result


def _seed_default_advice(conn: Any) -> None:
    """Insert hand-written seed advice with sentinel date (superseded by real data)."""
    seed_month = "2000-01"

    for (hz, m), advice_text in SEED_ADVICE.items():
        try:
            existing = conn.execute(
                """
                SELECT 1 FROM calibration_advice
                WHERE as_of_month = ? AND hazard_code = ? AND metric = ?
                """,
                [seed_month, hz, m],
            ).fetchone()
            if existing:
                continue
            conn.execute(
                """
                INSERT INTO calibration_advice
                    (as_of_month, hazard_code, metric, advice, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [seed_month, hz, m, advice_text, datetime.utcnow()],
            )
            LOGGER.info("Seeded default advice for %s/%s.", hz, m)
        except Exception as exc:
            LOGGER.warning("Failed to seed advice for %s/%s: %s", hz, m, exc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate concrete calibration advice from scoring data.",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="DuckDB URL (e.g. duckdb:///data/resolver.duckdb)",
    )
    parser.add_argument(
        "--seed-defaults",
        action="store_true",
        default=False,
        help="Insert hand-written seed advice for bootstrapping.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    db_url = args.db_url or _get_db_url_from_config()
    generate_calibration_advice(
        db_url=db_url,
        seed_defaults=args.seed_defaults,
    )


if __name__ == "__main__":
    main()
