# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Sequence

import duckdb
from duckdb import CatalogException

from pythia.buckets import BucketSpec, BUCKET_SPECS
from pythia.config import load as load_config

PYTHIA_DEFAULT_DB_URL = "duckdb:///data/resolver.duckdb"

logger = logging.getLogger(__name__)


def get_db_url() -> str:
    """Return the DuckDB URL Pythia should use.

    Precedence:
    1. PYTHIA_DB_URL environment variable
    2. app.db_url from config
    3. PYTHIA_DEFAULT_DB_URL fallback
    """

    try:
        cfg = load_config() or {}
    except Exception:
        cfg = {}

    app_cfg = cfg.get("app") or {}
    cfg_url = app_cfg.get("db_url")
    env_url = os.getenv("PYTHIA_DB_URL")

    if env_url:
        if env_url != cfg_url:
            logger.debug(
                "Using PYTHIA_DB_URL override for DuckDB (env wins over config): %s",
                env_url,
            )
        return env_url

    if cfg_url:
        logger.debug("Using DuckDB URL from config: %s", cfg_url)
        return cfg_url

    logger.debug("Using PYTHIA default DuckDB URL: %s", PYTHIA_DEFAULT_DB_URL)
    return PYTHIA_DEFAULT_DB_URL


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection for Pythia using the configured URL.

    For file-backed databases, we always open in read-write mode to avoid
    DuckDB's "different configuration" error when mixing read-only and
    read-write connections in the same process. For in-memory DBs we honour
    the caller's requested read_only flag.
    """

    url = get_db_url()
    if url.startswith("duckdb:///"):
        db_path = url[len("duckdb:///") :]
    else:
        db_path = url

    if db_path not in {":memory:"}:
        db_path_obj = Path(db_path)
        parent = db_path_obj.parent
        if parent:
            parent.mkdir(parents=True, exist_ok=True)

    if db_path == ":memory:":
        effective_read_only = read_only
        logger.debug(
            "Connecting to DuckDB at %s (requested read_only=%s, using_read_only=%s)",
            db_path,
            read_only,
            effective_read_only,
        )
    else:
        effective_read_only = False
        logger.debug(
            "Connecting to DuckDB at %s (requested read_only=%s, forcing_read_only=%s for file-backed DB)",
            db_path,
            read_only,
            effective_read_only,
        )

    return duckdb.connect(db_path, read_only=effective_read_only)


def _existing_columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    """Return a set of existing column names for a given table, lower-cased."""

    try:
        rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
    except Exception:
        return set()
    return {str(r[1]).lower() for r in rows}


def _ensure_table_and_columns(
    con: duckdb.DuckDBPyConnection,
    table: str,
    create_sql: str,
    required_columns: dict[str, str],
) -> None:
    """Ensure a table exists and contains required columns."""

    con.execute(create_sql)
    existing = _existing_columns(con, table)

    for col, col_type in required_columns.items():
        if col.lower() in existing:
            continue
        try:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
        except CatalogException:
            continue


def _ensure_hs_triage_table(con: duckdb.DuckDBPyConnection) -> None:
    """Ensure the hs_triage table exists for HS v2 triage outputs."""

    _ensure_table_and_columns(
        con,
        "hs_triage",
        """
        CREATE TABLE IF NOT EXISTS hs_triage (
            run_id TEXT NOT NULL,
            iso3 TEXT NOT NULL,
            hazard_code TEXT NOT NULL,
            tier TEXT NOT NULL,
            triage_score DOUBLE NOT NULL,
            need_full_spd BOOLEAN NOT NULL,
            drivers_json TEXT,
            regime_shifts_json TEXT,
            data_quality_json TEXT,
            scenario_stub TEXT,
            regime_change_likelihood DOUBLE,
            regime_change_magnitude DOUBLE,
            regime_change_score DOUBLE,
            regime_change_level INTEGER,
            regime_change_direction TEXT,
            regime_change_window TEXT,
            regime_change_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        {
            "run_id": "TEXT",
            "iso3": "TEXT",
            "hazard_code": "TEXT",
            "tier": "TEXT",
            "triage_score": "DOUBLE",
            "need_full_spd": "BOOLEAN",
            "drivers_json": "TEXT",
            "regime_shifts_json": "TEXT",
            "data_quality_json": "TEXT",
            "scenario_stub": "TEXT",
            "regime_change_likelihood": "DOUBLE",
            "regime_change_magnitude": "DOUBLE",
            "regime_change_score": "DOUBLE",
            "regime_change_level": "INTEGER",
            "regime_change_direction": "TEXT",
            "regime_change_window": "TEXT",
            "regime_change_json": "TEXT",
            "created_at": "TIMESTAMP",
        },
    )


def _ensure_question_research_table(con: duckdb.DuckDBPyConnection) -> None:
    """Ensure the question_research table exists for Researcher v2 outputs."""

    _ensure_table_and_columns(
        con,
        "question_research",
        """
        CREATE TABLE IF NOT EXISTS question_research (
            run_id TEXT NOT NULL,
            question_id TEXT NOT NULL,
            iso3 TEXT NOT NULL,
            hazard_code TEXT NOT NULL,
            metric TEXT NOT NULL,
            research_json TEXT NOT NULL,
            hs_evidence_json TEXT,
            question_evidence_json TEXT,
            merged_evidence_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        {
            "run_id": "TEXT",
            "question_id": "TEXT",
            "iso3": "TEXT",
            "hazard_code": "TEXT",
            "metric": "TEXT",
            "research_json": "TEXT",
            "hs_evidence_json": "TEXT",
            "question_evidence_json": "TEXT",
            "merged_evidence_json": "TEXT",
            "created_at": "TIMESTAMP",
        },
    )


def _ensure_question_run_metrics_table(con: duckdb.DuckDBPyConnection) -> None:
    """Ensure the question_run_metrics table exists for per-question metrics."""

    _ensure_table_and_columns(
        con,
        "question_run_metrics",
        """
        CREATE TABLE IF NOT EXISTS question_run_metrics (
            run_id TEXT NOT NULL,
            question_id TEXT NOT NULL,
            iso3 TEXT,
            hazard_code TEXT,
            metric TEXT,
            started_at_utc TIMESTAMP,
            finished_at_utc TIMESTAMP,
            wall_ms BIGINT,
            compute_ms BIGINT,
            queue_ms BIGINT,
            cost_usd DOUBLE,
            n_spd_models_expected INTEGER,
            n_spd_models_ok INTEGER,
            missing_model_ids_json TEXT,
            phase_max_ms_json TEXT,
            phase_cost_usd_json TEXT,
            PRIMARY KEY (run_id, question_id)
        );
        """,
        {
            "run_id": "TEXT",
            "question_id": "TEXT",
            "iso3": "TEXT",
            "hazard_code": "TEXT",
            "metric": "TEXT",
            "started_at_utc": "TIMESTAMP",
            "finished_at_utc": "TIMESTAMP",
            "wall_ms": "BIGINT",
            "compute_ms": "BIGINT",
            "queue_ms": "BIGINT",
            "cost_usd": "DOUBLE",
            "n_spd_models_expected": "INTEGER",
            "n_spd_models_ok": "INTEGER",
            "missing_model_ids_json": "TEXT",
            "phase_max_ms_json": "TEXT",
            "phase_cost_usd_json": "TEXT",
        },
    )


def _ensure_scenarios_table(con: duckdb.DuckDBPyConnection) -> None:
    """Ensure the scenarios table exists for Scenario Writer outputs."""

    _ensure_table_and_columns(
        con,
        "scenarios",
        """
        CREATE TABLE IF NOT EXISTS scenarios (
            run_id TEXT NOT NULL,
            iso3 TEXT NOT NULL,
            hazard_code TEXT NOT NULL,
            metric TEXT NOT NULL,
            scenario_type TEXT NOT NULL,
            bucket_label TEXT NOT NULL,
            probability DOUBLE NOT NULL,
            text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        {
            "run_id": "TEXT",
            "iso3": "TEXT",
            "hazard_code": "TEXT",
            "metric": "TEXT",
            "scenario_type": "TEXT",
            "bucket_label": "TEXT",
            "probability": "DOUBLE",
            "text": "TEXT",
            "created_at": "TIMESTAMP",
        },
    )


def _seed_bucket_definitions(
    con: duckdb.DuckDBPyConnection,
    metric: str,
    specs: Sequence[BucketSpec],
) -> None:
    """Ensure bucket_definitions rows exist for a metric."""

    metric_upper = metric.upper()
    rows = con.execute(
        """
        SELECT COUNT(*)
        FROM bucket_definitions
        WHERE upper(metric) = ?
        """,
        [metric_upper],
    ).fetchone()

    if rows and (rows[0] or 0) >= len(specs):
        return

    con.execute(
        """
        DELETE FROM bucket_definitions
        WHERE upper(metric) = ?
        """,
        [metric_upper],
    )

    for spec in specs:
        con.execute(
            """
            INSERT INTO bucket_definitions (
                metric, bucket_index, label, lower_bound, upper_bound
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                metric_upper,
                int(spec.idx),
                spec.label,
                None if spec.lower is None else float(spec.lower),
                None if spec.upper is None else float(spec.upper),
            ],
        )


def _seed_bucket_centroids(
    con: duckdb.DuckDBPyConnection,
    metric: str,
    specs: Sequence[BucketSpec],
) -> None:
    """Ensure wildcard centroids exist in bucket_centroids."""

    metric_upper = metric.upper()
    rows = con.execute(
        """
        SELECT COUNT(*)
        FROM bucket_centroids
        WHERE upper(metric) = ? AND hazard_code = '*'
        """,
        [metric_upper],
    ).fetchone()

    if rows and (rows[0] or 0) >= len(specs):
        return

    con.execute(
        """
        DELETE FROM bucket_centroids
        WHERE upper(metric) = ? AND hazard_code = '*'
        """,
        [metric_upper],
    )

    for spec in specs:
        con.execute(
            """
            INSERT INTO bucket_centroids (hazard_code, metric, bucket_index, centroid)
            VALUES (?, ?, ?, ?)
            """,
            ["*", metric_upper, int(spec.idx), float(spec.centroid)],
        )


def ensure_schema(con: Optional[duckdb.DuckDBPyConnection] = None) -> None:
    """
    Ensure all Pythia-related tables exist in DuckDB.

    This is idempotent and safe to call multiple times.
    """

    own_con = con is None
    if own_con:
        con = connect(read_only=False)

    assert con is not None

    try:
        _ensure_table_and_columns(
            con,
            "hs_runs",
            """
            CREATE TABLE IF NOT EXISTS hs_runs (
                hs_run_id TEXT,
                generated_at TIMESTAMP,
                git_sha TEXT,
                config_profile TEXT,
                countries_json TEXT,
                requested_countries_json TEXT,
                skipped_entries_json TEXT
            );
            """,
            {
                "hs_run_id": "TEXT",
                "generated_at": "TIMESTAMP",
                "git_sha": "TEXT",
                "config_profile": "TEXT",
                "countries_json": "TEXT",
                "requested_countries_json": "TEXT",
                "skipped_entries_json": "TEXT",
            },
        )

        _ensure_table_and_columns(
            con,
            "hs_scenarios",
            """
            CREATE TABLE IF NOT EXISTS hs_scenarios (
                hs_run_id TEXT,
                scenario_id TEXT,
                iso3 TEXT,
                hazard_code TEXT,
                scenario_title TEXT,
                likely_month TEXT,
                probability_pct DOUBLE,
                pin_best_guess DOUBLE,
                pa_best_guess DOUBLE,
                scenario_markdown TEXT,
                scenario_json TEXT
            );
            """,
            {
                "hs_run_id": "TEXT",
                "scenario_id": "TEXT",
                "iso3": "TEXT",
                "hazard_code": "TEXT",
                "scenario_title": "TEXT",
                "likely_month": "TEXT",
                "probability_pct": "DOUBLE",
                "pin_best_guess": "DOUBLE",
                "pa_best_guess": "DOUBLE",
                "scenario_markdown": "TEXT",
                "scenario_json": "TEXT",
            },
        )

        _ensure_table_and_columns(
            con,
            "hs_country_reports",
            """
            CREATE TABLE IF NOT EXISTS hs_country_reports (
                hs_run_id TEXT,
                iso3 TEXT,
                report_markdown TEXT,
                sources_json TEXT,
                grounded BOOLEAN,
                grounding_debug_json TEXT,
                structural_context TEXT,
                recent_signals_json TEXT
            );
            """,
            {
                "hs_run_id": "TEXT",
                "iso3": "TEXT",
                "report_markdown": "TEXT",
                "sources_json": "TEXT",
                "grounded": "BOOLEAN",
                "grounding_debug_json": "TEXT",
                "structural_context": "TEXT",
                "recent_signals_json": "TEXT",
            },
        )

        _ensure_question_run_metrics_table(con)

        _ensure_table_and_columns(
            con,
            "run_provenance",
            """
            CREATE TABLE IF NOT EXISTS run_provenance (
                run_id TEXT,
                hs_run_id TEXT,
                forecaster_run_id TEXT,
                artifact_run_id TEXT,
                artifact_workflow TEXT,
                artifact_name TEXT,
                db_sha256 TEXT,
                db_size_bytes BIGINT,
                facts_resolved_before BIGINT,
                facts_resolved_after BIGINT,
                facts_deltas_before BIGINT,
                facts_deltas_after BIGINT,
                snapshots_before BIGINT,
                snapshots_after BIGINT,
                hs_triage_before BIGINT,
                hs_triage_after BIGINT,
                questions_before BIGINT,
                questions_after BIGINT,
                forecasts_raw_before BIGINT,
                forecasts_raw_after BIGINT,
                forecasts_ensemble_before BIGINT,
                forecasts_ensemble_after BIGINT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            {
                "run_id": "TEXT",
                "hs_run_id": "TEXT",
                "forecaster_run_id": "TEXT",
                "artifact_run_id": "TEXT",
                "artifact_workflow": "TEXT",
                "artifact_name": "TEXT",
                "db_sha256": "TEXT",
                "db_size_bytes": "BIGINT",
                "facts_resolved_before": "BIGINT",
                "facts_resolved_after": "BIGINT",
                "facts_deltas_before": "BIGINT",
                "facts_deltas_after": "BIGINT",
                "snapshots_before": "BIGINT",
                "snapshots_after": "BIGINT",
                "hs_triage_before": "BIGINT",
                "hs_triage_after": "BIGINT",
                "questions_before": "BIGINT",
                "questions_after": "BIGINT",
                "forecasts_raw_before": "BIGINT",
                "forecasts_raw_after": "BIGINT",
                "forecasts_ensemble_before": "BIGINT",
                "forecasts_ensemble_after": "BIGINT",
                "created_at": "TIMESTAMP",
            },
        )

        _ensure_table_and_columns(
            con,
            "questions",
            """
            CREATE TABLE IF NOT EXISTS questions (
                question_id TEXT,
                hs_run_id TEXT,
                scenario_ids_json TEXT,
                iso3 TEXT,
                hazard_code TEXT,
                metric TEXT,
                target_month TEXT,
                window_start_date DATE,
                window_end_date DATE,
                wording TEXT,
                status TEXT,
                pythia_metadata_json TEXT
            );
            """,
            {
                "question_id": "TEXT",
                "hs_run_id": "TEXT",
                "scenario_ids_json": "TEXT",
                "iso3": "TEXT",
                "hazard_code": "TEXT",
                "metric": "TEXT",
                "target_month": "TEXT",
                "window_start_date": "DATE",
                "window_end_date": "DATE",
                "wording": "TEXT",
                "status": "TEXT",
                "pythia_metadata_json": "TEXT",
            },
        )

        _ensure_table_and_columns(
            con,
            "forecasts_ensemble",
            """
            CREATE TABLE IF NOT EXISTS forecasts_ensemble (
                run_id TEXT,
                question_id TEXT,
                iso3 TEXT,
                hazard_code TEXT,
                metric TEXT,
                model_name TEXT,
                month_index INTEGER,
                bucket_index INTEGER,
                probability DOUBLE,
                ev_value DOUBLE,
                weights_profile TEXT,
                created_at TIMESTAMP,
                status TEXT,
                human_explanation TEXT
            );
            """,
            {
                "run_id": "TEXT",
                "question_id": "TEXT",
                "iso3": "TEXT",
                "hazard_code": "TEXT",
                "metric": "TEXT",
                "model_name": "TEXT",
                "month_index": "INTEGER",
                "bucket_index": "INTEGER",
                "probability": "DOUBLE",
                "ev_value": "DOUBLE",
                "weights_profile": "TEXT",
                "created_at": "TIMESTAMP",
                "horizon_m": "INTEGER",
                "class_bin": "TEXT",
                "p": "DOUBLE",
                "status": "TEXT",
                "human_explanation": "TEXT",
            },
        )

        _ensure_table_and_columns(
            con,
            "forecasts_raw",
            """
            CREATE TABLE IF NOT EXISTS forecasts_raw (
                run_id TEXT,
                question_id TEXT,
                model_name TEXT,
                month_index INTEGER,
                bucket_index INTEGER,
                probability DOUBLE,
                ok BOOLEAN,
                elapsed_ms INTEGER,
                cost_usd DOUBLE,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                status TEXT,
                spd_json TEXT,
                human_explanation TEXT
            );
            """,
            {
                "run_id": "TEXT",
                "question_id": "TEXT",
                "model_name": "TEXT",
                "month_index": "INTEGER",
                "bucket_index": "INTEGER",
                "probability": "DOUBLE",
                "ok": "BOOLEAN",
                "elapsed_ms": "INTEGER",
                "cost_usd": "DOUBLE",
                "prompt_tokens": "INTEGER",
                "completion_tokens": "INTEGER",
                "total_tokens": "INTEGER",
                "status": "TEXT",
                "spd_json": "TEXT",
                "human_explanation": "TEXT",
            },
        )

        _ensure_table_and_columns(
            con,
            "question_context",
            """
            CREATE TABLE IF NOT EXISTS question_context (
                run_id TEXT,
                question_id TEXT,
                iso3 TEXT,
                hazard_code TEXT,
                metric TEXT,
                snapshot_start_month TEXT,
                snapshot_end_month TEXT,
                pa_history_json TEXT,
                context_json TEXT
            );
            """,
            {
                "run_id": "TEXT",
                "question_id": "TEXT",
                "iso3": "TEXT",
                "hazard_code": "TEXT",
                "metric": "TEXT",
                "snapshot_start_month": "TEXT",
                "snapshot_end_month": "TEXT",
                "pa_history_json": "TEXT",
                "context_json": "TEXT",
            },
        )

        _ensure_table_and_columns(
            con,
            "bucket_definitions",
            """
            CREATE TABLE IF NOT EXISTS bucket_definitions (
                metric TEXT,
                bucket_index INTEGER,
                label TEXT,
                lower_bound DOUBLE,
                upper_bound DOUBLE
            );
            """,
            {
                "metric": "TEXT",
                "bucket_index": "INTEGER",
                "label": "TEXT",
                "lower_bound": "DOUBLE",
                "upper_bound": "DOUBLE",
            },
        )

        con.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS ux_bucket_definitions
            ON bucket_definitions (metric, bucket_index)
            """
        )

        _ensure_table_and_columns(
            con,
            "bucket_centroids",
            """
            CREATE TABLE IF NOT EXISTS bucket_centroids (
                hazard_code TEXT,
                metric TEXT,
                bucket_index INTEGER,
                centroid DOUBLE
            );
            """,
            {
                "hazard_code": "TEXT",
                "metric": "TEXT",
                "bucket_index": "INTEGER",
                "centroid": "DOUBLE",
            },
        )

        con.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS ux_bucket_centroids
            ON bucket_centroids (hazard_code, metric, bucket_index)
            """
        )

        for metric, specs in BUCKET_SPECS.items():
            _seed_bucket_definitions(con, metric, specs)
            _seed_bucket_centroids(con, metric, specs)

        _ensure_table_and_columns(
            con,
            "llm_calls",
            """
            CREATE TABLE IF NOT EXISTS llm_calls (
                call_id TEXT,
                run_id TEXT,
                hs_run_id TEXT,
                question_id TEXT,
                call_type TEXT,
                phase TEXT,
                model_name TEXT,
                provider TEXT,
                model_id TEXT,
                prompt_text TEXT,
                response_text TEXT,
                parsed_json TEXT,
                usage_json TEXT,
                elapsed_ms INTEGER,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                cost_usd DOUBLE,
                error_text TEXT,
                status TEXT,
                error_type TEXT,
                error_message TEXT,
                hazard_scores_json TEXT,
                hazard_scores_parse_ok BOOLEAN,
                response_format TEXT,
                timestamp TIMESTAMP,
                iso3 TEXT,
                hazard_code TEXT,
                metric TEXT
            );
            """,
            {
                "call_id": "TEXT",
                "run_id": "TEXT",
                "hs_run_id": "TEXT",
                "question_id": "TEXT",
                "call_type": "TEXT",
                "phase": "TEXT",
                "model_name": "TEXT",
                "provider": "TEXT",
                "model_id": "TEXT",
                "prompt_text": "TEXT",
                "response_text": "TEXT",
                "parsed_json": "TEXT",
                "usage_json": "TEXT",
                "elapsed_ms": "INTEGER",
                "prompt_tokens": "INTEGER",
                "completion_tokens": "INTEGER",
                "total_tokens": "INTEGER",
                "cost_usd": "DOUBLE",
                "error_text": "TEXT",
                "status": "TEXT",
                "error_type": "TEXT",
                "error_message": "TEXT",
                "hazard_scores_json": "TEXT",
                "hazard_scores_parse_ok": "BOOLEAN",
                "response_format": "TEXT",
                "timestamp": "TIMESTAMP",
                "iso3": "TEXT",
                "hazard_code": "TEXT",
                "metric": "TEXT",
            },
        )

        _ensure_table_and_columns(
            con,
            "pm_checks",
            """
            CREATE TABLE IF NOT EXISTS pm_checks (
                pm_check_id TEXT,
                run_id TEXT,
                question_id TEXT,
                market_source TEXT,
                market_id TEXT,
                market_url TEXT,
                as_of_time TIMESTAMP,
                price DOUBLE,
                volume DOUBLE,
                liquidity DOUBLE,
                pm_json TEXT,
                llm_call_id TEXT,
                pm_summary_text TEXT
            );
            """,
            {
                "pm_check_id": "TEXT",
                "run_id": "TEXT",
                "question_id": "TEXT",
                "market_source": "TEXT",
                "market_id": "TEXT",
                "market_url": "TEXT",
                "as_of_time": "TIMESTAMP",
                "price": "DOUBLE",
                "volume": "DOUBLE",
                "liquidity": "DOUBLE",
                "pm_json": "TEXT",
                "llm_call_id": "TEXT",
                "pm_summary_text": "TEXT",
            },
        )

        _ensure_table_and_columns(
            con,
            "gtmc1_runs",
            """
            CREATE TABLE IF NOT EXISTS gtmc1_runs (
                gtmc1_run_id TEXT,
                run_id TEXT,
                question_id TEXT,
                active BOOLEAN,
                gtmc1_prob DOUBLE,
                coalition_rate DOUBLE,
                dispersion DOUBLE,
                median_of_final_medians DOUBLE,
                exceedance_ge_50 DOUBLE,
                num_runs INTEGER,
                median_rounds INTEGER,
                raw_reason TEXT,
                runs_ref TEXT,
                meta_json TEXT
            );
            """,
            {
                "gtmc1_run_id": "TEXT",
                "run_id": "TEXT",
                "question_id": "TEXT",
                "active": "BOOLEAN",
                "gtmc1_prob": "DOUBLE",
                "coalition_rate": "DOUBLE",
                "dispersion": "DOUBLE",
                "median_of_final_medians": "DOUBLE",
                "exceedance_ge_50": "DOUBLE",
                "num_runs": "INTEGER",
                "median_rounds": "INTEGER",
                "raw_reason": "TEXT",
                "runs_ref": "TEXT",
                "meta_json": "TEXT",
            },
        )

        _ensure_table_and_columns(
            con,
            "gtmc1_actors",
            """
            CREATE TABLE IF NOT EXISTS gtmc1_actors (
                gtmc1_run_id TEXT,
                actor_name TEXT,
                position DOUBLE,
                capability DOUBLE,
                salience DOUBLE,
                risk_threshold DOUBLE
            );
            """,
            {
                "gtmc1_run_id": "TEXT",
                "actor_name": "TEXT",
                "position": "DOUBLE",
                "capability": "DOUBLE",
                "salience": "DOUBLE",
                "risk_threshold": "DOUBLE",
            },
        )

        _ensure_table_and_columns(
            con,
            "meta_runs",
            """
            CREATE TABLE IF NOT EXISTS meta_runs (
                run_id TEXT,
                run_type TEXT,
                hs_run_id TEXT,
                forecaster_models_json TEXT,
                git_sha TEXT,
                config_profile TEXT,
                started_at TIMESTAMP,
                finished_at TIMESTAMP,
                total_cost_usd DOUBLE,
                total_tokens INTEGER,
                status TEXT,
                notes TEXT
            );
            """,
            {
                "run_id": "TEXT",
                "run_type": "TEXT",
                "hs_run_id": "TEXT",
                "forecaster_models_json": "TEXT",
                "git_sha": "TEXT",
                "config_profile": "TEXT",
                "started_at": "TIMESTAMP",
                "finished_at": "TIMESTAMP",
                "total_cost_usd": "DOUBLE",
                "total_tokens": "INTEGER",
                "status": "TEXT",
                "notes": "TEXT",
            },
        )

        _ensure_hs_triage_table(con)
        _ensure_question_research_table(con)
        _ensure_scenarios_table(con)
    finally:
        if own_con and con is not None:
            con.close()
