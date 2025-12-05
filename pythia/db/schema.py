from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import duckdb
from duckdb import CatalogException

from pythia.config import load as load_config

PA_CENTROIDS: tuple[float, ...] = (0.0, 30_000.0, 150_000.0, 375_000.0, 700_000.0)
FATALITIES_CENTROIDS: tuple[float, ...] = (0.0, 15.0, 62.0, 300.0, 700.0)

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

    con.execute(
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )


def _ensure_question_research_table(con: duckdb.DuckDBPyConnection) -> None:
    """Ensure the question_research table exists for Researcher v2 outputs."""

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS question_research (
            run_id TEXT NOT NULL,
            question_id TEXT NOT NULL,
            iso3 TEXT NOT NULL,
            hazard_code TEXT NOT NULL,
            metric TEXT NOT NULL,
            research_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )


def _ensure_scenarios_table(con: duckdb.DuckDBPyConnection) -> None:
    """Ensure the scenarios table exists for Scenario Writer outputs."""

    con.execute(
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
        """
    )


def _seed_pa_bucket_centroids(con: duckdb.DuckDBPyConnection) -> None:
    """Ensure wildcard PA centroids exist in bucket_centroids."""

    rows = con.execute(
        """
        SELECT COUNT(*)
        FROM bucket_centroids
        WHERE upper(metric) = 'PA' AND hazard_code = '*'
        """
    ).fetchone()

    if rows and (rows[0] or 0) >= 5:
        return

    con.execute(
        """
        DELETE FROM bucket_centroids
        WHERE upper(metric) = 'PA' AND hazard_code = '*'
        """
    )

    for idx, centroid in enumerate(PA_CENTROIDS, start=1):
        con.execute(
            """
            INSERT INTO bucket_centroids (hazard_code, metric, bucket_index, centroid)
            VALUES (?, ?, ?, ?)
            """,
            ["*", "PA", idx, float(centroid)],
        )


def _seed_fatalities_bucket_centroids(con: duckdb.DuckDBPyConnection) -> None:
    """Ensure wildcard fatalities centroids exist in bucket_centroids."""

    rows = con.execute(
        """
        SELECT COUNT(*)
        FROM bucket_centroids
        WHERE upper(metric) = 'FATALITIES' AND hazard_code = '*'
        """
    ).fetchone()

    if rows and (rows[0] or 0) >= 5:
        return

    con.execute(
        """
        DELETE FROM bucket_centroids
        WHERE upper(metric) = 'FATALITIES' AND hazard_code = '*'
        """
    )

    for idx, centroid in enumerate(FATALITIES_CENTROIDS, start=1):
        con.execute(
            """
            INSERT INTO bucket_centroids (hazard_code, metric, bucket_index, centroid)
            VALUES (?, ?, ?, ?)
            """,
            ["*", "FATALITIES", idx, float(centroid)],
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
                countries_json TEXT
            );
            """,
            {
                "hs_run_id": "TEXT",
                "generated_at": "TIMESTAMP",
                "git_sha": "TEXT",
                "config_profile": "TEXT",
                "countries_json": "TEXT",
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
                sources_json TEXT
            );
            """,
            {
                "hs_run_id": "TEXT",
                "iso3": "TEXT",
                "report_markdown": "TEXT",
                "sources_json": "TEXT",
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

        _seed_pa_bucket_centroids(con)
        _seed_fatalities_bucket_centroids(con)

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
