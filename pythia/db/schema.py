from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import duckdb
from duckdb import CatalogException

from pythia.config import load as load_config

PYTHIA_DEFAULT_DB_URL = "duckdb:///data/resolver.duckdb"


def get_db_url() -> str:
    """Return the DuckDB URL Pythia should use."""

    try:
        cfg = load_config() or {}
    except Exception:
        cfg = {}

    app_cfg = cfg.get("app") or {}
    return app_cfg.get("db_url") or os.getenv("PYTHIA_DB_URL") or PYTHIA_DEFAULT_DB_URL


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection for Pythia using the configured URL."""

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

    return duckdb.connect(db_path, read_only=read_only)


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
                created_at TIMESTAMP
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
                total_tokens INTEGER
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
            "llm_calls",
            """
            CREATE TABLE IF NOT EXISTS llm_calls (
                call_id TEXT,
                run_id TEXT,
                hs_run_id TEXT,
                question_id TEXT,
                call_type TEXT,
                model_name TEXT,
                provider TEXT,
                model_id TEXT,
                prompt_text TEXT,
                response_text TEXT,
                parsed_json TEXT,
                elapsed_ms INTEGER,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                cost_usd DOUBLE,
                error_text TEXT,
                timestamp TIMESTAMP
            );
            """,
            {
                "call_id": "TEXT",
                "run_id": "TEXT",
                "hs_run_id": "TEXT",
                "question_id": "TEXT",
                "call_type": "TEXT",
                "model_name": "TEXT",
                "provider": "TEXT",
                "model_id": "TEXT",
                "prompt_text": "TEXT",
                "response_text": "TEXT",
                "parsed_json": "TEXT",
                "elapsed_ms": "INTEGER",
                "prompt_tokens": "INTEGER",
                "completion_tokens": "INTEGER",
                "total_tokens": "INTEGER",
                "cost_usd": "DOUBLE",
                "error_text": "TEXT",
                "timestamp": "TIMESTAMP",
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
    finally:
        if own_con and con is not None:
            con.close()
