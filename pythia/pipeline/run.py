from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
import logging
import os
import subprocess
import sys
import uuid

import duckdb

from pythia.db.init import init as init_db
from pythia.config import load as load_cfg
from horizon_scanner.horizon_scanner import main as hs_main

_pool = ThreadPoolExecutor(max_workers=1)


def _update_ui_run(
    db_url: str,
    ui_run_id: str,
    *,
    started_at=None,
    finished_at=None,
    countries: list[str] | None = None,
    status: str | None = None,
    error: str | None = None,
) -> None:
    """
    Insert or update a ui_runs row for the given ui_run_id.

    - On first call (no existing row), inserts a new row.
    - On subsequent calls, updates provided fields.
    """
    db_path = db_url.replace("duckdb:///", "")
    try:
        con = duckdb.connect(db_path)
    except Exception as exc:
        logging.exception("ui_runs: failed to connect to DuckDB at %s: %r", db_path, exc)
        return

    try:
        # Ensure table exists (init_db should already have done this, but be defensive)
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS ui_runs (
              ui_run_id   TEXT PRIMARY KEY,
              started_at  TIMESTAMP,
              finished_at TIMESTAMP,
              countries   JSON,
              status      TEXT,
              error       TEXT,
              created_at  TIMESTAMP DEFAULT now()
            )
            """
        )

        # Does a row already exist?
        existing = con.execute(
            "SELECT 1 FROM ui_runs WHERE ui_run_id = ? LIMIT 1",
            [ui_run_id],
        ).fetchone()

        if not existing:
            con.execute(
                """
                INSERT INTO ui_runs (ui_run_id, started_at, finished_at, countries, status, error)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    ui_run_id,
                    started_at,
                    finished_at,
                    countries,
                    status,
                    error,
                ],
            )
        else:
            sets = []
            params: list = []
            if started_at is not None:
                sets.append("started_at = ?")
                params.append(started_at)
            if finished_at is not None:
                sets.append("finished_at = ?")
                params.append(finished_at)
            if countries is not None:
                sets.append("countries = ?")
                params.append(countries)
            if status is not None:
                sets.append("status = ?")
                params.append(status)
            if error is not None:
                sets.append("error = ?")
                params.append(error)

            if sets:
                sql = f"UPDATE ui_runs SET {', '.join(sets)} WHERE ui_run_id = ?"
                params.append(ui_run_id)
                con.execute(sql, params)

    except Exception:
        logging.exception("ui_runs: failed to insert/update row for %s", ui_run_id)
    finally:
        try:
            con.close()
        except Exception:
            pass


def _pipeline(ui_run_id: str, countries: list[str]):
    """
    End-to-end Pythia pipeline for a UI/API-triggered run.

    Steps:
      1) Ensure DuckDB schema exists (init_db).
      2) Insert a ui_runs row with status 'running'.
      3) Run Horizon Scanner to write scenarios + questions into DuckDB.
         - If `countries` is empty, HS falls back to hs_country_list.txt.
         - If non-empty, HS uses the provided list as country names.
      4) Run Forecaster in Pythia mode to generate SPD ensemble forecasts
         for all active questions in the `questions` table and write them
         into `forecasts_ensemble`.
      5) Update ui_runs.status to 'ok' or 'failed' with error message.

    NOTE: This runs inside a ThreadPoolExecutor worker; any expensive I/O
    (Gemini, LLM calls) happens off the main API thread.
    """
    try:
        cfg = load_cfg()
        app_cfg = cfg.get("app", {}) if isinstance(cfg, dict) else {}
        db_url = str(app_cfg.get("db_url", "")).strip()
    except Exception:
        db_url = ""

    if not db_url:
        # Fallback consistent with Horizon Scanner's default
        db_url = "duckdb:///data/resolver.duckdb"
        logging.warning("app.db_url missing from config; falling back to %s", db_url)
    else:
        logging.info("Using app.db_url from config in pipeline: %s", db_url)

    # Initial ui_runs row: queued/running
    try:
        _update_ui_run(
            db_url,
            ui_run_id,
            started_at=datetime.utcnow(),
            countries=countries or [],
            status="running",
            error=None,
        )
        logging.info("ui_runs: recorded start of ui_run_id=%s", ui_run_id)
    except Exception:
        logging.exception("ui_runs: failed to record start for ui_run_id=%s", ui_run_id)

    prev_ui = os.environ.get("PYTHIA_UI_RUN_ID")
    os.environ["PYTHIA_UI_RUN_ID"] = ui_run_id

    try:
        # 1) Ensure DuckDB schema exists (idempotent)
        try:
            init_db(db_url)
            logging.info("DuckDB schema initialised via init_db(%s).", db_url)
        except Exception as e:
            logging.exception("init_db(%s) failed; aborting pipeline.", db_url)
            try:
                _update_ui_run(
                    db_url,
                    ui_run_id,
                    finished_at=datetime.utcnow(),
                    status="failed",
                    error=f"init_db_failed:{e!r}",
                )
            except Exception:
                logging.exception("ui_runs: failed to record init_db failure for %s", ui_run_id)
            return

        # 2) Run Horizon Scanner to write hs_runs, hs_scenarios, questions
        try:
            # Horizon Scanner will also call init_db(db_url) internally and respect app.db_url.
            # If `countries` is empty, we pass None so HS falls back to hs_country_list.txt.
            hs_countries = countries or None
            hs_main(hs_countries)
            logging.info("Horizon Scanner completed in pipeline.")
        except Exception as e:
            logging.exception("Horizon Scanner failed inside pipeline; aborting before Forecaster.")
            try:
                _update_ui_run(
                    db_url,
                    ui_run_id,
                    finished_at=datetime.utcnow(),
                    status="failed",
                    error=f"horizon_scanner_failed:{e!r}",
                )
            except Exception:
                logging.exception("ui_runs: failed to record HS failure for %s", ui_run_id)
            return

        # 3) Run Forecaster in Pythia mode on active questions
        try:
            limit = 200  # hard cap for now; can be made configurable via cfg["forecaster"].get("max_questions", 200)
            cmd = [
                sys.executable,
                "-m",
                "forecaster.cli",
                "--mode",
                "pythia",
                "--limit",
                str(limit),
                "--purpose",
                "ui_pipeline",
            ]
            # Local SPD debugging: export PYTHIA_SPD_HARD_FAIL=1 and optionally
            # PYTHIA_DEBUG_SPD=1 before running the pipeline to surface full
            # tracebacks from the Forecaster subprocess.
            if os.environ.get("PYTHIA_SPD_HARD_FAIL") == "1":
                logging.info(
                    "SPD hard-fail mode enabled via PYTHIA_SPD_HARD_FAIL=1 for Forecaster subprocess."
                )
            if os.environ.get("PYTHIA_DEBUG_SPD") == "1":
                logging.info(
                    "SPD debug logging enabled via PYTHIA_DEBUG_SPD=1 for Forecaster subprocess."
                )
            logging.info("Starting Forecaster in Pythia mode: %s", " ".join(cmd))
            subprocess.run(cmd, check=True)
            logging.info("Forecaster (Pythia mode) completed successfully in pipeline.")
            try:
                _update_ui_run(
                    db_url,
                    ui_run_id,
                    finished_at=datetime.utcnow(),
                    status="ok",
                    error=None,
                )
            except Exception:
                logging.exception("ui_runs: failed to record success for %s", ui_run_id)
        except subprocess.CalledProcessError as e:
            logging.exception("Forecaster subprocess failed in pipeline: %s", e)
            try:
                _update_ui_run(
                    db_url,
                    ui_run_id,
                    finished_at=datetime.utcnow(),
                    status="failed",
                    error=f"forecaster_failed:{e!r}",
                )
            except Exception:
                logging.exception("ui_runs: failed to record Forecaster failure for %s", ui_run_id)
        except Exception as e:
            logging.exception("Unexpected error while running Forecaster in pipeline.")
            try:
                _update_ui_run(
                    db_url,
                    ui_run_id,
                    finished_at=datetime.utcnow(),
                    status="failed",
                    error=f"forecaster_unexpected_error:{e!r}",
                )
            except Exception:
                logging.exception("ui_runs: failed to record unexpected failure for %s", ui_run_id)
    finally:
        if prev_ui is None:
            os.environ.pop("PYTHIA_UI_RUN_ID", None)
        else:
            os.environ["PYTHIA_UI_RUN_ID"] = prev_ui


def enqueue_run(countries: list[str]) -> str:
    ui_run_id = f"ui_run_{date.today().isoformat()}_{uuid.uuid4().hex[:8]}"
    logging.info("enqueue_run: scheduling ui_run_id=%s for countries=%r", ui_run_id, countries)
    _pool.submit(_pipeline, ui_run_id, countries)
    return ui_run_id
