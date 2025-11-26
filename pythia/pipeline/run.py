from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import logging
import subprocess
import sys
import uuid

from pythia.db.init import init as init_db
from pythia.config import load as load_cfg
from horizon_scanner.horizon_scanner import main as hs_main

_pool = ThreadPoolExecutor(max_workers=1)


def _pipeline(countries: list[str]):
    """
    End-to-end Pythia pipeline for a UI/API-triggered run.

    Steps:
      1) Ensure DuckDB schema exists (init_db).
      2) Run Horizon Scanner to write scenarios + questions into DuckDB.
         - If `countries` is empty, HS falls back to hs_country_list.txt.
         - If non-empty, HS uses the provided list as country names.
      3) Run Forecaster in Pythia mode to generate SPD ensemble forecasts
         for all active questions in the `questions` table and write them
         into `forecasts_ensemble`.

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

    # 1) Ensure DuckDB schema exists (idempotent)
    try:
        init_db(db_url)
        logging.info("DuckDB schema initialised via init_db(%s).", db_url)
    except Exception as e:
        logging.exception("init_db(%s) failed; aborting pipeline.", db_url)
        return

    # 2) Run Horizon Scanner to write hs_runs, hs_scenarios, questions
    try:
        # Horizon Scanner will also call init_db(db_url) internally and respect app.db_url.
        # If `countries` is empty, we pass None so HS falls back to hs_country_list.txt.
        hs_countries = countries or None
        hs_main(hs_countries)
        logging.info("Horizon Scanner completed in pipeline.")
    except Exception:
        logging.exception("Horizon Scanner failed inside pipeline; aborting before Forecaster.")
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
        logging.info("Starting Forecaster in Pythia mode: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        logging.info("Forecaster (Pythia mode) completed successfully in pipeline.")
    except subprocess.CalledProcessError as e:
        logging.exception("Forecaster subprocess failed in pipeline: %s", e)
    except Exception:
        logging.exception("Unexpected error while running Forecaster in pipeline.")


def enqueue_run(countries: list[str]) -> str:
    run_id = f"ui_run_{date.today().isoformat()}_{uuid.uuid4().hex[:8]}"
    _pool.submit(_pipeline, countries)
    return run_id
