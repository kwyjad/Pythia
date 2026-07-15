# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""FastAPI application assembly for the Pythia API.

Decomposed (July 2026) into:

* ``pythia.api.core`` — shared infrastructure: the cached read-connection
  machinery, ``_execute`` (with retry-once connection resilience), table and
  bucket helpers, the heavy-request semaphore, CSV streaming, and parsing
  utilities.
* ``pythia.api.routes.*`` — route-group modules, each exposing ``router``.

This module keeps: app creation + CORS, the startup hook, the routes that
are entangled with connection state or test patch seams
(``/v1/admin/force_sync``, ``/v1/health``, ``/v1/version``, ``/v1/run``,
``/v1/ui_runs/{ui_run_id}``, ``/v1/countries``), the ``include_router``
calls (in the original registration order), and backward-compat re-exports
of every moved name so ``from pythia.api.app import X`` keeps working.

Mutable module state that used to live here (``_READ_CON``,
``_READ_CON_MTIME``, ``_LAST_SYNC_CHECK``, ``_POPULATION_BY_ISO3``,
``_COUNTRY_NAME_BY_ISO3``) now lives in ``pythia.api.core``. Reads and
writes of those names on this module are forwarded to core via module-class
properties (see the bottom of this file), so the legacy seam
``pythia.api.app._READ_CON = None`` still resets the real singleton.
"""

import json
import logging
import math
import re
import sys
import threading
import time
from datetime import datetime
from importlib.util import find_spec
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import os
import resource

import duckdb, pandas as pd
import numpy as np
from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse

from pythia.api.auth import require_admin_token
from pythia.api import db_sync as _db_sync_mod
from pythia.api.db_sync import (
    DbSyncError,
    db_was_refreshed,
    get_cached_latest_hs,
    get_cached_manifest,
    get_sync_status,
    maybe_sync_latest_db,
)
from pythia.api.models import (
    ContextBundle,
    ForecastBundle,
    HsBundle,
    LlmCallsBundle,
    QuestionBundleResponse,
)
from pythia.db.schema import connect as db_connect, ensure_schema
from pythia.db.util import ensure_llm_calls_columns
from pythia.config import load as load_cfg
from resolver.query.countries_index import compute_countries_index
from resolver.query.costs import (
    COST_COLUMNS,
    build_costs_monthly,
    build_costs_runs,
    build_costs_total,
    build_latencies_runs,
    build_run_runtimes,
)
from resolver.query.downloads import (
    build_ensemble_scores_export,
    build_forecast_spd_export,
    build_model_scores_export,
    build_rationale_export,
    build_triage_export,
)
from resolver.query.kpi_scopes import compute_countries_triaged_for_month_with_source
from resolver.query.questions_index import (
    compute_questions_forecast_summary,
    compute_questions_triage_summary,
)
from resolver.query.debug_ui import (
    _get_hs_triage_llm_calls_with_debug,
    _get_hs_triage_rows_with_debug,
    _list_hs_runs_with_debug,
    get_hs_triage_all,
    get_country_run_summary,
    list_hs_runs,
)
from resolver.query.resolver_ui import get_connector_last_updated, get_country_facts
from pythia.buckets import BUCKET_SPECS
from resolver.query import eiv_sql

# ---------------------------------------------------------------------------
# Shared infrastructure (moved to pythia.api.core). ``_core`` is imported as
# a module so live mutable state can be reached at call time; the ``from``
# import below re-exports every moved name for backward compatibility —
# tests and tools import/monkeypatch these via ``pythia.api.app``.
# ---------------------------------------------------------------------------
from pythia.api import core as _core
from pythia.api.core import (  # noqa: F401
    _DUCKDB_MEMORY_LIMIT,
    _DUCKDB_THREADS,
    _HEAVY_REQUEST_SEMAPHORE,
    _HealthAccessFilter,
    _ISO3_PATTERN,
    _READ_CON_LOCK,
    _SYNC_CHECK_INTERVAL_S,
    _acquire_heavy,
    _apply_json_fields,
    _bucket_centroids,
    _bucket_labels,
    _compile_named_params,
    _con,
    _concat_cost_tables,
    _count_distinct_active_questions,
    _country_name,
    _db_file_mtime,
    _ensure_read_connection,
    _execute,
    _execute_on,
    _fetch_one,
    _format_year_month_label,
    _is_connection_level_error,
    _json_sanitize,
    _latest_available_horizon,
    _latest_forecasted_target_month,
    _latest_questions_view,
    _load_country_registry,
    _load_population_registry,
    _maybe_refresh_db,
    _month_window,
    _nonnull_count,
    _open_duckdb_connection,
    _parse_year_month,
    _pick_col,
    _pick_timestamp_column,
    _population,
    _reopen_read_connection,
    _require_debug_token,
    _resolve_forecaster_run_id,
    _resolve_forecasts_ensemble_columns,
    _resolve_latest_questions_columns,
    _rows_from_cursor,
    _rows_from_df,
    _run_filter_cte,
    _safe_count_distinct,
    _safe_json_load,
    _shift_ym,
    _stream_csv,
    _table_columns,
    _table_exists,
    _table_has_columns,
    _test_filter,
    _validate_iso3_param,
)

app = FastAPI(title="Pythia API", version="1.0.0")
cors_origins_env = os.getenv("PYTHIA_CORS_ALLOW_ORIGINS", "*").strip()
cors_origins = (
    ["*"]
    if cors_origins_env in ("", "*")
    else [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins or ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
logger = logging.getLogger(__name__)


def _try_artifact_sync() -> None:
    """Fallback: sync DB from CI artifacts when Release-based sync is unavailable.

    Opt-in via ``PYTHIA_SYNC_FROM_ARTIFACTS=1``.  Requires the ``gh`` CLI.
    """
    if os.environ.get("PYTHIA_SYNC_FROM_ARTIFACTS", "").strip() not in ("1", "true", "yes"):
        return
    try:
        from scripts.sync_db import sync  # noqa: C0415

        sync()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Artifact-based DB sync failed: %s", exc)


@app.post("/v1/admin/force_sync")
def admin_force_sync(token: Optional[str] = Query(None)):
    """Force an immediate DB sync from the GitHub release, bypassing the throttle.

    Also forcibly reopens the DuckDB read connection if the on-disk DB file
    is newer than the connection's captured mtime — mirroring the mtime
    fallback in ``_maybe_refresh_db``. This matters because
    ``db_was_refreshed()`` is a consume-once flag: another request can
    consume it before this handler runs, which previously left the
    connection pinned to the pre-sync inode even though a fresh DB was on
    disk. Now we reopen whenever EITHER the flag is set OR the file mtime
    advanced past what the current connection captured.
    """
    # NOTE(code-motion): the connection-state globals this endpoint mutates
    # (_READ_CON, _READ_CON_MTIME, _LAST_SYNC_CHECK) now live in
    # pythia.api.core; the former ``global`` assignments are expressed as
    # attribute assignments on the core module — identical semantics, single
    # authoritative state. Kept out of the docstring so the OpenAPI
    # description stays byte-identical to the pre-decomposition schema.
    _require_debug_token(token)
    _core._LAST_SYNC_CHECK = None
    _db_sync_mod._LAST_SYNC_AT = None  # noqa: SLF001
    _db_sync_mod._LAST_MANIFEST = None  # noqa: SLF001
    try:
        manifest = maybe_sync_latest_db()
    except DbSyncError as exc:
        raise HTTPException(status_code=502, detail=f"Sync failed: {exc}") from exc
    flag_refreshed = db_was_refreshed()
    current_mtime = _db_file_mtime()
    mtime_newer = (
        current_mtime is not None
        and _core._READ_CON_MTIME is not None
        and current_mtime > _core._READ_CON_MTIME
    )
    db_refreshed = flag_refreshed or mtime_newer
    if db_refreshed:
        # Close-then-open (core._swap_read_connection): DuckDB caches database
        # instances per path, so opening the new connection while the old one
        # is still open would hand back the SAME stale instance reading the
        # pre-sync inode.
        if not _core._swap_read_connection():
            logger.warning("Failed to reopen DuckDB after force sync")
            raise HTTPException(status_code=502, detail="DB reopen failed")
        logger.info(
            "Force sync: DB refreshed and connection reopened (flag=%s mtime_newer=%s)",
            flag_refreshed, mtime_newer,
        )
    return {
        "status": "ok",
        "manifest": manifest,
        "db_refreshed": db_refreshed,
        "flag_refreshed": flag_refreshed,
        "mtime_newer": mtime_newer,
    }


@app.on_event("startup")
def _startup_sync():
    try:
        maybe_sync_latest_db()
    except DbSyncError as exc:
        logger.warning("DB sync failed during startup: %s", exc)
        _try_artifact_sync()
    con = None
    try:
        con = db_connect(read_only=False)
        con.execute(f"SET memory_limit='{_DUCKDB_MEMORY_LIMIT}'")
        con.execute(f"SET threads={_DUCKDB_THREADS}")
        ensure_schema(con)
        ensure_llm_calls_columns(con)
    except Exception as exc:  # noqa: BLE001
        logger.warning("DB schema sync failed during startup: %s", exc)
    finally:
        try:
            if con is not None:
                con.close()
        except Exception:
            pass


@app.get("/v1/health")
def health():
    """Liveness + DB-sync health.

    Always returns HTTP 200 so Render's health check does not kill the pod for
    a recoverable data-sync problem, but flips ``status`` to ``"degraded"`` and
    populates ``reason``/``sync`` when the most recent DB sync errored or the
    on-disk DB has drifted behind the published release. This makes a silent
    sync failure (e.g. a wedged temp path freezing the API on a stale DB)
    visible without trawling the server logs.
    """
    sync = get_sync_status()
    degraded = bool(sync.get("last_error")) or sync.get("in_sync") is False
    if not degraded:
        return {"ok": True, "status": "ok", "sync": sync}
    return {
        "ok": True,
        "status": "degraded",
        "reason": "db_sync_failing",
        "sync": sync,
    }


# Staleness probes (full-column MAX scans over 6 tables) are cached per DB
# version: the dashboard homepage is force-dynamic and calls /v1/version on
# every page view, which used to re-run all the scans each time. The DB only
# changes when the sync layer os.replace()s the file, which advances its
# mtime — the same freshness signal _maybe_refresh_db trusts.
_VERSION_PROBE_LOCK = threading.Lock()
# {include_test: (db_mtime, probes dict)} — at most two entries. Tests reset
# it by assigning None.
_VERSION_PROBE_CACHE: Optional[Dict[bool, tuple]] = None

# Probe keys carrying data (used for the "don't cache an all-None result"
# check; excludes the _hs_probe_ok bookkeeping flag).
_PROBE_DATA_KEYS = (
    "latest_forecast_month",
    "latest_forecast_run_id",
    "latest_scores_at",
    "latest_calibration_at",
    "latest_resolutions_at",
    "latest_advice_at",
    "latest_forecast_at",
    "latest_hs_run_id",
    "latest_hs_created_at",
)


def _compute_staleness_probes(include_test: bool = False) -> Dict[str, Any]:
    def _probe_test_filter(table: str) -> str:
        """Test-exclusion clause for a probe query ('' when include_test or
        the table has no is_test column — old DBs and minimal test fixtures)."""
        if include_test:
            return ""
        try:
            if _table_has_columns(_con(), table, ["is_test"]):
                return " AND COALESCE(is_test, FALSE) = FALSE"
        except Exception:
            return ""
        return ""

    probes: Dict[str, Any] = {}
    try:
        con = _con()
        fe_tf = _probe_test_filter("forecasts_ensemble")
        row = con.execute(
            "SELECT MAX(strftime(created_at, '%Y-%m')) FROM forecasts_ensemble "
            f"WHERE created_at IS NOT NULL{fe_tf}"
        ).fetchone()
        probes["latest_forecast_month"] = row[0] if row and row[0] else None
        run_row = con.execute(
            f"SELECT run_id FROM forecasts_ensemble WHERE run_id IS NOT NULL{fe_tf} "
            "ORDER BY run_id DESC LIMIT 1"
        ).fetchone()
        probes["latest_forecast_run_id"] = run_row[0] if run_row and run_row[0] else None
    except Exception:
        probes["latest_forecast_month"] = None
        probes["latest_forecast_run_id"] = None

    def _max_created_at(table: str) -> Optional[str]:
        try:
            con_local = _con()
            r = con_local.execute(
                f"SELECT strftime(MAX(created_at), '%Y-%m-%dT%H:%M:%S') "
                f"FROM {table} WHERE created_at IS NOT NULL{_probe_test_filter(table)}"
            ).fetchone()
            return r[0] if r and r[0] else None
        except Exception as exc:
            # A failed staleness probe silently under-reports latest_data_at;
            # log so the gap is diagnosable.
            logger.warning("staleness probe failed for %s: %r", table, exc)
            return None

    probes["latest_scores_at"] = _max_created_at("scores")
    probes["latest_calibration_at"] = _max_created_at("calibration_weights")
    probes["latest_resolutions_at"] = _max_created_at("resolutions")
    probes["latest_advice_at"] = _max_created_at("calibration_advice")
    probes["latest_forecast_at"] = _max_created_at("forecasts_ensemble")

    # Test-aware latest-HS probe. The manifest's latest_hs_run_id /
    # latest_hs_created_at (and the sync layer's DB introspection) have no
    # is_test concept, so with the test filter active /v1/version overrides
    # them with these values — a test-mode publish must not advance the
    # dashboard's "Last updated"/"Latest forecast scan" when Test is OFF.
    # _hs_probe_ok=True means the probe ran against a usable hs_runs table
    # (None values then mean "no matching runs", not "probe unavailable").
    probes["latest_hs_run_id"] = None
    probes["latest_hs_created_at"] = None
    probes["_hs_probe_ok"] = False
    try:
        con = _con()
        if _table_exists(con, "hs_runs"):
            cols = {
                r[0]
                for r in con.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'hs_runs'"
                ).fetchall()
            }
            id_col = next((c for c in ("hs_run_id", "run_id") if c in cols), None)
            ts_col = next(
                (c for c in ("created_at", "finished_at", "started_at", "generated_at") if c in cols),
                None,
            )
            if id_col and ts_col:
                r = con.execute(
                    f"SELECT {id_col}, strftime({ts_col}, '%Y-%m-%dT%H:%M:%S') FROM hs_runs "
                    f"WHERE {ts_col} IS NOT NULL{_probe_test_filter('hs_runs')} "
                    f"ORDER BY {ts_col} DESC LIMIT 1"
                ).fetchone()
                probes["_hs_probe_ok"] = True
                if r:
                    probes["latest_hs_run_id"] = r[0]
                    probes["latest_hs_created_at"] = r[1]
    except Exception as exc:
        logger.warning("staleness probe failed for hs_runs: %r", exc)
    return probes


def _staleness_probes(include_test: bool = False) -> Dict[str, Any]:
    global _VERSION_PROBE_CACHE
    # Cache key = the DB mtime captured when the read connection was OPENED
    # (core._READ_CON_MTIME), not the raw on-disk mtime. Right after a publish
    # swaps the DB file, the throttled _maybe_refresh_db may not have reopened
    # the connection yet; keying on the file mtime would compute probes against
    # the OLD inode and pin them under the NEW DB version until the next
    # publish (the cache-hit early return never recomputes). Keyed on the
    # connection's own mtime, pre-reopen requests cache under the old version
    # and the first post-reopen request recomputes against fresh data.
    try:
        _core._maybe_refresh_db()
        _core._ensure_read_connection()
    except Exception as exc:
        logger.warning("staleness probe connection setup failed: %r", exc)
    mtime = _core._READ_CON_MTIME
    with _VERSION_PROBE_LOCK:
        cache = _VERSION_PROBE_CACHE or {}
        entry = cache.get(include_test)
        if entry is not None and mtime is not None and entry[0] == mtime:
            return entry[1]
    probes = _compute_staleness_probes(include_test)
    # Don't cache an all-None result: it usually means a transient connection
    # failure, and caching it would blank the staleness fields until the next
    # DB refresh. (A genuinely empty DB re-probes each call — it's cheap there.)
    if mtime is not None and any(probes.get(k) is not None for k in _PROBE_DATA_KEYS):
        with _VERSION_PROBE_LOCK:
            if _VERSION_PROBE_CACHE is None:
                _VERSION_PROBE_CACHE = {}
            _VERSION_PROBE_CACHE[include_test] = (mtime, probes)
    return probes


@app.get("/v1/version")
def api_version(include_test: bool = Query(False)) -> Dict[str, Any]:
    try:
        manifest = maybe_sync_latest_db()
    except DbSyncError as exc:
        manifest = get_cached_manifest()
        if not manifest:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not manifest:
        raise HTTPException(status_code=503, detail="Manifest not available yet")
    result = dict(manifest)
    # Surface DB-sync health so a stale/drifted API is obvious at a glance.
    result["sync_status"] = get_sync_status()
    # Add DB staleness diagnostics so operators can verify the API has the
    # latest data. Probes are cached per DB version (see _staleness_probes).
    probes = _staleness_probes(include_test)
    result["latest_forecast_month"] = probes["latest_forecast_month"]
    result["latest_forecast_run_id"] = probes["latest_forecast_run_id"]

    # With the test filter active (the default), the manifest-derived latest-HS
    # fields are replaced by the test-aware hs_runs probe: the manifest and the
    # sync layer's introspection have no is_test concept, so a test-mode
    # publish would otherwise advance "Last updated"/"Latest forecast scan"
    # while the dashboard (Test OFF) shows no new data — the July 2026
    # "dashboard stuck on yesterday's test run" confusion. When the probe
    # cannot run (hs_runs missing/unusable), the manifest values stand.
    if not include_test and probes.get("_hs_probe_ok"):
        result["latest_hs_run_id"] = probes["latest_hs_run_id"]
        result["latest_hs_created_at"] = probes["latest_hs_created_at"]

    # "Last updated" should reflect the most recent *pipeline activity*, not just
    # the last Horizon Scanner run. The scoring/calibration loop updates the data
    # the dashboard shows (resolutions/scores/calibration) without creating a new
    # HS run, so latest_hs_created_at alone under-reports freshness. Surface a
    # unified latest_data_at = MAX(created_at) across the contributing tables.
    # NOTE: deliberately excludes the manifest's `created_utc` (publish/repackage
    # time), which is later than and distinct from when the data was computed.
    result["latest_scores_at"] = probes["latest_scores_at"]
    result["latest_calibration_at"] = probes["latest_calibration_at"]

    candidates = [
        result.get("latest_hs_created_at"),
        probes["latest_forecast_at"],
        probes["latest_resolutions_at"],
        result["latest_scores_at"],
        result["latest_calibration_at"],
        probes["latest_advice_at"],
    ]
    normalized = [str(c)[:19] for c in candidates if c]
    result["latest_data_at"] = max(normalized) if normalized else None
    return result


@app.post("/v1/run")
def start_run(payload: dict = Body(...), _=Depends(require_admin_token)):
    if os.getenv("PYTHIA_ALLOW_INPROCESS_RUN", "0").strip().lower() not in ("1", "true", "yes"):
        raise HTTPException(
            status_code=503,
            detail=(
                "In-process pipeline runs are disabled on this deployment "
                "(set PYTHIA_ALLOW_INPROCESS_RUN=1 to enable)"
            ),
        )
    # Deferred import: pythia.pipeline.run pulls in the full horizon_scanner
    # + calibration module tree (~100-300MB RSS), which must never load in
    # the memory-constrained API process unless a run is explicitly allowed.
    from pythia.pipeline.run import enqueue_run

    countries = payload.get("countries") or []
    run_id = enqueue_run(countries)
    return {"accepted": True, "run_id": run_id}


@app.get("/v1/ui_runs/{ui_run_id}")
def get_ui_run(ui_run_id: str):
    """
    Return status for a given ui_run_id created by /v1/run.

    Response shape:
      - found: bool
      - row: dict | None (full ui_runs row if found)
    """
    con = _con()
    rows = _rows_from_cursor(con.execute(
        "SELECT * FROM ui_runs WHERE ui_run_id = ?",
        [ui_run_id],
    ))
    if not rows:
        return {"found": False, "row": None}
    return {"found": True, "row": rows[0]}



# ---------------------------------------------------------------------------
# Route groups. Import order is irrelevant; ``include_router`` order below
# reproduces the original monolithic registration order exactly:
# admin/meta (defined above) -> questions -> calibration+performance ->
# forecasts -> risk index -> diagnostics -> /v1/countries (defined below) ->
# resolver explorer -> downloads -> costs.
# ---------------------------------------------------------------------------
from pythia.api.routes import (  # noqa: E402
    costs as _costs_routes,
    diagnostics as _diagnostics_routes,
    downloads as _downloads_routes,
    forecasts as _forecasts_routes,
    performance as _performance_routes,
    questions as _questions_routes,
    resolver_explorer as _resolver_explorer_routes,
    risk_index as _risk_index_routes,
    sibyl as _sibyl_routes,
)

app.include_router(_questions_routes.router)
app.include_router(_performance_routes.router)
app.include_router(_forecasts_routes.router)
app.include_router(_risk_index_routes.router)
app.include_router(_diagnostics_routes.router)


@app.get("/v1/countries")
def get_countries(
    metric_scope: Optional[str] = Query(None),
    year_month: Optional[str] = Query(None),
    forecaster_run_id: Optional[str] = Query(None, description="Forecaster run ID to scope countries"),
    include_test: bool = Query(False),
):
    con = _con()
    try:
        rows = compute_countries_index(
            con, metric_scope=metric_scope, year_month=year_month,
            forecaster_run_id=forecaster_run_id,
            include_test=include_test,
        )
        return {"rows": rows}
    except Exception:
        logger.exception("Failed to compute countries index, falling back.")

    if not _table_exists(con, "questions"):
        return {"rows": []}

    where_bits: list[str] = []
    params: list[str] = []
    if metric_scope:
        where_bits.append("UPPER(q.metric) = ?")
        params = [metric_scope.upper()]
    if not include_test and _table_has_columns(con, "questions", ["is_test"]):
        where_bits.append("COALESCE(q.is_test, FALSE) = FALSE")
    metric_filter = f"WHERE {' AND '.join(where_bits)}" if where_bits else ""

    if _table_exists(con, "forecasts_ensemble"):
        fe_test = (
            " AND COALESCE(fe.is_test, FALSE) = FALSE"
            if not include_test and _table_has_columns(con, "forecasts_ensemble", ["is_test"])
            else ""
        )
        sql = f"""
          SELECT
            q.iso3,
            COUNT(DISTINCT q.question_id) AS n_questions,
            COUNT(DISTINCT fe.question_id) AS n_forecasted
          FROM questions q
          LEFT JOIN forecasts_ensemble fe ON fe.question_id = q.question_id{fe_test}
          {metric_filter}
          GROUP BY q.iso3
          ORDER BY q.iso3
        """
    else:
        sql = f"""
          SELECT
            q.iso3,
            COUNT(DISTINCT q.question_id) AS n_questions,
            0 AS n_forecasted
          FROM questions q
          {metric_filter}
          GROUP BY q.iso3
          ORDER BY q.iso3
        """

    return {"rows": _rows_from_cursor(con.execute(sql, params))}


app.include_router(_resolver_explorer_routes.router)
app.include_router(_downloads_routes.router)
app.include_router(_costs_routes.router)
app.include_router(_sibyl_routes.router)

# ---------------------------------------------------------------------------
# Backward-compat re-exports of names moved into route modules.
# ---------------------------------------------------------------------------
from pythia.api.routes.questions import (  # noqa: E402,F401
    _build_llm_calls_bundle,
    _question_bundle_impl,
    _resolve_question_row,
    get_question_bundle,
    get_questions,
)
from pythia.api.routes.performance import (  # noqa: E402,F401
    get_calibration_advice,
    get_calibration_weights,
    performance_scores,
)
from pythia.api.routes.forecasts import (  # noqa: E402,F401
    get_forecasts_ensemble,
    get_forecasts_history,
    list_resolutions,
)
from pythia.api.routes.risk_index import (  # noqa: E402,F401
    _get_risk_index_binary,
    get_risk_index,
    rankings,
)
from pythia.api.routes.diagnostics import (  # noqa: E402,F401
    debug_hs_country_summary,
    debug_hs_runs,
    debug_hs_triage,
    debug_hs_triage_llm_calls,
    diagnostics_kpi_scopes,
    diagnostics_memory,
    diagnostics_run_summary,
    diagnostics_summary,
    hs_runs,
    hs_triage_all,
    resolution_rates,
)
from pythia.api.routes.resolver_explorer import (  # noqa: E402,F401
    _DB_SUMMARY_TABLES,
    _FRESHNESS_CANDIDATES,
    _SOURCE_CATEGORIES,
    _SOURCE_LABELS,
    _SOURCE_REGISTRY,
    _best_freshness_column,
    _resolver_query,
    _source_freshness,
    _source_row_count,
    _validated_columns,
    get_resolver_acaps,
    get_resolver_acled_monthly_fatalities,
    get_resolver_acled_political_events,
    get_resolver_conflict_forecasts,
    get_resolver_connector_status,
    get_resolver_country_facts,
    get_resolver_crisiswatch,
    get_resolver_db_summary,
    get_resolver_enso_state,
    get_resolver_facts_deltas,
    get_resolver_hdx_signals,
    get_resolver_reliefweb_reports,
    get_resolver_seasonal_forecasts,
    get_resolver_seasonal_tc_outlooks,
    get_resolver_source_data,
    get_resolver_source_inventory,
)
from pythia.api.routes.downloads import (  # noqa: E402,F401
    download_forecasts_csv,
    download_forecasts_xlsx,
    download_monthly_costs_csv,
    download_rationales_csv,
    download_run_costs_csv,
    download_scores_ensemble_bayesmc_csv,
    download_scores_ensemble_mean_csv,
    download_scores_model_csv,
    download_total_costs_csv,
    download_triage_csv,
)
from pythia.api.routes.costs import (  # noqa: E402,F401
    costs_latencies,
    costs_monthly,
    costs_run_runtimes,
    costs_runs,
    costs_total,
    llm_costs,
    llm_costs_summary,
)

# ---------------------------------------------------------------------------
# Live-state forwarding for legacy mutable globals.
#
# The names below are *rebound* (not mutated in place) by code that now lives
# in pythia.api.core, and several tests/tools assign them directly on this
# module (e.g. ``pythia.api.app._READ_CON = None`` to force a fresh
# connection, or replacing ``_POPULATION_BY_ISO3``). A plain
# ``from core import X`` re-export would break both directions: reads would
# go stale once core rebinds, and writes would only rebind the alias here.
# Module-class properties keep every read AND write on this module pointed at
# the single authoritative copy in pythia.api.core. (Assigning
# ``module.__class__`` to a ModuleType subclass is officially supported.)
# ---------------------------------------------------------------------------
import types as _types  # noqa: E402


class _AppModule(_types.ModuleType):
    """Module subclass forwarding legacy state attributes to pythia.api.core."""


def _forward_core_state(name: str) -> property:
    def _get(_module):
        return getattr(_core, name)

    def _set(_module, value):
        setattr(_core, name, value)

    return property(_get, _set)


for _state_name in (
    "_READ_CON",
    "_READ_CON_MTIME",
    "_LAST_SYNC_CHECK",
    "_POPULATION_BY_ISO3",
    "_COUNTRY_NAME_BY_ISO3",
):
    setattr(_AppModule, _state_name, _forward_core_state(_state_name))

sys.modules[__name__].__class__ = _AppModule
