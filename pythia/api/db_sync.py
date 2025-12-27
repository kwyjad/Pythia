# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from pythia.config import load as load_cfg
from resolver.db import duckdb_io

_DEFAULT_DB_ASSET = "resolver.duckdb"
_DEFAULT_MANIFEST_ASSET = "manifest.json"
_DEFAULT_SYNC_INTERVAL_S = 60

_LAST_MANIFEST: Optional[Dict[str, Any]] = None
_LAST_SYNC_AT: Optional[float] = None
_LATEST_RUNS: Optional[Dict[str, Any]] = None
_SYNC_LOCK = threading.Lock()

logger = logging.getLogger(__name__)


class DbSyncError(RuntimeError):
    pass


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _db_path_from_config() -> Path:
    cfg = load_cfg()
    db_url = cfg.get("app", {}).get("db_url")
    if not db_url:
        raise DbSyncError("Database URL not configured")
    if isinstance(db_url, str):
        return Path(db_url.replace("duckdb:///", "", 1))
    raise DbSyncError("Database URL is invalid")


def _build_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    token = _get_env("PYTHIA_GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def get_release_urls() -> Dict[str, str]:
    repo = _get_env("PYTHIA_DATA_REPO")
    tag = _get_env("PYTHIA_DATA_RELEASE_TAG")
    if not repo:
        raise DbSyncError("PYTHIA_DATA_REPO is not set")
    if not tag:
        raise DbSyncError("PYTHIA_DATA_RELEASE_TAG is not set")

    db_asset = _get_env("PYTHIA_DATA_DB_ASSET", _DEFAULT_DB_ASSET) or _DEFAULT_DB_ASSET
    manifest_asset = (
        _get_env("PYTHIA_DATA_MANIFEST_ASSET", _DEFAULT_MANIFEST_ASSET)
        or _DEFAULT_MANIFEST_ASSET
    )

    base = f"https://github.com/{repo}/releases/download/{tag}"
    return {
        "repo": repo,
        "tag": tag,
        "db_asset": db_asset,
        "manifest_asset": manifest_asset,
        "db_url": f"{base}/{db_asset}",
        "manifest_url": f"{base}/{manifest_asset}",
    }


def fetch_manifest() -> Dict[str, Any]:
    urls = get_release_urls()
    try:
        response = requests.get(urls["manifest_url"], headers=_build_headers(), timeout=30)
        response.raise_for_status()
    except Exception as exc:
        raise DbSyncError(f"Failed to download manifest: {exc}") from exc

    try:
        manifest = response.json()
    except json.JSONDecodeError as exc:
        raise DbSyncError(f"Manifest JSON decode failed: {exc}") from exc

    if not isinstance(manifest, dict):
        raise DbSyncError("Manifest JSON must be an object")

    manifest.setdefault("release_tag", urls["tag"])
    manifest.setdefault("db_asset", urls["db_asset"])
    manifest.setdefault("manifest_asset", urls["manifest_asset"])
    return manifest


def _manifest_key(manifest: Dict[str, Any]) -> Optional[str]:
    if "db_sha256" in manifest and manifest["db_sha256"]:
        return str(manifest["db_sha256"])
    if "latest_hs_run_id" in manifest and manifest["latest_hs_run_id"]:
        return str(manifest["latest_hs_run_id"])
    return None


def download_db_atomic(dest_path: Path) -> None:
    urls = get_release_urls()
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(urls["db_url"], headers=_build_headers(), stream=True, timeout=60)
        response.raise_for_status()
        with tmp_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    except Exception as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise DbSyncError(f"Failed to download DuckDB asset: {exc}") from exc

    os.replace(tmp_path, dest_path)


def get_cached_manifest() -> Optional[Dict[str, Any]]:
    if _LAST_MANIFEST is None:
        return None
    return _merge_manifest_with_latest_runs(dict(_LAST_MANIFEST))


def get_cached_latest_hs() -> Optional[Dict[str, Any]]:
    manifest = get_cached_manifest()
    if not manifest:
        return None
    run_id = manifest.get("latest_hs_run_id")
    created_at = manifest.get("latest_hs_created_at")
    if not run_id and not created_at:
        return None
    return {"run_id": run_id, "created_at": created_at, "meta": None}


def _merge_manifest_with_latest_runs(manifest: Dict[str, Any]) -> Dict[str, Any]:
    if not _LATEST_RUNS:
        return manifest
    merged = dict(manifest)
    merged.update(_LATEST_RUNS)
    return merged


def _db_url_from_path(db_path: Path) -> str:
    return f"duckdb:///{db_path}"


def _table_exists(conn: "duckdb_io.duckdb.DuckDBPyConnection", table: str) -> bool:
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = ?
        """,
        [table],
    ).fetchone()
    return bool(row and row[0])


def _table_columns(conn: "duckdb_io.duckdb.DuckDBPyConnection", table: str) -> set[str]:
    rows = conn.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'main' AND table_name = ?
        """,
        [table],
    ).fetchall()
    return {row[0] for row in rows}


def _pick_column(columns: set[str], candidates: list[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _fetch_latest_run(
    conn: "duckdb_io.duckdb.DuckDBPyConnection",
    table: str,
    id_candidates: list[str],
    time_candidates: list[str],
    where_clause: str = "",
) -> Optional[Dict[str, Any]]:
    if not _table_exists(conn, table):
        return None
    columns = _table_columns(conn, table)
    id_col = _pick_column(columns, id_candidates)
    time_col = _pick_column(columns, time_candidates)
    if not id_col or not time_col:
        return None
    where_sql = f"WHERE {where_clause}" if where_clause else ""
    row = conn.execute(
        f"""
        SELECT {id_col} AS run_id,
               {time_col} AS created_at
        FROM {table}
        {where_sql}
        ORDER BY {time_col} DESC NULLS LAST
        LIMIT 1
        """
    ).fetchone()
    if not row:
        return None
    return {"run_id": row[0], "created_at": row[1]}


def _refresh_latest_runs(db_path: Path) -> None:
    global _LATEST_RUNS
    try:
        conn = duckdb_io.get_db(_db_url_from_path(db_path))
    except Exception as exc:
        logger.warning("Failed to open DuckDB for latest run introspection: %s", exc)
        return
    try:
        latest: Dict[str, Any] = {}
        hs_latest = _fetch_latest_run(
            conn,
            "hs_runs",
            ["run_id", "hs_run_id"],
            ["created_at", "finished_at", "started_at", "generated_at"],
        )
        if hs_latest:
            latest["latest_hs_run_id"] = hs_latest["run_id"]
            latest["latest_hs_created_at"] = hs_latest["created_at"]

        forecast_latest = _fetch_latest_run(
            conn,
            "llm_calls",
            ["forecaster_run_id"],
            ["created_at"],
            where_clause="forecaster_run_id IS NOT NULL",
        )
        if not forecast_latest:
            forecast_latest = _fetch_latest_run(
                conn,
                "forecasts_raw",
                ["run_id"],
                ["created_at"],
            )
        if forecast_latest:
            latest["latest_forecast_run_id"] = forecast_latest["run_id"]
            latest["latest_forecast_created_at"] = forecast_latest["created_at"]

        _LATEST_RUNS = latest
    except Exception as exc:
        logger.warning("Failed to introspect latest runs: %s", exc)
    finally:
        duckdb_io.close_db(conn)


def maybe_sync_latest_db() -> Optional[Dict[str, Any]]:
    global _LAST_MANIFEST, _LAST_SYNC_AT
    global _LATEST_RUNS

    dest_path = _db_path_from_config()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    interval_s = int(_get_env("PYTHIA_DATA_SYNC_INTERVAL_S", str(_DEFAULT_SYNC_INTERVAL_S)) or 0)

    with _SYNC_LOCK:
        now = time.monotonic()
        if _LAST_SYNC_AT is not None and now - _LAST_SYNC_AT < interval_s:
            return get_cached_manifest()
        _LAST_SYNC_AT = now

        manifest = fetch_manifest()
        manifest_key = _manifest_key(manifest)
        last_key = _manifest_key(_LAST_MANIFEST) if _LAST_MANIFEST else None
        should_download = not dest_path.exists()
        if manifest_key and last_key:
            if manifest_key != last_key:
                should_download = True
        elif manifest_key and not last_key:
            should_download = True
        elif not manifest_key and _LAST_MANIFEST is None:
            should_download = not dest_path.exists()

        if should_download:
            download_db_atomic(dest_path)

        if should_download or _LATEST_RUNS is None:
            _refresh_latest_runs(dest_path)

        _LAST_MANIFEST = dict(manifest)
        return get_cached_manifest()
