# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
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
# Headroom kept free beyond the staged copy itself: DuckDB opens the swapped
# file and may write a WAL, and the export cache can share the disk.
_DISK_HEADROOM_BYTES = 64 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_LAST_MANIFEST: Optional[Dict[str, Any]] = None
_LAST_SYNC_AT: Optional[float] = None
_LATEST_RUNS: Optional[Dict[str, Any]] = None
_SYNC_LOCK = threading.Lock()
_DB_REFRESHED = threading.Event()

# Sync-health observability. These record the outcome of the most recent sync
# attempt so operators can see failures via /v1/version and /v1/health without
# having to read the server logs. ``_LAST_DOWNLOADED_KEY`` is the manifest key
# (db_sha256, falling back to latest_hs_run_id) of the DB currently on disk; it
# is compared against the live manifest to derive ``in_sync`` cheaply, without
# re-hashing the (large) DB file on every call.
_LAST_SYNC_ERROR: Optional[str] = None
_LAST_SYNC_OK_AT: Optional[str] = None
_LAST_SYNC_ATTEMPT_AT: Optional[str] = None
# Key (db_sha256, falling back to latest_hs_run_id) of the DB currently on disk.
_LAST_DOWNLOADED_KEY: Optional[str] = None
# Key of the newest manifest we have observed from the release — updated even
# when the subsequent DB download fails, so ``in_sync`` correctly reports drift
# (a new release the API has not yet managed to download). Kept separate from
# ``_LAST_MANIFEST`` because that one drives the download decision and must NOT
# advance on failure (or retries would stop).
_LAST_FETCHED_KEY: Optional[str] = None
# Key of the DB the API's read connection was opened on. Downloaded is not
# served: in Oct 2026 the API downloaded every release, reported in_sync,
# and kept answering from the file it booted with.
_SERVED_KEY: Optional[str] = None

logger = logging.getLogger(__name__)


class DbSyncError(RuntimeError):
    pass


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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
    headers = _build_headers()
    # Bust GitHub CDN caches so we always see the latest release assets.
    headers["Cache-Control"] = "no-cache, no-store"
    headers["Pragma"] = "no-cache"
    manifest_url = f"{urls['manifest_url']}?t={int(time.time())}"
    try:
        response = requests.get(manifest_url, headers=headers, timeout=30)
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


def _check_free_space(dest_path: Path, needed_bytes: Optional[int]) -> None:
    """Refuse a download the disk cannot hold, and say so in words.

    Atomic staging writes the new DB beside the old one, so the disk briefly
    holds two copies. When it cannot, the write dies part-way with ENOSPC,
    the temp file is removed, and the next poll tries again — forever, while
    the API keeps serving the old DB. That is a dashboard frozen on last
    month with nothing but a log line to say why, so the shortfall is raised
    before a byte is written, naming both numbers.
    """
    if not needed_bytes or needed_bytes <= 0:
        return
    try:
        free = shutil.disk_usage(dest_path.parent).free
    except OSError:
        return
    required = needed_bytes + _DISK_HEADROOM_BYTES
    if free < required:
        raise DbSyncError(
            "Insufficient disk space to stage the new DB: need "
            f"{required / 1e6:.0f} MB free on {dest_path.parent}, have "
            f"{free / 1e6:.0f} MB. The disk must hold two copies of the DB "
            "(the served one and the incoming one); enlarge it."
        )


def download_db_atomic(dest_path: Path, expected_sha256: Optional[str] = None) -> None:
    """Download the release DB to ``<dest>.tmp`` and swap it into place.

    When ``expected_sha256`` is a sha256 hex digest (the manifest's
    ``db_sha256``), the staged file is hashed and a mismatch is refused
    BEFORE the swap: a truncated or wrong file must never replace a good one.
    """
    urls = get_release_urls()
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    # Self-heal a wedged temp path. A crashed/interrupted download (or an older
    # code path / artifact sync) can leave ``<dest>.tmp`` behind as a *directory*
    # on a persistent disk. ``tmp_path.open("wb")`` would then fail with
    # ``[Errno 21] Is a directory`` on every subsequent sync, silently freezing
    # the API on a stale DB. Clear any pre-existing temp path (file or dir)
    # before writing so the download can always proceed.
    if tmp_path.is_dir() and not tmp_path.is_symlink():
        logger.warning("Removing stale temp directory before download: %s", tmp_path)
        shutil.rmtree(tmp_path, ignore_errors=True)
    elif tmp_path.exists() or tmp_path.is_symlink():
        try:
            tmp_path.unlink()
        except OSError as exc:
            logger.warning("Failed to remove stale temp file %s: %s", tmp_path, exc)

    # The destination must be a regular file for ``os.replace`` to swap it. If a
    # directory has somehow taken its place, fail loudly rather than wedge.
    if dest_path.is_dir() and not dest_path.is_symlink():
        raise DbSyncError(
            f"Destination DB path is a directory, not a file: {dest_path}"
        )

    headers = _build_headers()
    headers["Cache-Control"] = "no-cache, no-store"
    headers["Pragma"] = "no-cache"
    db_url = f"{urls['db_url']}?t={int(time.time())}"
    try:
        response = requests.get(db_url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()
        expected_len: Optional[int] = None
        try:
            raw_len = getattr(response, "headers", {}).get("Content-Length")
            expected_len = int(raw_len) if raw_len else None
        except (TypeError, ValueError, AttributeError):
            expected_len = None
        _check_free_space(dest_path, expected_len)
        want = (expected_sha256 or "").strip().lower()
        verify = bool(_SHA256_RE.match(want))
        digest = hashlib.sha256() if verify else None
        written = 0
        with tmp_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
                    written += len(chunk)
                    if digest is not None:
                        digest.update(chunk)
        if expected_len is not None and written != expected_len:
            raise DbSyncError(
                f"Truncated DB download: got {written} of {expected_len} bytes"
            )
        if digest is not None and digest.hexdigest() != want:
            raise DbSyncError(
                f"Downloaded DB sha256 {digest.hexdigest()[:12]}... does not match "
                f"the manifest's {want[:12]}...; refusing to swap it in"
            )
    except Exception as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        if isinstance(exc, DbSyncError):
            raise
        raise DbSyncError(f"Failed to download DuckDB asset: {exc}") from exc

    os.replace(tmp_path, dest_path)

    # Any WAL file present now belongs to the previous DB instance (which we
    # just replaced) or to the process that produced the downloaded artifact.
    # Replaying it against the new DB file risks an InternalException
    # ("Calling DatabaseManager::GetDefaultDatabase with no default database
    # set").  The fresh DB file is the authoritative snapshot, so the stale
    # WAL is expendable.
    wal_path = dest_path.with_suffix(dest_path.suffix + ".wal")
    try:
        if wal_path.exists():
            wal_path.unlink()
            logger.info("Removed stale WAL alongside refreshed DB: %s", wal_path)
    except OSError as exc:
        logger.warning("Failed to remove stale WAL %s: %s", wal_path, exc)


def db_was_refreshed() -> bool:
    """Check and clear the DB-refreshed flag.

    Returns ``True`` exactly once after ``maybe_sync_latest_db()``
    downloaded a new DB file, signalling callers to reopen connections.
    """
    if _DB_REFRESHED.is_set():
        _DB_REFRESHED.clear()
        return True
    return False


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
    """Populate ``_LATEST_RUNS`` from the current DB file on disk.

    Opens a FRESH direct DuckDB connection each call.  ``duckdb_io.get_db()``
    caches connections process-wide by path, but ``download_db_atomic``
    swaps the DB file inode via ``os.replace``; a cached connection keeps
    reading the old inode indefinitely, which would leave ``_LATEST_RUNS``
    stuck on pre-refresh runs even after a successful download.
    """
    global _LATEST_RUNS
    try:
        conn = duckdb_io.duckdb.connect(str(db_path), read_only=True)
    except Exception as exc:
        logger.warning("Failed to open DuckDB for latest run introspection: %s", exc)
        return
    try:
        # Cap this connection like the main read connection. Without an explicit
        # limit DuckDB allows itself ~80% of system RAM, and this runs right
        # after a ~1GB DB download — the worst moment for a memory spike.
        # (Env read directly: importing pythia.api.core here would be circular.)
        conn.execute(f"SET memory_limit='{os.getenv('PYTHIA_DUCKDB_MEMORY_LIMIT', '150MB')}'")
        conn.execute(f"SET threads={int(os.getenv('PYTHIA_DUCKDB_THREADS', '2'))}")
    except Exception:
        logger.warning("Failed to cap introspection connection memory", exc_info=True)
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
        try:
            conn.close()
        except Exception:
            pass


def mark_served() -> None:
    """Record that a read connection was just opened on the DB on disk."""
    global _SERVED_KEY
    _SERVED_KEY = _LAST_DOWNLOADED_KEY


def get_sync_status() -> Dict[str, Any]:
    """Return a snapshot of the most recent DB-sync outcome.

    ``in_sync`` is ``True`` when the key of the DB currently on disk matches the
    newest manifest key we have observed from the release (cheap — no file
    hashing). It is ``None`` when we have not yet observed both.
    """
    # Compare against what is SERVED once a connection has been opened;
    # before that, against what is on disk.
    current = _SERVED_KEY if _SERVED_KEY is not None else _LAST_DOWNLOADED_KEY
    if _LAST_FETCHED_KEY is None or current is None:
        in_sync: Optional[bool] = None
    else:
        in_sync = _LAST_FETCHED_KEY == current
    return {
        "last_error": _LAST_SYNC_ERROR,
        "last_ok_at": _LAST_SYNC_OK_AT,
        "last_attempt_at": _LAST_SYNC_ATTEMPT_AT,
        "manifest_key": _LAST_FETCHED_KEY,
        "downloaded_key": _LAST_DOWNLOADED_KEY,
        "served_key": _SERVED_KEY,
        "in_sync": in_sync,
    }


def background_sync_running() -> bool:
    return _BG_THREAD is not None and _BG_THREAD.is_alive()


def maybe_sync_latest_db(wait: bool = False) -> Optional[Dict[str, Any]]:
    """Fetch the release manifest and download a new DB when it changed.

    With ``wait=False`` (every request path) a caller that finds another
    sync in progress returns the cached manifest at once rather than queueing
    behind a multi-hundred-megabyte download; ``wait=True`` (startup, the
    background loop, ``force_sync``) blocks for the lock.
    """
    dest_path = _db_path_from_config()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    interval_s = int(_get_env("PYTHIA_DATA_SYNC_INTERVAL_S", str(_DEFAULT_SYNC_INTERVAL_S)) or 0)

    if not _SYNC_LOCK.acquire(blocking=wait):
        return get_cached_manifest()
    try:
        return _sync_locked(dest_path, interval_s)
    finally:
        _SYNC_LOCK.release()


def _sync_locked(dest_path: Path, interval_s: int) -> Optional[Dict[str, Any]]:
    global _LAST_MANIFEST, _LAST_SYNC_AT
    global _LATEST_RUNS
    global _LAST_SYNC_ERROR, _LAST_SYNC_OK_AT, _LAST_SYNC_ATTEMPT_AT
    global _LAST_DOWNLOADED_KEY, _LAST_FETCHED_KEY

    now = time.monotonic()
    if _LAST_SYNC_AT is not None and now - _LAST_SYNC_AT < interval_s:
        return get_cached_manifest()
    _LAST_SYNC_AT = now
    _LAST_SYNC_ATTEMPT_AT = _utc_now_iso()

    try:
        manifest = fetch_manifest()
        manifest_key = _manifest_key(manifest)
        # Record the newest release we've seen *before* attempting the
        # download, so a failed download leaves ``in_sync`` reporting drift.
        if manifest_key:
            _LAST_FETCHED_KEY = manifest_key
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
            expected_sha = manifest.get("db_sha256")
            if expected_sha:
                download_db_atomic(dest_path, expected_sha256=str(expected_sha))
            else:
                download_db_atomic(dest_path)
            _DB_REFRESHED.set()
            _LAST_DOWNLOADED_KEY = manifest_key
            logger.info("DB refreshed from release (key=%s)", manifest_key)
        elif _LAST_DOWNLOADED_KEY is None and manifest_key:
            # DB already on disk and matches the manifest (e.g. survived a
            # restart with a persistent disk); treat it as in-sync.
            _LAST_DOWNLOADED_KEY = manifest_key

        if should_download or _LATEST_RUNS is None:
            _refresh_latest_runs(dest_path)

        _LAST_MANIFEST = dict(manifest)
        _LAST_SYNC_ERROR = None
        _LAST_SYNC_OK_AT = _utc_now_iso()
        return get_cached_manifest()
    except DbSyncError as exc:
        _LAST_SYNC_ERROR = str(exc)
        raise


# ---------------------------------------------------------------------------
# Background sync
# ---------------------------------------------------------------------------
# Until Oct 2026 the API synced LAZILY: the only thing that ever fetched the
# manifest was an incoming request (``_con()`` / ``/v1/version``). A publish
# therefore reached the dashboard only after somebody visited, and that
# visitor's request then sat behind an ~800 MB download holding the sync
# lock. On a host that sleeps when idle, or a month with a failed download,
# nothing retried until the next visit — which is how the dashboard came to
# lag the release cycle after cycle. The loop below polls on its own clock
# and swaps the read connection itself, so the dashboard is current before
# anyone looks.

_BG_THREAD: Optional[threading.Thread] = None
_BG_STOP = threading.Event()


def background_sync_enabled() -> bool:
    raw = (_get_env("PYTHIA_DATA_BACKGROUND_SYNC", "1") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _background_sync_loop(on_refresh, interval_s: float) -> None:
    while not _BG_STOP.is_set():
        try:
            maybe_sync_latest_db(wait=True)
            if on_refresh is not None and db_was_refreshed():
                # Consume the flag here: the swap happens on this thread, and
                # the request path's mtime check then sees nothing newer.
                on_refresh()
        except DbSyncError as exc:
            logger.warning("Background DB sync failed: %s", exc)
        except Exception:  # noqa: BLE001 - the loop must never die
            logger.exception("Background DB sync crashed; will retry")
        if _BG_STOP.wait(interval_s):
            break


def start_background_sync(on_refresh=None) -> bool:
    """Start the background sync thread once per process.

    ``on_refresh`` is called after a new DB lands on disk; the API passes
    a function that swaps its read connection. Returns True if a thread was
    started. Disabled with ``PYTHIA_DATA_BACKGROUND_SYNC=0`` (tests, local
    runs without a release) and a no-op when no release repo is configured.
    """
    global _BG_THREAD
    if not background_sync_enabled():
        return False
    if not _get_env("PYTHIA_DATA_REPO") or not _get_env("PYTHIA_DATA_RELEASE_TAG"):
        return False
    if _BG_THREAD is not None and _BG_THREAD.is_alive():
        return False
    interval_s = max(
        15.0,
        float(_get_env("PYTHIA_DATA_SYNC_INTERVAL_S", str(_DEFAULT_SYNC_INTERVAL_S)) or 60),
    )
    _BG_STOP.clear()
    _BG_THREAD = threading.Thread(
        target=_background_sync_loop,
        args=(on_refresh, interval_s),
        name="pythia-db-sync",
        daemon=True,
    )
    _BG_THREAD.start()
    logger.info("Background DB sync started (interval=%ss)", interval_s)
    return True


def stop_background_sync(timeout: float = 5.0) -> None:
    global _BG_THREAD
    _BG_STOP.set()
    if _BG_THREAD is not None:
        _BG_THREAD.join(timeout)
    _BG_THREAD = None
