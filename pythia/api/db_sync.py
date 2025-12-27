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

_DEFAULT_DB_ASSET = "resolver.duckdb"
_DEFAULT_MANIFEST_ASSET = "manifest.json"
_DEFAULT_SYNC_INTERVAL_S = 60

_LAST_MANIFEST: Optional[Dict[str, Any]] = None
_LAST_SYNC_AT: Optional[float] = None
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
    return dict(_LAST_MANIFEST)


def maybe_sync_latest_db() -> Optional[Dict[str, Any]]:
    global _LAST_MANIFEST, _LAST_SYNC_AT

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

        _LAST_MANIFEST = dict(manifest)
        return get_cached_manifest()
