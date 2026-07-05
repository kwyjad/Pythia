# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Calibration-weight loading/caching and DB-URL config helpers (moved verbatim from cli.py)."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import threading
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only
    from forecaster.providers import ModelSpec


_PYTHIA_CFG_LOAD = None
if importlib.util.find_spec("pythia.config") is not None:
    _PYTHIA_CFG_LOAD = getattr(importlib.import_module("pythia.config"), "load", None)


def _pythia_db_url_from_config() -> str:
    """
    Best-effort helper to read the Pythia DuckDB URL from config or env.

    Priority:
      1. pythia.db.schema.get_db_url (if available)
      2. app.db_url from pythia.config
      3. PYTHIA_DB_URL environment variable
      4. default duckdb:///data/resolver.duckdb

    This helper is intentionally kept for backward compatibility with tests
    that monkeypatch it to point to a temporary DuckDB file.
    """

    try:
        from pythia.db.schema import get_db_url

        url = get_db_url()
        if url:
            return url
    except Exception:
        pass

    if _PYTHIA_CFG_LOAD is not None:
        try:
            cfg = _PYTHIA_CFG_LOAD()
            app_cfg = cfg.get("app", {}) if isinstance(cfg, dict) else {}
            db_url = str(app_cfg.get("db_url", "")).strip()
            if db_url:
                return db_url
        except Exception:
            pass

    env_url = os.getenv("PYTHIA_DB_URL", "").strip()
    if env_url:
        return env_url

    return "duckdb:///data/resolver.duckdb"


def _pythia_db_path_from_config() -> str:
    """Return a filesystem path for the configured DuckDB database."""

    db_url = _pythia_db_url_from_config()
    if db_url.startswith("duckdb:///"):
        return db_url.replace("duckdb:///", "", 1)
    return db_url


# --------------------------------------------------------------------------------
# Calibration weights loader (optional legacy file fallback).
# --------------------------------------------------------------------------------

def _load_calibration_weights_file() -> Dict[str, Any]:
    path = os.getenv("CALIB_WEIGHTS_PATH", "")
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_calibration_weights_db(
    hazard_code: str,
    metric: str,
) -> Optional[Dict[str, float]]:
    try:
        from resolver.db import duckdb_io
    except Exception:
        return None

    hz = (hazard_code or "").upper().strip()
    mt = (metric or "").upper().strip()
    if not hz or not mt:
        return None

    db_url = _pythia_db_url_from_config() or os.getenv("RESOLVER_DB_URL", "").strip()
    if not db_url:
        return None

    conn = None
    try:
        conn = duckdb_io.get_db(db_url)
    except Exception:
        return None

    try:
        row = conn.execute(
            """
            SELECT as_of_month
            FROM calibration_weights
            WHERE hazard_code = ? AND metric = ?
            ORDER BY as_of_month DESC
            LIMIT 1
            """,
            [hz, mt],
        ).fetchone()
        if not row:
            return None
        as_of_month = str(row[0])

        rows = conn.execute(
            """
            SELECT model_name, weight
            FROM calibration_weights
            WHERE hazard_code = ? AND metric = ? AND as_of_month = ?
            ORDER BY COALESCE(model_name, '')
            """,
            [hz, mt, as_of_month],
        ).fetchall()
        if not rows:
            return None

        weights: Dict[str, float] = {}
        for model_name, weight in rows:
            if model_name is None:
                continue
            weights[str(model_name)] = float(weight)
        if not weights:
            return None

        if os.getenv("PYTHIA_DEBUG_DB", "0") == "1":
            print(
                "[forecaster] loaded calibration weights for hazard="
                f"{hz} metric={mt} as_of={as_of_month}: {weights}"
            )
        return weights
    except Exception:
        return None
    finally:
        try:
            duckdb_io.close_db(conn)
        except Exception:
            pass


def _calibration_weights_enabled() -> bool:
    """Whether calibration weights are applied to ensemble aggregation."""
    return os.getenv("PYTHIA_USE_CALIBRATION_WEIGHTS", "1") != "0"


# Per-process cache: one DB read per (hazard, metric) per run instead of one
# per question. Values may be None (no weights available). Guarded by a lock:
# the SPD phase runs questions on worker threads, and an unguarded check-then-
# set raced duplicate DB reads for the same (hazard, metric).
_CALIB_WEIGHTS_CACHE: Dict[Tuple[str, str], Optional[Dict[str, float]]] = {}
_CALIB_WEIGHTS_LOCK = threading.Lock()


def _reset_calibration_weights_cache() -> None:
    """Clear cached calibration weights (call at run start so a long-lived
    process picks up freshly computed weights)."""
    with _CALIB_WEIGHTS_LOCK:
        _CALIB_WEIGHTS_CACHE.clear()


def _member_weight_key(ms: ModelSpec) -> str:
    """Disambiguated member key matching _write_spd_members_v2_to_db naming."""
    return f"{getattr(ms, 'name', '')} ({getattr(ms, 'model_id', '')})"
