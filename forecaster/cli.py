from __future__ import annotations
"""
cli.py — Forecaster runner (Pythia-only question sources)

WHAT THIS FILE DOES (high level, in plain English)
--------------------------------------------------
- Fetches Pythia Horizon Scanner questions (or local test questions).
- For each question:
  1) Runs the RESEARCH step to build a compact research brief.
  2) Classifies the question (primary/secondary topic + "strategic?" score).
     - If it's strategic *and* the question is binary, we try GTMC1.
  3) Builds a forecasting prompt and asks each LLM model in your ensemble for a forecast.
  4) Aggregates model outputs with a Bayesian Monte Carlo layer ("BMC"); optionally fuses GTMC1 for binary.
  5) Records *everything* into ONE wide CSV row via io_logs.write_unified_row(...).

- Additionally, it runs an **ablation** pass ("no-research") so you can quantify the
  value of the research component. Those results are logged into dedicated CSV columns.

- It also logs three ensemble **variants** for diagnostics:
  (a) no_gtmc1            → BMC aggregation without the GTMC1 signal,
  (b) uniform_weights     → treat all LLMs equally,
  (c) no_bmc_no_gtmc1     → a very simple average of model outputs (no BMC, no GTMC1).
"""

import argparse
import asyncio
import importlib
import importlib.util
import json
import os
import re
import logging
import time
import traceback
from datetime import datetime
from contextlib import ExitStack
from typing import Any, Dict, List, Optional, Tuple
import inspect

import numpy as np

from pathlib import Path
from pythia.db.schema import connect, ensure_schema

LOG = logging.getLogger(__name__)

_PYTHIA_CFG_LOAD = None
if importlib.util.find_spec("pythia.config") is not None:
    _PYTHIA_CFG_LOAD = getattr(importlib.import_module("pythia.config"), "load", None)

try:
    from pythia.llm_profiles import get_current_models as _get_llm_profile_models
except Exception:
    _get_llm_profile_models = None  # type: ignore


# Hazard codes for which GTMC1 is relevant (adjust as needed for your schema)
CONFLICT_HAZARD_CODES = {
    "CONFLICT",
    "POLITICAL_VIOLENCE",
    "CIVIL_CONFLICT",
    "URBAN_CONFLICT",
}

# SPD buckets for Pythia PA/PIN forecasts (order must match prompts & aggregation)
SPD_CLASS_BINS_PA = [
    "<10k",
    "10k-<50k",
    "50k-<250k",
    "250k-<500k",
    ">=500k",
]

# SPD buckets for conflict fatalities forecasts (per month)
SPD_CLASS_BINS_FATALITIES = [
    "<5",
    "5-<25",
    "25-<100",
    "100-<500",
    ">=500",
]

# Backwards-compatibility alias; PA remains the default bucket scheme
SPD_CLASS_BINS = SPD_CLASS_BINS_PA


def _extract_pythia_meta(post: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract Pythia-specific metadata attached by _load_pythia_questions(...).

    Returns a dict with keys:
      - iso3
      - hazard_code
      - metric
      - target_month
    Missing fields are normalized to "".
    """
    return {
        "iso3": str(post.get("pythia_iso3") or "").upper(),
        "hazard_code": str(post.get("pythia_hazard_code") or "").upper(),
        "metric": str(post.get("pythia_metric") or "").upper(),
        "target_month": str(post.get("pythia_target_month") or ""),
    }


def _advise_poetry_lock_if_needed():
    # Dev convenience: if Poetry complains about a stale lock, print the fix.
    import os
    if os.getenv("CI"):
        return  # CI already handles regeneration
    # Lightweight hint only; we don't try to run Poetry here.
    os.environ.setdefault("PYTHIA_LOCK_HINT_SHOWN", "0")


def _safe_json_load(s: str):
    try:
        import json as _json
        return _json.loads(s)
    except Exception:
        return None


def _as_dict(obj: Any) -> Dict[str, Any]:
    """Return a dict from obj. If obj is a JSON string, parse it. Else {}."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, (str, bytes)):
        try:
            parsed = json.loads(obj)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _must_dict(name: str, obj: Any) -> Dict[str, Any]:
    d = _as_dict(obj)
    if not d:
        # Keep this very explicit so CI logs are helpful and not a vague AttributeError
        raise RuntimeError(f"{name} is not a dict after coercion (type={type(obj).__name__})")
    return d


# ---- Forecaster internals (all relative imports) --------------------------------
from .config import ist_iso
from .prompts import (
    build_binary_prompt,
    build_numeric_prompt,
    build_mcq_prompt,
    build_spd_prompt,
    build_spd_prompt_fatalities,
    build_spd_prompt_pa,
)
from .providers import DEFAULT_ENSEMBLE, _get_or_client, llm_semaphore
from .ensemble import (
    EnsembleResult,
    MemberOutput,
    _normalize_spd_keys,
    run_ensemble_binary,
    run_ensemble_mcq,
    run_ensemble_numeric,
    run_ensemble_spd,
    sanitize_mcq_vector,
)
from .aggregate import (
    SPD_BUCKET_CENTROIDS_DEFAULT,
    SPD_BUCKET_CENTROIDS_FATALITIES_DEFAULT,
    aggregate_binary,
    aggregate_mcq,
    aggregate_numeric,
    aggregate_spd,
)
from .research import run_research_async

# --- Corrected seen_guard import ---
try:
    from . import seen_guard
except ImportError as e:
    print(f"[warn] seen_guard not available ({e!r}); continuing without duplicate protection.")
    seen_guard = None

from . import GTMC1

# --- seen_guard import shim (ensures a callable filter_unseen_posts exists) ---
try:
    try:
        # When cli.py is executed as a module
        from .seen_guard import SeenGuard  # type: ignore
    except Exception:
        # When cli.py is executed as a script from repo root
        from seen_guard import SeenGuard  # type: ignore

    _sg = SeenGuard()

    def filter_unseen_posts(posts):
        # Adapter to old call-site name; calls the actual class method.
        return _sg.filter_fresh_posts(posts)

except Exception as e:
    print(f"[seen_guard] disabled ({e}); processing all posts returned.")
    def filter_unseen_posts(posts):
        return posts
# --- end seen_guard import shim ---



# Unified CSV helpers (single file)
from .io_logs import ensure_unified_csv, write_unified_row, write_human_markdown, finalize_and_commit

# --------------------------------------------------------------------------------
# Small utility helpers (safe JSON, timing, clipping, etc.)
# --------------------------------------------------------------------------------

# --- SeenGuard wiring (robust to different shapes/APIs) -----------------------
def _load_seen_guard():
    """
    Try to load a SeenGuard instance from seen_guard.py in a robust way.
    Will look for common instance names and fall back to constructing SeenGuard.
    Returns: guard instance or None
    """
    try:
        import seen_guard as sg_mod
    except Exception:
        return None

    # Prefer a ready-made instance exported from the module
    for attr in ("_GUARD", "GUARD", "guard"):
        guard = getattr(sg_mod, attr, None)
        if guard is not None:
            return guard

    # Fallback: instantiate if class is available
    try:
        SG = getattr(sg_mod, "SeenGuard", None)
        if SG is not None:
            cooldown = int(os.getenv("SEEN_COOLDOWN_HOURS", "24"))
            path = os.getenv("SEEN_GUARD_PATH", "forecast_logs/state/seen_forecasts.jsonl")
            return SG(Path(path), cooldown_hours=cooldown)
    except Exception:
        pass

    return None


def _apply_seen_guard(guard, posts):
    """
    Call the first matching method on guard to filter posts.
    Accepts either a return of (posts, dup_count) or just posts.
    """
    if not guard or not posts:
        return posts, 0

    candidates = [
        "filter_fresh_posts",
        "filter_unseen_posts",
        "filter_posts",
        "filter_recent_posts",
        "filter_new_posts",
        "filter",  # very generic, last
    ]

    last_err = None
    for name in candidates:
        if hasattr(guard, name):
            fn = getattr(guard, name)
            try:
                # Try simple positional call
                result = fn(posts)
            except TypeError:
                # Try kwargs form if implemented that way
                try:
                    result = fn(posts=posts)
                except Exception as e:
                    last_err = e
                    continue
            except Exception as e:
                last_err = e
                continue

            # Normalize return
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], list):
                return result
            if isinstance(result, list):
                return result, 0

            # Unexpected return shape; treat as no-op
            return posts, 0

    # If we got here, no callable matched or all failed
    if last_err:
        raise last_err
    return posts, 0
# ----------------------------------------------------------------------------- 

# Time in milliseconds since start_time
def _ms(start_time: float) -> int:
    return int(round((time.time() - start_time) * 1000))

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _sanitize_markdown_chunks(chunks: List[Any]) -> List[str]:
    """Return a list of strings suitable for ``"\n\n".join(...)``.

    The markdown builder collects many diagnostic entries, some of which
    originate from optional integrations (GTMC1 raw dumps, prediction-market
    lookups, etc.).  When any of those helpers return ``None`` we previously
    propagated the ``None`` directly into the markdown list.  Later, when we
    attempted to join the chunks we hit ``TypeError: sequence item X: expected
    str instance, NoneType found``.  This helper drops ``None`` entries and
    coerces any remaining values to strings so the join is always safe.
    """

    sanitized: List[str] = []
    for chunk in chunks:
        if chunk is None:
            continue
        if isinstance(chunk, str):
            sanitized.append(chunk)
            continue
        try:
            sanitized.append(str(chunk))
        except Exception:
            # If ``str(chunk)`` itself fails we silently drop the entry; the
            # surrounding debug output already makes it clear something odd
            # happened, and failing to write the human log is worse.
            continue
    return sanitized


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


def _write_spd_ensemble_to_db(
    *,
    run_id: str,
    question_id: str,
    spd_main: Dict[str, List[float]],
    metric: str,
    hazard_code: str,
    iso3: str = "",
    ev_main: Optional[Dict[str, Any]] = None,
    weights_profile: str = "",
) -> None:
    """
    Persist SPD ensemble into forecasts_ensemble.

    spd_main: dict like {"month_1": [p1..p5], ..., "month_6": [p1..p5]}
    ev_main:  dict like {"month_1": ev_value, ...} (optional)
    """

    metric_up = (metric or "").upper()
    hz_up = (hazard_code or "").upper()

    if metric_up == "FATALITIES" and (hz_up.startswith("CONFLICT") or hz_up in CONFLICT_HAZARD_CODES):
        class_bins = SPD_CLASS_BINS_FATALITIES
    else:
        class_bins = SPD_CLASS_BINS_PA

    from .ensemble import _normalize_spd_keys  # local import to avoid cycles

    spd_main = _normalize_spd_keys(spd_main, n_months=6, n_buckets=len(class_bins))

    try:
        import duckdb
    except Exception as exc:  # noqa: BLE001
        print(
            f"[warn] duckdb is required to write SPD ensemble (question_id={question_id}): {type(exc).__name__}: {exc}"
        )
        return

    db_url = _pythia_db_url_from_config()
    db_path = db_url[len("duckdb:///") :] if db_url.startswith("duckdb:///") else db_url

    try:
        con = duckdb.connect(db_path)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[warn] Failed to open DB for SPD ensemble write (question_id={question_id}): {type(exc).__name__}: {exc}"
        )
        return

    now = datetime.utcnow()

    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS forecasts_ensemble (
                horizon_m INTEGER,
                class_bin VARCHAR,
                p DOUBLE,
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
            """
        )

        con.execute(
            "DELETE FROM forecasts_ensemble WHERE question_id = ? AND run_id = ?;",
            [question_id, run_id],
        )

        for month_idx in range(1, 7):
            key = f"month_{month_idx}"
            probs = spd_main.get(key) or []
            if not isinstance(probs, (list, tuple)):
                continue
            ev_val = None
            if ev_main and key in ev_main:
                try:
                    ev_val = float(ev_main[key])
                except Exception:
                    ev_val = None

            for bucket_idx, prob in enumerate(probs, start=1):
                class_bin = class_bins[bucket_idx - 1] if 0 <= bucket_idx - 1 < len(class_bins) else str(bucket_idx)
                try:
                    con.execute(
                        """
                        INSERT INTO forecasts_ensemble (
                            horizon_m,
                            class_bin,
                            p,
                            run_id,
                            question_id,
                            iso3,
                            hazard_code,
                            metric,
                            month_index,
                            bucket_index,
                            probability,
                            ev_value,
                            weights_profile,
                            created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        [
                            month_idx,
                            class_bin,
                            float(prob),
                            run_id,
                            question_id,
                            iso3,
                            hz_up,
                            metric_up,
                            month_idx,
                            bucket_idx,
                            float(prob),
                            ev_val if ev_val is not None and bucket_idx == 1 else None,
                            weights_profile,
                            now,
                        ],
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[warn] Failed to write forecasts_ensemble row for q={question_id} month={month_idx}: {exc}"
                    )
    finally:
        try:
            con.close()
        except Exception:
            pass


def _write_spd_raw_to_db(
    *,
    run_id: str,
    question_id: str,
    iso3: str,
    hazard_code: str,
    metric: str,
    ens_res: EnsembleResult,
) -> None:
    """
    Write per-model SPD forecasts into forecasts_raw for this question.

    Uses PA buckets for PA metrics and conflict fatalities buckets for metric="FATALITIES"
    on conflict hazards.
    """

    metric_up = (metric or "").upper()
    hz_up = (hazard_code or "").upper()

    if metric_up == "FATALITIES" and (hz_up.startswith("CONFLICT") or hz_up in CONFLICT_HAZARD_CODES):
        class_bins = SPD_CLASS_BINS_FATALITIES
    else:
        class_bins = SPD_CLASS_BINS_PA

    try:
        con = connect(read_only=False)
        ensure_schema(con)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[warn] Failed to open DB for SPD raw write (question_id={question_id}): {type(exc).__name__}: {exc}"
        )
        return

    try:
        con.execute(
            "DELETE FROM forecasts_raw WHERE run_id = ? AND question_id = ?;",
            [run_id, question_id],
        )

        for m in ens_res.members:
            model_name = getattr(m, "name", "")
            ok = bool(getattr(m, "ok", False))
            elapsed_ms = getattr(m, "elapsed_ms", 0) or 0
            cost_usd = getattr(m, "cost_usd", 0.0) or 0.0
            prompt_tokens = getattr(m, "prompt_tokens", 0) or 0
            completion_tokens = getattr(m, "completion_tokens", 0) or 0
            total_tokens = getattr(m, "total_tokens", prompt_tokens + completion_tokens) or 0

            if not isinstance(getattr(m, "parsed", None), dict):
                continue

            parsed = _normalize_spd_keys(m.parsed, n_months=6, n_buckets=len(class_bins))
            for month_idx in range(1, 7):
                key = f"month_{month_idx}"
                probs = parsed.get(key) or []
                if not isinstance(probs, (list, tuple)):
                    continue
                for bucket_idx, prob in enumerate(probs, start=1):
                    try:
                        con.execute(
                            """
                            INSERT INTO forecasts_raw (
                                run_id,
                                question_id,
                                model_name,
                                month_index,
                                bucket_index,
                                probability,
                                ok,
                                elapsed_ms,
                                cost_usd,
                                prompt_tokens,
                                completion_tokens,
                                total_tokens
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                            """,
                            [
                                run_id,
                                question_id,
                                model_name,
                                month_idx,
                                bucket_idx,
                                float(prob),
                                ok,
                                elapsed_ms,
                                cost_usd,
                                prompt_tokens,
                                completion_tokens,
                                total_tokens,
                            ],
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"[warn] Failed to write forecasts_raw row for q={question_id} month={month_idx}: {exc}"
                        )
    finally:
        try:
            con.close()
        except Exception:
            pass


def _load_pa_history_block(
    iso3: str,
    hazard_code: str,
    *,
    months: int = 36,
) -> tuple[str, Dict[str, Any]]:
    """
    Best-effort PA history block for the research bundle.

    Reads up to `months` rows from `facts_resolved` for the given iso3 + hazard_code
    where metric is PA-like (affected). Returns (markdown_block, meta_dict).

    On any error or if no rows are found, returns ("", {"pa_history_error": "reason"}).
    """
    import duckdb

    iso3 = (iso3 or "").upper().strip()
    hz = (hazard_code or "").upper().strip()
    if not iso3 or not hz:
        return "", {"pa_history_error": "missing_iso3_or_hazard"}

    db_url = _pythia_db_url_from_config() or os.getenv("RESOLVER_DB_URL", "").strip()
    if not db_url:
        return "", {"pa_history_error": "missing_db_url"}

    db_path = db_url.replace("duckdb:///", "")
    try:
        con = duckdb.connect(db_path, read_only=True)
    except Exception as exc:
        return "", {"pa_history_error": f"connect_failed:{exc!r}"}

    try:
        # NOTE: metric canonicalization: `affected` is the canonical PA metric.
        # We accept a few synonyms to be robust to partial migrations.
        sql = """
            SELECT ym, value
            FROM facts_resolved
            WHERE iso3 = ?
              AND hazard_code = ?
              AND lower(metric) IN ('affected','people_affected','pa')
            ORDER BY ym DESC
            LIMIT ?
        """
        rows = con.execute(sql, [iso3, hz, int(months)]).fetchall()
    except Exception as exc:
        con.close()
        return "", {"pa_history_error": f"query_failed:{exc!r}"}
    finally:
        try:
            con.close()
        except Exception:
            pass

    if not rows:
        return "", {"pa_history_error": "no_rows"}

    # Oldest → newest for readability
    rows = list(reversed(rows))
    lines = [
        "## Resolver 36-month PA history",
        "",
        "| Month (ym) | People affected |",
        "|---|---|",
    ]
    months_list: list[str] = []
    history_detail: list[Dict[str, Any]] = []
    for ym, value in rows:
        try:
            ym_str = str(ym)
            val = "" if value is None else f"{int(value):,}"
        except Exception:
            ym_str = str(ym)
            val = str(value)
        months_list.append(ym_str)
        try:
            numeric_val = None if value is None else float(value)
        except Exception:
            numeric_val = None
        history_detail.append({"ym": ym_str, "value": numeric_val})
        lines.append(f"| {ym_str} | {val} |")

    block = "\n".join(lines)
    meta = {
        "pa_history_error": "",
        "pa_history_rows": len(rows),
        "pa_history_months": months_list,
        "pa_history_rows_detail": history_detail,
    }
    return block, meta


def _load_bucket_centroids(
    hazard_code: str,
    metric: str,
    class_bins: Optional[List[str]] = None,
) -> Optional[List[float]]:
    """
    Look up data-driven centroids for SPD buckets from `bucket_centroids`.

    Returns a list of floats ordered to match class_bins (or SPD_CLASS_BINS_PA),
    or None if no centroids are available for this (hazard_code, metric).
    """
    try:
        from resolver.db import duckdb_io
    except Exception:
        return None

    hz = (hazard_code or "").upper().strip()
    mt = (metric or "").upper().strip()
    bins = class_bins or SPD_CLASS_BINS_PA
    if not mt:
        return None

    db_url = _pythia_db_url_from_config() or os.getenv("RESOLVER_DB_URL", "").strip()
    if not db_url:
        if os.getenv("PYTHIA_DEBUG_DB", "0") == "1":
            print(f"[forecaster] no db_url while loading bucket centroids for hazard={hz!r}, metric={mt!r}")
        return None

    con = None
    try:
        con = duckdb_io.get_db(db_url)
    except Exception as exc:
        if os.getenv("PYTHIA_DEBUG_DB", "0") == "1":
            print(f"[forecaster] failed to connect to DuckDB for bucket centroids: {exc!r}")
        return None

    try:
        try:
            con.execute("SELECT 1 FROM bucket_centroids LIMIT 1")
        except Exception:
            if os.getenv("PYTHIA_DEBUG_DB", "0") == "1":
                print("[forecaster] bucket_centroids table not found; using default centroids")
            return None

        rows = con.execute(
            """
            SELECT class_bin, ev
            FROM bucket_centroids
            WHERE metric = ?
              AND hazard_code = ?
            """,
            [mt, hz],
        ).fetchall()

        if not rows:
            if os.getenv("PYTHIA_DEBUG_DB", "0") == "1":
                print(
                    f"[forecaster] no bucket_centroids rows for hazard={hz!r}, metric={mt!r}; "
                    "falling back to defaults"
                )
            return None

        by_bin = {cb: float(ev) for (cb, ev) in rows}
        centroids: List[float] = []
        for bin_label in bins:
            if bin_label not in by_bin:
                if os.getenv("PYTHIA_DEBUG_DB", "0") == "1":
                    print(
                        f"[forecaster] bucket_centroids missing bin={bin_label!r} "
                        f"for hazard={hz!r}, metric={mt!r}; using defaults"
                    )
                return None
            centroids.append(by_bin[bin_label])

        if os.getenv("PYTHIA_DEBUG_DB", "0") == "1":
            print(
                f"[forecaster] loaded data-driven centroids for hazard={hz!r}, metric={mt!r}: "
                f"{centroids}"
            )
        return centroids
    finally:
        try:
            duckdb_io.close_db(con)
        except Exception:
            pass


def _load_pythia_questions(limit: int) -> List[dict]:
    """
    Load active Horizon Scanner questions from DuckDB and adapt them to the
    question 'post' shape expected by _run_one_question_body.

    For now, we treat each PA question as a numeric question:
      - q['type'] = 'numeric'
      - title = wording from Horizon Scanner
      - description/criteria left blank (research still works with title alone)
    """

    from datetime import datetime

    import duckdb
    from resolver.db import duckdb_io

    max_limit = max(1, int(limit))
    db_url = _pythia_db_url_from_config() or duckdb_io.DEFAULT_DB_URL
    conn = duckdb_io.get_db(db_url)
    try:
        sql = """
            SELECT
                question_id,
                hs_run_id,
                scenario_ids_json,
                iso3,
                hazard_code,
                metric,
                target_month,
                window_start_date,
                window_end_date,
                wording,
                status,
                pythia_metadata_json
            FROM questions
            WHERE status = 'active'
            ORDER BY iso3, hazard_code, metric, target_month, question_id
            LIMIT ?
        """
        try:
            cursor = conn.execute(sql, [max_limit])
        except duckdb.BinderException as exc:
            LOG.error("BinderException in _load_pythia_questions: %s", exc)
            LOG.error("SQL attempted in _load_pythia_questions:\n%s", sql.strip())
            try:
                cols = conn.execute("PRAGMA table_info('questions');").fetchall()
                LOG.error("questions table columns: %s", cols)
            except Exception as debug_exc:
                LOG.error("Failed to introspect questions table: %s", debug_exc)
            raise
        rows = cursor.fetchall()
        description = getattr(cursor, "description", []) or []
    finally:
        duckdb_io.close_db(conn)

    cols = [c[0] for c in description]
    posts: List[dict] = []
    for row in rows:
        rec = dict(zip(cols, row))
        qid = rec.get("question_id")
        iso3 = (rec.get("iso3") or "").upper()
        hz = (rec.get("hazard_code") or "").upper()
        metric = rec.get("metric") or "PA"
        target_month = rec.get("target_month") or ""
        wording = rec.get("wording") or ""
        hs_run_id = rec.get("hs_run_id")

        scenario_ids = _safe_json_load(rec.get("scenario_ids_json") or "[]") or []
        pythia_meta = _as_dict(rec.get("pythia_metadata_json") or {})

        question = {
            "id": qid,
            "title": wording,
            "type": "spd",
            "possibilities": {"type": "spd"},
        }

        post = {
            "id": qid,
            "question": question,
            "description": "",
            "pythia_iso3": iso3,
            "pythia_hazard_code": hz,
            "pythia_metric": metric,
            "pythia_target_month": target_month,
            "pythia_status": rec.get("status"),
            "pythia_hs_run_id": hs_run_id,
            "pythia_scenario_ids": scenario_ids,
            "pythia_window_start_date": rec.get("window_start_date"),
            "pythia_window_end_date": rec.get("window_end_date"),
            "pythia_metadata": pythia_meta,
            "created_time_iso": datetime.utcnow().isoformat(),
        }
        posts.append(post)

    return posts

def _maybe_dump_raw_gtmc1(content: str, *, run_id: str, question_id: str) -> Optional[str]:
    """
    If PYTHIA_DEBUG_RAW=1, write the raw LLM JSON-ish text we received for the
    GTMC1 actor table to a file in gtmc_logs/ and return the path. Otherwise None.
    """
    if os.getenv("PYTHIA_DEBUG_RAW", "0") != "1":
        return None
    try:
        os.makedirs("gtmc_logs", exist_ok=True)
        path = os.path.join("gtmc_logs", f"{run_id}_q{question_id}_actors_raw.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path
    except Exception:
        return None

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


def _load_calibration_advice_db(
    hazard_code: str,
    metric: str,
) -> Optional[str]:
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
            SELECT advice
            FROM calibration_advice
            WHERE hazard_code = ? AND metric = ?
            ORDER BY as_of_month DESC
            LIMIT 1
            """,
            [hz, mt],
        ).fetchone()
        if not row:
            return None
        advice = row[0]
        return str(advice)
    except Exception:
        return None
    finally:
        try:
            duckdb_io.close_db(conn)
        except Exception:
            pass

def _choose_weights_for_question(calib: Dict[str, Any], class_primary: str, qtype: str) -> Tuple[Dict[str, float], str]:
    model_names = [ms.name for ms in DEFAULT_ENSEMBLE]
    # 1) class-conditional
    try:
        by_class = calib.get("by_class", {})
        w = by_class.get(class_primary or "", {}).get(qtype, {})
        if isinstance(w, dict) and w:
            out = {m: float(w.get(m, 0.0)) for m in model_names}
            s = sum(out.values()) or 0.0
            if s > 0:
                return out, f"class_conditional:{class_primary}:{qtype}"
    except Exception:
        pass
    # 2) global
    try:
        glob = calib.get("global", {})
        w = glob.get(qtype, {})
        if isinstance(w, dict) and w:
            out = {m: float(w.get(m, 0.0)) for m in model_names}
            s = sum(out.values()) or 0.0
            if s > 0:
                return out, f"global:{qtype}"
    except Exception:
        pass
    # 3) uniform
    return ({m: 1.0 for m in model_names}, "uniform")

# --------------------------------------------------------------------------------
# Shape helpers
# --------------------------------------------------------------------------------

def _get_possibilities(q: dict) -> dict:
    return (q.get("possibilities") or q.get("range") or {})

def _get_options_list(q: dict) -> List[str]:
    if isinstance(q.get("options"), list):
        out = []
        for opt in q["options"]:
            if isinstance(opt, dict):
                out.append(str(opt.get("label") or opt.get("name") or ""))
            else:
                out.append(str(opt))
        return out
    poss = _get_possibilities(q)
    if isinstance(poss.get("options"), list):
        return [str(x.get("name") if isinstance(x, dict) else x) for x in poss["options"]]
    if isinstance(poss.get("scale", {}).get("options"), list):
        return [str(x.get("name") if isinstance(x, dict) else x) for x in poss["scale"]["options"]]
    return []

def _is_discrete(q: dict) -> bool:
    poss = _get_possibilities(q)
    q_type = (poss.get("type") or q.get("type") or "").lower()
    if q_type == "discrete":
        return True
    if q_type == "numeric" and isinstance(poss.get("scale", {}).get("values"), list):
        return True
    return False

def _discrete_values(q: dict) -> List[float]:
    poss = _get_possibilities(q)
    values = poss.get("scale", {}).get("values") or poss.get("values")
    if not values:
        return []
    return [float(v) for v in values]

# --------------------------------------------------------------------------------
# Simple, no-BMC fallback aggregators for the diagnostic variant "no_bmc_no_gtmc1"
# --------------------------------------------------------------------------------

def _simple_average_binary(members: List[MemberOutput]) -> Optional[float]:
    vals = [float(m.parsed) for m in members if m.ok and isinstance(m.parsed, (int, float))]
    if not vals:
        return None
    return float(np.mean([_clip01(v) for v in vals]))

def _simple_average_mcq(members: List[MemberOutput], n_opts: int) -> Optional[List[float]]:
    vecs: List[List[float]] = []
    for m in members:
        if m.ok and isinstance(m.parsed, list) and len(m.parsed) == n_opts:
            v = np.asarray(m.parsed, dtype=float)
            v = np.clip(v, 0.0, 1.0)
            s = float(v.sum())
            if s > 0:
                vecs.append((v / s).tolist())
    if not vecs:
        return None
    mean = np.mean(np.asarray(vecs), axis=0)
    mean = np.clip(mean, 1e-9, 1.0)
    mean = mean / float(mean.sum())
    return mean.tolist()

def _simple_average_numeric(members: List[MemberOutput]) -> Optional[Dict[str, float]]:
    p10s, p50s, p90s = [], [], []
    for m in members:
        if m.ok and isinstance(m.parsed, dict):
            d = m.parsed
            if "P10" in d and "P90" in d:
                p10s.append(float(d["P10"]))
                p90s.append(float(d["P90"]))
                p50s.append(float(d.get("P50", 0.5*(float(d["P10"]) + float(d["P90"])))))
    if not p10s:
        return None
    return {
        "P10": float(np.mean(p10s)),
        "P50": float(np.mean(p50s)) if p50s else 0.5 * (float(np.mean(p10s)) + float(np.mean(p90s))),
        "P90": float(np.mean(p90s)),
    }

# --------------------------------------------------------------------------------
# Core orchestration for ONE question → produce a single CSV row
# --------------------------------------------------------------------------------

async def _run_one_question_body(
    post: dict,
    *,
    run_id: str,
    purpose: str,
    calib: Dict[str, Any],
    seen_guard_state: Dict[str, Any],
    seen_guard_run_report: Optional[Dict[str, Any]] = None,
) -> None:
    t_start_total = time.time()
    _post_original = post
    try:
    
        post = _must_dict("post", post)
        q = _must_dict("q", post.get("question"))

        required = ("title", "type")
        missing = [k for k in required if not str(q.get(k, "")).strip()]
        if missing:
            raise RuntimeError(f"question payload missing required keys: {missing}")

        # Metaculus posts use integer IDs; Pythia Horizon Scanner posts use hex string IDs.
        # Try to coerce to int for backwards compatibility, but fall back to 0 so hex IDs don't crash.
        raw_post_id = post.get("id") or post.get("post_id") or 0
        try:
            post_id = int(raw_post_id)
        except (TypeError, ValueError):
            post_id = 0
        question_id_raw = q.get("id") or post.get("id") or post.get("post_id") or ""
        question_id = str(question_id_raw)
    
        seen_guard_enabled = bool(seen_guard_state.get("enabled", False))
        seen_guard_lock_acquired = seen_guard_state.get("lock_acquired")
        seen_guard_lock_error = str(seen_guard_state.get("lock_error") or "")
        
        title = str(q.get("title") or post.get("title") or "").strip()
        url = str(post.get("question_url") or "")
        qtype = (q.get("type") or "binary").strip()
        description = str(post.get("description") or q.get("description") or "")
        criteria = str(q.get("resolution_criteria") or q.get("fine_print") or q.get("resolution") or "")
        units = q.get("unit") or q.get("units") or ""
        tournament_id = post.get("pythia_hs_run_id") or ""
    
        # Options / discrete values
        options = _get_options_list(q)
        n_options = len(options) if qtype == "multiple_choice" else 0
        discrete_values = _discrete_values(q) if qtype in ("numeric", "discrete") and _is_discrete(q) else []
        ev_main: Optional[Dict[str, Any]] = None

        pmeta = _extract_pythia_meta(post)
        metric_up = (pmeta.get("metric") or "").upper()
        hz_code = (pmeta.get("hazard_code") or "").upper()
        hz_is_conflict = bool(hz_code and (hz_code in CONFLICT_HAZARD_CODES or hz_code.startswith("CONFLICT")))

        # ------------------ 1) Research step (LLM brief + sources appended) ---------
        t0 = time.time()
        research_text, research_meta = await run_research_async(
            run_id=run_id,
            question_id=str(question_id),
            title=title,
            description=description,
            criteria=criteria,
            qtype=qtype,
            options=options if qtype == "multiple_choice" else None,
            units=str(units) if units else None,
            slug=f"q{question_id}",
        )

        # Normalize meta to a dict so downstream `.get(...)` calls never crash.
        research_meta = _as_dict(research_meta)

        t_research_ms = _ms(t0)

        # Supplement research with PA history when we have iso3 + hazard_code
        pa_block = ""
        pa_meta: Dict[str, Any] = {}
        if pmeta.get("iso3") and pmeta.get("hazard_code"):
            pa_block, pa_meta = _load_pa_history_block(
                pmeta["iso3"],
                pmeta["hazard_code"],
            )
            if pa_block:
                research_text = f"{research_text}\n\n{pa_block}"
        # Merge PA meta into research_meta under a clear prefix
        for key, value in pa_meta.items():
            if key.startswith("pa_history_"):
                research_meta[key] = value
            else:
                research_meta[f"pa_history_{key}"] = value

        if qtype == "spd" and pmeta.get("iso3") and pmeta.get("hazard_code"):
            history_rows = []
            raw_history = pa_meta.get("pa_history_rows_detail")
            if isinstance(raw_history, list):
                for item in raw_history:
                    if not isinstance(item, dict):
                        continue
                    history_rows.append(
                        {
                            "ym": str(item.get("ym") or ""),
                            "value": item.get("value"),
                            "source": str(item.get("source") or ""),
                        }
                    )

            snapshot_start = history_rows[0]["ym"] if history_rows else ""
            snapshot_end = history_rows[-1]["ym"] if history_rows else ""
            context_extra = {
                "history_len": len(history_rows),
                "summary_text": pa_block,
            }

            try:
                con = connect(read_only=False)
                ensure_schema(con)
                con.execute(
                    "DELETE FROM question_context WHERE run_id = ? AND question_id = ?;",
                    [run_id, str(question_id)],
                )
                con.execute(
                    """
                    INSERT INTO question_context (
                        run_id,
                        question_id,
                        iso3,
                        hazard_code,
                        metric,
                        snapshot_start_month,
                        snapshot_end_month,
                        pa_history_json,
                        context_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    [
                        run_id,
                        str(question_id),
                        pmeta.get("iso3", "").upper(),
                        pmeta.get("hazard_code", "").upper(),
                        metric_up,
                        snapshot_start,
                        snapshot_end,
                        json.dumps(history_rows, ensure_ascii=False),
                        json.dumps(context_extra, ensure_ascii=False),
                    ],
                )
                con.close()
            except Exception as ctx_exc:  # noqa: BLE001
                print(
                    f"[warn] Failed to write question_context for Q{question_id}: {type(ctx_exc).__name__}: {ctx_exc}"
                )

        calib_advice_text = None
        if hz_code and metric_up:
            calib_advice_text = _load_calibration_advice_db(hz_code, metric_up)

        if calib_advice_text:
            research_text = (
                f"{research_text}\n\n## Calibration guidance for this hazard/metric\n{calib_advice_text}"
            )


        # ------------------ 2) Hazard-based "classification" (for GTMC1 gate) -----
        is_conflict_hazard = hz_is_conflict

        # We still publish classifier-like fields to keep CSV schema stable,
        # but they are now cheap deterministic values.
        class_primary = hz_code or ""
        class_secondary = ""
        is_strategic = is_conflict_hazard
        strategic_score = 1.0 if is_conflict_hazard else 0.0
        classifier_source = "hazard_code"
        classifier_rationale = (
            "hazard_code in CONFLICT_HAZARD_CODES"
            if is_conflict_hazard
            else "non-conflict hazard or missing hazard_code"
        )
        classifier_cost = 0.0

        # ------------------ 3) Optional GTMC1 (binary + conflict hazards only) ------
        # NOTE: for now, GTMC1 still only runs for binary questions. When we
        # move to SPD questions, we may relax the qtype guard.
        gtmc1_active = bool(is_conflict_hazard and qtype == "binary")
        actors_table: Optional[List[Dict[str, Any]]] = None
        gtmc1_signal: Dict[str, Any] = {}
        gtmc1_policy_sentence: str = ""
        t_gtmc1_ms = 0
    
        # Raw-dump debugging fields (only populated on failure / deactivation)
        gtmc1_raw_dump_path: str = ""
        gtmc1_raw_excerpt: str = ""
        gtmc1_raw_reason: str = ""

        if gtmc1_active:
            try:
                # Use the same async OpenAI client as other calls; model comes from config
                from .config import OPENAI_API_KEY
                from .providers import _get_or_client  # async OpenAI client

                client = _get_or_client()
                if client is None or not OPENAI_API_KEY:
                    gtmc1_active = False
                else:
                    prompt = f"""You are a research analyst preparing inputs for a Bruce Bueno de Mesquita-style
    game-theoretic bargaining model (BDM/Scholz). Identify actors and quantitative inputs on four dimensions.
    TITLE:
    {title}
    CONTEXT:
    {description}
    LATEST RESEARCH:
    {research_text}
    INSTRUCTIONS
    1) Define a POLICY CONTINUUM 0–100 for this question:
       0 = outcome least favorable to YES resolution; 100 = most favorable to YES resolution.
    2) Identify 3–8 ACTORS that materially influence the outcome (government, opposition, factions,
       mediators, veto players, firms, unions, external patrons).
    3) For each actor, provide:
       - "position" (0–100)
       - "capability" (0–100)
       - "salience" (0–100)
       - "risk_threshold" (0.00–0.10)
    4) OUTPUT STRICT JSON ONLY; NO commentary; schema:
    {{
      "policy_continuum": "Short one-sentence description of the 0–100 axis.",
      "actors": [
        {{"name":"Government","position":62,"capability":70,"salience":80,"risk_threshold":0.04}},
        {{"name":"Opposition","position":35,"capability":60,"salience":85,"risk_threshold":0.05}}
      ]
    }}
    Constraints: All numbers within ranges; 3–8 total actors; valid JSON.
    """
                    t_gt0 = time.time()
                    profile_models = {}
                    if _get_llm_profile_models is not None:
                        try:
                            profile_models = _get_llm_profile_models()
                        except Exception:
                            profile_models = {}
                    default_gtmc1_model = profile_models.get("openai", "gpt-5.1-pro")
                    async with llm_semaphore:
                        resp = await client.chat.completions.create(
                            model=os.getenv("GTMC1_MODEL_ID", default_gtmc1_model),
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.2,
                        )
                    text = (resp.choices[0].message.content or "").strip()
                    raw_text_for_debug = text  # keep exactly what the LLM sent
                    try:
                        data = json.loads(re.sub(r"^```json\s*|\s*```$", "", text, flags=re.S))
                    except Exception:
                        data = {}
                        gtmc1_active = False
                        gtmc1_raw_reason = "json_parse_error"
                        # Dump raw if requested; otherwise keep a short excerpt for the human log
                        gtmc1_raw_dump_path = _maybe_dump_raw_gtmc1(raw_text_for_debug, run_id=run_id, question_id=question_id) or ""
                        if not gtmc1_raw_dump_path:
                            # Use the same limit used elsewhere for model raw content
                            MAX_RAW = int(os.getenv("HUMAN_LOG_MODEL_RAW_MAX_CHARS", "5000"))
                            gtmc1_raw_excerpt = raw_text_for_debug[:MAX_RAW]
    
                    actors = data.get("actors") or []
                    gtmc1_policy_sentence = str(data.get("policy_continuum") or "").strip()
                    cleaned: List[Dict[str, Any]] = []
                    for a in actors:
                        try:
                            nm = str(a.get("name") or "").strip()
                            pos = float(a.get("position")); cap = float(a.get("capability"))
                            sal = float(a.get("salience")); thr = float(a.get("risk_threshold"))
                            if not nm: continue
                            if not (0.0 <= pos <= 100.0): continue
                            if not (0.0 <= cap <= 100.0): continue
                            if not (0.0 <= sal <= 100.0): continue
                            if not (0.0 <= thr <= 0.10): continue
                            cleaned.append({
                                "name": nm, "position": pos, "capability": cap,
                                "salience": sal, "risk_threshold": thr
                            })
                        except Exception:
                            continue
                    if len(cleaned) >= 3:
                        actors_table = cleaned
                        gtmc1_signal, _df_like = await asyncio.to_thread(
                            GTMC1.run_monte_carlo_from_actor_table,
                            actor_rows=actors_table,
                            num_runs=60,
                            log_dir="gtmc_logs",
                            run_slug=f"q{question_id}",
                        )
                    else:
                        gtmc1_active = False
                        gtmc1_raw_reason = "actors_lt_3"
                        gtmc1_raw_dump_path = _maybe_dump_raw_gtmc1(raw_text_for_debug, run_id=run_id, question_id=question_id) or ""
                        if not gtmc1_raw_dump_path:
                            MAX_RAW = int(os.getenv("HUMAN_LOG_MODEL_RAW_MAX_CHARS", "5000"))
                            gtmc1_raw_excerpt = raw_text_for_debug[:MAX_RAW]
            except Exception:
                gtmc1_active = False
                t_gtmc1_ms = 0

        # If GTMC1 succeeded, append a short summary to the research bundle
        if gtmc1_active and gtmc1_signal:
            try:
                prob_yes = gtmc1_signal.get("gtmc1_prob") \
                    or gtmc1_signal.get("prob_yes") \
                    or gtmc1_signal.get("exceedance_ge_50")
                coal_rate = gtmc1_signal.get("coalition_rate")
                disp = gtmc1_signal.get("dispersion")

                lines = [
                    "## GTMC1 scenario analysis (bargaining model)",
                    "",
                    f"- Policy continuum: {gtmc1_policy_sentence or '(not specified)'}",
                ]
                if prob_yes is not None:
                    lines.append(f"- GTMC1-estimated probability of YES-aligned outcome: {float(prob_yes):.2f}")
                if coal_rate is not None:
                    lines.append(f"- Coalition formation rate in simulations: {float(coal_rate):.2f}")
                if disp is not None:
                    lines.append(f"- Dispersion of actor positions (0–1): {float(disp):.2f}")

                gtmc1_block = "\n".join(lines)
                research_text = f"{research_text}\n\n{gtmc1_block}"
            except Exception:
                # On any formatting error, keep GTMC1 out of the research bundle but don't crash.
                pass
    
        # ------------------ 4) Build main prompts (WITH research) -------------------
        if qtype == "binary":
            main_prompt = build_binary_prompt(title, description, research_text, criteria)
        elif qtype == "multiple_choice":
            main_prompt = build_mcq_prompt(title, options, description, research_text, criteria)
        elif qtype == "spd":
            if metric_up == "FATALITIES" and hz_is_conflict:
                main_prompt = build_spd_prompt_fatalities(title, description, research_text, criteria)
            else:
                main_prompt = build_spd_prompt_pa(title, description, research_text, criteria)
        else:
            main_prompt = build_numeric_prompt(title, str(units or ""), description, research_text, criteria)
    
        # ------------------ 5) Ensemble calls (WITH research) -----------------------
        t0 = time.time()
        if qtype == "binary":
            ens_res = await run_ensemble_binary(main_prompt, DEFAULT_ENSEMBLE)
        elif qtype == "multiple_choice":
            ens_res = await run_ensemble_mcq(main_prompt, n_options, DEFAULT_ENSEMBLE)
        elif qtype == "spd":
            ens_res = await run_ensemble_spd(
                main_prompt,
                DEFAULT_ENSEMBLE,
                run_id=run_id,
                question_id=str(question_id),
            )
        else:
            ens_res = await run_ensemble_numeric(main_prompt, DEFAULT_ENSEMBLE)
        t_ensemble_ms = _ms(t0)
    
        # ------------------ 6) Choose calibration weights & aggregate ---------------
        calib_weights_map: Dict[str, float] = {}
        weights_profile = "uniform"

        if qtype == "spd" and hz_code and metric_up:
            db_weights = _load_calibration_weights_db(hz_code, metric_up)
            if db_weights:
                calib_weights_map = db_weights
                weights_profile = f"db:{hz_code}:{metric_up}"

        if not calib_weights_map:
            calib_weights_map, weights_profile = _choose_weights_for_question(
                _load_calibration_weights_file(), class_primary=class_primary, qtype=qtype
            )
    
        # MAIN aggregation (with optional GTMC1 for binary)
        if qtype == "binary":
            final_main, bmc_summary = aggregate_binary(ens_res, gtmc1_signal if gtmc1_active else None, calib_weights_map)
        elif qtype == "multiple_choice":
            vec_main, bmc_summary = aggregate_mcq(ens_res, n_options, calib_weights_map)
            final_main = {options[i]: vec_main[i] for i in range(n_options)} if n_options else {}
        elif qtype == "spd":
            bucket_labels = SPD_CLASS_BINS_PA
            default_centroids = SPD_BUCKET_CENTROIDS_DEFAULT
            if metric_up == "FATALITIES" and hz_is_conflict:
                bucket_labels = SPD_CLASS_BINS_FATALITIES
                default_centroids = SPD_BUCKET_CENTROIDS_FATALITIES_DEFAULT

            bucket_centroids = None
            if pmeta.get("hazard_code") and pmeta.get("metric"):
                bucket_centroids = _load_bucket_centroids(
                    pmeta["hazard_code"],
                    pmeta["metric"],
                    class_bins=bucket_labels,
                )

            if bucket_centroids is None:
                bucket_centroids = default_centroids

            try:
                # Normal SPD aggregation path
                spd_main, ev_dict, bmc_summary = aggregate_spd(
                    ens_res,
                    weights=calib_weights_map,
                    bucket_centroids=bucket_centroids,
                )
                from .ensemble import _normalize_spd_keys  # local import to avoid cycles

                spd_main = _normalize_spd_keys(spd_main, n_months=6, n_buckets=len(bucket_labels))
                final_main = spd_main
                ev_main = ev_dict
            except KeyError as exc:
                # Contain schema/parse bugs like KeyError('\n     "month_1"') and fall back.
                try:
                    raw_keys = set()
                    ens_obj = locals().get("ens_res")
                    # We don't need the exact class; just look for .members
                    if hasattr(ens_obj, "members"):
                        for _m in ens_obj.members:
                            if isinstance(_m, MemberOutput) and isinstance(_m.parsed, dict):
                                raw_keys.update(str(k) for k in _m.parsed.keys())
                except Exception:
                    raw_keys = set()

                offending = str(exc)
                print(
                    f"[spd] KeyError during SPD aggregation for question_id={question_id!r}: {offending!r}. "
                    f"raw_spd_keys={sorted(raw_keys)!r}. Falling back to uniform SPD across 6 months."
                )

                # Build a conservative uniform SPD: all buckets equal for all 6 months.
                n_buckets = len(bucket_labels)
                uniform_vec = [1.0 / float(n_buckets)] * n_buckets
                spd_main = {f"month_{i}": list(uniform_vec) for i in range(1, 7)}

                # No meaningful expected values if aggregation failed; keep it empty.
                ev_main = {}

                # Tag BMC summary so we can see this in CSV / logs.
                bmc_summary = {
                    "method": "spd_keyerror_fallback",
                    "error": offending,
                }

                final_main = spd_main
        else:
            quantiles_main, bmc_summary = aggregate_numeric(ens_res, calib_weights_map)
            final_main = dict(quantiles_main)

        # If this is a Pythia SPD question, write ensemble SPD into DuckDB
        if qtype == "spd" and isinstance(final_main, dict):
            # Heuristic: presence of Pythia metadata marks Pythia mode
            if "pythia_iso3" in post or "pythia_hazard_code" in post:
                try:
                    _write_spd_ensemble_to_db(
                        run_id=run_id,
                        question_id=str(question_id),
                        iso3=pmeta.get("iso3", ""),
                        hazard_code=pmeta.get("hazard_code", ""),
                        metric=pmeta.get("metric", ""),
                        spd_main=final_main,
                        ev_main=ev_main,
                        weights_profile=weights_profile,
                    )
                except Exception as exc:
                    print(f"[warn] Failed to write SPD ensemble to DB for question {question_id}: {exc}")
                try:
                    _write_spd_raw_to_db(
                        run_id=run_id,
                        question_id=str(question_id),
                        iso3=pmeta.get("iso3", ""),
                        hazard_code=pmeta.get("hazard_code", ""),
                        metric=pmeta.get("metric", ""),
                        ens_res=ens_res,
                    )
                except Exception as exc:
                    print(f"[warn] Failed to write SPD RAW to DB for question {question_id}: {exc}")

        bmc_summary = _as_dict(bmc_summary)

        # ------------------ 7) Diagnostic variants (WITH research) ------------------
        # NOTE: aggregate_binary now ignores gtmc1_signal; these variants are retained
        # only for schema continuity and weight-comparison diagnostics.
        if qtype == "binary":
            v_nogtmc1, _ = aggregate_binary(ens_res, None, calib_weights_map)
            v_uniform, _ = aggregate_binary(ens_res, gtmc1_signal if gtmc1_active else None, {m.name: 1.0 for m in DEFAULT_ENSEMBLE})
            v_simple = _simple_average_binary(ens_res.members)
        elif qtype == "multiple_choice":
            v_nogtmc1_vec, _ = aggregate_mcq(ens_res, n_options, calib_weights_map)
            v_nogtmc1 = {options[i]: v_nogtmc1_vec[i] for i in range(n_options)} if n_options else {}
            v_uniform_vec, _ = aggregate_mcq(ens_res, n_options, {m.name: 1.0 for m in DEFAULT_ENSEMBLE})
            v_uniform = {options[i]: v_uniform_vec[i] for i in range(n_options)} if n_options else {}
            v_simple_vec = _simple_average_mcq(ens_res.members, n_options)
            v_simple = {options[i]: v_simple_vec[i] for i in range(n_options)} if (n_options and v_simple_vec) else {}
        elif qtype == "spd":
            v_nogtmc1, ev_nogtmc1, _ = aggregate_spd(
                ens_res,
                weights=calib_weights_map,
                bucket_centroids=bucket_centroids,
            )
            v_uniform, ev_uniform, _ = aggregate_spd(
                ens_res,
                weights={m.name: 1.0 for m in DEFAULT_ENSEMBLE},
                bucket_centroids=bucket_centroids,
            )
            v_simple = v_nogtmc1
        else:
            v_nogtmc1, _ = aggregate_numeric(ens_res, calib_weights_map)
            v_uniform, _ = aggregate_numeric(ens_res, {m.name: 1.0 for m in DEFAULT_ENSEMBLE})
            v_simple = _simple_average_numeric(ens_res.members) or {}
    
        # ------------------ 8) Ablation pass: NO RESEARCH ---------------------------
        if qtype == "spd":
            # Skip ablation for SPD to avoid doubling LLM cost
            ab_main = final_main
            ab_uniform = final_main
            ab_simple = final_main
        elif qtype == "binary":
            ab_prompt = build_binary_prompt(title, description, "", criteria)
            ens_res_ab = await run_ensemble_binary(ab_prompt, DEFAULT_ENSEMBLE)
            ab_main, _ = aggregate_binary(ens_res_ab, None, calib_weights_map)
            ab_uniform, _ = aggregate_binary(ens_res_ab, None, {m.name: 1.0 for m in DEFAULT_ENSEMBLE})
            ab_simple = _simple_average_binary(ens_res_ab.members)
        elif qtype == "multiple_choice":
            ab_prompt = build_mcq_prompt(title, options, description, "", criteria)
            ens_res_ab = await run_ensemble_mcq(ab_prompt, n_options, DEFAULT_ENSEMBLE)
            ab_vec, _ = aggregate_mcq(ens_res_ab, n_options, calib_weights_map)
            ab_main = {options[i]: ab_vec[i] for i in range(n_options)} if n_options else {}
            ab_uniform_vec, _ = aggregate_mcq(ens_res_ab, n_options, {m.name: 1.0 for m in DEFAULT_ENSEMBLE})
            ab_uniform = {options[i]: ab_uniform_vec[i] for i in range(n_options)} if n_options else {}
            ab_simple_vec = _simple_average_mcq(ens_res_ab.members, n_options)
            ab_simple = {options[i]: ab_simple_vec[i] for i in range(n_options)} if (n_options and ab_simple_vec) else {}
        else:
            ab_prompt = build_numeric_prompt(title, str(units or ""), description, "", criteria)
            ens_res_ab = await run_ensemble_numeric(ab_prompt, DEFAULT_ENSEMBLE)
            ab_main, _ = aggregate_numeric(ens_res_ab, calib_weights_map)
            ab_uniform, _ = aggregate_numeric(ens_res_ab, {m.name: 1.0 for m in DEFAULT_ENSEMBLE})
            ab_simple = _simple_average_numeric(ens_res_ab.members) or {}
    
        # ------------------ 9) Build ONE wide CSV row and write it ------------------
        ensure_unified_csv()
    
        row: Dict[str, Any] = {
            # Run metadata
            "run_id": run_id,
            "run_time_iso": ist_iso(),
            "purpose": purpose,
            "git_sha": os.getenv("GIT_SHA", ""),
            "config_profile": "default",
            "weights_profile": "class_calibration",
            "openrouter_models_json": [
                {"name": ms.name, "provider": ms.provider, "model_id": ms.model_id, "weight": ms.weight}
                for ms in DEFAULT_ENSEMBLE
            ],
    
            # Question metadata
            "question_id": str(question_id),
            "question_url": url,
            "question_title": title,
            "question_type": qtype,
            "tournament_id": tournament_id if isinstance(tournament_id, str) else str(tournament_id),
            "created_time_iso": post.get("creation_time") or q.get("creation_time") or "",
            "closes_time_iso": post.get("close_time") or q.get("close_time") or "",
            "resolves_time_iso": post.get("scheduled_resolve_time") or q.get("scheduled_resolve_time") or "",
    
            # Classification
            "class_primary": class_primary,
            "class_secondary": class_secondary or "",
            "is_strategic": str(is_strategic),
            "strategic_score": f"{strategic_score:.3f}",
            "classifier_source": classifier_source,
            "classifier_rationale": classifier_rationale,
    
            # Research
            "research_llm": research_meta.get("research_llm", ""),
            "research_source": research_meta.get("research_source", ""),
            "research_query": research_meta.get("research_query", ""),
            "research_n_raw": str(research_meta.get("research_n_raw", "")),
            "research_n_kept": str(research_meta.get("research_n_kept", "")),
            "research_cached": research_meta.get("research_cached", ""),
            "research_error": research_meta.get("research_error", ""),
    
    
            # Options/values
            "n_options": str(n_options if qtype == "multiple_choice" else 0),
            "options_json": options if qtype == "multiple_choice" else "",
            "discrete_values_json": discrete_values if (qtype in ("numeric", "discrete") and discrete_values) else "",
        }
    
        row["seen_guard_triggered"] = (
            "1"
            if seen_guard_enabled and bool(seen_guard_lock_acquired)
            else ("0" if seen_guard_enabled else "")
        )
    
        # Per-model outputs
        for i, ms in enumerate(DEFAULT_ENSEMBLE):
            mo: Optional[MemberOutput] = None
            if isinstance(ens_res, EnsembleResult) and i < len(ens_res.members):
                mo = ens_res.members[i]
    
            ok = bool(mo and mo.ok)
            row[f"model_ok__{ms.name}"] = "1" if ok else "0"
            row[f"model_time_ms__{ms.name}"] = str(getattr(mo, "elapsed_ms", 0) or "")
    
            if ok and mo is not None:
                if qtype == "binary" and isinstance(mo.parsed, (float, int)):
                    row[f"binary_prob__{ms.name}"] = f"{_clip01(float(mo.parsed)):.6f}"
                elif qtype == "multiple_choice" and isinstance(mo.parsed, list):
                    row[f"mcq_json__{ms.name}"] = mo.parsed
                elif qtype == "spd" and isinstance(mo.parsed, dict):
                    row[f"spd_json__{ms.name}"] = mo.parsed
                elif qtype in ("numeric", "discrete") and isinstance(mo.parsed, dict):
                    p10 = _safe_float(mo.parsed.get("P10"))
                    p50 = _safe_float(mo.parsed.get("P50"))
                    p90 = _safe_float(mo.parsed.get("P90"))
                    if p10 is not None: row[f"numeric_p10__{ms.name}"] = f"{p10:.6f}"
                    if p50 is not None: row[f"numeric_p50__{ms.name}"] = f"{p50:.6f}"
                    if p90 is not None: row[f"numeric_p90__{ms.name}"] = f"{p90:.6f}"
    
            row[f"cost_usd__{ms.name}"] = f"{getattr(mo,'cost_usd',0.0):.6f}" if mo else ""
    
        # Ensemble (main)
        if qtype == "binary" and isinstance(final_main, float):
            row["binary_prob__ensemble"] = f"{_clip01(final_main):.6f}"
        elif qtype == "multiple_choice" and isinstance(final_main, dict):
            row["mcq_json__ensemble"] = final_main
            for j in range(min(15, n_options)):
                row[f"mcq_{j+1}__ensemble"] = f"{_clip01(float(final_main.get(options[j], 0.0))):.6f}"
        elif qtype == "spd" and isinstance(final_main, dict):
            row["spd_json__ensemble"] = final_main
            if isinstance(ev_main, dict):
                row["spd_ev_json__ensemble"] = ev_main
        elif qtype in ("numeric", "discrete") and isinstance(final_main, dict):
            for k in ("P10", "P50", "P90"):
                if k in final_main:
                    row[f"numeric_{k.lower()}__ensemble"] = f"{float(final_main[k]):.6f}"
    
        # Variants (WITH research)
        def _fill_variant(tag: str, val: Any):
            if qtype == "binary" and isinstance(val, float):
                row[f"binary_prob__ensemble_{tag}"] = f"{_clip01(val):.6f}"
            elif qtype == "multiple_choice" and isinstance(val, dict):
                row[f"mcq_json__ensemble_{tag}"] = val
            elif qtype == "spd" and isinstance(val, dict):
                row[f"spd_json__ensemble_{tag}"] = val
            elif qtype in ("numeric", "discrete") and isinstance(val, dict):
                for k in ("P10", "P50", "P90"):
                    if k in val:
                        row[f"numeric_{k.lower()}__ensemble_{tag}"] = f"{float(val[k]):.6f}"
    
        _fill_variant("no_gtmc1", v_nogtmc1)
        _fill_variant("uniform_weights", v_uniform)
        if qtype == "binary":
            _fill_variant("no_bmc_no_gtmc1", v_simple)  # float for binary
        else:
            _fill_variant("no_bmc_no_gtmc1", v_simple if isinstance(v_simple, dict) else v_simple)
    
        # Ablation (NO research)
        row["ablation_no_research"] = "1"
        if qtype == "binary" and isinstance(ab_main, float):
            row["binary_prob__ensemble_no_research"] = f"{_clip01(ab_main):.6f}"
        elif qtype == "multiple_choice" and isinstance(ab_main, dict):
            row["mcq_json__ensemble_no_research"] = ab_main
            for j in range(min(15, n_options)):
                row[f"mcq_{j+1}__ensemble_no_research"] = f"{_clip01(float(ab_main.get(options[j], 0.0))):.6f}"
        elif qtype == "spd" and isinstance(ab_main, dict):
            row["spd_json__ensemble_no_research"] = ab_main
        elif qtype in ("numeric", "discrete") and isinstance(ab_main, dict):
            for k in ("P10", "P50", "P90"):
                if k in ab_main:
                    row[f"numeric_{k.lower()}__ensemble_no_research"] = f"{float(ab_main[k]):.6f}"
    
        def _fill_ablation_variant(tag: str, val: Any):
            if qtype == "binary" and isinstance(val, float):
                row[f"binary_prob__ensemble_no_research_{tag}"] = f"{_clip01(val):.6f}"
            elif qtype == "multiple_choice" and isinstance(val, dict):
                row[f"mcq_json__ensemble_no_research_{tag}"] = val
            elif qtype == "spd" and isinstance(val, dict):
                row[f"spd_json__ensemble_no_research_{tag}"] = val
            elif qtype in ("numeric", "discrete") and isinstance(val, dict):
                for k in ("P10", "P50", "P90"):
                    if k in val:
                        row[f"numeric_{k.lower()}__ensemble_no_research_{tag}"] = f"{float(val[k]):.6f}"
    
        _fill_ablation_variant("no_gtmc1", ab_main)
        _fill_ablation_variant("uniform_weights", ab_uniform)
        _fill_ablation_variant("no_bmc_no_gtmc1", ab_simple if isinstance(ab_simple, dict) else ({"P50": ab_simple} if isinstance(ab_simple, float) else ab_simple))

        # Diagnostics, timings, weights used
        gtmc1_signal = _as_dict(gtmc1_signal)
        row.update({
            "gtmc1_active": "1" if gtmc1_active else "0",
            "actors_cached": "0",
            "gtmc1_actor_count": str(len(actors_table) if actors_table else 0),
            "gtmc1_coalition_rate": (gtmc1_signal.get("coalition_rate") if gtmc1_signal else ""),
            "gtmc1_exceedance_ge_50": (gtmc1_signal.get("exceedance_ge_50") if gtmc1_signal else ""),
            "gtmc1_dispersion": (gtmc1_signal.get("dispersion") if gtmc1_signal else ""),
            "gtmc1_median_rounds": (gtmc1_signal.get("median_rounds") if gtmc1_signal else ""),
            "gtmc1_num_runs": (gtmc1_signal.get("num_runs") if gtmc1_signal else ""),
            "gtmc1_policy_sentence": gtmc1_policy_sentence or "",
            "gtmc1_signal_json": gtmc1_signal or "",
    
            "bmc_summary_json": "",
    
            "cdf_steps_clamped": "",
            "cdf_upper_open_adjusted": "",
            "prob_sum_renormalized": "",

            "t_research_ms": str(t_research_ms),
            "t_ensemble_ms": str(t_ensemble_ms),
            "t_gtmc1_ms": str(t_gtmc1_ms),
            "t_total_ms": str(_ms(t_start_total)),

            "resolved": "",
            "resolved_time_iso": "",
            "resolved_outcome_label": "",
            "resolved_value": "",
            "score_brier": "",
            "score_log": "",
            "score_crps": "",
    
            "score_brier__no_research": "",
            "score_log__no_research": "",
            "score_crps__no_research": "",
    
            "weights_profile_applied": weights_profile,
            "weights_per_model_json": calib_weights_map,
            "dedupe_hash": "",
            "seen_guard_triggered": "",
        })
    
        # Human-readable markdown log
        MAX_RAW_CHARS = int(os.getenv("HUMAN_LOG_MODEL_RAW_MAX_CHARS","5000"))
        RESEARCH_MAX = int(os.getenv("HUMAN_LOG_RESEARCH_MAX_CHARS","20000"))
        md = []
        md.append(f"# {title} (QID: {question_id})")
        md.append(f"- Type: {qtype}")
        md.append(f"- URL: {url}")
        md.append(f"- Classifier: {class_primary} | strategic={is_strategic} (score={strategic_score:.2f})")
        md.append("### SeenGuard")
        lock_status = "n/a"
        if seen_guard_enabled:
            lock_status = "acquired" if seen_guard_lock_acquired else "not_acquired"
        md.append(f"- enabled={seen_guard_enabled} | lock_status={lock_status}")
        if seen_guard_run_report:
            before = seen_guard_run_report.get("before")
            skipped = seen_guard_run_report.get("skipped")
            after = seen_guard_run_report.get("after")
            md.append(f"- run_filter: before={before} | skipped={skipped} | after={after}")
            if seen_guard_run_report.get("error"):
                md.append(f"- filter_error={seen_guard_run_report['error']}")
        debug_note = "lock disabled"
        if seen_guard_enabled:
            debug_note = "lock acquired" if seen_guard_lock_acquired else "lock fallback"
        if seen_guard_lock_error:
            debug_note += f" | error={seen_guard_lock_error}"
        md.append(f"- debug_note={debug_note}")
    
        md.append("## Research (summary)")
        md.append((research_text or "").strip()[:RESEARCH_MAX])
        # Research (debug)
        try:
            _r_src   = research_meta.get("research_source","")
            _r_llm   = research_meta.get("research_llm","")
            _r_q     = research_meta.get("research_query","")
            _r_raw   = research_meta.get("research_n_raw","")
            _r_kept  = research_meta.get("research_n_kept","")
            _r_cache = research_meta.get("research_cached","")
            _r_err   = research_meta.get("research_error","")
            md.append("### Research (debug)")
            _r_cost  = research_meta.get("research_cost_usd", 0.0)
            md.append(
                f"- source={_r_src} | llm={_r_llm} | cached={_r_cache} | "
                f"n_raw={_r_raw} | n_kept={_r_kept} | cost=${float(_r_cost):.6f}"
            )
    
            if _r_q:
                md.append(f"- query: {_r_q}")
            if _r_err:
                md.append(f"- error: {_r_err}")
        except Exception:
            pass
    
        # --- GTMC1 (debug) --------------------------------------------------------
        try:
            md.append("### GTMC1 (debug)")
            # Basic flags
            md.append(f"- strategic_class={is_strategic} | strategic_score={strategic_score:.2f} | source={classifier_source}")
            md.append(f"- gtmc1_active={gtmc1_active} | qtype={qtype} | t_ms={t_gtmc1_ms}")
    
            # Actor extraction outcome
            _n_actors = len(actors_table) if actors_table else 0
            md.append(f"- actors_parsed={_n_actors}")
    
            # Key Monte Carlo outputs (if any)
            _sig = gtmc1_signal or {}
            _ex = _sig.get("exceedance_ge_50")
            _coal = _sig.get("coalition_rate")
            _med = _sig.get("median_of_final_medians")
            _disp = _sig.get("dispersion")
    
            md.append(f"- exceedance_ge_50={_ex} | coalition_rate={_coal} | median={_med} | dispersion={_disp}")
            _runs_csv = _sig.get("runs_csv")
            if _runs_csv:
                md.append(f"- runs_csv={_runs_csv}")
            _meta_json = _sig.get("meta_json")
            if _meta_json:
                md.append(f"- meta_json={_meta_json}")
    
            # If GTMC1 was expected but didn’t apply, say why (best effort).
            if is_conflict_hazard and qtype == "binary" and not gtmc1_active:
                md.append("- note=GTMC1 gate opened (conflict hazard) but deactivated later (client/JSON/actors<3).")
            # If we captured raw (on failure), surface it.
            if gtmc1_raw_reason:
                md.append(f"- raw_reason={gtmc1_raw_reason}")
            if gtmc1_raw_dump_path or gtmc1_raw_excerpt:
                md.append("### GTMC1 (raw)")
                if gtmc1_raw_dump_path:
                    md.append(f"- raw_file={gtmc1_raw_dump_path}")
                if gtmc1_raw_excerpt:
                    md.append("```json")
                    md.append(gtmc1_raw_excerpt)
                    md.append("```")
        except Exception as _gtmc1_dbg_ex:
            md.append(f"- gtmc1_debug_error={type(_gtmc1_dbg_ex).__name__}: {str(_gtmc1_dbg_ex)[:200]}")
        # --------------------------------------------------------------------------
    
        # --- GTMC1 (actors used) ---------------------------------------------------
        # Show the actual table we fed into GTMC1 so you can audit inputs later.
        if gtmc1_active and actors_table:
            try:
                md.append("### GTMC1 (actors used)")
                md.append("| Actor | Position | Capability | Salience | Risk thresh |")
                md.append("|---|---:|---:|---:|---:|")
                for a in actors_table:
                    md.append(
                        f"| {a['name']} | {float(a['position']):.0f} | "
                        f"{float(a['capability']):.0f} | {float(a['salience']):.0f} | "
                        f"{float(a['risk_threshold']):.3f} |"
                    )
            except Exception as _gtmc1_tbl_ex:
                md.append(f"- actors_table_render_error={type(_gtmc1_tbl_ex).__name__}: {str(_gtmc1_tbl_ex)[:160]}")
    
        # --- Ensemble outputs (compact) --------------------------------------------
        try:
            md.append("### Ensemble (model outputs)")
            for m in ens_res.members:
                if not isinstance(m, MemberOutput):
                    continue
                _line = f"- {m.name}: ok={m.ok} t_ms={getattr(m,'elapsed_ms',0)}"
                if qtype == "binary" and m.ok and isinstance(m.parsed, (float, int)):
                    _line += f" p={_clip01(float(m.parsed)):.4f}"
                elif qtype == "multiple_choice" and m.ok and isinstance(m.parsed, list):
                    # just show top-3
                    try:
                        vec = [float(x) for x in m.parsed]
                        idxs = np.argsort(vec)[::-1][:3]
                        _line += " top3=" + ", ".join([f"{options[i]}:{_clip01(vec[i]):.3f}" for i in idxs])
                    except Exception:
                        pass
                elif qtype == "spd" and m.ok and isinstance(m.parsed, dict):
                    try:
                        m1 = m.parsed.get("month_1")
                        if isinstance(m1, list) and len(m1) >= 3:
                            _line += " month_1=" + ", ".join([f"{float(x):.2f}" for x in m1[:3]]) + " …"
                    except Exception:
                        pass
                elif qtype in ("numeric", "discrete") and m.ok and isinstance(m.parsed, dict):
                    p10 = _safe_float(m.parsed.get("P10"))
                    p50 = _safe_float(m.parsed.get("P50"))
                    p90 = _safe_float(m.parsed.get("P90"))
                    if p10 is not None and p90 is not None:
                        if p50 is None:
                            p50 = 0.5 * (p10 + p90)
                        _line += f" P10={p10:.3f}, P50={p50:.3f}, P90={p90:.3f}"
                md.append(_line)
        except Exception as _ens_dbg_ex:
            md.append(f"- ensemble_debug_error={type(_ens_dbg_ex).__name__}: {str(_ens_dbg_ex)[:200]}")
    
        # --- Per-model details: reasoning + usage/cost --------------------------------
        try:
            MODEL_RAW_MAX = int(os.getenv("HUMAN_LOG_MODEL_RAW_MAX_CHARS", "5000"))
            md.append("")
            md.append("### Per-model (raw + usage/cost)")
    
            for m in ens_res.members:
                if not isinstance(m, MemberOutput):
                    continue
                md.append(f"#### {m.name}")
                md.append(
                    f"- ok={m.ok} | t_ms={getattr(m,'elapsed_ms',0)} | "
                    f"tokens: prompt={getattr(m,'prompt_tokens',0)}, "
                    f"completion={getattr(m,'completion_tokens',0)}, "
                    f"total={getattr(m,'total_tokens',0)} | "
                    f"cost=${float(getattr(m,'cost_usd',0.0)):.6f}"
                )
                if getattr(m, "error", None):
                    md.append(f"- error={str(m.error)[:240]}")
                if getattr(m, "raw_text", None):
                    raw = (m.raw_text or "").strip()
                    if raw:
                        md.append("```md")
                        md.append(raw[:MODEL_RAW_MAX])
                        md.append("```")
        except Exception as _pm_ex:
            md.append(f"- per_model_dump_error={type(_pm_ex).__name__}: {str(_pm_ex)[:200]}")
    
        # --- Aggregation summary (BMC) ---------------------------------------------
        try:
            md.append("### Aggregation (BMC)")
            # Make the BMC summary JSON-safe and also visible in the human log
            bmc_json = {}
            if isinstance(bmc_summary, dict):
                # strip large arrays already removed; copy select keys if present
                for k in ("mean", "var", "std", "n_evidence", "p10", "p50", "p90"):
                    if k in bmc_summary:
                        bmc_json[k] = bmc_summary[k]
            # Put a human line:
            if qtype == "binary" and isinstance(final_main, float):
                md.append(f"- final_probability={_clip01(final_main):.4f}")
            elif qtype == "multiple_choice" and isinstance(final_main, dict):
                # show top-3
                items = sorted(final_main.items(), key=lambda kv: kv[1], reverse=True)[:3]
                md.append("- final_top3=" + ", ".join([f"{k}:{_clip01(float(v)):.3f}" for k, v in items]))
            elif qtype == "spd" and isinstance(final_main, dict):
                md.append("### SPD Forecast (5 buckets × 6 months)")
                for m_idx in range(1, 7):
                    key = f"month_{m_idx}"
                    probs = final_main.get(key)
                    if not isinstance(probs, list) or len(probs) != 5:
                        continue
                    line = " | ".join(f"{float(p):.2f}" for p in probs)
                    md.append(f"- {key}: {line}")
                if isinstance(ev_main, dict):
                    md.append("### Expected people affected (per month)")
                    for key, val in sorted(ev_main.items()):
                        md.append(f"- {key}: {float(val):,.0f}")
            elif qtype in ("numeric", "discrete") and isinstance(final_main, dict):
                _p10 = final_main.get("P10"); _p50 = final_main.get("P50"); _p90 = final_main.get("P90")
                md.append(f"- final_quantiles: P10={_p10}, P50={_p50}, P90={_p90}")
            md.append(f"- bmc_summary={json.dumps(bmc_json)}")
        except Exception as _bmc_dbg_ex:
            md.append(f"- bmc_debug_error={type(_bmc_dbg_ex).__name__}: {str(_bmc_dbg_ex)[:200]}")
    
        # --------------------------------------------------------------------------
        # Attach BMC summary into CSV row (JSON), then persist both CSV + human log
        # --------------------------------------------------------------------------
        try:
            if isinstance(bmc_summary, dict):
                row["bmc_summary_json"] = {k: v for k, v in bmc_summary.items() if k != "samples"}
        except Exception:
            # keep whatever default is in row already
            pass
    
        # Write human-readable markdown file
        try:
            safe_md = _sanitize_markdown_chunks(md)
            if len(safe_md) < len(md):
                dropped = len(md) - len(safe_md)
                print(f"[warn] Dropped {dropped} non-string markdown line(s) for Q{question_id}.")
            write_human_markdown(run_id=run_id, question_id=question_id, content="\n\n".join(safe_md))
        except Exception as _md_ex:
            print(f"[warn] failed to write human markdown for Q{question_id}: {type(_md_ex).__name__}: {str(_md_ex)[:180]}")
    
        # Finally, write the unified CSV row
        write_unified_row(row)
        print("✔ logged to forecasts.csv")
        return
    
    
    except Exception as _e:
        _post_t = type(_post_original).__name__
        try:
            _q_t = type(q).__name__
        except Exception:
            _q_t = "unknown"
        try:
            _cls_t = type(cls_info).__name__  # may be undefined earlier; that's fine
        except Exception:
            _cls_t = "unknown"

        _err_t = type(_e).__name__
        _err_msg = str(_e)[:200]

        # Detect SPD questions
        is_spd = False
        try:
            poss = (q.get("possibilities") or {}) if isinstance(q, dict) else {}
            qt = (poss.get("type") or (q.get("type") if isinstance(q, dict) else "") or "").lower()
            is_spd = (qt == "spd")
        except Exception:
            is_spd = False

        spd_keys = None
        if is_spd:
            try:
                keys_set = set()
                ens_obj = locals().get("ens_res")
                if hasattr(ens_obj, "members"):
                    for _m in getattr(ens_obj, "members", []):
                        if isinstance(_m, MemberOutput) and isinstance(_m.parsed, dict):
                            keys_set.update(_m.parsed.keys())
                final_obj = locals().get("final_main")
                if isinstance(final_obj, dict):
                    keys_set.update(final_obj.keys())
                spd_keys = sorted(list(keys_set)) if keys_set else []
            except Exception:
                spd_keys = None

        # Core log message
        try:
            msg = (
                f"[error] run_one_question internal failure "
                f"(post_type={_post_t}, q_type={_q_t}, cls_info_type={_cls_t}): "
                f"{_err_t}: {_err_msg}"
            )
            if spd_keys is not None:
                msg += f" | spd_keys={spd_keys!r}"
            print(msg)
            traceback.print_exc()
        except Exception:
            pass

        # --- SPD soft-fail toggle ---
        hard_fail = os.getenv("PYTHIA_SPD_HARD_FAIL", "0") == "1"

        if is_spd and isinstance(_e, KeyError) and not hard_fail:
            print(
                f"[spd] soft-fail KeyError in SPD question; "
                f"skipping question without raising (post_type={_post_t}, q_type={_q_t})."
            )
            return

        raise RuntimeError(
            f"run_one_question failed (post={_post_t}, q={_q_t}, cls_info={_cls_t})"
        ) from _e


async def run_one_question(
    post: dict,
    *,
    run_id: str,
    purpose: str,
    calib: Dict[str, Any],
    seen_guard_run_report: Optional[Dict[str, Any]] = None,
) -> None:
    post = _must_dict("post", post)
    q = _as_dict(post.get("question"))
    question_id_raw = q.get("id") or post.get("id") or post.get("post_id") or ""
    question_id = str(question_id_raw)

    seen_guard_state: Dict[str, Any] = {
        "enabled": bool(seen_guard),
        "lock_acquired": None,
        "lock_error": "",
    }

    lock_stack = ExitStack()
    try:
        if seen_guard:
            try:
                acquired = lock_stack.enter_context(seen_guard.lock(question_id))
                seen_guard_state["lock_acquired"] = bool(acquired)
                if not acquired:
                    print(f"[seen_guard] QID {question_id} is locked by another process; skipping.")
                    return
            except Exception as _sg_lock_ex:
                seen_guard_state["lock_error"] = f"{type(_sg_lock_ex).__name__}: {str(_sg_lock_ex)[:160]}"
                seen_guard_state["lock_acquired"] = False
                print(f"[seen_guard] lock error for QID {question_id}: {seen_guard_state['lock_error']}")

        await _run_one_question_body(
            post=post,
            run_id=run_id,
            purpose=purpose,
            calib=calib,
            seen_guard_state=seen_guard_state,
            seen_guard_run_report=seen_guard_run_report,
        )
    finally:
        lock_stack.close()


# ==============================================================================
# Top-level runner (fetch posts, iterate, and commit logs)
# ==============================================================================

async def run_job(mode: str, limit: int, purpose: str, *, questions_file: str = "data/test_questions.json") -> None:
    """
    Fetch a batch of posts and process them one by one.
    Supports:
      - mode="pythia": reads Horizon Scanner questions from DuckDB
      - mode="test_questions": reads local JSON of test posts
    """
    # --- local imports to keep this function self-contained ---------------
    import os, json, inspect
    from pathlib import Path

    def _istamp():
        # Use Istanbul-tz stamp from config if available, else UTC-ish fallback
        try:
            from .config import IST_TZ
            from datetime import datetime
            return datetime.now(IST_TZ).strftime("%Y%m%d-%H%M%S")
        except Exception:
            from datetime import datetime, timezone
            return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    run_id = _istamp()
    os.environ["PYTHIA_FORECASTER_RUN_ID"] = run_id
    print("----------------------------------------------------------------------------------------")

    # --- load helpers from this module scope --------------------------------
    # They already exist below in this file; just reference them:
    #   ensure_unified_csv(), run_one_question(...), _load_calibration_weights_file()
    #   finalize_and_commit()

    # --- load questions ------------------------------------------------------
    posts: List[dict] = []
    fetch_limit = max(1, limit)

    if mode == "test_questions":
        qfile = Path(questions_file)
        if not qfile.exists():
            raise FileNotFoundError(f"Questions file not found: {qfile}")

        with qfile.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            posts = data
        elif isinstance(data, dict):
            posts = data.get("results") or data.get("posts") or []
        else:
            posts = []

        print(f"[info] Loaded {len(posts)} test post(s) from {qfile.as_posix()}.")
    elif mode == "pythia":
        print("[info] Loading Pythia questions from DuckDB...")
        posts = _load_pythia_questions(fetch_limit)
        print(f"[info] Loaded {len(posts)} question(s) from DuckDB (Pythia mode).")
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # --- SeenGuard wiring (handles both package and top-level) ---------------
    def _load_seen_guard():
        """
        Try to import a SeenGuard instance/class from forecaster.seen_guard or seen_guard.
        Return an instance or None.
        """
        sg_mod = None
        # Prefer relative (inside package)
        try:
            from . import seen_guard as _sg
            sg_mod = _sg
        except Exception:
            # Fall back to absolute names
            for modname in ("forecaster.seen_guard", "seen_guard"):
                try:
                    sg_mod = importlib.import_module(modname)
                    break
                except Exception:
                    continue

        if sg_mod is None:
            return None

        # If module exposes a ready-made instance, use it
        for attr in ("_GUARD", "GUARD", "guard"):
            guard = getattr(sg_mod, attr, None)
            if guard is not None:
                return guard

        # Else instantiate SeenGuard(csv_path/state_file/lock_dir via env defaults)
        SG = getattr(sg_mod, "SeenGuard", None)
        if SG is not None and inspect.isclass(SG):
            try:
                return SG()  # it reads env defaults internally
            except Exception:
                return None

        return None

    def _apply_seen_guard(guard, posts_list):
        """
        Call whichever filter method exists; normalize return to (posts, dup_count).
        """
        if guard is None or not posts_list:
            return posts_list, 0

        candidates = [
            "filter_unseen_posts",     # your current API
            "filter_fresh_posts",      # earlier suggestion
            "filter_posts",
            "filter_recent_posts",
            "filter_new_posts",
            "filter",                  # very generic, last
        ]
        last_err = None
        for name in candidates:
            if hasattr(guard, name):
                fn = getattr(guard, name)
                try:
                    # most APIs: fn(posts)
                    result = fn(posts_list)
                except TypeError:
                    try:
                        # named arg fallback
                        result = fn(posts=posts_list)
                    except Exception as e:
                        last_err = e
                        continue
                except Exception as e:
                    last_err = e
                    continue

                # normalize return
                if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], list):
                    return result
                if isinstance(result, list):
                    # compute naive dup_count
                    return result, max(0, len(posts_list) - len(result))

                # unexpected shape → treat as no-op
                return posts_list, 0

        if last_err:
            raise last_err
        return posts_list, 0

    # Try to activate seen guard
    seen_guard_run_report: Dict[str, Any] = {
        "enabled": False,
        "before": len(posts),
        "after": len(posts),
        "skipped": 0,
        "error": "",
    }
    try:
        guard = _load_seen_guard()
        if guard is None:
            print("[seen_guard] not active; processing all posts returned.")
        else:
            seen_guard_run_report["enabled"] = True
            seen_guard_run_report["before"] = len(posts)
            before = len(posts)
            posts, dup_count = _apply_seen_guard(guard, posts)
            after = len(posts)
            if not isinstance(dup_count, int):
                dup_count = max(0, before - after)
            seen_guard_run_report["skipped"] = int(dup_count)
            seen_guard_run_report["after"] = after
            print(f"[seen_guard] {dup_count} duplicate(s) skipped; {after} fresh post(s) remain.")
    except Exception as _sg_ex:
        seen_guard_run_report["error"] = f"{type(_sg_ex).__name__}: {str(_sg_ex)[:200]}"
        print(f"[seen_guard] disabled due to error: {type(_sg_ex).__name__}: {str(_sg_ex)[:200]}")

    # Ensure CSV exists before we start
    ensure_unified_csv()

    # Process each post
    if not posts:
        print("[info] No posts to process.")
        try:
            finalize_and_commit()
            print("[logs] finalize_and_commit: done")
        except Exception as e:
            print(f"[warn] finalize_and_commit failed: {type(e).__name__}: {str(e)[:180]}")
        return

    batch = posts[: max(1, limit)]
    for idx, raw_post in enumerate(batch, start=1):
        post = raw_post
        if not isinstance(post, dict):
            print(
                f"[error] Skipping entry #{idx}: unexpected post type "
                f"{type(raw_post).__name__}"
            )
            continue

        q = post.get("question") or {}
        qid = q.get("id") or post.get("id") or "?"
        title = (q.get("title") or post.get("title") or "").strip()
        print("")
        print("----------------------------------------------------------------------------------------")
        print(f"[{idx}/{len(batch)}] ❓ {title}  (QID: {qid})")
        try:
            await run_one_question(
                post,
                run_id=run_id,
                purpose=purpose,
                calib=_load_calibration_weights_file(),
                seen_guard_run_report=seen_guard_run_report,
            )
        except Exception as e:
            print(f"[error] run_one_question failed for QID {qid}: {type(e).__name__}: {str(e)[:200]}")

    # Commit logs to git if configured
    try:
        finalize_and_commit()
        print("[logs] finalize_and_commit: done")
    except Exception as e:
        print(f"[warn] finalize_and_commit failed: {type(e).__name__}: {str(e)[:180]}")


# ==============================================================================
# CLI entrypoint
# ==============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forecaster runner")
    p.add_argument(
        "--mode",
        default="pythia",
        choices=["pythia", "test_questions"],
        help=(
            "Question source: 'pythia' (DuckDB questions table) or "
            "'test_questions' (local JSON)."
        ),
    )
    p.add_argument("--limit", type=int, default=20, help="Max posts to fetch/process")
    p.add_argument("--purpose", default="ad_hoc", help="String tag recorded in CSV/logs")
    p.add_argument("--questions-file", default="data/test_questions.json",
                   help="When --mode test_questions, path to JSON payload")
    return p.parse_args()

def main() -> None:
    try:
        _advise_poetry_lock_if_needed()
    except Exception:
        # Never block the run because of the hint
        pass
    args = _parse_args()
    print("🚀 Forecaster ensemble starting…")
    print(f"Mode: {args.mode} | Limit: {args.limit} | Purpose: {args.purpose}")
    try:
        ensure_schema()
        asyncio.run(
            run_job(
                mode=args.mode,
                limit=args.limit,
                purpose=args.purpose,
                questions_file=args.questions_file,
            )
        )
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"[fatal] {type(e).__name__}: {str(e)[:200]}")
        raise


if __name__ == "__main__":
    main()
