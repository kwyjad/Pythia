# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Analysis report routes: /v1/analysis_reports/* (Nov 2026).

The skill analyses, resolver reviews and forecast pipeline reviews are
written into Google Drive by scheduled sessions. ``sync_analysis_reports.yml``
copies them into their own GitHub release (``analysis-reports``, never the
data release) with a PDF of each and an ``index.json``; these routes read that
index and nothing in Drive. The API holds no Drive credential and must not.

- ``GET /v1/analysis_reports``: the index grouped by calendar month, newest
  first: per month the scoring rounds (skill + resolver report) and the
  forecast runs (forecast review), reruns nested as versions, and
  placeholders for a report that is expected and not yet written.
- ``GET /v1/analysis_reports/{report_id}/content``: the Markdown.
- ``GET /v1/analysis_reports/{report_id}/download?format=md|pdf``: 302 to the
  release asset.

A failed index fetch serves the last good copy; with none, an empty list and
``available: false``, never a 500. An id is looked up in the index before any
URL is built from it.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

import requests
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse

logger = logging.getLogger(__name__)

router = APIRouter()

DEFAULT_TAG = "analysis-reports"
INDEX_TTL_S = 600
CONTENT_CACHE_MAX = 64
#: Forecast runs before this date predate the forecast pipeline review, so
#: they get no "review pending" placeholder.
FORECAST_REVIEWS_SINCE = "2026-10-01"
TYPES = ("skill", "resolver", "forecast")
DISPLAY_NAMES = {
    "skill": "Skill analysis",
    "resolver": "Resolver pipeline review",
    "forecast": "Forecast pipeline review",
}

_LOCK = threading.Lock()
_INDEX: Dict[str, Any] = {"data": None, "fetched_at": 0.0, "last_error": None}
_CONTENT: Dict[str, str] = {}


def _release_base() -> Optional[str]:
    repo = (os.getenv("PYTHIA_DATA_REPO") or "").strip()
    if not repo:
        return None
    tag = (os.getenv("PYTHIA_ANALYSIS_REPORTS_TAG") or DEFAULT_TAG).strip() or DEFAULT_TAG
    return f"https://github.com/{repo}/releases/download/{tag}"


def _fetch(url: str, *, timeout: float = 10.0) -> requests.Response:
    """HTTP seam (tests patch this)."""
    headers = {"Cache-Control": "no-cache"}
    token = os.getenv("PYTHIA_GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp


def reset_cache() -> None:
    with _LOCK:
        _INDEX.update({"data": None, "fetched_at": 0.0, "last_error": None})
        _CONTENT.clear()


def load_index(now: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """The index, cached for ten minutes; the last good copy on a failure."""
    now = time.time() if now is None else now
    with _LOCK:
        if _INDEX["data"] is not None and now - _INDEX["fetched_at"] < INDEX_TTL_S:
            return _INDEX["data"]
        base = _release_base()
        if base is None:
            _INDEX["last_error"] = "PYTHIA_DATA_REPO is not set"
            return _INDEX["data"]
        try:
            data = _fetch(f"{base}/index.json?t={int(now)}").json()
            if not isinstance(data, dict) or not isinstance(data.get("reports"), list):
                raise ValueError("index.json has no reports list")
            _INDEX.update({"data": data, "fetched_at": now, "last_error": None})
        except Exception as exc:  # noqa: BLE001 - degrade, never 500
            logger.warning("analysis reports: index fetch failed: %s", exc)
            _INDEX["last_error"] = str(exc)[:200]
            # Try again soon, but keep serving what we had.
            _INDEX["fetched_at"] = now - INDEX_TTL_S + 60
        return _INDEX["data"]


# ---------------------------------------------------------------------------
# Grouping (pure)
# ---------------------------------------------------------------------------


def _version_key(rec: Dict[str, Any]) -> tuple:
    return (rec.get("rerun_date") or "", rec.get("prepared") or "")


def _entry(versions: List[Dict[str, Any]]) -> Dict[str, Any]:
    ordered = sorted(versions, key=_version_key, reverse=True)
    latest = dict(ordered[0])
    latest["available"] = True
    latest["expected"] = True
    latest["versions"] = [
        {k: v.get(k) for k in ("report_id", "rerun_date", "prepared", "md_asset", "pdf_asset",
                               "title", "verdict", "dry_run")}
        for v in ordered
    ]
    return latest


def _placeholder(kind: str, period: str, **extra: Any) -> Dict[str, Any]:
    return {"type": kind, "display_name": DISPLAY_NAMES[kind], "period": period,
            "expected": True, "available": False, "versions": [], **extra}


def _matches(rec: Dict[str, Any], types: Optional[List[str]], year: Optional[str],
             include_dry_runs: bool) -> bool:
    if types and rec.get("type") not in types:
        return False
    if year and not str(rec.get("period") or "").startswith(year):
        return False
    if not include_dry_runs and rec.get("dry_run"):
        return False
    return True


def group_reports(reports: List[Dict[str, Any]], *, forecast_runs: Optional[List[Dict[str, Any]]] = None,
                  types: Optional[List[str]] = None, year: Optional[str] = None,
                  include_dry_runs: bool = True) -> List[Dict[str, Any]]:
    """Calendar months, newest first; inside each, scoring rounds and forecast runs.

    ``forecast_runs`` (production forecaster runs from the DB, each with
    ``fc_run_id`` and ``date``) adds a "review pending" placeholder for a run
    with no forecast review. A scoring round missing one of its two reports
    gets a placeholder for it.
    """
    by_base: Dict[str, List[Dict[str, Any]]] = {}
    for rec in reports:
        if rec.get("type") not in TYPES or not rec.get("report_id"):
            continue
        by_base.setdefault(rec.get("rerun_of") or rec["report_id"], []).append(rec)
    entries = [_entry(v) for v in by_base.values()]

    cycles: Dict[str, Dict[str, Any]] = {}

    def cycle(period: str) -> Dict[str, Any]:
        return cycles.setdefault(period, {"period": period, "scoring_rounds": {}, "forecast_runs": {}})

    for e in entries:
        if not _matches(e, types, year, include_dry_runs):
            continue
        chain = e.get("chain") or {}
        c = cycle(str(e.get("period") or "unknown"))
        if e["type"] in ("skill", "resolver"):
            key = str(chain.get("publish_run_id") or f"unknown-{e['period']}")
            rnd = c["scoring_rounds"].setdefault(key, {
                "publish_run_id": chain.get("publish_run_id"),
                "date": chain.get("date"),
                "folder_name": chain.get("folder_name"),
                "reports": {},
            })
            rnd["reports"][e["type"]] = e
        else:
            fc = chain.get("fc_run_id") or e.get("file_fc_run_id") or e["report_id"]
            run = c["forecast_runs"].setdefault(fc, {
                "fc_run_id": fc,
                "date": chain.get("date") or e.get("file_date"),
                "publish_run_id": chain.get("publish_run_id"),
                "reports": [],
            })
            run["reports"].append(e)

    show_forecast = not types or "forecast" in types
    for run in forecast_runs or []:
        fc = run.get("fc_run_id")
        date = str(run.get("date") or "")
        if not fc or not date or date < FORECAST_REVIEWS_SINCE:
            continue
        if not show_forecast or (year and not date.startswith(year)):
            continue
        if any(fc in c["forecast_runs"] for c in cycles.values()):
            continue
        c = cycle(date[:7])
        c["forecast_runs"][fc] = {
            "fc_run_id": fc, "date": date, "publish_run_id": None,
            "reports": [_placeholder("forecast", date[:7], pending=True)],
        }

    out: List[Dict[str, Any]] = []
    for period in sorted(cycles, reverse=True):
        c = cycles[period]
        rounds = []
        for rnd in sorted(c["scoring_rounds"].values(), key=lambda r: r.get("date") or "", reverse=True):
            reports = []
            for kind in ("skill", "resolver"):
                if types and kind not in types:
                    continue
                reports.append(rnd["reports"].get(kind) or _placeholder(kind, period))
            rounds.append({**rnd, "reports": reports})
        runs = sorted(c["forecast_runs"].values(), key=lambda r: (r.get("date") or "", r["fc_run_id"]),
                      reverse=True)
        out.append({"period": period, "scoring_rounds": rounds, "forecast_runs": runs})
    return out


# ---------------------------------------------------------------------------
# Production forecast runs (for "review pending")
# ---------------------------------------------------------------------------


def production_forecast_runs() -> List[Dict[str, Any]]:
    """Production forecaster runs since reviews began, from the served DB.

    Any failure (older DB, missing table or column, no DB at all) means no
    placeholders, never an error."""
    try:
        from pythia.api.core import _con, _execute, _table_exists, _table_has_columns

        con = _con()
        if not _table_exists(con, "forecasts_ensemble"):
            return []
        if not _table_has_columns(con, "forecasts_ensemble", ["run_id", "created_at"]):
            return []
        test = (" AND COALESCE(is_test, FALSE) = FALSE"
                if _table_has_columns(con, "forecasts_ensemble", ["is_test"]) else "")
        rows = _execute(con, f"""
            SELECT run_id, CAST(MIN(created_at) AS DATE) AS d
            FROM forecasts_ensemble
            WHERE run_id LIKE 'fc\\_%' ESCAPE '\\'{test}
            GROUP BY run_id
            HAVING MIN(created_at) >= CAST(? AS TIMESTAMP)
        """, [FORECAST_REVIEWS_SINCE]).fetchall()
        return [{"fc_run_id": str(r[0]), "date": str(r[1])} for r in rows if r[0] and r[1]]
    except Exception as exc:  # noqa: BLE001
        logger.info("analysis reports: no forecast runs for placeholders (%s)", exc)
        return []


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _find(report_id: str) -> Dict[str, Any]:
    index = load_index() or {}
    for rec in index.get("reports") or []:
        if isinstance(rec, dict) and rec.get("report_id") == report_id:
            return rec
    raise HTTPException(status_code=404, detail="Unknown report")


@router.get("/v1/analysis_reports")
def list_analysis_reports(
    type: Optional[List[str]] = Query(None, description="skill, resolver or forecast (repeatable)"),
    year: Optional[str] = Query(None, pattern=r"^\d{4}$"),
    include_dry_runs: bool = Query(True),
):
    types = [t for t in (type or []) if t in TYPES] or None
    index = load_index()
    if index is None:
        return {"available": False, "generated_at": None, "cycles": [], "years": []}
    reports = [r for r in index.get("reports") or [] if isinstance(r, dict)]
    cycles = group_reports(reports, forecast_runs=production_forecast_runs(), types=types,
                           year=year, include_dry_runs=include_dry_runs)
    years = sorted({str(r.get("period") or "")[:4] for r in reports if r.get("period")}, reverse=True)
    return {
        "available": True,
        "generated_at": index.get("generated_at"),
        "stale": _INDEX.get("last_error") is not None,
        "years": years,
        "cycles": cycles,
    }


@router.get("/v1/analysis_reports/{report_id}/content")
def analysis_report_content(report_id: str):
    rec = _find(report_id)
    base = _release_base()
    asset = rec.get("md_asset")
    if not base or not asset:
        raise HTTPException(status_code=404, detail="Report text unavailable")
    key = str(rec.get("sha256") or report_id)
    with _LOCK:
        cached = _CONTENT.get(key)
    if cached is None:
        try:
            cached = _fetch(f"{base}/{asset}", timeout=20).text
        except Exception as exc:  # noqa: BLE001
            logger.warning("analysis reports: content fetch failed for %s: %s", report_id, exc)
            raise HTTPException(status_code=503, detail="Report text unavailable right now") from exc
        with _LOCK:
            if len(_CONTENT) >= CONTENT_CACHE_MAX:
                _CONTENT.pop(next(iter(_CONTENT)))
            _CONTENT[key] = cached
    return {"report_id": report_id, "title": rec.get("title"), "markdown": cached}


@router.get("/v1/analysis_reports/{report_id}/download")
def analysis_report_download(report_id: str, format: str = Query("pdf", pattern=r"^(md|pdf)$")):
    rec = _find(report_id)
    base = _release_base()
    asset = rec.get("md_asset") if format == "md" else rec.get("pdf_asset")
    if not base or not asset:
        raise HTTPException(status_code=404, detail=f"No {format} for this report")
    return RedirectResponse(f"{base}/{asset}", status_code=302)
