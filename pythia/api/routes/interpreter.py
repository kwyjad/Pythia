# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Interpreter routes: /v1/interpreter/*.

Serves the plain-language reports stored in ``interpretations`` (written by
interpreter/run.py) and the per-ISO3 attention values for the interpreter
page's map (read from ``forecast_deviation``).

Contracts:
- every "latest" selection threads ``include_test`` and filters
  ``COALESCE(is_test, FALSE) = FALSE`` when it is off;
- a pre-interpreter DB (no ``interpretations`` table, or no rows) returns
  ``has_interpretation: false`` — never a 500 — so the frontend renders an
  explicit empty state rather than zeros;
- prose placeholders are resolved SERVER-side via interpreter.render (the
  same implementation the stored markdown used), so the dashboard never
  re-implements figure formatting. Import-light: interpreter.render pulls
  only stdlib + interpreter.lexicon (see test_api_lazy_pipeline_import.py's
  constraint on this process).
"""

import json
import logging
import math
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from pythia.api.core import (
    _con,
    _execute,
    _rows_from_cursor,
    _table_exists,
    _test_filter,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_LN2 = math.log(2.0)

# Best-available aggregate per question — mirrors the dashboard's
# STANDARD_MODEL_PREFERENCE and compute_deviation's consumer ordering.
_MODEL_PREFERENCE = ("ensemble_bayesmc_v2", "ensemble_mean_v2", "track2_flash")

_KINDS = ("current", "scored", "combined")

# Columns served in list views (everything except the heavy content/figure
# payloads).
_LIGHT_COLUMNS = (
    "interpretation_id, kind, run_id, hs_run_id, scored_run_id, version, "
    "template_version, model_name, thinking_level, status, cost_usd, "
    "input_tokens, output_tokens, created_at, is_test"
)


def _maybe_json(raw: Any) -> Any:
    if raw is None or raw == "":
        return None
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def _shape_full(row: Dict[str, Any]) -> Dict[str, Any]:
    """One interpretation row -> API shape (content resolved server-side)."""
    content = _maybe_json(row.pop("content_json", None))
    figures = _maybe_json(row.pop("figures_json", None)) or {}
    validation = _maybe_json(row.pop("validation_json", None))

    content_resolved = None
    unresolved: List[str] = []
    if isinstance(content, dict):
        try:
            from interpreter.render import FigureResolver, resolve_content

            resolver = FigureResolver(
                per_question=figures.get("per_question") or {},
                global_figures=figures.get("global") or {},
            )
            content_resolved = resolve_content(content, resolver)
            unresolved = resolver.misses
        except Exception:  # noqa: BLE001 - serve the raw content over a 500
            logger.exception("interpreter content resolution failed")
            content_resolved = content

    row["content"] = content
    row["content_resolved"] = content_resolved
    row["validation"] = validation
    row["unresolved_figures"] = unresolved
    return row


@router.get("/v1/interpreter/latest")
def interpreter_latest(
    kind: str = Query("combined"),
    include_test: bool = Query(False),
):
    """The newest interpretation of this kind (highest version of the newest
    run). ``has_interpretation: false`` when none exists."""
    if kind not in _KINDS:
        raise HTTPException(status_code=400, detail=f"kind must be one of {_KINDS}")
    con = _con()
    if not _table_exists(con, "interpretations"):
        return {"has_interpretation": False, "kind": kind}
    rows = _rows_from_cursor(
        _execute(
            con,
            f"""
            SELECT * FROM interpretations
            WHERE kind = ?{_test_filter(include_test)}
            ORDER BY created_at DESC, version DESC
            LIMIT 1
            """,
            [kind],
        )
    )
    if not rows:
        return {"has_interpretation": False, "kind": kind}
    return {
        "has_interpretation": True,
        "kind": kind,
        "interpretation": _shape_full(rows[0]),
    }


@router.get("/v1/interpreter/versions")
def interpreter_versions(
    kind: Optional[str] = Query(None),
    run_id: Optional[str] = Query(None),
    include_test: bool = Query(False),
):
    """The version list for the selector (light rows, newest first)."""
    con = _con()
    if not _table_exists(con, "interpretations"):
        return {"rows": []}
    clauses = ""
    params: List[Any] = []
    if kind:
        if kind not in _KINDS:
            raise HTTPException(status_code=400, detail=f"kind must be one of {_KINDS}")
        clauses += " AND kind = ?"
        params.append(kind)
    if run_id:
        clauses += " AND COALESCE(run_id, scored_run_id, '') = ?"
        params.append(run_id)
    rows = _rows_from_cursor(
        _execute(
            con,
            f"""
            SELECT {_LIGHT_COLUMNS} FROM interpretations
            WHERE 1=1{clauses}{_test_filter(include_test)}
            ORDER BY created_at DESC, version DESC
            """,
            params,
        )
    )
    return {"rows": rows}


@router.get("/v1/interpreter/attention_map")
def interpreter_attention_map(
    run_id: Optional[str] = Query(None),
    include_test: bool = Query(False),
):
    """Per-ISO3 attention values for the interpreter map.

    ``attention`` is the country's maximum js_vs_baserate (best-available
    aggregate per question) scaled by its ln 2 maximum to [0, 1] — coloured
    by ATTENTION, not raw risk (a risk-coloured map would just repeat the
    risk index page). Every table touched is test-filtered.
    """
    con = _con()
    if not _table_exists(con, "forecast_deviation") or not _table_exists(con, "questions"):
        return {"has_data": False, "run_id": None, "rows": []}

    if not run_id:
        rows = _execute(
            con,
            f"""
            SELECT MAX(fd.run_id) FROM forecast_deviation fd
            JOIN questions q ON q.question_id = fd.question_id
            WHERE fd.run_id IS NOT NULL AND fd.run_id <> ''
              {_test_filter(include_test, "fd")}{_test_filter(include_test, "q")}
            """,
        ).fetchall()
        run_id = rows[0][0] if rows and rows[0] and rows[0][0] else None
    if not run_id:
        return {"has_data": False, "run_id": None, "rows": []}

    pref_cases = " ".join(
        f"WHEN model_name = '{m}' THEN {i}" for i, m in enumerate(_MODEL_PREFERENCE)
    )
    rows = _rows_from_cursor(
        _execute(
            con,
            f"""
            WITH preferred AS (
                SELECT fd.*, ROW_NUMBER() OVER (
                    PARTITION BY fd.question_id
                    ORDER BY CASE {pref_cases} ELSE 99 END
                ) AS rn
                FROM forecast_deviation fd
                JOIN questions q ON q.question_id = fd.question_id
                WHERE fd.run_id = ?
                  {_test_filter(include_test, "fd")}{_test_filter(include_test, "q")}
            ),
            ranked AS (
                SELECT p.*, ROW_NUMBER() OVER (
                    PARTITION BY p.iso3
                    ORDER BY COALESCE(p.js_vs_baserate, -1) DESC
                ) AS iso_rank
                FROM preferred p WHERE p.rn = 1
            )
            SELECT r.iso3,
                   r.js_vs_baserate,
                   r.hazard_code,
                   r.metric,
                   r.eiv_nominal,
                   agg.n_questions
            FROM ranked r
            JOIN (
                SELECT iso3, COUNT(*) AS n_questions
                FROM preferred WHERE rn = 1 GROUP BY iso3
            ) agg ON agg.iso3 = r.iso3
            WHERE r.iso_rank = 1
            ORDER BY r.iso3
            """,
            [run_id],
        )
    )
    for r in rows:
        js = r.get("js_vs_baserate")
        r["attention"] = (
            max(0.0, min(float(js) / _LN2, 1.0)) if js is not None else None
        )
    return {"has_data": bool(rows), "run_id": run_id, "rows": rows}


# The dynamic path is declared LAST so the literal routes above always win.
@router.get("/v1/interpreter/{interpretation_id}")
def interpreter_get(interpretation_id: str):
    """One report by id (any status — the frontend banners non-ok rows)."""
    con = _con()
    if not _table_exists(con, "interpretations"):
        raise HTTPException(status_code=404, detail="no interpretations in this DB")
    rows = _rows_from_cursor(
        _execute(
            con,
            "SELECT * FROM interpretations WHERE interpretation_id = ? "
            "ORDER BY version DESC LIMIT 1",
            [interpretation_id],
        )
    )
    if not rows:
        raise HTTPException(status_code=404, detail="interpretation not found")
    return {"interpretation": _shape_full(rows[0])}
