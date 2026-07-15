# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Forecast and resolution routes:
/v1/forecasts/ensemble, /v1/forecasts/history, /v1/resolutions.

Endpoint functions moved verbatim from pythia.api.app (July 2026
decomposition); shared helpers come from pythia.api.core.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Query

from pythia.api.core import (
    _con,
    _execute,
    _latest_questions_view,
    _rows_from_cursor,
    _table_exists,
    _test_filter,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Row cap for the unfiltered JSON endpoints: _rows_from_cursor materializes
# every row as a Python dict, so an unbounded pull of forecasts_ensemble can
# take hundreds of MB. The default is generous (well above current data
# volume); explicit ?limit= raises it to the hard max.
_DEFAULT_ROW_CAP = 200_000
_MAX_ROW_CAP = 500_000


def _capped_rows(con, sql: str, params, cap: int) -> dict:
    """Execute with LIMIT cap+1; truncate to cap and flag if exceeded."""
    sql += f" LIMIT {cap + 1}"
    rows = _rows_from_cursor(_execute(con, sql, params))
    if len(rows) > cap:
        return {"rows": rows[:cap], "truncated": True}
    return {"rows": rows}


@router.get("/v1/forecasts/ensemble")
def get_forecasts_ensemble(
    iso3: Optional[str] = Query(None),
    hazard_code: Optional[str] = Query(None),
    metric: Optional[str] = Query(None),
    target_month: Optional[str] = Query(None),
    horizon_m: Optional[int] = Query(None),
    latest_only: bool = Query(True),
    include_test: bool = Query(False),
    limit: Optional[int] = Query(None, ge=1, le=_MAX_ROW_CAP),
):
    con = _con()
    params = {}
    if iso3:
        params["iso3"] = iso3
    if hazard_code:
        params["hazard_code"] = hazard_code
    if metric:
        params["metric"] = metric
    if target_month:
        params["target_month"] = target_month
    if horizon_m is not None:
        params["horizon_m"] = horizon_m

    if latest_only:
        cte, _ = _latest_questions_view(
            con,
            iso3=iso3,
            hazard_code=hazard_code,
            metric=metric,
            target_month=target_month,
            status=None,
            include_test=include_test,
        )
        sql = cte + """
        SELECT
          fe.question_id,
          q.iso3,
          q.hazard_code,
          q.metric,
          q.target_month,
          fe.horizon_m,
          fe.class_bin,
          fe.p,
          fe.aggregator,
          fe.ensemble_version
        FROM forecasts_ensemble fe
        JOIN latest_q q ON fe.question_id = q.question_id
        """
        where_bits = []
        where_bits.append("q.rn = 1")
        if horizon_m is not None:
            where_bits.append("fe.horizon_m = :horizon_m")
        if where_bits:
            sql += " WHERE " + " AND ".join(where_bits)
        sql += " ORDER BY q.iso3, q.hazard_code, q.metric, q.target_month, fe.horizon_m, fe.class_bin"
        return _capped_rows(con, sql, params, limit or _DEFAULT_ROW_CAP)

    # latest_only=False: historical view (all runs)
    sql = f"""
      SELECT
        fe.question_id,
        q.iso3,
        q.hazard_code,
        q.metric,
        q.target_month,
        q.run_id,
        fe.horizon_m,
        fe.class_bin,
        fe.p,
        fe.aggregator,
        fe.ensemble_version
      FROM forecasts_ensemble fe
      JOIN questions q ON fe.question_id = q.question_id
      WHERE 1=1{_test_filter(include_test, "fe")}
    """
    if iso3:
        sql += " AND q.iso3 = :iso3"
    if hazard_code:
        sql += " AND q.hazard_code = :hazard_code"
    if metric:
        sql += " AND UPPER(q.metric) = UPPER(:metric)"
    if target_month:
        sql += " AND q.target_month = :target_month"
    if horizon_m is not None:
        sql += " AND fe.horizon_m = :horizon_m"

    sql += " ORDER BY q.target_month, q.iso3, q.hazard_code, q.metric, q.run_id, fe.horizon_m, fe.class_bin"
    return _capped_rows(con, sql, params, limit or _DEFAULT_ROW_CAP)


@router.get("/v1/forecasts/history")
def get_forecasts_history(
    iso3: str = Query(...),
    hazard_code: str = Query(...),
    metric: str = Query(...),
    target_month: str = Query(...),
    include_test: bool = Query(False),
    limit: Optional[int] = Query(None, ge=1, le=_MAX_ROW_CAP),
):
    """
    Return all historical ensemble forecasts for a given question concept
    (iso3, hazard_code, metric, target_month), grouped by HS run.

    Each row includes:
      - run_id
      - hs_run_created_at
      - horizon_m
      - class_bin
      - p
    """
    con = _con()
    params = {
        "iso3": iso3,
        "hazard_code": hazard_code,
        "metric": metric,
        "target_month": target_month,
    }

    sql = f"""
      SELECT
        q.run_id,
        h.created_at AS hs_run_created_at,
        fe.question_id,
        fe.horizon_m,
        fe.class_bin,
        fe.p
      FROM forecasts_ensemble fe
      JOIN questions q ON fe.question_id = q.question_id
      JOIN hs_runs h ON q.run_id = h.run_id
      WHERE q.iso3 = :iso3
        AND q.hazard_code = :hazard_code
        AND UPPER(q.metric) = UPPER(:metric)
        AND q.target_month = :target_month
        {_test_filter(include_test, "fe")}
      ORDER BY h.created_at, fe.horizon_m, fe.class_bin
    """
    return _capped_rows(con, sql, params, limit or _DEFAULT_ROW_CAP)


@router.get("/v1/resolutions")
def list_resolutions(iso3: str, month: str, metric: str = "PIN"):
    con = _con()
    if not _table_exists(con, "resolutions"):
        return {"rows": []}
    qsql = "SELECT question_id FROM questions WHERE iso3=? AND target_month=? AND metric=?"
    qids = [r[0] for r in con.execute(qsql, [iso3.upper(), month, metric]).fetchall()]
    if not qids:
        return {"rows": []}
    inlist = ",".join(["?"] * len(qids))
    return {"rows": _rows_from_cursor(con.execute(
        f"SELECT * FROM resolutions WHERE question_id IN ({inlist})",
        qids,
    ))}
