# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Risk-index routes: /v1/risk_index and /v1/rankings.

Endpoint functions moved verbatim from pythia.api.app (July 2026
decomposition); shared helpers come from pythia.api.core.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from pythia.api import core as _core
from pythia.api.core import (
    _con,
    _country_name,
    _execute,
    _latest_forecasted_target_month,
    _load_population_registry,
    _population,
    _resolve_forecasts_ensemble_columns,
    _rows_from_cursor,
    _run_filter_cte,
    _table_exists,
    _table_has_columns,
    _test_filter,
)
from pythia.buckets import BUCKET_SPECS
from resolver.query import eiv_sql

logger = logging.getLogger(__name__)

router = APIRouter()

def _get_risk_index_binary(
    con,
    metric_upper: str,
    hazard_code_upper: Optional[str],
    target_month: Optional[str],
    horizon_m: int,
    duration_m: int,
    normalize: bool,
    forecaster_run_id: Optional[str],
    include_test: bool = False,
) -> dict:
    """Risk index for binary EVENT_OCCURRENCE questions.

    The 'risk value' is the ensemble P(event) stored in bucket_1.
    No centroid multiplication is needed.
    """
    horizon_col, bucket_col, prob_col = _resolve_forecasts_ensemble_columns(con)
    if not horizon_col or not bucket_col or not prob_col:
        return {
            "metric": metric_upper, "target_month": target_month,
            "horizon_m": horizon_m, "duration_m": duration_m,
            "normalize": normalize, "rows": [], "metric_type": "binary",
        }

    if not target_month:
        target_month = _latest_forecasted_target_month(
            con, metric_upper, horizon_col, horizon_m,
            include_test=include_test,
            forecaster_run_id=forecaster_run_id,
        )
        if not target_month:
            _tf = _test_filter(include_test)
            _run_clause = ""
            _fallback_params: dict[str, Any] = {"metric": metric_upper}
            if forecaster_run_id:
                _run_clause = " AND question_id IN (SELECT question_id FROM forecasts_ensemble WHERE run_id = :run_id)"
                _fallback_params["run_id"] = forecaster_run_id
            row = _execute(
                con,
                f"SELECT MAX(target_month) FROM questions WHERE UPPER(metric)=:metric{_tf}{_run_clause}",
                _fallback_params,
            ).fetchone()
            target_month = row[0] if row and row[0] else None

    if not target_month:
        return {
            "metric": metric_upper, "target_month": None,
            "horizon_m": 6, "duration_m": duration_m,
            "normalize": normalize, "rows": [], "metric_type": "binary",
        }

    run_cte, run_join = _run_filter_cte(con, forecaster_run_id)
    run_cte_sql = f", {run_cte}" if run_cte else ""

    model_name_available = _table_has_columns(con, "forecasts_ensemble", ["model_name"])

    # For binary questions, bucket_1 = P(yes). We detect bucket_1 via
    # bucket_index=1 or class_bin label matching.
    bucket_is_label = bucket_col == "class_bin"
    if bucket_is_label:
        bucket_filter = f"AND fe.{bucket_col} IN ('1', 'bucket_1', 'yes')"
    else:
        bucket_filter = f"AND fe.{bucket_col} = 1"

    # Model selection CTE (prefer ensemble_bayesmc_v2 > ensemble_mean_v2)
    model_cte = ""
    from_alias = "fe"
    if model_name_available:
        model_cte = f"""
        , base_b AS (
          SELECT
            q.question_id, q.iso3, q.hazard_code,
            fe.{horizon_col} AS m,
            fe.{prob_col} AS p,
            COALESCE(fe.model_name, '') AS model_name
          FROM forecasts_ensemble fe
          JOIN q ON q.question_id = fe.question_id
          {run_join}
          WHERE fe.{horizon_col} BETWEEN 1 AND 6
            AND fe.{prob_col} IS NOT NULL
            {bucket_filter}
        ),
        chosen_model_b AS (
          SELECT question_id,
            CASE
              WHEN SUM(CASE WHEN model_name = 'ensemble_bayesmc_v2' THEN 1 ELSE 0 END) > 0
                THEN 'ensemble_bayesmc_v2'
              WHEN SUM(CASE WHEN model_name = 'ensemble_mean_v2' THEN 1 ELSE 0 END) > 0
                THEN 'ensemble_mean_v2'
              ELSE MIN(model_name)
            END AS chosen_model
          FROM base_b GROUP BY question_id
        ),
        binary_probs AS (
          SELECT base_b.iso3, base_b.hazard_code, base_b.m, base_b.p
          FROM base_b
          JOIN chosen_model_b USING (question_id)
          WHERE base_b.model_name = chosen_model_b.chosen_model
        )
        """
        from_alias = "binary_probs"
    else:
        model_cte = f"""
        , binary_probs AS (
          SELECT q.iso3, q.hazard_code,
                 fe.{horizon_col} AS m,
                 fe.{prob_col} AS p
          FROM forecasts_ensemble fe
          JOIN q ON q.question_id = fe.question_id
          {run_join}
          WHERE fe.{horizon_col} BETWEEN 1 AND 6
            AND fe.{prob_col} IS NOT NULL
            {bucket_filter}
        )
        """
        from_alias = "binary_probs"

    _tf = _test_filter(include_test)
    sql = f"""
    WITH q AS (
      SELECT question_id, iso3, hazard_code, metric, target_month
      FROM questions
      WHERE UPPER(metric) = :metric
        AND target_month = :target_month
        AND (:hazard_code IS NULL OR UPPER(hazard_code) = UPPER(:hazard_code))
        {_tf}
    )
    {run_cte_sql}
    {model_cte},
    monthly AS (
      SELECT iso3, m, MAX(p) AS prob
      FROM {from_alias}
      GROUP BY iso3, m
    ),
    hazards AS (
      SELECT iso3, COUNT(DISTINCT hazard_code) AS n_hazards_forecasted
      FROM q GROUP BY iso3
    ),
    pivoted AS (
      SELECT
        iso3,
        MAX(CASE WHEN m = 1 THEN prob END) AS m1,
        MAX(CASE WHEN m = 2 THEN prob END) AS m2,
        MAX(CASE WHEN m = 3 THEN prob END) AS m3,
        MAX(CASE WHEN m = 4 THEN prob END) AS m4,
        MAX(CASE WHEN m = 5 THEN prob END) AS m5,
        MAX(CASE WHEN m = 6 THEN prob END) AS m6,
        MAX(prob) AS total
      FROM monthly
      GROUP BY iso3
    )
    SELECT
      p.iso3,
      h.n_hazards_forecasted,
      p.m1, p.m2, p.m3, p.m4, p.m5, p.m6,
      p.total,
      NULL AS population,
      p.m1 AS m1_pc, p.m2 AS m2_pc, p.m3 AS m3_pc,
      p.m4 AS m4_pc, p.m5 AS m5_pc, p.m6 AS m6_pc,
      p.total AS total_pc
    FROM pivoted p
    LEFT JOIN hazards h ON h.iso3 = p.iso3
    ORDER BY p.total DESC NULLS LAST
    """
    params = {
        "metric": metric_upper,
        "target_month": target_month,
        "hazard_code": hazard_code_upper,
    }
    rows = _rows_from_cursor(_execute(con, sql, params))

    for row in rows:
        row["country_name"] = _country_name(row.get("iso3", ""))
        row["metric_type"] = "binary"
        # For binary, per-capita IS the raw probability (already a rate)
        if row.get("total") is not None:
            row["expected_value"] = row["total"]
            row["per_capita"] = row["total"]

    return {
        "metric": metric_upper,
        "target_month": target_month,
        "horizon_m": 6,
        "duration_m": duration_m,
        "normalize": normalize,
        "forecaster_run_id": forecaster_run_id,
        "rows": rows,
        "metric_type": "binary",
    }


@router.get("/v1/risk_index")
def get_risk_index(
    metric: str = Query("PA", description="Metric to rank on, e.g. 'PA'"),
    hazard_code: Optional[str] = Query(None, description="Optional hazard code filter"),
    target_month: Optional[str] = Query(None, description="Target month 'YYYY-MM'"),
    horizon_m: int = Query(1, ge=1, le=6, description="Forecast horizon in months ahead"),
    duration_m: int = Query(6, ge=1, le=12, description="Duration in months (metadata only)"),
    normalize: bool = Query(True, description="If true, include per-capita ranking"),
    agg: str = Query("surge", description="Aggregation mode: surge (default) or burden (legacy)"),
    alpha: float = Query(0.1, ge=0, le=1, description="Surge blending weight"),
    forecaster_run_id: Optional[str] = Query(None, description="Forecaster run ID to scope results"),
    model: Optional[str] = Query(
        None,
        description=(
            "Override the forecast source model_name (e.g. 'sibyl' for the "
            "parallel deep-research track); default = ensemble preference"
        ),
    ),
    include_test: bool = Query(False),
):
    con = _con()
    metric_upper = (metric or "").strip().upper() or "PA"
    model_override = (model or "").strip() or None
    hazard_code_upper = (hazard_code or "").strip().upper() or None
    is_pa = metric_upper == "PA"
    is_fatalities = metric_upper == "FATALITIES"
    is_binary = metric_upper == "EVENT_OCCURRENCE"
    is_phase3 = metric_upper == "PHASE3PLUS_IN_NEED"

    if is_binary:
        return _get_risk_index_binary(
            con, metric_upper, hazard_code_upper, target_month,
            horizon_m, duration_m, normalize, forecaster_run_id,
            include_test=include_test,
        )

    if not (is_pa or is_fatalities or is_phase3):
        return {
            "metric": metric_upper,
            "target_month": target_month,
            "horizon_m": horizon_m,
            "duration_m": duration_m,
            "normalize": normalize,
            "rows": [],
        }

    horizon_col, bucket_col, prob_col = _resolve_forecasts_ensemble_columns(con)
    if not horizon_col or not bucket_col or not prob_col:
        return {
            "metric": metric_upper,
            "target_month": target_month,
            "horizon_m": horizon_m,
            "duration_m": duration_m,
            "normalize": normalize,
            "rows": [],
        }

    if not target_month:
        target_month = _latest_forecasted_target_month(
            con, metric_upper, horizon_col, horizon_m,
            include_test=include_test,
            forecaster_run_id=forecaster_run_id,
        )
        if not target_month:
            _tf = _test_filter(include_test)
            _run_clause = ""
            _fallback_params: dict[str, Any] = {"metric": metric_upper}
            if forecaster_run_id:
                _run_clause = " AND question_id IN (SELECT question_id FROM forecasts_ensemble WHERE run_id = :run_id)"
                _fallback_params["run_id"] = forecaster_run_id
            row = _execute(
                con,
                f"SELECT MAX(target_month) FROM questions WHERE UPPER(metric)=:metric{_tf}{_run_clause}",
                _fallback_params,
            ).fetchone()
            target_month = row[0] if row and row[0] else None

    if not target_month:
        return {
            "metric": metric_upper,
            "target_month": None,
            "horizon_m": 6,
            "duration_m": duration_m,
            "normalize": normalize,
            "rows": [],
        }

    db_population_map: dict[str, int] = {}
    populations_available = False
    populations_table_available = _table_exists(con, "populations") and _table_has_columns(
        con, "populations", ["iso3", "population", "year"]
    )
    if populations_table_available:
        for pop_row in _rows_from_cursor(_execute(
            con,
            """
            SELECT iso3, population
            FROM (
              SELECT iso3, population,
                     ROW_NUMBER() OVER (PARTITION BY iso3 ORDER BY year DESC) AS rn
              FROM populations
              WHERE population IS NOT NULL AND population > 0
            )
            WHERE rn = 1
            """,
        )):
            iso = str(pop_row.get("iso3", "")).strip().upper()
            if not iso:
                continue
            try:
                pop_value = int(float(pop_row.get("population", 0)))
            except Exception:
                continue
            if pop_value <= 0:
                continue
            db_population_map[iso] = pop_value
        populations_available = bool(db_population_map)
    agg_mode = "surge" if (is_pa or is_phase3) else "burden"
    registry_available = False
    if (normalize or agg_mode == "surge") and not populations_available:
        _load_population_registry()
        # NOTE(code-motion): _POPULATION_BY_ISO3 is *rebound* by
        # _load_population_registry, so it must be read live through the
        # core module namespace rather than via an import-time alias.
        registry_available = bool(_core._POPULATION_BY_ISO3)
        if not registry_available:
            logger.debug("Population registry empty; per-capita values unavailable.")
    if agg is not None and str(agg).strip():
        requested_agg = str(agg).strip().lower()
        if requested_agg not in {"surge", "burden"}:
            raise HTTPException(status_code=400, detail="agg must be 'surge' or 'burden'")

    centroids_available = _table_exists(con, "bucket_centroids") and _table_has_columns(
        con, "bucket_centroids", ["metric", "hazard_code", "bucket_index", "centroid"]
    )

    pop_cte = ""
    pop_select = """
      , NULL AS population
      , NULL AS m1_pc
      , NULL AS m2_pc
      , NULL AS m3_pc
      , NULL AS m4_pc
      , NULL AS m5_pc
      , NULL AS m6_pc
      , NULL AS total_pc
    """
    pop_join = ""
    pop_join_monthly = ""
    if populations_available:
        pop_cte = """
        , pop AS (
          SELECT iso3, population
          FROM (
            SELECT iso3, population,
                   ROW_NUMBER() OVER (PARTITION BY iso3 ORDER BY year DESC) AS rn
            FROM populations
            WHERE population IS NOT NULL AND population > 0
          )
          WHERE rn = 1
        )
        """
        pop_select = """
          , pop.population
          , CASE
              WHEN :normalize AND pop.population IS NOT NULL AND pop.population != 0
                THEN p.m1 / pop.population
              ELSE NULL
            END AS m1_pc
          , CASE
              WHEN :normalize AND pop.population IS NOT NULL AND pop.population != 0
                THEN p.m2 / pop.population
              ELSE NULL
            END AS m2_pc
          , CASE
              WHEN :normalize AND pop.population IS NOT NULL AND pop.population != 0
                THEN p.m3 / pop.population
              ELSE NULL
            END AS m3_pc
          , CASE
              WHEN :normalize AND pop.population IS NOT NULL AND pop.population != 0
                THEN p.m4 / pop.population
              ELSE NULL
            END AS m4_pc
          , CASE
              WHEN :normalize AND pop.population IS NOT NULL AND pop.population != 0
                THEN p.m5 / pop.population
              ELSE NULL
            END AS m5_pc
          , CASE
              WHEN :normalize AND pop.population IS NOT NULL AND pop.population != 0
                THEN p.m6 / pop.population
              ELSE NULL
            END AS m6_pc
          , CASE
              WHEN :normalize AND pop.population IS NOT NULL AND pop.population != 0
                THEN p.total / pop.population
              ELSE NULL
            END AS total_pc
        """
        pop_join = "LEFT JOIN pop ON pop.iso3 = p.iso3"
        pop_join_monthly = "LEFT JOIN pop ON pop.iso3 = ms.iso3"

    bucket_is_label = bucket_col == "class_bin"
    centroid_join = ""
    centroid_expr = "NULL"
    if centroids_available:
        centroid_join, centroid_expr, _bucket_index_expr = eiv_sql.build_centroid_join(
            base_alias="fe",
            metric_expr=":metric",
            hazard_expr="fe.hazard_code",
            bucket_expr=f"fe.{bucket_col}",
            bucket_is_label=bucket_is_label,
        )
    else:
        bucket_index_expr = eiv_sql.bucket_index_expr(
            ":metric", f"fe.{bucket_col}", bucket_is_label=bucket_is_label
        )
        centroid_expr = eiv_sql.fallback_centroid_expr(":metric", bucket_index_expr)

    run_cte, run_join = _run_filter_cte(con, forecaster_run_id)
    run_cte_sql = f", {run_cte}" if run_cte else ""

    model_name_available = _table_has_columns(con, "forecasts_ensemble", ["model_name"])
    base_cte = ""
    per_row_cte = f"""
        , per_row AS (
          SELECT
            q.iso3,
            q.hazard_code,
            q.metric,
            fe.{horizon_col} AS m,
            fe.{bucket_col} AS b,
            fe.{prob_col} AS p,
            COALESCE({centroid_expr}, 0) AS centroid
          FROM forecasts_ensemble fe
          JOIN q ON q.question_id = fe.question_id
          {run_join}
          {centroid_join}
      WHERE fe.{horizon_col} BETWEEN 1 AND 6
        AND fe.{bucket_col} IS NOT NULL
        AND fe.{prob_col} IS NOT NULL
    )
    """
    if model_name_available:
        base_cte = f"""
        , base AS (
          SELECT
            q.question_id,
            q.iso3,
            q.hazard_code,
            q.metric,
            fe.{horizon_col} AS {horizon_col},
            fe.{bucket_col} AS {bucket_col},
            fe.{prob_col} AS {prob_col},
            COALESCE(fe.model_name, '') AS model_name
          FROM forecasts_ensemble fe
          JOIN q ON q.question_id = fe.question_id
          {run_join}
          WHERE fe.{horizon_col} BETWEEN 1 AND 6
            AND fe.{bucket_col} IS NOT NULL
            AND fe.{prob_col} IS NOT NULL
        )
        """
        if model_override:
            # Explicit source override (e.g. model=sibyl): only questions
            # that have rows under that model_name contribute.
            base_cte += """
        , filtered AS (
          SELECT base.*
          FROM base
          WHERE base.model_name = :model_override
        )
        """
        else:
            base_cte += """
        , chosen_model AS (
          SELECT
            question_id,
            CASE
              WHEN SUM(CASE WHEN model_name = 'ensemble_bayesmc_v2' THEN 1 ELSE 0 END) > 0
                THEN 'ensemble_bayesmc_v2'
              WHEN SUM(CASE WHEN model_name = 'ensemble_mean_v2' THEN 1 ELSE 0 END) > 0
                THEN 'ensemble_mean_v2'
              ELSE MIN(model_name)
            END AS chosen_model
          FROM base
          GROUP BY question_id
        ),
        filtered AS (
          SELECT base.*
          FROM base
          JOIN chosen_model USING (question_id)
          WHERE base.model_name = chosen_model.chosen_model
        )
        """
        per_row_cte = f"""
        , per_row AS (
          SELECT
            fe.iso3,
            fe.hazard_code,
            fe.metric,
            fe.{horizon_col} AS m,
            fe.{bucket_col} AS b,
            fe.{prob_col} AS p,
            COALESCE({centroid_expr}, 0) AS centroid
          FROM filtered fe
          {centroid_join}
        )
        """

    if is_pa:
        if populations_available:
            monthly_eiv_expr = """
              CASE
                WHEN pop.population IS NOT NULL
                  THEN LEAST(pop.population, ms.max_eiv + :alpha * (ms.sum_eiv - ms.max_eiv))
                ELSE ms.max_eiv + :alpha * (ms.sum_eiv - ms.max_eiv)
              END
            """
        else:
            monthly_eiv_expr = """
              ms.max_eiv + :alpha * (ms.sum_eiv - ms.max_eiv)
            """
    else:
        monthly_eiv_expr = "ms.sum_eiv"

    _tf = _test_filter(include_test)
    sql = f"""
    WITH q AS (
      SELECT question_id, iso3, hazard_code, metric, target_month
      FROM questions
      WHERE UPPER(metric) = :metric
        AND target_month = :target_month
        AND (:hazard_code IS NULL OR UPPER(hazard_code) = UPPER(:hazard_code))
        {_tf}
    )
    {run_cte_sql}
    {pop_cte}
    {base_cte}
    {per_row_cte},
    monthly_hazards AS (
      SELECT
        iso3,
        hazard_code,
        m,
        SUM(p * centroid) AS eiv
      FROM per_row
      GROUP BY iso3, hazard_code, m
    ),
    monthly_summary AS (
      SELECT
        iso3,
        m,
        SUM(eiv) AS sum_eiv,
        MAX(eiv) AS max_eiv
      FROM monthly_hazards
      GROUP BY iso3, m
    ),
    monthly AS (
      SELECT
        ms.iso3,
        ms.m,
        {monthly_eiv_expr} AS eiv
      FROM monthly_summary ms
      {pop_join_monthly}
    ),
    hazards AS (
      SELECT iso3, COUNT(DISTINCT hazard_code) AS n_hazards_forecasted
      FROM q
      GROUP BY iso3
    ),
    pivoted AS (
      SELECT
        iso3,
        SUM(CASE WHEN m = 1 THEN eiv ELSE 0 END) AS m1,
        SUM(CASE WHEN m = 2 THEN eiv ELSE 0 END) AS m2,
        SUM(CASE WHEN m = 3 THEN eiv ELSE 0 END) AS m3,
        SUM(CASE WHEN m = 4 THEN eiv ELSE 0 END) AS m4,
        SUM(CASE WHEN m = 5 THEN eiv ELSE 0 END) AS m5,
        SUM(CASE WHEN m = 6 THEN eiv ELSE 0 END) AS m6,
        CASE
          WHEN :agg = 'surge' THEN MAX(eiv)
          ELSE SUM(eiv)
        END AS total
      FROM monthly
      GROUP BY iso3
    )
    SELECT
      p.iso3,
      h.n_hazards_forecasted,
      p.m1, p.m2, p.m3, p.m4, p.m5, p.m6,
      p.total
      {pop_select}
    FROM pivoted p
    LEFT JOIN hazards h ON h.iso3 = p.iso3
    {pop_join}
    ORDER BY p.total DESC NULLS LAST
    """
    params = {
        "metric": metric_upper,
        "target_month": target_month,
        "hazard_code": hazard_code_upper,
        "agg": agg_mode,
        "alpha": alpha,
    }
    if model_name_available and model_override:
        params["model_override"] = model_override
    if populations_available:
        params["normalize"] = normalize
    rows = _rows_from_cursor(_execute(con, sql, params))
    for row in rows:
        row.setdefault("population", None)
        row.setdefault("m1_pc", None)
        row.setdefault("m2_pc", None)
        row.setdefault("m3_pc", None)
        row.setdefault("m4_pc", None)
        row.setdefault("m5_pc", None)
        row.setdefault("m6_pc", None)
        row.setdefault("total_pc", None)
        pop_value = row.get("population")
        if not isinstance(pop_value, (int, float)) or pop_value <= 0:
            pop_value = db_population_map.get(row.get("iso3", ""))
        if normalize and (not pop_value or pop_value <= 0):
            pop_value = _population(row.get("iso3", ""))
        if pop_value and pop_value > 0:
            row["population"] = pop_value
            if agg_mode == "surge":
                capped_months = []
                for key in ["m1", "m2", "m3", "m4", "m5", "m6"]:
                    value = row.get(key)
                    if value is None:
                        capped_months.append(None)
                        continue
                    capped_value = min(float(value), float(pop_value))
                    row[key] = capped_value
                    capped_months.append(capped_value)
                capped_values = [v for v in capped_months if v is not None]
                if capped_values:
                    row["total"] = max(capped_values)
            if normalize:
                row["m1_pc"] = (
                    row["m1"] / pop_value if row.get("m1") is not None else None
                )
                row["m2_pc"] = (
                    row["m2"] / pop_value if row.get("m2") is not None else None
                )
                row["m3_pc"] = (
                    row["m3"] / pop_value if row.get("m3") is not None else None
                )
                row["m4_pc"] = (
                    row["m4"] / pop_value if row.get("m4") is not None else None
                )
                row["m5_pc"] = (
                    row["m5"] / pop_value if row.get("m5") is not None else None
                )
                row["m6_pc"] = (
                    row["m6"] / pop_value if row.get("m6") is not None else None
                )
                row["total_pc"] = (
                    row["total"] / pop_value if row.get("total") is not None else None
                )

    # Back-compat: keep the v1 keys the frontend expects.
    for row in rows:
        # Prefer existing values if present, otherwise map from new names.
        if row.get("expected_value") is None:
            if row.get("total") is not None:
                row["expected_value"] = row["total"]
            elif row.get("eiv_total") is not None:
                row["expected_value"] = row["eiv_total"]

        if row.get("per_capita") is None:
            if row.get("total_pc") is not None:
                row["per_capita"] = row["total_pc"]
            elif row.get("eiv_total_pc") is not None:
                row["per_capita"] = row["eiv_total_pc"]

    for row in rows:
        row["country_name"] = _country_name(row.get("iso3", ""))
        row["metric_type"] = "spd"

    return {
        "metric": metric_upper,
        "target_month": target_month,
        "horizon_m": 6,
        "duration_m": duration_m,
        "normalize": normalize,
        "forecaster_run_id": forecaster_run_id,
        "model": model_override,
        "rows": rows,
        "metric_type": "spd",
    }


@router.get("/v1/rankings")
def rankings(
    month: str,
    metric: str = "PIN",
    normalize: bool = True,
    forecaster_run_id: Optional[str] = Query(None, description="Forecaster run ID to scope results"),
    include_test: bool = Query(False),
):
    con = _con()
    run_cte, run_join = _run_filter_cte(con, forecaster_run_id)
    run_cte_sql = f"{run_cte}," if run_cte else ""
    # Fallback centroid CASE derived from the canonical BUCKET_SPECS (labels
    # are code-controlled constants, not user input). PA specs cover the
    # legacy PIN metric alias.
    _fallback_specs = BUCKET_SPECS.get(
        (metric or "").upper(), BUCKET_SPECS["PA"]
    )
    centroid_case = " ".join(
        f"WHEN '{s.label}' THEN {float(s.centroid):g}" for s in _fallback_specs
    )
    sql = f"""
    WITH {run_cte_sql}
    ev AS (
      SELECT q.iso3, fe.horizon_m,
             SUM(
               fe.p * COALESCE(
                 bc.ev,
                 CASE fe.class_bin
                   {centroid_case}
                 END
               )
             ) AS ev_pin
      FROM forecasts_ensemble fe
      JOIN questions q ON q.question_id=fe.question_id
      {run_join}
      LEFT JOIN bucket_centroids bc
        ON bc.metric = q.metric
       AND bc.class_bin = fe.class_bin
      AND bc.hazard_code = q.hazard_code
      WHERE q.metric=? AND q.target_month=?{_test_filter(include_test, "fe")}
      GROUP BY 1,2
    ), pop AS (
      SELECT iso3, MAX_BY(population, year) AS population
      FROM populations GROUP BY 1
    )
    SELECT ev.iso3, ev.horizon_m,
           ev.ev_pin AS expected_value,
           CASE WHEN ? THEN ev.ev_pin/NULLIF(pop.population,0) ELSE NULL END AS per_capita
    FROM ev LEFT JOIN pop ON ev.iso3=pop.iso3
      ORDER BY (CASE WHEN ? THEN per_capita ELSE expected_value END) DESC
    """
    return {"rows": _rows_from_cursor(con.execute(sql, [metric, month, normalize, normalize]))}
