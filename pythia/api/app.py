from fastapi import FastAPI, Depends, Body, Query
from typing import Optional
import duckdb, pandas as pd
from pythia.api.auth import require_token
from pythia.config import load as load_cfg
from pythia.pipeline.run import enqueue_run

app = FastAPI(title="Pythia API", version="1.0.0")


def _con():
    db_url = load_cfg()["app"]["db_url"].replace("duckdb:///", "")
    return duckdb.connect(db_url, read_only=True)


def _latest_questions_view(
    iso3: Optional[str] = None,
    hazard_code: Optional[str] = None,
    metric: Optional[str] = None,
    target_month: Optional[str] = None,
    status: Optional[str] = None,
) -> str:
    """
    Returns a SQL string for a 'latest questions' CTE called latest_q, parameterised
    by filters. The idea:

      - Identify question concepts: (iso3, hazard_code, metric, target_month)
      - For each concept, pick the question with the latest hs_runs.created_at
        (i.e. latest HS run).
      - Join questions q with hs_runs h to get run timestamps.

    NOTE: This helper builds only the CTE string; you still need to bind the same
    filter parameters to the main query.
    """
    # We build filters into both the inner and outer query for simplicity
    where_bits = []
    if iso3:
        where_bits.append("q.iso3 = :iso3")
    if hazard_code:
        where_bits.append("q.hazard_code = :hazard_code")
    if metric:
        where_bits.append("UPPER(q.metric) = UPPER(:metric)")
    if target_month:
        where_bits.append("q.target_month = :target_month")
    if status:
        where_bits.append("q.status = :status")

    where_clause = ""
    if where_bits:
        where_clause = "WHERE " + " AND ".join(where_bits)

    cte = f"""
    WITH latest_q AS (
      SELECT q.*
      FROM questions q
      JOIN hs_runs h ON q.run_id = h.run_id
      JOIN (
        SELECT
          iso3,
          hazard_code,
          metric,
          target_month,
          MAX(h.created_at) AS latest_run
        FROM questions q
        JOIN hs_runs h ON q.run_id = h.run_id
        {where_clause}
        GROUP BY 1,2,3,4
      ) x
      ON q.iso3 = x.iso3
     AND q.hazard_code = x.hazard_code
     AND q.metric = x.metric
     AND q.target_month = x.target_month
     AND h.created_at = x.latest_run
      {where_clause}
    )
    """
    return cte


@app.get("/v1/health")
def health():
    return {"ok": True}


@app.post("/v1/run")
def start_run(payload: dict = Body(...), _=Depends(require_token)):
    countries = payload.get("countries") or []
    run_id = enqueue_run(countries)
    return {"accepted": True, "run_id": run_id}


@app.get("/v1/ui_runs/{ui_run_id}")
def get_ui_run(ui_run_id: str, _=Depends(require_token)):
    """
    Return status for a given ui_run_id created by /v1/run.

    Response shape:
      - found: bool
      - row: dict | None (full ui_runs row if found)
    """
    con = _con()
    df = con.execute(
        "SELECT * FROM ui_runs WHERE ui_run_id = ?",
        [ui_run_id],
    ).fetchdf()
    if df.empty:
        return {"found": False, "row": None}
    row = df.to_dict(orient="records")[0]
    return {"found": True, "row": row}


@app.get("/v1/questions")
def get_questions(
    iso3: Optional[str] = Query(None),
    hazard_code: Optional[str] = Query(None),
    metric: Optional[str] = Query(None),
    target_month: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    run_id: Optional[str] = Query(None),
    latest_only: bool = Query(False),
    _=Depends(require_token),
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
    if status:
        params["status"] = status
    if run_id:
        params["run_id"] = run_id

    if not latest_only:
        where_bits = []
        if iso3:
            where_bits.append("iso3 = :iso3")
        if hazard_code:
            where_bits.append("hazard_code = :hazard_code")
        if metric:
            where_bits.append("UPPER(metric) = UPPER(:metric)")
        if target_month:
            where_bits.append("target_month = :target_month")
        if status:
            where_bits.append("status = :status")
        if run_id:
            where_bits.append("run_id = :run_id")

        sql = "SELECT * FROM questions"
        if where_bits:
            sql += " WHERE " + " AND ".join(where_bits)
        sql += " ORDER BY target_month, iso3, hazard_code, metric, run_id"
        df = con.execute(sql, params).fetchdf()
        return {"rows": df.to_dict(orient="records")}

    # latest_only=True: one row per concept (iso3, hazard, metric, target_month) from latest run
    cte = _latest_questions_view(
        iso3=iso3,
        hazard_code=hazard_code,
        metric=metric,
        target_month=target_month,
        status=status,
    )
    sql = cte + """
    SELECT *
    FROM latest_q
    """
    if run_id:
        sql += " WHERE run_id = :run_id"
    sql += " ORDER BY target_month, iso3, hazard_code, metric"
    df = con.execute(sql, params).fetchdf()
    return {"rows": df.to_dict(orient="records")}


@app.get("/v1/calibration/weights")
def get_calibration_weights(
    hazard_code: Optional[str] = Query(None),
    metric: Optional[str] = Query(None),
    as_of_month: Optional[str] = Query(None, description="YYYY-MM; if omitted, use latest"),
    _=Depends(require_token),
):
    """
    Return calibration weights per model for the given hazard_code/metric/as_of_month.

    If as_of_month is omitted, we use the latest as_of_month in calibration_weights
    (optionally filtered by hazard_code/metric).
    """
    con = _con()

    # Resolve as_of_month if not given
    params: dict = {}
    where_bits = []

    if hazard_code:
        where_bits.append("hazard_code = :hazard_code")
        params["hazard_code"] = hazard_code.upper()
    if metric:
        where_bits.append("metric = :metric")
        params["metric"] = metric.upper()

    base_where = ""
    if where_bits:
        base_where = "WHERE " + " AND ".join(where_bits)

    if not as_of_month:
        # Pick latest as_of_month given any hazard/metric filters
        sql_latest = f"""
          SELECT as_of_month
          FROM calibration_weights
          {base_where}
          ORDER BY as_of_month DESC
          LIMIT 1
        """
        row = con.execute(sql_latest, params).fetchone()
        if not row:
            return {"found": False, "as_of_month": None, "rows": []}
        as_of_month = row[0]

    # Now fetch rows for this as_of_month + filters
    params["as_of_month"] = as_of_month
    where_full = ["as_of_month = :as_of_month"]
    if hazard_code:
        where_full.append("hazard_code = :hazard_code")
    if metric:
        where_full.append("metric = :metric")

    sql = """
      SELECT
        as_of_month,
        hazard_code,
        metric,
        model_name,
        weight,
        n_questions,
        n_samples,
        avg_brier,
        avg_log,
        avg_crps,
        created_at
      FROM calibration_weights
    """
    sql += " WHERE " + " AND ".join(where_full)
    sql += " ORDER BY hazard_code, metric, model_name"

    df = con.execute(sql, params).fetchdf()

    if df.empty:
        return {"found": False, "as_of_month": as_of_month, "rows": []}

    # We return rows, plus the resolved as_of_month for convenience
    return {
        "found": True,
        "as_of_month": as_of_month,
        "rows": df.to_dict(orient="records"),
    }


@app.get("/v1/calibration/advice")
def get_calibration_advice(
    hazard_code: Optional[str] = Query(None),
    metric: Optional[str] = Query(None),
    as_of_month: Optional[str] = Query(None, description="YYYY-MM; if omitted, use latest"),
    _=Depends(require_token),
):
    """
    Return calibration advice text per (hazard_code, metric, as_of_month).

    - If hazard_code/metric are omitted, returns advice for all rows at the chosen as_of_month.
    - If as_of_month is omitted, uses the latest as_of_month present in calibration_advice
      (optionally filtered by hazard_code/metric).
    """
    con = _con()

    params: dict = {}
    where_bits = []

    if hazard_code:
        where_bits.append("hazard_code = :hazard_code")
        params["hazard_code"] = hazard_code.upper()
    if metric:
        where_bits.append("metric = :metric")
        params["metric"] = metric.upper()

    base_where = ""
    if where_bits:
        base_where = "WHERE " + " AND ".join(where_bits)

    if not as_of_month:
        sql_latest = f"""
          SELECT as_of_month
          FROM calibration_advice
          {base_where}
          ORDER BY as_of_month DESC
          LIMIT 1
        """
        row = con.execute(sql_latest, params).fetchone()
        if not row:
            return {"found": False, "as_of_month": None, "rows": []}
        as_of_month = row[0]

    params["as_of_month"] = as_of_month
    where_full = ["as_of_month = :as_of_month"]
    if hazard_code:
        where_full.append("hazard_code = :hazard_code")
    if metric:
        where_full.append("metric = :metric")

    sql = """
      SELECT
        as_of_month,
        hazard_code,
        metric,
        advice,
        created_at
      FROM calibration_advice
    """
    sql += " WHERE " + " AND ".join(where_full)
    sql += " ORDER BY hazard_code, metric"

    df = con.execute(sql, params).fetchdf()

    if df.empty:
        return {"found": False, "as_of_month": as_of_month, "rows": []}

    return {
        "found": True,
        "as_of_month": as_of_month,
        "rows": df.to_dict(orient="records"),
    }


@app.get("/v1/forecasts/ensemble")
def get_forecasts_ensemble(
    iso3: Optional[str] = Query(None),
    hazard_code: Optional[str] = Query(None),
    metric: Optional[str] = Query(None),
    target_month: Optional[str] = Query(None),
    horizon_m: Optional[int] = Query(None),
    latest_only: bool = Query(True),
    _=Depends(require_token),
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
        cte = _latest_questions_view(
            iso3=iso3,
            hazard_code=hazard_code,
            metric=metric,
            target_month=target_month,
            status=None,
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
        if horizon_m is not None:
            where_bits.append("fe.horizon_m = :horizon_m")
        if where_bits:
            sql += " WHERE " + " AND ".join(where_bits)
        sql += " ORDER BY q.iso3, q.hazard_code, q.metric, q.target_month, fe.horizon_m, fe.class_bin"
        df = con.execute(sql, params).fetchdf()
        return {"rows": df.to_dict(orient="records")}

    # latest_only=False: historical view (all runs)
    sql = """
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
      WHERE 1=1
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
    df = con.execute(sql, params).fetchdf()
    return {"rows": df.to_dict(orient="records")}


@app.get("/v1/forecasts/history")
def get_forecasts_history(
    iso3: str = Query(...),
    hazard_code: str = Query(...),
    metric: str = Query(...),
    target_month: str = Query(...),
    _=Depends(require_token),
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

    sql = """
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
      ORDER BY h.created_at, fe.horizon_m, fe.class_bin
    """
    df = con.execute(sql, params).fetchdf()
    return {"rows": df.to_dict(orient="records")}


@app.get("/v1/resolutions")
def list_resolutions(iso3: str, month: str, metric: str = "PIN", _=Depends(require_token)):
    con = _con()
    qsql = "SELECT question_id FROM questions WHERE iso3=? AND target_month=? AND metric=?"
    qids = [r[0] for r in con.execute(qsql, [iso3.upper(), month, metric]).fetchall()]
    if not qids:
        return {"rows": []}
    inlist = ",".join(["?"] * len(qids))
    df = con.execute(
        f"SELECT * FROM resolutions WHERE question_id IN ({inlist})",
        qids,
    ).fetchdf()
    return {"rows": df.to_dict(orient="records")}


@app.get("/v1/risk_index")
def get_risk_index(
    metric: str = Query("PA", description="Metric to rank on, e.g. 'PA'"),
    target_month: str = Query(..., description="Target month 'YYYY-MM'"),
    horizon_m: int = Query(1, ge=1, le=6, description="Forecast horizon in months ahead"),
    normalize: bool = Query(True, description="If true, include per-capita ranking"),
    _=Depends(require_token),
):
    """
    Country-level risk index for a given metric/target_month/horizon.

    For each country (iso3), this sums expected value (centroid-based) across all
    questions with the given metric and target_month at the specified horizon_m.
    It then optionally normalises by population.

    Returns:
      - iso3
      - expected_value (EV of metric, summed over hazards)
      - per_capita (EV / population) if normalize=true
    """
    con = _con()
    params = {
        "metric": metric.upper(),
        "target_month": target_month,
        "horizon_m": horizon_m,
        "normalize": normalize,
    }

    sql = """
    WITH ev AS (
      SELECT q.iso3, fe.horizon_m,
             SUM(
               fe.p * COALESCE(
                 bc.ev,
                 CASE fe.class_bin
                   WHEN '<10k' THEN 5000
                   WHEN '10k-<50k' THEN 25000
                   WHEN '50k-<250k' THEN 120000
                   WHEN '250k-<500k' THEN 350000
                   WHEN '>=500k' THEN 700000
                 END
               )
             ) AS ev_value
      FROM forecasts_ensemble fe
      JOIN questions q ON q.question_id = fe.question_id
      LEFT JOIN bucket_centroids bc
        ON bc.metric = q.metric
       AND bc.class_bin = fe.class_bin
       AND bc.hazard_code = q.hazard_code
      WHERE UPPER(q.metric) = :metric
        AND q.target_month = :target_month
        AND fe.horizon_m = :horizon_m
      GROUP BY 1,2
    ), pop AS (
      SELECT iso3, MAX_BY(population, year) AS population
      FROM populations GROUP BY 1
    )
    SELECT ev.iso3, ev.horizon_m,
           ev.ev_value AS expected_value,
           CASE WHEN :normalize THEN ev.ev_value/NULLIF(pop.population,0) ELSE NULL END AS per_capita
    FROM ev LEFT JOIN pop ON ev.iso3 = pop.iso3
    ORDER BY (CASE WHEN :normalize THEN per_capita ELSE expected_value END) DESC
    """

    df = con.execute(sql, params).fetchdf()
    return {
        "metric": metric.upper(),
        "target_month": target_month,
        "horizon_m": horizon_m,
        "normalize": normalize,
        "rows": df.to_dict(orient="records"),
    }


@app.get("/v1/rankings")
def rankings(month: str, metric: str = "PIN", normalize: bool = True, _=Depends(require_token)):
    con = _con()
    sql = """
    WITH ev AS (
      SELECT q.iso3, fe.horizon_m,
             SUM(
               fe.p * COALESCE(
                 bc.ev,
                 CASE fe.class_bin
                   WHEN '<10k' THEN 5000
                   WHEN '10k-<50k' THEN 25000
                   WHEN '50k-<250k' THEN 120000
                   WHEN '250k-<500k' THEN 350000
                   WHEN '>=500k' THEN 700000
                 END
               )
             ) AS ev_pin
      FROM forecasts_ensemble fe
      JOIN questions q ON q.question_id=fe.question_id
      LEFT JOIN bucket_centroids bc
        ON bc.metric = q.metric
       AND bc.class_bin = fe.class_bin
      AND bc.hazard_code = q.hazard_code
      WHERE q.metric=? AND q.target_month=?
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
    df = con.execute(sql, [metric, month, normalize, normalize]).fetchdf()
    return {"rows": df.to_dict(orient="records")}


@app.get("/v1/diagnostics/summary")
def diagnostics_summary(_=Depends(require_token)):
    """
    Return a high-level summary of Pythia's state:

      - question counts by status
      - number of questions with forecasts (ensemble)
      - number of questions with resolutions
      - number of questions with scores
      - latest HS run (hs_runs)
      - latest calibration as_of_month (calibration_weights)
    """
    con = _con()

    q_counts = con.execute(
        "SELECT status, COUNT(*) AS n FROM questions GROUP BY status"
    ).fetchdf().to_dict(orient="records")

    q_with_forecast = con.execute(
        "SELECT COUNT(DISTINCT question_id) AS n FROM forecasts_ensemble"
    ).fetchone()[0]

    q_with_resolutions = con.execute(
        "SELECT COUNT(DISTINCT question_id) AS n FROM resolutions"
    ).fetchone()[0]

    q_with_scores = con.execute(
        "SELECT COUNT(DISTINCT question_id) AS n FROM scores"
    ).fetchone()[0]

    hs_row = con.execute(
        "SELECT run_id, created_at, meta FROM hs_runs ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if hs_row:
        latest_hs = {
            "run_id": hs_row[0],
            "created_at": hs_row[1],
            "meta": hs_row[2],
        }
    else:
        latest_hs = None

    cal_row = con.execute(
        "SELECT as_of_month, MAX(created_at) FROM calibration_weights GROUP BY as_of_month ORDER BY as_of_month DESC LIMIT 1"
    ).fetchone()
    if cal_row:
        latest_calibration = {
            "as_of_month": cal_row[0],
            "created_at": cal_row[1],
        }
    else:
        latest_calibration = None

    return {
        "questions_by_status": q_counts,
        "questions_with_forecasts": int(q_with_forecast),
        "questions_with_resolutions": int(q_with_resolutions),
        "questions_with_scores": int(q_with_scores),
        "latest_hs_run": latest_hs,
        "latest_calibration": latest_calibration,
    }


@app.get("/v1/llm/costs")
def llm_costs(
    component: str | None = Query(None),
    model: str | None = Query(None),
    since: str | None = Query(None),
    limit: int = Query(200, ge=1, le=5000),
    _=Depends(require_token),
):
    """
    Return recent LLM call cost/usage rows from llm_calls.

    Optional filters:
      - component: "HS" | "Researcher" | "Forecaster" | etc.
      - model: model_name (exact match)
      - since: ISO timestamp (created_at >= since)
    """
    con = _con()
    sql = "SELECT * FROM llm_calls WHERE 1=1"
    params: list = []

    if component:
        sql += " AND component = ?"
        params.append(component)
    if model:
        sql += " AND model_name = ?"
        params.append(model)
    if since:
        sql += " AND created_at >= ?"
        params.append(since)

    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    df = con.execute(sql, params).fetchdf()
    return {"rows": df.to_dict(orient="records")}
