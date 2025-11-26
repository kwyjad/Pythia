from fastapi import FastAPI, Depends, Body, Query
import duckdb, pandas as pd
from pythia.api.auth import require_token
from pythia.config import load as load_cfg
from pythia.pipeline.run import enqueue_run

app = FastAPI(title="Pythia API", version="1.0.0")


def _con():
    db_url = load_cfg()["app"]["db_url"].replace("duckdb:///", "")
    return duckdb.connect(db_url, read_only=True)


@app.get("/v1/health")
def health():
    return {"ok": True}


@app.post("/v1/run")
def start_run(payload: dict = Body(...), _=Depends(require_token)):
    countries = payload.get("countries") or []
    run_id = enqueue_run(countries)
    return {"accepted": True, "run_id": run_id}


@app.get("/v1/questions")
def list_questions(
    iso3: str | None = Query(None),
    month: str | None = Query(None),
    hazard_code: str | None = Query(None),
    _=Depends(require_token),
):
    con = _con()
    sql = "SELECT * FROM questions WHERE 1=1"
    params = []
    if iso3:
        sql += " AND iso3=?"
        params.append(iso3.upper())
    if month:
        sql += " AND target_month=?"
        params.append(month)
    if hazard_code:
        sql += " AND hazard_code=?"
        params.append(hazard_code.upper())
    df = con.execute(sql, params).fetchdf()
    return {"rows": df.to_dict(orient="records")}


@app.get("/v1/forecasts/ensemble")
def list_ensemble(iso3: str, target_month: str, metric: str = "PIN", _=Depends(require_token)):
    con = _con()
    qsql = "SELECT question_id FROM questions WHERE iso3=? AND target_month=? AND metric=?"
    qids = [r[0] for r in con.execute(qsql, [iso3.upper(), target_month, metric]).fetchall()]
    if not qids:
        return {"rows": []}
    inlist = ",".join(["?"] * len(qids))
    df = con.execute(
        f"SELECT * FROM forecasts_ensemble WHERE question_id IN ({inlist})",
        qids,
    ).fetchdf()
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
