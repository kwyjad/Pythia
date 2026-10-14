# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""This run beside the previous two production runs.

Most operational regressions are invisible in absolute terms and obvious as
a delta. A batch fallback rate of 32% reads as a number; beside last
month's 0% it reads as a collapse. A CrisisWatch table holding two editions
reads as data; beside last month's five it reads as a source that stopped.

Only production runs are compared. A test run's cost and coverage are
shaped by whatever was being tested, so putting one in this table would
manufacture a regression out of a two-country smoke run.

Everything comes from the travelling database, which carries the previous
cycles' rows — so this needs no artifact, no API and no network.
"""

from __future__ import annotations

from typing import Any

METRICS = (
    "questions_generated",
    "countries_covered",
    "hs_triage_rows",
    "n_llm_calls",
    "n_llm_errors",
    "total_cost_usd",
    "total_tokens",
    "cost_forecast_usd",
    "cost_hs_usd",
    "cost_sibyl_usd",
    "n_batches",
    "batch_fallback_pct",
    "batch_tier_cost_pct",
    "batchable_cost_usd",
    "non_batchable_cost_usd",
    "cache_hit_rate_pct",
    "forecasts_raw_rows",
    "forecasts_ensemble_rows",
    "crisiswatch_editions",
    "crisiswatch_latest_edition",
)


def _columns(con, table: str) -> set[str]:
    try:
        # table_info yields (cid, name, ...) — the NAME is column 1, and reading
        # column 0 gives the ordinal, so every lookup silently missed.
        return {r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()}
    except Exception:
        return set()


def _scalar(con, sql: str, params: list[Any], default: Any = 0) -> Any:
    try:
        row = con.execute(sql, params).fetchone()
    except Exception:
        return default
    if not row or row[0] is None:
        return default
    return row[0]


def recent_production_hs_runs(con, *, current_hs_run_id: str | None, limit: int = 3) -> list[str]:
    """The newest ``limit`` production HS runs, current one first."""

    cols = _columns(con, "hs_runs")
    if not cols:
        return [current_hs_run_id] if current_hs_run_id else []
    test_filter = "WHERE COALESCE(is_test, FALSE) = FALSE" if "is_test" in cols else ""
    order_col = "generated_at" if "generated_at" in cols else "hs_run_id"
    try:
        rows = con.execute(
            f"SELECT hs_run_id FROM hs_runs {test_filter} ORDER BY {order_col} DESC LIMIT ?",
            [limit + 2],
        ).fetchall()
    except Exception:
        return [current_hs_run_id] if current_hs_run_id else []
    out: list[str] = []
    if current_hs_run_id:
        out.append(current_hs_run_id)
    for (rid,) in rows:
        rid = str(rid)
        if rid and rid not in out:
            out.append(rid)
        if len(out) >= limit:
            break
    return out


def sibyl_run_for(con, hs_run_id: str | None) -> str | None:
    """The newest Sibyl run of the HS run, or None."""
    if not hs_run_id or "sibyl_run_id" not in _columns(con, "sibyl_runs"):
        return None
    order = "created_at" if "created_at" in _columns(con, "sibyl_runs") else "sibyl_run_id"
    try:
        row = con.execute(
            f"SELECT sibyl_run_id FROM sibyl_runs WHERE hs_run_id = ? ORDER BY {order} DESC LIMIT 1",
            [hs_run_id],
        ).fetchone()
    except Exception:
        return None
    return str(row[0]) if row and row[0] else None


def run_llm_where(hs_run_id: str | None, fc_run_id: str | None,
                  sibyl_run_id: str | None = None) -> tuple[str, list[Any]]:
    """The one definition of "this run's LLM calls" (Oct 2026).

    The HS run, the forecaster run and the Sibyl run, by ``run_id``, and
    every row stamped with the HS run (Sibyl's steps carry it). The debug
    bundle's call tables and its cost total use the same clause, so the
    summary total and this table's ``total_cost_usd`` agree by construction.
    """
    run_ids = [i for i in (fc_run_id, sibyl_run_id, hs_run_id) if i]
    if not run_ids:
        return "FALSE", []
    clause = f"run_id IN ({','.join('?' for _ in run_ids)})"
    params: list[Any] = list(run_ids)
    if hs_run_id:
        clause = f"({clause} OR hs_run_id = ?)"
        params.append(hs_run_id)
    return clause, params


def batchable_sql(con) -> str:
    """Rows a batch family could have carried (scripts/ci/batch_economics.py)."""
    try:
        from scripts.ci.batch_economics import _batchable_sql  # noqa: PLC0415

        return _batchable_sql(con)
    except Exception:
        return "(phase IN ('spd_v2', 'binary_v2'))"


def _phase_bucket(phase: Any) -> str:
    try:
        from resolver.query.costs import phase_group  # noqa: PLC0415

        return phase_group(phase)
    except Exception:
        return str(phase or "other")


def _forecaster_run_for(con, hs_run_id: str) -> str | None:
    if not _columns(con, "forecasts_ensemble") or not _columns(con, "questions"):
        return None
    try:
        from pythia.run_questions import in_run_clause  # noqa: PLC0415

        clause, n = in_run_clause(con, "q")
        row = None
        if _columns(con, "run_questions"):
            row = con.execute(
                "SELECT MAX(forecaster_run_id) FROM run_questions "
                "WHERE hs_run_id = ? AND forecaster_run_id IS NOT NULL",
                [hs_run_id],
            ).fetchone()
        if not (row and row[0]):
            row = con.execute(
                f"""
                SELECT run_id FROM forecasts_ensemble
                WHERE question_id IN (SELECT q.question_id FROM questions q WHERE {clause})
                GROUP BY run_id ORDER BY COUNT(*) DESC LIMIT 1
                """,
                [hs_run_id] * n,
            ).fetchone()
    except Exception:
        return None
    return str(row[0]) if row and row[0] else None


def _metrics_for_run(con, hs_run_id: str) -> dict[str, Any]:
    fc_run_id = _forecaster_run_for(con, hs_run_id)
    sibyl_run_id = sibyl_run_for(con, hs_run_id)
    ids = [i for i in (hs_run_id, fc_run_id) if i]
    llm_where, llm_params = run_llm_where(hs_run_id, fc_run_id, sibyl_run_id)

    out: dict[str, Any] = {
        "hs_run_id": hs_run_id,
        "forecaster_run_id": fc_run_id or "",
        "sibyl_run_id": sibyl_run_id or "",
    }
    from pythia.run_questions import in_run_clause  # noqa: PLC0415

    clause, n = in_run_clause(con, "q")
    out["questions_generated"] = _scalar(
        con, f"SELECT COUNT(*) FROM questions q WHERE {clause}", [hs_run_id] * n
    )
    out["countries_covered"] = _scalar(
        con, "SELECT COUNT(DISTINCT iso3) FROM hs_triage WHERE run_id = ?", [hs_run_id]
    )
    out["hs_triage_rows"] = _scalar(
        con, "SELECT COUNT(*) FROM hs_triage WHERE run_id = ?", [hs_run_id]
    )
    out["n_llm_calls"] = _scalar(con, f"SELECT COUNT(*) FROM llm_calls WHERE {llm_where}", llm_params)
    out["n_llm_errors"] = _scalar(
        con,
        f"SELECT COUNT(*) FROM llm_calls WHERE {llm_where} AND error_text IS NOT NULL AND error_text <> ''",
        llm_params,
    )
    out["total_cost_usd"] = round(
        float(_scalar(con, f"SELECT SUM(COALESCE(cost_usd,0)) FROM llm_calls WHERE {llm_where}", llm_params, 0.0)),
        4,
    )
    out["total_tokens"] = int(
        _scalar(
            con,
            "SELECT SUM(COALESCE(TRY_CAST(json_extract_string(usage_json,'$.total_tokens') AS BIGINT),0)) "
            f"FROM llm_calls WHERE {llm_where}",
            llm_params,
            0,
        )
    )
    for label, phases in (
        ("cost_forecast_usd", ("spd_v2", "binary_v2", "scenario_v2")),
        ("cost_hs_usd", ("hs_triage",)),
        ("cost_sibyl_usd", ("sibyl",)),
    ):
        p_ph = ",".join("?" for _ in phases)
        out[label] = round(
            float(
                _scalar(
                    con,
                    f"SELECT SUM(COALESCE(cost_usd,0)) FROM llm_calls "
                    f"WHERE {llm_where} AND phase IN ({p_ph})",
                    llm_params + list(phases),
                    0.0,
                )
            ),
            4,
        )

    if _columns(con, "llm_batches"):
        ph2 = ",".join("?" for _ in ids)
        out["n_batches"] = _scalar(
            con,
            f"SELECT COUNT(*) FROM llm_batches WHERE hs_run_id IN ({ph2}) OR run_id IN ({ph2})",
            ids + ids,
        )
        n_requests = float(
            _scalar(
                con,
                f"SELECT SUM(COALESCE(n_requests,0)) FROM llm_batches "
                f"WHERE hs_run_id IN ({ph2}) OR run_id IN ({ph2})",
                ids + ids,
                0.0,
            )
        )
        n_fallback = float(
            _scalar(
                con,
                f"SELECT SUM(COALESCE(n_fallback_sync,0)) FROM llm_batches "
                f"WHERE hs_run_id IN ({ph2}) OR run_id IN ({ph2})",
                ids + ids,
                0.0,
            )
        )
        out["batch_fallback_pct"] = round(100.0 * n_fallback / n_requests, 1) if n_requests else 0.0
    else:
        out["n_batches"] = 0
        out["batch_fallback_pct"] = ""

    batch_cost = float(
        _scalar(
            con,
            "SELECT SUM(CASE WHEN json_extract_string(usage_json,'$.service_tier') = 'batch' "
            f"THEN COALESCE(cost_usd,0) ELSE 0 END) FROM llm_calls WHERE {llm_where}",
            llm_params,
            0.0,
        )
    )
    # Over the work a batch family could have carried only: grounding,
    # scenarios, adversarial synthesis and Sibyl are never batched, and
    # counting them made a fully batched run read as one third full price.
    bsql = batchable_sql(con)
    batchable = float(_scalar(
        con, f"SELECT SUM(COALESCE(cost_usd,0)) FROM llm_calls WHERE {llm_where} AND {bsql}",
        llm_params, 0.0))
    out["batchable_cost_usd"] = round(batchable, 4)
    out["non_batchable_cost_usd"] = round(float(out["total_cost_usd"] or 0.0) - batchable, 4)
    out["batch_tier_cost_pct"] = round(100.0 * batch_cost / batchable, 1) if batchable else 0.0

    # Per phase (the Costs page's buckets) and per model, as their own rows.
    by_phase: dict[str, float] = {}
    by_model: dict[str, float] = {}
    try:
        rows = con.execute(
            f"SELECT phase, COALESCE(model_id, model_name, 'unknown'), SUM(COALESCE(cost_usd,0)) "
            f"FROM llm_calls WHERE {llm_where} GROUP BY 1, 2",
            llm_params,
        ).fetchall()
    except Exception:
        rows = []
    for phase, model, cost in rows:
        key = _phase_bucket(phase)
        by_phase[key] = by_phase.get(key, 0.0) + float(cost or 0.0)
        by_model[str(model)] = by_model.get(str(model), 0.0) + float(cost or 0.0)
    out["_cost_by_phase"] = {k: round(v, 4) for k, v in by_phase.items()}
    out["_cost_by_model"] = {k: round(v, 4) for k, v in by_model.items()}

    prompt_tokens = float(
        _scalar(
            con,
            "SELECT SUM(COALESCE(TRY_CAST(json_extract_string(usage_json,'$.prompt_tokens') AS BIGINT),0)) "
            f"FROM llm_calls WHERE {llm_where}",
            llm_params,
            0.0,
        )
    )
    hit_tokens = float(
        _scalar(
            con,
            "SELECT SUM(COALESCE(TRY_CAST(json_extract_string(usage_json,'$.cache_read_input_tokens') AS BIGINT),0) "
            "+ COALESCE(TRY_CAST(json_extract_string(usage_json,'$.cached_tokens') AS BIGINT),0)) "
            f"FROM llm_calls WHERE {llm_where}",
            llm_params,
            0.0,
        )
    )
    out["cache_hit_rate_pct"] = round(100.0 * hit_tokens / prompt_tokens, 1) if prompt_tokens else 0.0

    if fc_run_id:
        out["forecasts_raw_rows"] = _scalar(
            con, "SELECT COUNT(*) FROM forecasts_raw WHERE run_id = ?", [fc_run_id]
        )
        out["forecasts_ensemble_rows"] = _scalar(
            con, "SELECT COUNT(*) FROM forecasts_ensemble WHERE run_id = ?", [fc_run_id]
        )
    else:
        out["forecasts_raw_rows"] = 0
        out["forecasts_ensemble_rows"] = 0

    # Connector freshness AS IT IS NOW, not as it was at that run: the DB
    # keeps no per-run snapshot of it. For CrisisWatch, which is versioned
    # by edition rather than overwritten, the count is still meaningful.
    if _columns(con, "crisiswatch_entries"):
        out["crisiswatch_editions"] = _scalar(
            con, "SELECT COUNT(DISTINCT (year * 100 + COALESCE(month,0))) FROM crisiswatch_entries", []
        )
        latest = _scalar(
            con, "SELECT MAX(year * 100 + COALESCE(month,0)) FROM crisiswatch_entries", [], 0
        )
        out["crisiswatch_latest_edition"] = (
            f"{int(latest)//100}-{int(latest)%100:02d}" if latest else ""
        )
    else:
        out["crisiswatch_editions"] = 0
        out["crisiswatch_latest_edition"] = ""
    return out


def collect(con, *, current_hs_run_id: str | None, n_runs: int = 3) -> list[dict[str, Any]]:
    """One row per metric, one column per run. Never raises."""

    runs = recent_production_hs_runs(con, current_hs_run_id=current_hs_run_id, limit=n_runs)
    if not runs:
        return []
    per_run = []
    for rid in runs:
        try:
            per_run.append(_metrics_for_run(con, rid))
        except Exception as exc:  # noqa: BLE001
            per_run.append({"hs_run_id": rid, "error": str(exc)})

    rows: list[dict[str, Any]] = []
    header = {"metric": "forecaster_run_id"}
    for idx, run in enumerate(per_run):
        header[f"run_{idx}__{run['hs_run_id']}"] = run.get("forecaster_run_id", "")
    rows.append(header)
    sib = {"metric": "sibyl_run_id"}
    for idx, run in enumerate(per_run):
        sib[f"run_{idx}__{run['hs_run_id']}"] = run.get("sibyl_run_id", "")
    rows.append(sib)
    for metric in METRICS:
        row: dict[str, Any] = {"metric": metric}
        for idx, run in enumerate(per_run):
            row[f"run_{idx}__{run['hs_run_id']}"] = run.get(metric, "")
        rows.append(row)
    for prefix, key in (("cost_phase:", "_cost_by_phase"), ("cost_model:", "_cost_by_model")):
        names = sorted({n for run in per_run for n in (run.get(key) or {})})
        for name in names:
            row = {"metric": prefix + name}
            for idx, run in enumerate(per_run):
                row[f"run_{idx}__{run['hs_run_id']}"] = (run.get(key) or {}).get(name, 0.0)
            rows.append(row)
    return rows
