from __future__ import annotations

import json
import logging
from datetime import date

import duckdb
import pandas as pd

from resolver.db.duckdb_io import upsert_dataframe
from pythia.utils.ids import scenario_id as sid, question_id as qid

ALLOWED = set(["FL", "DR", "TC", "HW", "ACO", "ACE", "DI", "CU", "EC", "PHE"])


def _to_target_month(today: date, months_ahead: int) -> str:
    y, m = today.year, today.month + months_ahead
    y += (m - 1) // 12
    m = ((m - 1) % 12) + 1
    return f"{y:04d}-{m:02d}"


def upsert_hs_payload(
    db_url: str,
    run_meta: dict,
    scenarios: list[dict],
    *,
    today: date,
    horizon_months: int,
):
    con = duckdb.connect(db_url.replace("duckdb:///", ""))
    # audit
    upsert_dataframe(con, "hs_runs", pd.DataFrame([run_meta]), keys=["run_id"])

    scenario_rows, question_rows = [], []
    for sc in scenarios:
        hz = sc.get("hazard_code", "").strip().upper()
        if hz not in ALLOWED:
            continue
        iso3 = sc["iso3"].upper()
        s_id = sid(iso3, hz, sc.get("title", ""), sc.get("json", {}))
        target_month = _to_target_month(today, horizon_months)

        scenario_rows.append(
            {
                "scenario_id": s_id,
                "run_id": run_meta["run_id"],
                "iso3": iso3,
                "country_name": sc.get("country_name", ""),
                "hazard_code": hz,
                "hazard_label": sc.get("hazard_label", ""),
                "likely_window_month": sc.get("likely_window_month", ""),
                "markdown": sc.get("markdown", ""),
                "json": sc.get("json", {}),
            }
        )

        best = sc.get("best_guess") or {}

        metric = "PA"
        wording = (
            f"How many people will be affected in {iso3} due to {hz} by {target_month}?"
        )
        q_id = qid(iso3, hz, metric, target_month, wording)
        question_rows.append(
            {
                "question_id": q_id,
                "scenario_id": s_id,
                "run_id": run_meta["run_id"],
                "iso3": iso3,
                "country_name": sc.get("country_name", ""),
                "hazard_code": hz,
                "hazard_label": sc.get("hazard_label", ""),
                "metric": metric,
                "target_month": target_month,
                "wording": wording,
                "best_guess_value": float(best.get(metric) or 0),
                "hs_json": {"source": "HS", "raw": sc.get("json", {})},
                "status": "active",
            }
        )

    # --- Normalize JSON payloads for DuckDB and add diagnostics ---

    # Log counts before upsert
    logging.info(
        "Horizon Scanner DB payload prepared: %d scenario_rows, %d question_rows for run_id=%s",
        len(scenario_rows),
        len(question_rows),
        run_meta["run_id"],
    )

    # Normalize scenario JSON to valid JSON strings (if not already)
    for row in scenario_rows:
        raw = row.get("json", {})
        if isinstance(raw, str):
            # Assume caller already provided a JSON string
            continue
        try:
            row["json"] = json.dumps(raw, ensure_ascii=False)
        except TypeError:
            logging.warning(
                "Failed to JSON-serialize scenario json for scenario_id=%s; falling back to empty object.",
                row.get("scenario_id"),
            )
            row["json"] = "{}"

    # Normalize question hs_json to valid JSON strings (if not already)
    for row in question_rows:
        raw = row.get("hs_json", {})
        if isinstance(raw, str):
            # Assume already JSON; if it's a Python repr, DuckDB will complain and our logs will show it.
            continue
        try:
            row["hs_json"] = json.dumps(raw, ensure_ascii=False)
        except TypeError:
            logging.warning(
                "Failed to JSON-serialize hs_json for question_id=%s; falling back to empty object.",
                row.get("question_id"),
            )
            row["hs_json"] = "{}"

    # Extra diagnostics: show one example question payload after normalization
    if question_rows:
        sample = question_rows[0]
        logging.info(
            "Example question row before upsert | question_id=%s | iso3=%s | hazard=%s | metric=%s | hs_json_preview=%r",
            sample.get("question_id"),
            sample.get("iso3"),
            sample.get("hazard_code"),
            sample.get("metric"),
            (str(sample.get("hs_json"))[:200] + "...")
            if sample.get("hs_json") is not None
            else None,
        )

    # --- Upsert into DuckDB ---

    if scenario_rows:
        upsert_dataframe(
            con,
            "hs_scenarios",
            pd.DataFrame(scenario_rows),
            keys=["scenario_id"],
        )
    if question_rows:
        upsert_dataframe(
            con,
            "questions",
            pd.DataFrame(question_rows),
            keys=["question_id"],
        )

    con.close()
