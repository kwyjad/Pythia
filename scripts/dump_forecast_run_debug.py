from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

from forecaster.cli import SPD_CLASS_BINS_FATALITIES, SPD_CLASS_BINS_PA
from forecaster.ensemble import (
    SPD_BUCKET_CENTROIDS_DEFAULT,
    SPD_BUCKET_CENTROIDS_FATALITIES,
    SPD_BUCKET_CENTROIDS_PA,
    _load_bucket_centroids_db,
)
from resolver.db import duckdb_io

DB_URL = os.environ.get("RESOLVER_DB_URL", duckdb_io.DEFAULT_DB_URL)
OUT_PATH = Path("forecast_run_debug.md")


def _connect():
    try:
        return duckdb_io.get_db(DB_URL)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"[error] Failed to open DuckDB at {DB_URL}: {type(exc).__name__}: {exc}")


def _get_latest_run_id(con) -> str | None:
    row = con.execute(
        """
        SELECT run_id, MAX(created_at) AS t
        FROM forecasts_ensemble
        GROUP BY run_id
        ORDER BY t DESC
        LIMIT 1
        """
    ).fetchone()
    return row[0] if row else None


def _get_question_types_for_run(con, run_id: str) -> Dict[Tuple[str, str], str]:
    rows = con.execute(
        """
        SELECT q.hazard_code, q.metric, q.question_id
        FROM forecasts_ensemble fe
        JOIN questions q ON fe.question_id = q.question_id
        WHERE fe.run_id = ?
        GROUP BY q.hazard_code, q.metric, q.question_id
        ORDER BY q.hazard_code, q.metric, q.question_id
        """,
        [run_id],
    ).fetchall()

    mapping: Dict[Tuple[str, str], str] = {}
    for hz, metric, qid in rows:
        key = ((hz or "").upper(), (metric or "").upper())
        if key not in mapping:
            mapping[key] = str(qid)
    return mapping


def _load_question_meta(con, question_id: str) -> dict:
    row = con.execute(
        """
        SELECT question_id, hs_run_id, iso3, hazard_code, metric,
               target_month, window_start_date, window_end_date, wording, pythia_metadata_json
        FROM questions
        WHERE question_id = ?
        """,
        [question_id],
    ).fetchone()
    if not row:
        return {}
    cols = [
        "question_id",
        "hs_run_id",
        "iso3",
        "hazard_code",
        "metric",
        "target_month",
        "window_start_date",
        "window_end_date",
        "wording",
        "pythia_metadata_json",
    ]
    return dict(zip(cols, row))


def _load_ensemble_spd(con, run_id: str, question_id: str):
    rows = con.execute(
        """
        SELECT month_idx, bucket_index, probability, ev_value
        FROM forecasts_ensemble
        WHERE run_id = ?
          AND question_id = ?
          AND status = 'ok'
          AND month_idx IS NOT NULL
        ORDER BY month_idx, bucket_index
        """,
        [run_id, question_id],
    ).fetchall()

    spd: Dict[int, List[float]] = {}
    ev: Dict[int, float] = {}
    for month_idx, bucket_idx, prob, ev_val in rows:
        if month_idx is None:
            continue
        m = int(month_idx)
        b = int(bucket_idx)
        probs = spd.setdefault(m, [0.0] * 5)
        if 1 <= b <= 5:
            probs[b - 1] = float(prob or 0.0)
        if ev_val is not None and m not in ev:
            ev[m] = float(ev_val)
    return spd, ev


def _load_model_spd_and_usage(con, run_id: str, question_id: str):
    rows = con.execute(
        """
        SELECT model_name, month_index, bucket_index, probability,
               ok, elapsed_ms, cost_usd, prompt_tokens, completion_tokens, total_tokens
        FROM forecasts_raw
        WHERE run_id = ? AND question_id = ?
        ORDER BY model_name, month_index, bucket_index
        """,
        [run_id, question_id],
    ).fetchall()

    data: Dict[str, dict] = {}
    for (
        model_name,
        month_idx,
        bucket_idx,
        prob,
        ok,
        elapsed_ms,
        cost_usd,
        prompt_tokens,
        completion_tokens,
        total_tokens,
    ) in rows:
        name = str(model_name)
        m = int(month_idx)
        b = int(bucket_idx)
        entry = data.setdefault(
            name,
            {
                "spd": {},
                "ok": bool(ok),
                "elapsed_ms": int(elapsed_ms or 0),
                "cost_usd": float(cost_usd or 0.0),
                "prompt_tokens": int(prompt_tokens or 0),
                "completion_tokens": int(completion_tokens or 0),
                "total_tokens": int(total_tokens or 0),
            },
        )
        spd = entry["spd"].setdefault(m, [0.0] * 5)
        if 1 <= b <= 5:
            spd[b - 1] = float(prob or 0.0)
    return data


def _load_llm_calls(con, run_id: str, question_id: str):
    rows = con.execute(
        """
        SELECT model_name, provider, model_id,
               prompt_text, response_text,
               elapsed_ms, prompt_tokens, completion_tokens, total_tokens, cost_usd, error_text
        FROM llm_calls
        WHERE run_id = ?
          AND question_id = ?
          AND call_type = 'forecast'
        ORDER BY timestamp
        """,
        [run_id, question_id],
    ).fetchall()

    calls: Dict[str, dict] = {}
    for (
        model_name,
        provider,
        model_id,
        prompt_text,
        response_text,
        elapsed_ms,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        cost_usd,
        error_text,
    ) in rows:
        name = str(model_name)
        calls[name] = {
            "provider": provider,
            "model_id": model_id,
            "prompt_text": prompt_text,
            "response_text": response_text,
            "elapsed_ms": int(elapsed_ms or 0),
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "total_tokens": int(total_tokens or 0),
            "cost_usd": float(cost_usd or 0.0),
            "error_text": error_text,
        }
    return calls


def _resolve_spd_bucket_config(hazard_code: str, metric: str) -> tuple[list[str], list[float], str]:
    metric_up = (metric or "").upper()
    labels = SPD_CLASS_BINS_FATALITIES if metric_up == "FATALITIES" else SPD_CLASS_BINS_PA

    if metric_up == "PA":
        default_centroids = SPD_BUCKET_CENTROIDS_PA
        default_source_label = "default_pa"
    elif metric_up == "FATALITIES":
        default_centroids = SPD_BUCKET_CENTROIDS_FATALITIES
        default_source_label = "default_fatalities"
    else:
        default_centroids = SPD_BUCKET_CENTROIDS_DEFAULT
        default_source_label = "default_generic"

    bucket_centroids_db = _load_bucket_centroids_db(hazard_code or "", metric_up, labels)
    if bucket_centroids_db is not None:
        return list(labels), list(bucket_centroids_db), "db"

    return list(labels), list(default_centroids), default_source_label


def main():
    con = _connect()

    try:
        run_id = os.environ.get("PYTHIA_FORECAST_RUN_ID") or _get_latest_run_id(con)
        if not run_id:
            print("[warn] No forecasts_ensemble entries found; nothing to dump.")
            return

        mapping = _get_question_types_for_run(con, run_id)
        if not mapping:
            print(f"[warn] No questions found for run_id={run_id}.")
            return
    finally:
        duckdb_io.close_db(con)

    lines: List[str] = []
    lines.append(f"# Pythia Forecast Run Debug (run_id={run_id})")
    lines.append("")

    for (hz, metric), question_id in sorted(mapping.items()):
        con = _connect()
        try:
            qmeta = _load_question_meta(con, question_id)
            ensemble_spd, ensemble_ev = _load_ensemble_spd(con, run_id, question_id)
            model_spd = _load_model_spd_and_usage(con, run_id, question_id)
            llm_calls = _load_llm_calls(con, run_id, question_id)

            iso3 = qmeta.get("iso3", "")
            wording = qmeta.get("wording", "")
            window = f"{qmeta.get('window_start_date','')} → {qmeta.get('window_end_date','')}"

            question_lines: List[str] = []

            question_lines.append(f"## Type: {hz}/{metric} — question_id `{question_id}`")
            question_lines.append("")
            question_lines.append("### Question")
            question_lines.append("")
            question_lines.append(f"- ISO3: `{iso3}`")
            question_lines.append(f"- Hazard: `{hz}`")
            question_lines.append(f"- Metric: `{metric}`")
            question_lines.append(f"- Window: `{window}`")
            question_lines.append("")
            if wording:
                question_lines.append("**Wording:**")
                question_lines.append("")
                question_lines.append(str(wording))
                question_lines.append("")

            if not ensemble_spd:
                nf_rows = con.execute(
                    """
                    SELECT COUNT(*) FROM forecasts_ensemble
                    WHERE run_id = ? AND question_id = ? AND status = 'no_forecast'
                    """,
                    [run_id, question_id],
                ).fetchone()
                if nf_rows and nf_rows[0] > 0:
                    question_lines.append("### Ensemble SPD & EV")
                    question_lines.append("")
                    question_lines.append("_No forecast generated for this question (status = `no_forecast`)._")
                    question_lines.append("")
                else:
                    continue
            else:
                question_lines.append("### Ensemble SPD & EV")
                question_lines.append("")
                question_lines.append("| Month | B1 p | B2 p | B3 p | B4 p | B5 p | EV |")
                question_lines.append("|---|---:|---:|---:|---:|---:|---:|")
                for m in sorted(ensemble_spd.keys()):
                    probs = ensemble_spd[m]
                    ev_val = ensemble_ev.get(m)
                    row = [str(m)] + [f"{p:.3f}" for p in probs] + [f"{ev_val:,.0f}" if ev_val is not None else ""]
                    question_lines.append("| " + " | ".join(row) + " |")
                question_lines.append("")

            labels, centroids, centroid_source = _resolve_spd_bucket_config(hz or "", metric or "")
            question_lines.append("**SPD bucket configuration**")
            if labels:
                labels_str = ", ".join(str(x) for x in labels)
                question_lines.append(f"- Bucket labels: [{labels_str}]")
            if centroids:
                centroids_str = ", ".join(f"{float(x):g}" for x in centroids)
                question_lines.append(f"- Centroids used (for EV): [{centroids_str}]")
            if centroid_source:
                question_lines.append(f"- Centroid source: `{centroid_source}`")
            question_lines.append("")

            question_lines.append("### Per-model SPD & LLM metadata")
            question_lines.append("")
            for model_name in sorted(model_spd.keys()):
                entry = model_spd[model_name]
                spd = entry["spd"]
                llm = llm_calls.get(model_name, {})
                question_lines.append(f"#### Model: {model_name}")
                question_lines.append("")
                question_lines.append(
                    f"- ok={int(entry['ok'])}, elapsed_ms={entry['elapsed_ms']}, "
                    f"tokens={entry['total_tokens']}, cost=${entry['cost_usd']:.6f}"
                )
                if llm:
                    question_lines.append(f"- provider={llm.get('provider')} model_id={llm.get('model_id')}")
                if llm and llm.get("error_text"):
                    question_lines.append(f"- error: {llm['error_text']}")
                question_lines.append("")
                question_lines.append("| Month | B1 p | B2 p | B3 p | B4 p | B5 p |")
                question_lines.append("|---|---:|---:|---:|---:|---:|")
                for m in sorted(spd.keys()):
                    probs = spd[m]
                    row = [str(m)] + [f"{p:.3f}" for p in probs]
                    question_lines.append("| " + " | ".join(row) + " |")
                question_lines.append("")
                if llm and llm.get("response_text"):
                    raw = (llm["response_text"] or "").strip()
                    question_lines.append("**Full model response:**")
                    question_lines.append("")
                    question_lines.append("```md")
                    question_lines.append(raw)
                    question_lines.append("```")
                    question_lines.append("")
            question_lines.append("---")
            question_lines.append("")

            lines.extend(question_lines)
        finally:
            duckdb_io.close_db(con)

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[info] Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
