from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

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
        SELECT month_index, bucket_index, probability, ev_value
        FROM forecasts_ensemble
        WHERE run_id = ? AND question_id = ?
        ORDER BY month_index, bucket_index
        """,
        [run_id, question_id],
    ).fetchall()

    spd: Dict[int, List[float]] = {}
    ev: Dict[int, float] = {}
    for month_idx, bucket_idx, prob, ev_val in rows:
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

        lines: List[str] = []
        lines.append(f"# Pythia Forecast Run Debug (run_id={run_id})")
        lines.append("")

        for (hz, metric), question_id in sorted(mapping.items()):
            qmeta = _load_question_meta(con, question_id)
            ensemble_spd, ensemble_ev = _load_ensemble_spd(con, run_id, question_id)
            model_spd = _load_model_spd_and_usage(con, run_id, question_id)
            llm_calls = _load_llm_calls(con, run_id, question_id)

            iso3 = qmeta.get("iso3", "")
            wording = qmeta.get("wording", "")
            window = f"{qmeta.get('window_start_date','')} → {qmeta.get('window_end_date','')}"

            lines.append(f"## Type: {hz}/{metric} — question_id `{question_id}`")
            lines.append("")
            lines.append("### Question")
            lines.append("")
            lines.append(f"- ISO3: `{iso3}`")
            lines.append(f"- Hazard: `{hz}`")
            lines.append(f"- Metric: `{metric}`")
            lines.append(f"- Window: `{window}`")
            lines.append("")
            if wording:
                lines.append("**Wording:**")
                lines.append("")
                lines.append(str(wording))
                lines.append("")

            lines.append("### Ensemble SPD & EV")
            lines.append("")
            lines.append("| Month | B1 p | B2 p | B3 p | B4 p | B5 p | EV |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for m in sorted(ensemble_spd.keys()):
                probs = ensemble_spd[m]
                ev_val = ensemble_ev.get(m)
                row = [str(m)] + [f"{p:.3f}" for p in probs] + [f"{ev_val:,.0f}" if ev_val is not None else ""]
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

            lines.append("### Per-model SPD & LLM metadata")
            lines.append("")
            for model_name in sorted(model_spd.keys()):
                entry = model_spd[model_name]
                spd = entry["spd"]
                llm = llm_calls.get(model_name, {})
                lines.append(f"#### Model: {model_name}")
                lines.append("")
                lines.append(
                    f"- ok={int(entry['ok'])}, elapsed_ms={entry['elapsed_ms']}, "
                    f"tokens={entry['total_tokens']}, cost=${entry['cost_usd']:.6f}"
                )
                if llm:
                    lines.append(f"- provider={llm.get('provider')} model_id={llm.get('model_id')}")
                if llm and llm.get("error_text"):
                    lines.append(f"- error: {llm['error_text']}")
                lines.append("")
                lines.append("| Month | B1 p | B2 p | B3 p | B4 p | B5 p |")
                lines.append("|---|---:|---:|---:|---:|---:|")
                for m in sorted(spd.keys()):
                    probs = spd[m]
                    row = [str(m)] + [f"{p:.3f}" for p in probs]
                    lines.append("| " + " | ".join(row) + " |")
                lines.append("")
                if llm and llm.get("response_text"):
                    raw = (llm["response_text"] or "").strip()
                    lines.append("**Full model response:**")
                    lines.append("")
                    lines.append("```md")
                    lines.append(raw)
                    lines.append("```")
                    lines.append("")
            lines.append("---")
            lines.append("")

        OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
        print(f"[info] Wrote {OUT_PATH}")
    finally:
        duckdb_io.close_db(con)


if __name__ == "__main__":
    main()
