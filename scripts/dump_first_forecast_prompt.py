from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
from resolver.db import duckdb_io

try:
    # Prefer shared config if available
    from pythia.db.schema import get_db_url
except ImportError:  # pragma: no cover - optional dependency
    get_db_url = None  # type: ignore


OUT_PATH = Path("debug_first_forecast_prompt.md")


def _resolve_db_url() -> str:
    if get_db_url is not None:
        try:
            return get_db_url()
        except Exception:
            pass
    return os.getenv("PYTHIA_DB_URL") or duckdb_io.DEFAULT_DB_URL


def _connect_duckdb():
    db_url = _resolve_db_url()
    try:
        con = duckdb_io.get_db(db_url)
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        raise SystemExit(f"[error] Failed to connect to DuckDB at {db_url!r}: {exc}") from exc
    return con


def _get_latest_forecast_run_id(con) -> Optional[str]:
    """Return the most recent run_id for which we have at least one forecast call."""

    row = con.execute(
        """
        SELECT run_id
        FROM llm_calls
        WHERE call_type = 'forecast'
          AND run_id IS NOT NULL
          AND run_id <> ''
        ORDER BY timestamp DESC
        LIMIT 1
        """
    ).fetchone()
    return row[0] if row else None


def _get_question_types_for_run(con, run_id: str) -> Dict[Tuple[str, str], str]:
    """Map (hazard_code, metric) to one representative question_id for the run."""

    rows = con.execute(
        """
        SELECT DISTINCT
            q.hazard_code,
            q.metric,
            q.question_id
        FROM forecasts_ensemble fe
        JOIN questions q
          ON fe.question_id = q.question_id
        WHERE fe.run_id = ?
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


def _get_first_forecast_call_for_question(
    con, run_id: str, question_id: str
) -> Optional[Dict[str, Any]]:
    """Return the first forecast call (chronologically) for a given question in a run."""

    cols = [
        "call_id",
        "question_id",
        "model_name",
        "provider",
        "model_id",
        "prompt_text",
        "response_text",
        "elapsed_ms",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost_usd",
        "timestamp",
    ]
    row = con.execute(
        """
        SELECT call_id,
               question_id,
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
               timestamp
        FROM llm_calls
        WHERE run_id = ?
          AND question_id = ?
          AND call_type = 'forecast'
        ORDER BY timestamp
        LIMIT 1
        """,
        [run_id, question_id],
    ).fetchone()

    if not row:
        return None

    return {c: v for c, v in zip(cols, row)}


def _get_research_calls_for_question(
    con, run_id: str, question_id: str
) -> List[Dict[str, Any]]:
    cols = [
        "call_id",
        "model_name",
        "provider",
        "model_id",
        "prompt_text",
        "response_text",
        "elapsed_ms",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost_usd",
        "timestamp",
    ]
    rows = con.execute(
        """
        SELECT call_id,
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
               timestamp
        FROM llm_calls
        WHERE run_id = ?
          AND question_id = ?
          AND call_type = 'research'
        ORDER BY timestamp
        """,
        [run_id, question_id],
    ).fetchall()
    return [dict(zip(cols, r)) for r in rows]


def _get_question_metadata(con, question_id: str) -> Dict[str, Any]:
    row = con.execute(
        """
        SELECT
            question_id,
            hs_run_id,
            iso3,
            hazard_code,
            metric,
            target_month,
            window_start_date,
            window_end_date,
            wording,
            pythia_metadata_json
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
    q = dict(zip(cols, row))
    try:
        q["pythia_metadata"] = json.loads(q.get("pythia_metadata_json") or "{}")
    except Exception:
        q["pythia_metadata"] = {}
    return q


def _get_question_context(con, run_id: str, question_id: str) -> Dict[str, Any]:
    row = con.execute(
        """
        SELECT iso3,
               hazard_code,
               metric,
               snapshot_start_month,
               snapshot_end_month,
               pa_history_json,
               context_json
        FROM question_context
        WHERE run_id = ?
          AND question_id = ?
        """,
        [run_id, question_id],
    ).fetchone()

    if not row:
        return {}

    cols = [
        "iso3",
        "hazard_code",
        "metric",
        "snapshot_start_month",
        "snapshot_end_month",
        "pa_history_json",
        "context_json",
    ]
    d = dict(zip(cols, row))

    try:
        d["pa_history"] = json.loads(d.get("pa_history_json") or "[]")
    except Exception:
        d["pa_history"] = []

    try:
        d["context"] = json.loads(d.get("context_json") or "{}")
    except Exception:
        d["context"] = {}

    return d


def _load_spd_from_forecasts_ensemble(
    con, run_id: str, question_id: str
) -> Tuple[Dict[int, List[float]], Dict[int, float]]:
    """
    Load the ensemble SPD and EV per month from forecasts_ensemble.

    Returns:
      ensemble_probs: {month_index: [p1..p5]}
      ensemble_ev: {month_index: ev_value}
    """

    rows = con.execute(
        """
        SELECT month_index, bucket_index, probability, ev_value
        FROM forecasts_ensemble
        WHERE run_id = ? AND question_id = ?
        ORDER BY month_index, bucket_index
        """,
        [run_id, question_id],
    ).fetchall()

    ensemble_probs: Dict[int, List[float]] = {}
    ensemble_ev: Dict[int, float] = {}

    for month_idx, bucket_idx, p, ev_val in rows:
        m = int(month_idx)
        b = int(bucket_idx)
        probs = ensemble_probs.setdefault(m, [0.0] * 5)
        if 1 <= b <= 5:
            probs[b - 1] = float(p or 0.0)

        if ev_val is not None and m not in ensemble_ev:
            ensemble_ev[m] = float(ev_val)

    return ensemble_probs, ensemble_ev


def _load_bucket_centroids(con, hazard_code: str, metric: str) -> List[float]:
    """
    Load bucket centroids for this hazard/metric from bucket_centroids.

    Returns a list [c1..c5], or [] if none found.
    """

    hz = (hazard_code or "").upper()
    m = (metric or "").upper()

    try:
        rows = con.execute(
            """
            SELECT bucket_index, centroid
            FROM bucket_centroids
            WHERE upper(hazard_code) = ?
              AND upper(metric) = ?
            ORDER BY bucket_index
            """,
            [hz, m],
        ).fetchall()
    except duckdb.CatalogException:
        return []
    except Exception:
        return []

    if not rows:
        return []

    centroids = [0.0] * 5
    for bucket_idx, centroid in rows:
        b = int(bucket_idx)
        if 1 <= b <= 5:
            centroids[b - 1] = float(centroid or 0.0)
    return centroids


def _fmt_tokens_and_cost(row: Dict[str, Any]) -> str:
    pt = int(row.get("prompt_tokens") or 0)
    ct = int(row.get("completion_tokens") or 0)
    tt = int(row.get("total_tokens") or 0)
    cost = float(row.get("cost_usd") or 0.0)
    ms = int(row.get("elapsed_ms") or 0)
    return f"tokens: prompt={pt}, completion={ct}, total={tt}; cost=${cost:.6f}; time={ms}ms"


def _truncate(s: str, max_len: int = 2000) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"\n\n[... truncated, {len(s) - max_len} chars omitted ...]\n"


def _write_placeholder(message: str) -> None:
    OUT_PATH.write_text(
        "\n".join(["# Pythia Forecast Prompt Debug", "", message]), encoding="utf-8"
    )
    print(message)
    print(f"[info] Wrote placeholder markdown to {OUT_PATH}")


def main() -> None:
    con = None
    try:
        con = _connect_duckdb()
    except SystemExit as e:
        _write_placeholder(str(e))
        return

    try:
        run_id = _get_latest_forecast_run_id(con)
        if not run_id:
            _write_placeholder("[warn] No forecast calls found in llm_calls; nothing to dump.")
            return
        print(f"[info] Using latest forecast run_id={run_id}")

        qtype_map = _get_question_types_for_run(con, run_id)
        if not qtype_map:
            _write_placeholder(f"[warn] No forecast entries linked to questions for run_id={run_id}.")
            return

        lines: List[str] = []

        lines.append("# Pythia Forecast Prompts Debug (by question type)")
        lines.append("")
        lines.append(f"- **run_id:** `{run_id}`")
        type_labels = [f"{hz}/{metric}" for hz, metric in sorted(qtype_map.keys())]
        lines.append(f"- **question types covered:** {', '.join(type_labels)}")
        lines.append("")

        for hz_metric in sorted(qtype_map.keys()):
            hz, metric = hz_metric
            question_id = qtype_map[hz_metric]
            lines.append(f"## Type: {hz}/{metric} — question_id `{question_id}`")
            lines.append("")

            fc = _get_first_forecast_call_for_question(con, run_id, question_id)
            research_calls = _get_research_calls_for_question(con, run_id, question_id)
            qmeta = _get_question_metadata(con, question_id)
            qctx = _get_question_context(con, run_id, question_id)
            ensemble_probs, ensemble_ev = _load_spd_from_forecasts_ensemble(
                con, run_id, question_id
            )
            centroids = _load_bucket_centroids(con, qmeta.get("hazard_code"), qmeta.get("metric"))

            if not fc:
                lines.append(f"_No forecast calls found for question `{question_id}` in this run._")
                lines.append("")
                continue

            lines.append("### Run / Question")
            lines.append("")
            lines.append(f"- **run_id:** `{run_id}`")
            lines.append(f"- **question_id:** `{question_id}`")
            lines.append(
                f"- **model:** `{fc['provider']}/{fc['model_id']}` (`{fc['model_name']}`)"
            )

            if qmeta:
                lines.append(f"- **iso3:** `{(qmeta.get('iso3') or '').upper()}`")
                lines.append(f"- **hazard_code:** `{qmeta.get('hazard_code') or ''}`")
                lines.append(f"- **metric:** `{qmeta.get('metric') or ''}`")
                lines.append(f"- **target_month:** `{qmeta.get('target_month') or ''}`")
                ws = qmeta.get("window_start_date") or ""
                we = qmeta.get("window_end_date") or ""
                if ws or we:
                    lines.append(f"- **window:** `{ws}` → `{we}`")
                if qmeta.get("wording"):
                    lines.append("")
                    lines.append("#### Question wording")
                    lines.append("")
                    lines.append(str(qmeta["wording"]))

            lines.append("")
            lines.append("### Research Calls")
            lines.append("")

            if research_calls:
                for idx, rc in enumerate(research_calls, start=1):
                    lines.append(f"#### Research call {idx}")
                    lines.append("")
                    lines.append(
                        f"- **model:** `{rc['provider']}/{rc['model_id']}` (`{rc['model_name']}`)"
                    )
                    lines.append(f"- **usage:** {_fmt_tokens_and_cost(rc)}")
                    lines.append("")
                    lines.append("##### Research prompt (truncated)")
                    lines.append("")
                    lines.append("```text")
                    lines.append(_truncate(rc.get("prompt_text") or "", 2000))
                    lines.append("```")
                    lines.append("")
                    lines.append("##### Research response (truncated)")
                    lines.append("")
                    lines.append("```text")
                    lines.append(_truncate(rc.get("response_text") or "", 2000))
                    lines.append("```")
                    lines.append("")
            else:
                lines.append("_No research calls recorded for this question/run._")
                lines.append("")

            lines.append("### Resolver 36-month Snapshot")
            lines.append("")

            if qctx:
                lines.append(
                    f"- **snapshot:** `{qctx.get('snapshot_start_month','')}` → "
                    f"`{qctx.get('snapshot_end_month','')}`"
                )
                history = qctx.get("pa_history") or []
                if history:
                    lines.append("")
                    lines.append("| Month | PA | Source |")
                    lines.append("|---|---:|---|")
                    for h in sorted(history, key=lambda item: str(item.get("ym") or "")):
                        ym = h.get("ym") or ""
                        val = h.get("value") or ""
                        source = h.get("source") or ""
                        lines.append(f"| {ym} | {val} | {source} |")
                    lines.append("")
                else:
                    lines.append("_No PA history found._")
                    lines.append("")

                ctx_extra = qctx.get("context") or {}
                fatality_history = ctx_extra.get("fatalities_history") if isinstance(ctx_extra, dict) else []
                if fatality_history:
                    lines.append("#### Fatalities history (Resolver)")
                    lines.append("")
                    lines.append("| Month | Fatalities | Source |")
                    lines.append("|---|---:|---|")
                    for h in sorted(fatality_history, key=lambda item: str(item.get("ym") or "")):
                        ym = h.get("ym") or ""
                        val = h.get("value") or ""
                        source = h.get("source") or ""
                        lines.append(f"| {ym} | {val} | {source} |")
                    lines.append("")

                if ctx_extra:
                    lines.append("#### Resolver context JSON")
                    lines.append("")
                    lines.append("```json")
                    try:
                        lines.append(json.dumps(ctx_extra, indent=2, ensure_ascii=False))
                    except Exception:
                        lines.append(str(ctx_extra))
                    lines.append("```")
                lines.append("")
            else:
                lines.append("_No Resolver context found for this question/run._")
                lines.append("")

            lines.append("## SPD Ensemble (from forecasts_ensemble)")
            lines.append("")

            if ensemble_probs:
                header = "| Month | B1 p | B2 p | B3 p | B4 p | B5 p | EV (db) |"
                if centroids:
                    header += " EV (spd×centroids) |"
                lines.append(header)
                if centroids:
                    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
                else:
                    lines.append("|---|---:|---:|---:|---:|---:|---:|")

                for month_idx in sorted(ensemble_probs.keys()):
                    probs = ensemble_probs[month_idx]
                    ev_db = ensemble_ev.get(month_idx)
                    row = [f"{month_idx}"]
                    for p in probs:
                        row.append(f"{p:.3f}")

                    if ev_db is not None:
                        row.append(f"{ev_db:,.0f}")
                    else:
                        row.append("")

                    if centroids:
                        ev_calc = sum(p * c for p, c in zip(probs, centroids))
                        row.append(f"{ev_calc:,.0f}")

                    lines.append("| " + " | ".join(row) + " |")
                lines.append("")
            else:
                lines.append("_No forecasts_ensemble rows found for this question/run._")
                lines.append("")

            if centroids:
                lines.append("### Bucket centroids (from bucket_centroids)")
                lines.append("")
                lines.append("| Bucket | Centroid |")
                lines.append("|---:|---:|")
                for idx, c in enumerate(centroids, start=1):
                    lines.append(f"| {idx} | {c:,.0f} |")
                lines.append("")
            else:
                lines.append("_No bucket_centroids found for this hazard/metric._")
                lines.append("")

            lines.append("### Forecast Prompt Text (Full)")
            lines.append("")
            lines.append("```text")
            lines.append(fc.get("prompt_text") or "")
            lines.append("```")
            lines.append("")

        OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
        print(f"[info] Wrote forecast prompt markdown to {OUT_PATH}")
    finally:
        duckdb_io.close_db(con)


if __name__ == "__main__":
    main()
