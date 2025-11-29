from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _get_first_forecast_call_for_run(con, run_id: str) -> Optional[Dict[str, Any]]:
    """Return the first forecast call (chronologically) for a given run_id."""

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
          AND call_type = 'forecast'
        ORDER BY timestamp
        LIMIT 1
        """,
        [run_id],
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

        fc = _get_first_forecast_call_for_run(con, run_id)
        if not fc:
            _write_placeholder(f"[warn] No forecast calls for run_id={run_id}; nothing to dump.")
            return

        question_id = str(fc["question_id"])
        research_calls = _get_research_calls_for_question(con, run_id, question_id)
        qmeta = _get_question_metadata(con, question_id)
        qctx = _get_question_context(con, run_id, question_id)
    finally:
        duckdb_io.close_db(con)

    lines: List[str] = []

    lines.append("# Pythia Forecast Prompt Debug")
    lines.append("")
    lines.append("## Run / Question")
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
            lines.append("### Question wording")
            lines.append("")
            lines.append(str(qmeta["wording"]))

    lines.append("")
    lines.append("## Research Calls")
    lines.append("")

    if research_calls:
        for idx, rc in enumerate(research_calls, start=1):
            lines.append(f"### Research call {idx}")
            lines.append("")
            lines.append(
                f"- **model:** `{rc['provider']}/{rc['model_id']}` (`{rc['model_name']}`)"
            )
            lines.append(f"- **usage:** {_fmt_tokens_and_cost(rc)}")
            lines.append("")
            lines.append("#### Research prompt (truncated)")
            lines.append("")
            lines.append("```text")
            lines.append(_truncate(rc.get("prompt_text") or "", 2000))
            lines.append("```")
            lines.append("")
            lines.append("#### Research response (truncated)")
            lines.append("")
            lines.append("```text")
            lines.append(_truncate(rc.get("response_text") or "", 2000))
            lines.append("```")
            lines.append("")
    else:
        lines.append("_No research calls recorded for this question/run._")
        lines.append("")

    lines.append("## Resolver 36-month Snapshot")
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
        if ctx_extra:
            lines.append("### Resolver context JSON")
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

    lines.append("## Forecast Prompt Text (Full)")
    lines.append("")
    lines.append("```text")
    lines.append(fc.get("prompt_text") or "")
    lines.append("```")
    lines.append("")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[info] Wrote first forecast prompt markdown to {OUT_PATH}")


if __name__ == "__main__":
    main()
