# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

"""Per-stage health snapshot + cost attribution for the staged pipeline.

Before this script, every interpreted diagnostic lived at the final stage:
grounding health and Brave breaker state were computed only inside the S4 debug
bundle, and that bundle is gated `if: success()`. So a Brave breaker trip at
hs_submit — which blocks ungrounded hazards from forecasting — stayed invisible
for hours, and a pipeline that stalled mid-way produced no health reporting at
all for work already paid for.

Emits ``diagnostics/stage_health__<stage>.json`` plus a Markdown summary,
covering four things a stage can answer about itself:

  1. grounding health — including the no-backend sentinels, so a breaker trip
     is countable rather than inferred from a costs page
  2. RC level distribution, once hs_triage has rows for the run
  3. is_test consistency across the stamped tables — the boolean-input coercion
     hazard means a production month can be silently stamped as test data
  4. cost attribution by phase / call_type / model, both run-scoped and
     windowed to this stage

This deliberately does NOT import scripts.dump_pythia_debug_bundle: that module
pulls pandas and the whole forecaster tree, which is far more import surface
than an early stage should take on. The grounding sentinels are imported from
horizon_scanner.llm_logging when available and fall back to literals otherwise.

DIAGNOSTIC ONLY: must never fail a pipeline stage. Stages 2..N of a live
pipeline run whatever is on main, so every query is guarded and main() always
returns 0.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import duckdb

DEFAULT_DB_URL = "duckdb:///data/resolver.duckdb"

try:  # keep import surface minimal; literals are the contract either way
    from horizon_scanner.llm_logging import (
        GROUNDING_BREAKER_MODEL_ID,
        GROUNDING_FAILED_MODEL_ID,
        GROUNDING_UNAVAILABLE_MODEL_ID,
    )
except Exception:  # pragma: no cover - exercised only in sparse envs
    GROUNDING_FAILED_MODEL_ID = "grounding-failed"
    GROUNDING_BREAKER_MODEL_ID = "grounding-breaker-tripped"
    GROUNDING_UNAVAILABLE_MODEL_ID = "grounding-unavailable"

# These are naming sentinels, not models: a call logged under one of them got
# no grounding evidence back from any backend.
_NO_BACKEND_IDS = {
    GROUNDING_FAILED_MODEL_ID: "all backends returned nothing",
    GROUNDING_BREAKER_MODEL_ID: "Brave circuit breaker tripped",
    GROUNDING_UNAVAILABLE_MODEL_ID: "no backend available (missing key)",
}

# Tables whose is_test stamp must agree for a run. Each maps to the column
# carrying the run id.
_IS_TEST_TABLES = {
    "hs_runs": "hs_run_id",
    "hs_triage": "run_id",
    "llm_calls": "hs_run_id",
    "hs_hazard_tail_packs": "hs_run_id",
    "hs_adversarial_checks": "hs_run_id",
}


def _q(con, sql: str, params: Optional[list] = None) -> List[tuple]:
    try:
        return list(con.execute(sql, params or []).fetchall())
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"[warn] stage_health query failed: {type(exc).__name__}: {exc}")
        return []


def _has_table(con, table: str) -> bool:
    try:
        con.execute(f"SELECT 1 FROM {table} LIMIT 1")
        return True
    except Exception:
        return False


def _grounding(con, hs_run_id: str, since: Optional[str]) -> Dict[str, Any]:
    """RC + triage grounding health, keyed off the synthetic hazard_code encoding."""
    if not _has_table(con, "llm_calls"):
        return {"available": False}

    out: Dict[str, Any] = {"available": True}
    for label, pattern in (("rc", "GROUNDING\\_%"), ("triage", "TRIAGE\\_GROUNDING\\_%")):
        where = ["phase = 'hs_triage'", f"UPPER(hazard_code) LIKE '{pattern}' ESCAPE '\\'"]
        params: List[Any] = []
        # RC's pattern would also match TRIAGE_GROUNDING_*; exclude it explicitly.
        if label == "rc":
            where.append("UPPER(hazard_code) NOT LIKE 'TRIAGE\\_GROUNDING\\_%' ESCAPE '\\'")
        if hs_run_id:
            where.append("hs_run_id = ?")
            params.append(hs_run_id)
        if since:
            where.append("timestamp >= ?")
            params.append(since)

        rows = _q(
            con,
            f"""
            SELECT COALESCE(model_id, '(null)'), COUNT(*),
                   SUM(CASE WHEN error_text IS NOT NULL AND error_text <> '' THEN 1 ELSE 0 END)
            FROM llm_calls WHERE {' AND '.join(where)}
            GROUP BY 1 ORDER BY 2 DESC
            """,
            params,
        )
        by_model = {str(r[0]): {"n_calls": int(r[1]), "n_errors": int(r[2] or 0)} for r in rows}
        no_backend = {
            mid: {"n_calls": v["n_calls"], "reason": _NO_BACKEND_IDS[mid]}
            for mid, v in by_model.items()
            if mid in _NO_BACKEND_IDS
        }
        total = sum(v["n_calls"] for v in by_model.values())
        n_no_backend = sum(v["n_calls"] for v in no_backend.values())
        out[label] = {
            "n_calls": total,
            "by_model": by_model,
            "n_no_backend": n_no_backend,
            "no_backend_by_reason": no_backend,
            "no_backend_pct": round(100.0 * n_no_backend / total, 1) if total else 0.0,
        }
    return out


def _rc_levels(con, hs_run_id: str) -> Dict[str, Any]:
    if not _has_table(con, "hs_triage"):
        return {"available": False}
    where, params = [], []
    if hs_run_id:
        where.append("run_id = ?")
        params.append(hs_run_id)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    rows = _q(
        con,
        f"""
        SELECT COALESCE(CAST(regime_change_level AS VARCHAR), '(null)'), COUNT(*)
        FROM hs_triage {clause} GROUP BY 1 ORDER BY 1
        """,
        params,
    )
    tracks = _q(
        con,
        f"SELECT COALESCE(CAST(track AS VARCHAR),'(null)'), COUNT(*) FROM hs_triage {clause} GROUP BY 1 ORDER BY 1",
        params,
    )
    return {
        "available": True,
        "n_rows": sum(int(r[1]) for r in rows),
        "by_level": {str(r[0]): int(r[1]) for r in rows},
        "by_track": {str(r[0]): int(r[1]) for r in tracks},
    }


def _is_test_consistency(con, hs_run_id: str, expected: Optional[bool]) -> Dict[str, Any]:
    """Every table stamped for this run should agree on is_test.

    The poller dispatches stages over REST, where a workflow_dispatch boolean
    can arrive as the string "false" — so a production month being stamped as
    test data (or the reverse) is a live hazard worth asserting per stage.
    """
    per_table: Dict[str, Any] = {}
    seen: set = set()
    for table, run_col in _IS_TEST_TABLES.items():
        if not _has_table(con, table):
            continue
        where, params = [], []
        if hs_run_id:
            where.append(f"{run_col} = ?")
            params.append(hs_run_id)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        rows = _q(
            con,
            f"SELECT COALESCE(CAST(is_test AS VARCHAR),'(null)'), COUNT(*) FROM {table} {clause} GROUP BY 1",
            params,
        )
        if not rows:
            continue
        counts = {str(r[0]): int(r[1]) for r in rows}
        per_table[table] = counts
        for k, v in counts.items():
            if v:
                seen.add(k.lower())

    consistent = len(seen) <= 1
    matches_expected = None
    if expected is not None and seen:
        want = "true" if expected else "false"
        matches_expected = seen == {want}
    return {
        "per_table": per_table,
        "distinct_values": sorted(seen),
        "consistent": consistent,
        "expected": expected,
        "matches_expected": matches_expected,
    }


def _cost(con, hs_run_id: str, fc_run_id: str, since: Optional[str]) -> Dict[str, Any]:
    """Spend by phase / call_type / model, run-scoped and windowed to this stage.

    llm_calls accumulates in the travelling DB, so an unscoped SUM would report
    every prior stage's spend as this stage's.
    """
    if not _has_table(con, "llm_calls"):
        return {"available": False}

    run_where, run_params = [], []
    ids = [i for i in (hs_run_id, fc_run_id) if i]
    if ids:
        placeholders = ",".join("?" for _ in ids)
        run_where.append(f"(hs_run_id IN ({placeholders}) OR run_id IN ({placeholders}))")
        run_params = ids + ids

    def rollup(extra_where: List[str], extra_params: List[Any], by: str) -> List[Dict[str, Any]]:
        where = run_where + extra_where
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        rows = _q(
            con,
            f"""
            SELECT COALESCE({by}, '(none)'), COUNT(*), SUM(COALESCE(cost_usd,0)),
                   SUM(COALESCE(total_tokens,0))
            FROM llm_calls {clause} GROUP BY 1 ORDER BY 3 DESC
            """,
            run_params + extra_params,
        )
        return [
            {"key": str(r[0]), "n_calls": int(r[1]), "cost_usd": round(float(r[2] or 0), 4),
             "tokens": int(r[3] or 0)}
            for r in rows
        ]

    stage_where, stage_params = ([], [])
    if since:
        stage_where, stage_params = (["timestamp >= ?"], [since])

    def total(where: List[str], params: List[Any]) -> Dict[str, Any]:
        w = run_where + where
        clause = f"WHERE {' AND '.join(w)}" if w else ""
        rows = _q(
            con,
            f"SELECT COUNT(*), SUM(COALESCE(cost_usd,0)), SUM(COALESCE(total_tokens,0)) FROM llm_calls {clause}",
            run_params + params,
        )
        if not rows:
            return {"n_calls": 0, "cost_usd": 0.0, "tokens": 0}
        n, c, t = rows[0]
        return {"n_calls": int(n or 0), "cost_usd": round(float(c or 0), 4), "tokens": int(t or 0)}

    return {
        "available": True,
        "since": since,
        "run_total": total([], []),
        "stage_total": total(stage_where, stage_params) if since else None,
        "run_by_phase": rollup([], [], "phase"),
        "stage_by_phase": rollup(stage_where, stage_params, "phase") if since else None,
        "stage_by_call_type": rollup(stage_where, stage_params, "call_type") if since else None,
        "stage_by_model": rollup(stage_where, stage_params, "model_id") if since else None,
    }


def _markdown(rep: Dict[str, Any]) -> str:
    L: List[str] = [f"### Stage health — `{rep['stage']}`", ""]
    if rep.get("hs_run_id"):
        L.append(f"hs_run: `{rep['hs_run_id']}`")
        L.append("")

    g = rep.get("grounding") or {}
    if g.get("available"):
        L.append("**Grounding**")
        L.append("")
        rows = [(k, g.get(k) or {}) for k in ("rc", "triage")]
        rows = [(k, s) for k, s in rows if s.get("n_calls")]
        if rows:
            L.append("| pass | calls | no-backend | % |")
            L.append("|---|--:|--:|--:|")
            for k, s in rows:
                L.append(f"| {k} | {s['n_calls']} | {s['n_no_backend']} | {s['no_backend_pct']}% |")
        else:
            # An empty table is ambiguous between "grounding failed entirely"
            # and "grounding happened in an earlier stage". Say which.
            scope = "this stage's window" if rep.get("since") else "this run"
            L.append(f"_No grounding calls in {scope} — grounding runs in the hs_submit "
                     f"and hs_rc_collect stages._")
        L.append("")
        for k in ("rc", "triage"):
            for mid, v in ((g.get(k) or {}).get("no_backend_by_reason") or {}).items():
                L.append(f"- `{k}`: {v['n_calls']} call(s) — {v['reason']}")
        L.append("")

    rc = rep.get("rc_levels") or {}
    if rc.get("available") and rc.get("n_rows"):
        lv = ", ".join(f"L{k}={v}" for k, v in sorted(rc["by_level"].items()))
        tr = ", ".join(f"T{k}={v}" for k, v in sorted(rc["by_track"].items()))
        L.append(f"**RC levels** ({rc['n_rows']} triage rows): {lv} · tracks: {tr}")
        L.append("")

    it = rep.get("is_test") or {}
    if it.get("per_table"):
        status = "consistent" if it["consistent"] else "**INCONSISTENT**"
        exp = it.get("matches_expected")
        extra = "" if exp is None else (" · matches expected" if exp else " · **does NOT match expected**")
        L.append(f"**is_test**: {status} — values {it['distinct_values']}{extra}")
        # Always render the per-table counts, not just on failure. When this
        # first fired in production the summary said "INCONSISTENT" but the
        # breakdown was only in the JSON artifact — and artifacts are exactly
        # what a restricted-network reader cannot fetch, so the finding was
        # visible but undiagnosable.
        L.append("")
        L.append("| table | " + " | ".join(sorted({k for c in it["per_table"].values() for k in c})) + " |")
        keys = sorted({k for c in it["per_table"].values() for k in c})
        L.append("|---|" + "--:|" * len(keys))
        for table, counts in sorted(it["per_table"].items()):
            L.append(f"| {table} | " + " | ".join(str(counts.get(k, 0)) for k in keys) + " |")
        L.append("")

    c = rep.get("cost") or {}
    if c.get("available"):
        rt = c["run_total"]
        L.append(f"**Cost** — run to date: {rt['n_calls']} calls, ${rt['cost_usd']}, {rt['tokens']} tokens")
        st = c.get("stage_total")
        if st:
            L.append("")
            L.append(f"This stage (since {c['since']}): {st['n_calls']} calls, ${st['cost_usd']}, {st['tokens']} tokens")
            rows = [r for r in (c.get("stage_by_phase") or []) if r["n_calls"]]
            if rows:
                L.append("")
                L.append("| phase | calls | cost | tokens |")
                L.append("|---|--:|--:|--:|")
                for r in rows:
                    L.append(f"| {r['key']} | {r['n_calls']} | ${r['cost_usd']} | {r['tokens']} |")
        L.append("")
    return "\n".join(L)


def _emit_summary(text: str) -> None:
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"[warn] could not write step summary: {type(exc).__name__}: {exc}")


def _expected_is_test() -> Optional[bool]:
    raw = os.getenv("PYTHIA_TEST_MODE")
    if raw is None or raw == "":
        return None
    return raw.strip().lower() in ("1", "true", "yes")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=DEFAULT_DB_URL)
    p.add_argument("--stage", default=os.getenv("PIPELINE_STAGE", "unknown"))
    p.add_argument("--hs-run-id", default="")
    p.add_argument("--fc-run-id", default="")
    p.add_argument("--since", default="", help="ISO timestamp marking this stage's start")
    p.add_argument("--out", default="")
    args = p.parse_args(argv)

    db = args.db
    db_path = db[len("duckdb:///"):] if db.startswith("duckdb:///") else db
    if not os.path.exists(db_path):
        print(f"[warn] DuckDB database not found at {db_path}")
        return 0

    try:
        con = duckdb.connect(db_path, read_only=True)
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"[warn] could not open DuckDB at {db_path}: {type(exc).__name__}: {exc}")
        return 0

    since = args.since.strip() or None
    hs_run_id = args.hs_run_id.strip()
    try:
        rep = {
            "stage": args.stage,
            "hs_run_id": hs_run_id,
            "fc_run_id": args.fc_run_id.strip(),
            "since": since,
            "grounding": _grounding(con, hs_run_id, since),
            "rc_levels": _rc_levels(con, hs_run_id),
            "is_test": _is_test_consistency(con, hs_run_id, _expected_is_test()),
            "cost": _cost(con, hs_run_id, args.fc_run_id.strip(), since),
        }
    finally:
        try:
            con.close()
        except Exception:
            pass

    out = args.out or f"diagnostics/stage_health__{args.stage}.json"
    try:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=2, default=str)
        print(f"wrote {out}")
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"[warn] could not write {out}: {type(exc).__name__}: {exc}")

    md = _markdown(rep)
    print(md)
    _emit_summary(md)

    # Warnings: each names a condition that silently degrades a run.
    g = rep["grounding"]
    if g.get("available"):
        for k in ("rc", "triage"):
            s = g.get(k) or {}
            breaker = (s.get("no_backend_by_reason") or {}).get(GROUNDING_BREAKER_MODEL_ID)
            if breaker:
                print(
                    f"::warning title=Brave breaker tripped::{breaker['n_calls']} {k} grounding "
                    f"call(s) at stage {args.stage} got no evidence because the Brave circuit "
                    f"breaker was tripped; ungrounded hazards are blocked from forecasting."
                )
            if s.get("n_calls") and s.get("no_backend_pct", 0) >= 50.0:
                print(
                    f"::warning title=Grounding largely failed::{s['no_backend_pct']}% of {k} "
                    f"grounding calls at stage {args.stage} returned no backend evidence."
                )

    it = rep["is_test"]
    if it["per_table"] and not it["consistent"]:
        print(
            f"::warning title=is_test inconsistent::stage {args.stage} has mixed is_test values "
            f"{it['distinct_values']} across stamped tables for this run."
        )
    if it.get("matches_expected") is False:
        print(
            f"::warning title=is_test does not match PYTHIA_TEST_MODE::expected "
            f"{it['expected']} but stamped values are {it['distinct_values']} at stage {args.stage}."
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
