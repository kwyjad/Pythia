# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Before and after for the ACE/PA outcomes that are only a lower bound.

Read-only (``duckdb.connect(read_only=True)``; nothing created). For every
ACE/PA resolution whose ``source_desc`` names a lower bound (the IOM DTM
rung), lists per scored model the exact scores stored today (brier, log,
crps, computed as if the bound were the answer) beside the censored scores
``compute_scores`` now writes (brier_censored, log_censored, crps_censored),
recomputed here from the stored forecasts with the same functions. A bound
in bucket 0 states nothing and gets no censored score.

    python -m scripts.ci.censored_rescore_report --db data/resolver.duckdb \\
        --out-dir diagnostics/censored_rescore
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import duckdb

from pythia.tools import compute_scores as cs
from pythia.tools.scoring_class import CENSORED_HAZARD_METRICS, is_lower_bound_source

AGGREGATES = ("ensemble_mean_v2", "ensemble_bayesmc_v2", "track2_flash")


def _table_exists(con, name: str) -> bool:
    try:
        con.execute(f"SELECT 1 FROM {name} LIMIT 0")
        return True
    except Exception:  # noqa: BLE001
        return False


def build(con) -> Dict[str, Any]:
    if not _table_exists(con, "resolutions"):
        return {"stub": True, "reason": "no resolutions table"}
    pairs = ", ".join(f"('{h}', '{m}')" for h, m in sorted(CENSORED_HAZARD_METRICS))
    res = con.execute(
        f"""
        SELECT r.question_id, r.horizon_m, r.value, r.source_desc, upper(q.metric)
        FROM resolutions r JOIN questions q ON q.question_id = r.question_id
        WHERE (upper(q.hazard_code), upper(q.metric)) IN ({pairs})
        ORDER BY r.question_id, r.horizon_m
        """
    ).fetchall()
    rows: List[Dict[str, Any]] = []
    bucket0 = 0
    for qid, h, value, desc, metric in res:
        if not is_lower_bound_source(desc) or value is None:
            continue
        k = cs._bucket_index(float(value), metric)
        if k is None:
            continue
        stored = {}
        if _table_exists(con, "scores"):
            for model, run_id, st, v in con.execute(
                "SELECT model_name, run_id, score_type, value FROM scores "
                "WHERE question_id = ? AND horizon_m = ?", [qid, h],
            ).fetchall():
                stored.setdefault((model, run_id), {})[st] = v
        models = con.execute(
            "SELECT DISTINCT model_name, run_id FROM forecasts_raw "
            "WHERE question_id = ? AND month_index = ?", [qid, h],
        ).fetchall() if _table_exists(con, "forecasts_raw") else []
        if k == 0:
            bucket0 += 1
        bins = cs._class_bins(metric, "ACE")
        for model, run_id in sorted(models, key=lambda m: (str(m[0]), str(m[1]))):
            spd = cs._load_spd(con, question_id=qid, horizon_m=h, class_bins=bins,
                               model_name=model, run_id=run_id)
            before = stored.get((model, run_id), {})
            row = {
                "question_id": qid, "horizon_m": h, "value": value, "bucket": k,
                "source_desc": desc, "model": model, "run_id": run_id,
                "before": {t: before.get(t) for t in ("brier", "log", "crps")},
                "after": None,
            }
            if spd and k > 0:
                row["after"] = {
                    "brier_censored": cs._brier_censored(spd, k),
                    "log_censored": cs._log_score_censored(spd, k),
                    "crps_censored": cs._rps_censored(spd, k),
                }
                row["exact_now"] = {"brier": cs._brier(spd, k), "log": cs._log_score(spd, k),
                                    "crps": cs._rps(spd, k)}
            rows.append(row)
    return {"rows": rows, "n_resolutions": len({(r["question_id"], r["horizon_m"]) for r in rows}),
            "bucket_0_unscored": bucket0}


def _f(v) -> str:
    return "" if v is None else f"{float(v):.3f}"


def render(rep: Dict[str, Any]) -> str:
    out = ["# ACE/PA lower-bound outcomes: exact scores before, censored scores after", ""]
    if rep.get("stub"):
        return "\n".join(out + [f"**{rep['reason']}.**", ""])
    rows = rep["rows"]
    out += [f"{rep['n_resolutions']} lower-bound resolution(s); {rep['bucket_0_unscored']} in "
            "bucket 0 (no score: the bound states nothing). Before = the exact scores stored now, "
            "as if the bound were the answer; after = the censored scores this change writes. "
            "Different scales: compare within a column pair, never across.", ""]
    agg = [r for r in rows if r["model"] in AGGREGATES]
    out += ["## Aggregates", "",
            "| question | h | value | bucket | model | brier | log | crps | brier_c | log_c | crps_c |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in agg:
        a = r["after"] or {}
        b = r["before"]
        out.append(f"| {r['question_id']} | {r['horizon_m']} | {r['value']:,.0f} | {r['bucket']} | "
                   f"{r['model']} | {_f(b['brier'])} | {_f(b['log'])} | {_f(b['crps'])} | "
                   f"{_f(a.get('brier_censored'))} | {_f(a.get('log_censored'))} | {_f(a.get('crps_censored'))} |")
    out += ["", f"All models: {len(rows)} (resolution, model, run) rows; JSON has each.", ""]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", required=True)
    ap.add_argument("--out-dir", default="diagnostics/censored_rescore")
    args = ap.parse_args(argv)
    path = args.db[len("duckdb:///"):] if args.db.startswith("duckdb:///") else args.db
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        con = duckdb.connect(path, read_only=True)
        try:
            rep = build(con)
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001 - a report never fails a job
        rep = {"stub": True, "reason": f"{type(exc).__name__}: {exc}"}
    (out / "censored_rescore.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    (out / "censored_rescore.md").write_text(render(rep), encoding="utf-8")
    print(render(rep))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
