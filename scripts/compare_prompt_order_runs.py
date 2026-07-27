# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Compare two pipeline runs for the prompt-order (V3) A/B validation.

Usage:
    python -m scripts.compare_prompt_order_runs --db data/resolver.duckdb \
        --run-a fc_<baseline> --run-b fc_<v3> \
        [--hs-run-a hs_<baseline> --hs-run-b hs_<v3>]

Gate for enabling PYTHIA_PROMPT_V3_ORDER in production: run one
PYTHIA_TEST_MODE=1 pipeline with V3 order on the same DB snapshot as the
prior (legacy-order) run, then compare:

- per-(question, model, month) SPD Jensen-Shannon divergence and mean |dp|
  per bucket between the two runs (drift should sit within the noise floor —
  establish it by comparing two same-order runs if in doubt);
- parse-failure / member-dropout rates from llm_calls (should not worsen by
  more than ~1pp);
- RC level distribution and triage tier histogram (should sit within
  month-to-month variance);
- binary P(yes) deltas for EVENT_OCCURRENCE questions.

The script is advisory: it prints a report and always exits 0 (the reviewer
makes the call). Stdlib + duckdb only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict


def _js_divergence(p: list[float], q: list[float]) -> float:
    """Jensen-Shannon divergence (base 2, in [0, 1]) between two SPDs."""

    eps = 1e-12
    p = [max(eps, float(x)) for x in p]
    q = [max(eps, float(x)) for x in q]
    ps, qs = sum(p), sum(q)
    p = [x / ps for x in p]
    q = [x / qs for x in q]
    m = [(a + b) / 2.0 for a, b in zip(p, q)]

    def _kl(a: list[float], b: list[float]) -> float:
        return sum(x * math.log2(x / y) for x, y in zip(a, b))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _load_spds(con, run_id: str) -> dict:
    """{(question_id, model_name, month_index): [p_1..p_K]} from forecasts_raw."""

    rows = con.execute(
        """
        SELECT question_id, model_name, month_index, bucket_index, probability
        FROM forecasts_raw
        WHERE run_id = ?
        ORDER BY question_id, model_name, month_index, bucket_index
        """,
        [run_id],
    ).fetchall()
    out: dict = defaultdict(dict)
    for qid, model, month, bidx, prob in rows:
        out[(qid, model, month)][int(bidx)] = float(prob or 0.0)
    return {
        key: [probs[i] for i in sorted(probs)]
        for key, probs in out.items()
    }


def _call_stats(con, run_id: str) -> dict:
    rows = con.execute(
        """
        SELECT COUNT(*) AS n,
               SUM(CASE WHEN COALESCE(error_text, '') <> '' THEN 1 ELSE 0 END) AS n_err
        FROM llm_calls
        WHERE run_id = ? AND phase IN ('spd_v2', 'binary_v2')
        """,
        [run_id],
    ).fetchone()
    n = int(rows[0] or 0)
    n_err = int(rows[1] or 0)
    return {"calls": n, "errors": n_err, "error_rate": (n_err / n) if n else 0.0}


def _rc_triage_hist(con, hs_run_id: str) -> dict:
    rc = dict(
        con.execute(
            """
            SELECT COALESCE(regime_change_level, 0) AS lvl, COUNT(*)
            FROM hs_triage WHERE hs_run_id = ? GROUP BY 1 ORDER BY 1
            """,
            [hs_run_id],
        ).fetchall()
    )
    tier = dict(
        con.execute(
            """
            SELECT COALESCE(tier, '?') AS tier, COUNT(*)
            FROM hs_triage WHERE hs_run_id = ? GROUP BY 1 ORDER BY 1
            """,
            [hs_run_id],
        ).fetchall()
    )
    return {"rc_levels": {int(k): int(v) for k, v in rc.items()},
            "tiers": {str(k): int(v) for k, v in tier.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--run-a", required=True, help="baseline forecaster run_id")
    ap.add_argument("--run-b", required=True, help="candidate forecaster run_id")
    ap.add_argument("--hs-run-a", default=None)
    ap.add_argument("--hs-run-b", default=None)
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    import duckdb

    con = duckdb.connect(args.db, read_only=True)
    report: dict = {"run_a": args.run_a, "run_b": args.run_b}

    spds_a = _load_spds(con, args.run_a)
    spds_b = _load_spds(con, args.run_b)
    shared = sorted(set(spds_a) & set(spds_b))
    report["spd_pairs_compared"] = len(shared)
    report["spd_only_in_a"] = len(set(spds_a) - set(spds_b))
    report["spd_only_in_b"] = len(set(spds_b) - set(spds_a))

    jsds: list[float] = []
    mean_abs_dp: list[float] = []
    per_model: dict = defaultdict(list)
    for key in shared:
        pa, pb = spds_a[key], spds_b[key]
        if len(pa) != len(pb) or not pa:
            continue
        jsd = _js_divergence(pa, pb)
        jsds.append(jsd)
        mean_abs_dp.append(sum(abs(a - b) for a, b in zip(pa, pb)) / len(pa))
        per_model[key[1]].append(jsd)

    def _summ(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0}
        vals = sorted(vals)
        return {
            "n": len(vals),
            "mean": sum(vals) / len(vals),
            "p50": vals[len(vals) // 2],
            "p90": vals[int(len(vals) * 0.9)],
            "max": vals[-1],
        }

    report["jsd"] = _summ(jsds)
    report["mean_abs_dp"] = _summ(mean_abs_dp)
    report["jsd_by_model"] = {m: _summ(v) for m, v in sorted(per_model.items())}
    report["llm_calls_a"] = _call_stats(con, args.run_a)
    report["llm_calls_b"] = _call_stats(con, args.run_b)

    if args.hs_run_a and args.hs_run_b:
        report["hs_a"] = _rc_triage_hist(con, args.hs_run_a)
        report["hs_b"] = _rc_triage_hist(con, args.hs_run_b)

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"Prompt-order A/B comparison: {args.run_a} (A) vs {args.run_b} (B)")
    print(f"  SPD vectors compared: {report['spd_pairs_compared']} "
          f"(only-in-A: {report['spd_only_in_a']}, only-in-B: {report['spd_only_in_b']})")
    j = report["jsd"]
    if j.get("n"):
        print(f"  JS divergence: mean={j['mean']:.4f} p50={j['p50']:.4f} "
              f"p90={j['p90']:.4f} max={j['max']:.4f}")
        d = report["mean_abs_dp"]
        print(f"  Mean |dp| per bucket: mean={d['mean']:.4f} p90={d['p90']:.4f}")
        print("  By model:")
        for m, s in report["jsd_by_model"].items():
            print(f"    {m}: n={s['n']} mean={s.get('mean', 0):.4f} max={s.get('max', 0):.4f}")
    else:
        print("  No overlapping SPD vectors — check run ids / DB snapshot.")
    a, b = report["llm_calls_a"], report["llm_calls_b"]
    print(f"  SPD/binary call errors: A {a['errors']}/{a['calls']} "
          f"({a['error_rate']:.1%})  B {b['errors']}/{b['calls']} ({b['error_rate']:.1%})")
    if b["error_rate"] > a["error_rate"] + 0.01:
        print("  [warn] candidate error rate worse by >1pp — investigate before enabling V3")
    if "hs_a" in report:
        print(f"  RC levels A: {report['hs_a']['rc_levels']}  B: {report['hs_b']['rc_levels']}")
        print(f"  Tiers A: {report['hs_a']['tiers']}  B: {report['hs_b']['tiers']}")
    print("Advisory only — reviewer decides. Compare against a baseline-vs-baseline "
          "rerun to establish the temperature noise floor before judging drift.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
