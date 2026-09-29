# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""Does a regime-change flag make Track 1 members SHIFT their forecast, or SPREAD it?

Read-only. Reads ``forecasts_raw.reasoning_trace_json`` for Track 1 SPD ensemble
members and compares the stated PRIOR with the month-1 POSTERIOR the member wrote,
grouped by metric, RC level (0 vs 1+) and RC direction.

Per group it reports:

* sharpness: max bucket probability and Shannon entropy (bits), prior and posterior,
  and the change;
* shift: the signed change in expected bucket index, prior to posterior;
* spread: the change in probability on buckets more than one away from the prior's
  modal bucket, below and above separately;
* for resolved questions, month-1 Brier and RPS of prior, posterior and climatology
  (the ``base_rate_spd`` anchor ``score_baselines`` scores);
* whether the prior is already wider than the base-rate SPD (entropy, prior minus base).

FATALITIES outcomes are read from ``acled_monthly_fatalities`` (all event types) rather
than from ``resolutions``, so a database written before the 29 September fix still
answers against the series the question resolves on. A month the series did not cover
globally, or a country it never names, has no outcome.

The hypothesis holds when RC 1+ posteriors lose sharpness (entropy up) much more than
they shift, and more than RC 0 posteriors do.

    python -m scripts.ci.rc_shift_diagnostic --db resolver.duckdb [--out table.md]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

AGGREGATES = {"ensemble_mean_v2", "ensemble_bayesmc_v2", "track2_flash", "sibyl"}
SPD_METRICS = ("FATALITIES", "PA", "PHASE3PLUS_IN_NEED")


def _norm(p: Sequence[float]) -> Optional[List[float]]:
    vals = [max(0.0, float(x or 0.0)) for x in p]
    s = sum(vals)
    if s <= 0:
        return None
    return [v / s for v in vals]


def entropy_bits(p: Sequence[float]) -> float:
    return -sum(x * math.log2(x) for x in p if x > 0)


def expected_index(p: Sequence[float]) -> float:
    return sum(i * x for i, x in enumerate(p))


def far_mass(p: Sequence[float], mode: int) -> tuple:
    """Mass on buckets more than one away from ``mode``: (below, above)."""
    below = sum(x for i, x in enumerate(p) if i < mode - 1)
    above = sum(x for i, x in enumerate(p) if i > mode + 1)
    return below, above


def brier(p: Sequence[float], k: int) -> float:
    return sum((x - (1.0 if i == k else 0.0)) ** 2 for i, x in enumerate(p))


def rps(p: Sequence[float], k: int) -> float:
    n = len(p)
    if n < 2:
        return 0.0
    cf = co = 0.0
    total = 0.0
    for i in range(n - 1):
        cf += p[i]
        co += 1.0 if i == k else 0.0
        total += (cf - co) ** 2
    return total / (n - 1)


def rc_direction_group(direction: Any) -> str:
    d = str(direction or "").strip().lower()
    return d if d in {"up", "down", "mixed"} else "unclear"


def compare(prior: Sequence[float], post: Sequence[float]) -> Dict[str, float]:
    mode = max(range(len(prior)), key=lambda i: prior[i])
    b0, a0 = far_mass(prior, mode)
    b1, a1 = far_mass(post, mode)
    return {
        "max_prior": max(prior),
        "max_post": max(post),
        "h_prior": entropy_bits(prior),
        "h_post": entropy_bits(post),
        "shift": expected_index(post) - expected_index(prior),
        "far_below": b1 - b0,
        "far_above": a1 - a0,
    }


def _fatalities_outcomes(con, before_ym: str) -> Dict[tuple, float]:
    """(iso3, 'YYYY-MM') -> all-types deaths, zero for a covered quiet month.

    Only months strictly before ``before_ym`` count: the current month is partial.
    """
    try:
        rows = con.execute(
            "SELECT upper(iso3), strftime(month, '%Y-%m'), SUM(fatalities) "
            "FROM acled_monthly_fatalities WHERE strftime(month, '%Y-%m') < ? GROUP BY 1, 2",
            [before_ym],
        ).fetchall()
    except Exception:
        return {}
    out = {(r[0], r[1]): float(r[2] or 0.0) for r in rows}
    out["__months__"] = {r[1] for r in rows}  # type: ignore[assignment]
    out["__iso3s__"] = {r[0] for r in rows}  # type: ignore[assignment]
    return out


def _outcome_value(metric, iso3, ym, qid, fat, res) -> Optional[float]:
    if metric == "FATALITIES":
        if not fat or ym not in fat["__months__"] or iso3 not in fat["__iso3s__"]:
            return None
        return fat.get((iso3, ym), 0.0)
    return res.get(qid)


def load_rows(con) -> List[Dict[str, Any]]:
    q = """
        SELECT fr.run_id, fr.question_id, fr.model_name, fr.reasoning_trace_json,
               q.metric, q.iso3, q.hazard_code, CAST(q.window_start_date AS VARCHAR),
               ht.regime_change_level, ht.regime_change_direction, q.track
        FROM forecasts_raw fr
        JOIN questions q USING (question_id)
        LEFT JOIN hs_triage ht
          ON ht.run_id = q.hs_run_id AND ht.iso3 = q.iso3 AND ht.hazard_code = q.hazard_code
        WHERE q.track IN (1, 2)
          AND q.metric IN ('FATALITIES', 'PA', 'PHASE3PLUS_IN_NEED')
          AND fr.reasoning_trace_json IS NOT NULL
          AND COALESCE(q.is_test, FALSE) = FALSE
        GROUP BY ALL
    """
    traces = con.execute(q).fetchall()
    post = defaultdict(dict)
    for run_id, qid, model, b, p in con.execute(
        "SELECT run_id, question_id, model_name, bucket_index, probability FROM forecasts_raw "
        "WHERE month_index = 1 AND reasoning_trace_json IS NOT NULL"
    ).fetchall():
        post[(run_id, qid, model)][int(b)] = float(p or 0.0)
    out = []
    # One run per (question, model): the latest, as every score summary does.
    latest: Dict[tuple, str] = {}
    for t in traces:
        k = (t[1], t[2])
        if k not in latest or str(t[0]) > latest[k]:
            latest[k] = str(t[0])
    for run_id, qid, model, trace_json, metric, iso3, hz, ws, lvl, dirn, track in traces:
        if model in AGGREGATES and model != "track2_flash":
            continue
        if latest.get((qid, model)) != str(run_id):
            continue
        try:
            prior = _norm(json.loads(trace_json)["prior"]["spd"])
        except Exception:
            continue
        pb = post.get((run_id, qid, model))
        if not prior or not pb:
            continue
        posterior = _norm([pb.get(i, 0.0) for i in range(1, len(prior) + 1)])
        if not posterior or len(posterior) != len(prior):
            continue
        out.append({
            "run_id": run_id, "question_id": qid, "model": model, "metric": metric,
            "iso3": (iso3 or "").upper(), "hazard": hz, "ym": (ws or "")[:7],
            "track": int(track or 0),
            "rc": "1+" if (lvl or 0) >= 1 else "0",
            "dir": rc_direction_group(dirn),
            "prior": prior, "post": posterior,
        })
    return out


def attach_outcomes(con, rows: List[Dict[str, Any]], today_ym: Optional[str] = None) -> None:
    from pythia.tools.base_rate_spd import _bucket_index_for_value, base_rate_spd

    if today_ym is None:
        import datetime as _dt

        today_ym = _dt.date.today().strftime("%Y-%m")
    fat = _fatalities_outcomes(con, today_ym)
    res = {}
    try:
        for qid, v in con.execute("SELECT question_id, value FROM resolutions WHERE horizon_m = 1").fetchall():
            res[qid] = v
    except Exception:
        pass
    base_cache: Dict[str, Any] = {}
    for r in rows:
        key = r["question_id"]
        if key not in base_cache:
            try:
                probs, _src, _d = base_rate_spd(con, r["iso3"], r["hazard"], r["metric"], r["ym"])
            except Exception:
                probs = []
            base_cache[key] = _norm(probs) if probs else None
        r["base"] = base_cache[key]
        val = _outcome_value(r["metric"], r["iso3"], r["ym"], key, fat, res)
        r["bucket"] = None if val is None else _bucket_index_for_value(val, r["metric"])


def summarise(rows: Iterable[Dict[str, Any]], key) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    out = []
    for g, rs in sorted(groups.items()):
        cmp = [compare(r["prior"], r["post"]) for r in rs]
        mean = lambda k: sum(c[k] for c in cmp) / len(cmp)  # noqa: E731
        s: Dict[str, Any] = {
            "group": g, "n": len(rs), "n_questions": len({r["question_id"] for r in rs}),
            "max_prior": mean("max_prior"), "max_post": mean("max_post"),
            "h_prior": mean("h_prior"), "h_post": mean("h_post"),
            "dh": mean("h_post") - mean("h_prior"),
            "shift": mean("shift"), "abs_shift": sum(abs(c["shift"]) for c in cmp) / len(cmp),
            "far_below": mean("far_below"), "far_above": mean("far_above"),
        }
        with_base = [r for r in rs if r.get("base") and len(r["base"]) == len(r["prior"])]
        s["h_base"] = (sum(entropy_bits(r["base"]) for r in with_base) / len(with_base)) if with_base else None
        s["prior_minus_base_h"] = (
            sum(entropy_bits(r["prior"]) - entropy_bits(r["base"]) for r in with_base) / len(with_base)
            if with_base else None
        )
        res = [r for r in with_base if r.get("bucket") is not None]
        s["n_resolved"] = len(res)
        s["n_resolved_q"] = len({r["question_id"] for r in res})
        if res:
            for label, field in (("prior", "prior"), ("post", "post"), ("clim", "base")):
                s[f"brier_{label}"] = sum(brier(r[field], r["bucket"]) for r in res) / len(res)
                s[f"rps_{label}"] = sum(rps(r[field], r["bucket"]) for r in res) / len(res)
        out.append(s)
    return out


def _f(x, nd=3):
    return "—" if x is None else f"{x:+.{nd}f}" if nd and isinstance(x, float) and x < 0 else ("—" if x is None else f"{x:.{nd}f}")


def render(title: str, summary: List[Dict[str, Any]]) -> str:
    head = ("| group | n (q) | maxP prior→post | H prior→post (Δ) | shift E[k] (|shift|) "
            "| far below / above Δ | H prior − base | resolved | Brier prior / post / clim | RPS prior / post / clim |")
    lines = [f"### {title}", "", head, "|" + "---|" * 10]
    for s in summary:
        g = " · ".join(str(x) for x in s["group"])
        br = rp = "—"
        if s["n_resolved"]:
            br = f"{s['brier_prior']:.3f} / {s['brier_post']:.3f} / {s['brier_clim']:.3f}"
            rp = f"{s['rps_prior']:.3f} / {s['rps_post']:.3f} / {s['rps_clim']:.3f}"
        lines.append(
            f"| {g} | {s['n']} ({s['n_questions']}) | {s['max_prior']:.2f}→{s['max_post']:.2f} "
            f"| {s['h_prior']:.2f}→{s['h_post']:.2f} ({s['dh']:+.2f}) "
            f"| {s['shift']:+.2f} ({s['abs_shift']:.2f}) "
            f"| {s['far_below']:+.3f} / {s['far_above']:+.3f} "
            f"| {_f(s['prior_minus_base_h'], 2)} | {s['n_resolved']} ({s['n_resolved_q']}) | {br} | {rp} |"
        )
    return "\n".join(lines)


def run(con, today_ym: Optional[str] = None) -> str:
    rows = load_rows(con)
    attach_outcomes(con, rows, today_ym)
    parts = [
        f"Rows: {len(rows)} SPD forecasts with a parseable prior (Track 1 members; Track 2's "
        "single model as the RC-0 comparator). Latest run per question and model. "
        "Resolved = month 1 has an outcome."
    ]
    for metric in SPD_METRICS:
        mr = [r for r in rows if r["metric"] == metric]
        if not mr:
            continue
        parts.append(render(f"{metric}: by track and RC level", summarise(mr, lambda r: (f"T{r['track']}", r["rc"]))))
        t1 = [r for r in mr if r["track"] == 1]
        parts.append(render(f"{metric}: Track 1 by RC direction", summarise(t1, lambda r: (r["rc"], r["dir"]))))
    return "\n\n".join(parts) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", required=True)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    import duckdb

    con = duckdb.connect(args.db, read_only=True)
    text = run(con)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
