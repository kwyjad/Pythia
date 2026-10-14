# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Which grounding sources the models actually cite, by stage (Oct 2026).

Brave charges per query, and the HS stages send about 1,900 a run (RC
grounding 998, triage grounding about 650, adversarial about 210). This
reads the latest production HS runs and, per stage, reports the queries
and their cost, the sources a pack carried, the share of packs whose
model answer cites at least one of them, and where the cited sources sat
in the pack (ranks 1-5, 6-10, beyond 10).

A source is "cited" when the first five words of its title appear in the
answer: the RC answer for RC grounding, the triage answer for triage
grounding, and the counter-evidence ``source`` fields for an adversarial
check. Triage answers name agencies (FEWS NET, WFP) rather than article
titles, so its share is a floor.

Read-only. Always exits 0.

Usage:
    python -m scripts.ci.grounding_citation --db data/resolver.duckdb [--runs 2] [--out x.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

STAGES = ("rc_grounding", "triage_grounding", "adversarial_search")
_ANSWER_CALL = {"rc_grounding": "rc_pass_1", "triage_grounding": "triage_pass_1"}


def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(text or "").lower()).strip()


def title_key(title: Any) -> Optional[str]:
    """The first five words of a title, before any " | " or " - " suffix."""
    t = str(title or "").split(" | ")[0].split(" - ")[0]
    words = _norm(t).split()
    return " ".join(words[:5]) if len(words) >= 3 else None


def _rank_bucket(rank: int) -> str:
    return "rank_1_5" if rank <= 5 else ("rank_6_10" if rank <= 10 else "rank_over_10")


def _cited(sources: List[Any], answer: str) -> List[int]:
    ranks = []
    for i, s in enumerate(sources):
        title = s.get("title") if isinstance(s, dict) else s
        k = title_key(title)
        if k and k in answer:
            ranks.append(i + 1)
    return ranks


def measure_run(con, hs_run_id: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"hs_run_id": hs_run_id, "stages": {}}
    spend = {
        str(ct): (int(n or 0), float(c or 0.0), int(q or 0))
        for ct, n, c, q in con.execute(
            "SELECT call_type, COUNT(*), SUM(cost_usd), "
            "SUM(TRY_CAST(json_extract(usage_json, '$.web_search_requests') AS INTEGER)) "
            "FROM llm_calls WHERE hs_run_id = ? AND provider = 'brave' GROUP BY 1",
            [hs_run_id],
        ).fetchall()
    }
    answers: Dict[tuple, str] = {}
    for ct, hz, iso, txt in con.execute(
        "SELECT call_type, hazard_code, iso3, response_text FROM llm_calls "
        "WHERE hs_run_id = ? AND call_type IN ('rc_pass_1', 'triage_pass_1')",
        [hs_run_id],
    ).fetchall():
        parts = str(hz or "").split("_")
        if len(parts) >= 2:
            answers[(str(ct), str(iso), parts[1])] = _norm(txt)

    def _stage(name: str, rows: List[tuple]) -> Dict[str, Any]:
        n_packs = n_sources = n_citing = n_cited = 0
        buckets: Counter = Counter()
        for sources, answer in rows:
            n_packs += 1
            n_sources += len(sources)
            ranks = _cited(sources, answer)
            n_cited += len(ranks)
            n_citing += bool(ranks)
            for r in ranks:
                buckets[_rank_bucket(r)] += 1
        calls, cost, queries = spend.get(name, (0, 0.0, 0))
        return {
            "brave_queries": queries,
            "brave_cost_usd": round(cost, 3),
            "packs": n_packs,
            "avg_sources": round(n_sources / n_packs, 1) if n_packs else None,
            "share_packs_citing": round(n_citing / n_packs, 3) if n_packs else None,
            "citations": n_cited,
            "citations_by_rank": dict(buckets),
        }

    for name in ("rc_grounding", "triage_grounding"):
        rows = []
        for iso, hz, sj in con.execute(
            "SELECT iso3, hazard_code, sources_json FROM hs_hazard_tail_packs "
            "WHERE hs_run_id = ? AND query LIKE ?",
            [hs_run_id, name + ":%"],
        ).fetchall():
            answer = answers.get((_ANSWER_CALL[name], str(iso), str(hz)))
            if answer is not None:
                rows.append((json.loads(sj or "[]"), answer))
        out["stages"][name] = _stage(name, rows)

    rows = []
    try:
        for sj, pj in con.execute(
            "SELECT sources_json, payload_json FROM hs_adversarial_checks WHERE hs_run_id = ?",
            [hs_run_id],
        ).fetchall():
            payload = json.loads(pj or "{}")
            cited = " ".join(
                _norm(c.get("source"))
                for c in (payload.get("counter_evidence") or [])
                if isinstance(c, dict)
            )
            rows.append((json.loads(sj or "[]"), cited))
    except Exception:  # noqa: BLE001 — a DB with no checks table
        pass
    out["stages"]["adversarial_search"] = _stage("adversarial_search", rows)
    return out


def latest_production_runs(con, n: int) -> List[str]:
    return [
        str(r[0])
        for r in con.execute(
            "SELECT hs_run_id FROM hs_runs WHERE COALESCE(is_test, FALSE) = FALSE "
            "ORDER BY generated_at DESC LIMIT ?",
            [n],
        ).fetchall()
    ]


def render(results: List[Dict[str, Any]]) -> str:
    lines = ["## Grounding citation by stage", "",
             "| run | stage | queries | cost | packs | sources/pack | packs citing | ranks 1-5 | 6-10 | >10 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for res in results:
        for name, st in res["stages"].items():
            b = st["citations_by_rank"]
            share = "" if st["share_packs_citing"] is None else f"{100 * st['share_packs_citing']:.0f}%"
            lines.append(
                f"| {res['hs_run_id']} | {name} | {st['brave_queries']} | ${st['brave_cost_usd']:.2f} | "
                f"{st['packs']} | {st['avg_sources'] or ''} | {share} | {b.get('rank_1_5', 0)} | "
                f"{b.get('rank_6_10', 0)} | {b.get('rank_over_10', 0)} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    path = args.db[len("duckdb:///"):] if args.db.startswith("duckdb:///") else args.db
    results: List[Dict[str, Any]] = []
    try:
        import duckdb  # noqa: PLC0415

        con = duckdb.connect(path, read_only=True)
        try:
            results = [measure_run(con, r) for r in latest_production_runs(con, args.runs)]
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001
        print(f"::warning title=Grounding citation::{type(exc).__name__}: {exc}")
    print(render(results))
    if args.out:
        try:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
