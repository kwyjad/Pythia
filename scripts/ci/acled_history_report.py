# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Evidence for the ACLED history backfill (Resolver Update, ``acled_history``).

``acled_monthly_fatalities`` began in 2025-08, so the 36-month window the
climatology and level-and-volatility references read held about thirteen
months. The backfill reaches back years; this script records what that did:

* ``snapshot`` — the reference scores for one question epoch's ACE/FATALITIES
  questions, as a CSV, taken BEFORE anything is written and again after.
* ``verify``   — rows and countries per month, the rows written before their
  month ended (must be none), and the months ``source_coverage`` reports live.
* ``compare``  — the two snapshots side by side: mean Brier, RPS (``crps``)
  and log per reference, over the (question, horizon) pairs both carry.

Read-only apart from ``verify`` rebuilding ``source_coverage`` (which
``compute_resolutions`` rebuilds on every run anyway). Always exits 0: it is
evidence, and evidence must never cost the ingest beside it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import duckdb

#: Reference forecasters reported (order is the column order of the table).
REFERENCES = (
    "__ext_climatology",
    "__ext_level_volatility",
    "__ext_level_transition",
    "__ext_persistence",
    "__ext_uniform",
)
SCORE_TYPES = ("brier", "crps", "log")
SNAPSHOT_COLUMNS = ("question_id", "iso3", "horizon_m", "model_name", "score_type", "value")


def _connect(db: str) -> duckdb.DuckDBPyConnection:
    path = db[len("duckdb:///"):] if db.startswith("duckdb:///") else db
    return duckdb.connect(path)


def _has_table(con, name: str) -> bool:
    try:
        con.execute(f"SELECT 1 FROM {name} LIMIT 0")
        return True
    except Exception:  # noqa: BLE001
        return False


def snapshot_rows(con, epoch: str) -> List[Dict[str, object]]:
    """Reference score rows for the epoch's ACE/FATALITIES questions."""
    if not (_has_table(con, "scores") and _has_table(con, "questions")):
        return []
    placeholders = ",".join("?" for _ in REFERENCES)
    rows = con.execute(
        f"""
        SELECT s.question_id, q.iso3, s.horizon_m, s.model_name, s.score_type, s.value
        FROM scores s
        JOIN questions q ON q.question_id = s.question_id
        WHERE upper(q.hazard_code) = 'ACE' AND upper(q.metric) = 'FATALITIES'
          AND (s.question_id LIKE ? OR substr(CAST(q.window_start_date AS VARCHAR), 1, 7) = ?)
          AND s.model_name IN ({placeholders})
          AND COALESCE(q.is_test, FALSE) = FALSE
        ORDER BY 1, 3, 4, 5
        """,
        [f"%_{epoch}", epoch, *REFERENCES],
    ).fetchall()
    return [dict(zip(SNAPSHOT_COLUMNS, r)) for r in rows]


def write_snapshot(rows: Sequence[Dict[str, object]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(SNAPSHOT_COLUMNS))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_snapshot(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        out = []
        for r in csv.DictReader(fh):
            r["horizon_m"] = int(r["horizon_m"])
            r["value"] = float(r["value"])
            out.append(r)
        return out


def _index(rows) -> Dict[Tuple[str, str], Dict[Tuple[str, int], float]]:
    by: Dict[Tuple[str, str], Dict[Tuple[str, int], float]] = defaultdict(dict)
    for r in rows:
        by[(str(r["model_name"]), str(r["score_type"]))][(str(r["question_id"]), int(r["horizon_m"]))] = float(r["value"])
    return by


def summarise(rows, keys: Optional[set] = None) -> Dict[str, Dict[str, Dict[str, float]]]:
    """{model: {score_type: {"mean": m, "n": n, "questions": q}}} over ``keys`` when given."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for (model, st), cells in _index(rows).items():
        vals = [v for k, v in cells.items() if keys is None or k in keys]
        if not vals:
            continue
        qs = {k[0] for k in cells if keys is None or k in keys}
        out.setdefault(model, {})[st] = {
            "mean": sum(vals) / len(vals), "n": len(vals), "questions": len(qs),
        }
    return out


def common_keys(rows, models: Sequence[str], score_type: str = "brier") -> set:
    idx = _index(rows)
    sets = [set(idx.get((m, score_type), {})) for m in models if idx.get((m, score_type))]
    if not sets:
        return set()
    keys = sets[0]
    for s in sets[1:]:
        keys &= s
    return keys


def _fmt(cell: Optional[Dict[str, float]]) -> str:
    return "—" if not cell else f"{cell['mean']:.3f} (n={int(cell['n'])})"


def comparison_markdown(before, after, epoch: str) -> str:
    lines = [f"### ACE/FATALITIES reference scores, epoch {epoch}", ""]
    b, a = summarise(before), summarise(after)
    lines += ["Each reference over its own scored (question, horizon) pairs.", "",
              "| reference | Brier before | Brier after | RPS before | RPS after | log before | log after |",
              "|---|---|---|---|---|---|---|"]
    for m in REFERENCES:
        if m not in b and m not in a:
            continue
        row = [m]
        for st in SCORE_TYPES:
            row += [_fmt(b.get(m, {}).get(st)), _fmt(a.get(m, {}).get(st))]
        lines.append("| " + " | ".join(row) + " |")
    trio = ["__ext_climatology", "__ext_level_volatility", "__ext_level_transition"]
    keys = common_keys(after, [m for m in trio if summarise(after).get(m)])
    if keys:
        paired = summarise(after, keys)
        nq = len({k[0] for k in keys})
        lines += ["", f"After the backfill, paired over the {len(keys)} (question, horizon) "
                  f"pairs ({nq} questions) the three level/climatology references all scored:", "",
                  "| reference | Brier | RPS | log |", "|---|---|---|---|"]
        for m in trio:
            if m in paired:
                lines.append("| " + " | ".join(
                    [m] + [f"{paired[m][st]['mean']:.3f}" if st in paired[m] else "—" for st in SCORE_TYPES]
                ) + " |")
    return "\n".join(lines) + "\n"


def verify(con, since: str) -> Dict[str, object]:
    """Per-month rows/countries, partial rows, and live coverage months."""
    from pythia.tools.base_rate_spd import ACLED_COMPLETE_MONTH_SQL

    has_upd = any(r[1] == "updated_at" for r in con.execute(
        "PRAGMA table_info('acled_monthly_fatalities')").fetchall())
    partial = f"NOT ({ACLED_COMPLETE_MONTH_SQL})" if has_upd else "FALSE"
    rows = con.execute(
        f"""
        SELECT strftime(month, '%Y-%m') AS ym, COUNT(*) AS n, COUNT(DISTINCT iso3) AS c,
               SUM(fatalities) AS f, SUM(CASE WHEN {partial} THEN 1 ELSE 0 END) AS p
        FROM acled_monthly_fatalities
        WHERE strftime(month, '%Y-%m') >= ?
        GROUP BY 1 ORDER BY 1
        """,
        [since],
    ).fetchall()
    months = [{"month": r[0], "rows": int(r[1]), "countries": int(r[2]),
               "fatalities": int(r[3] or 0), "partial_rows": int(r[4] or 0)} for r in rows]
    by_year: Dict[str, Dict[str, int]] = {}
    for m in months:
        y = by_year.setdefault(m["month"][:4], {"months": 0, "rows": 0, "fatalities": 0})
        y["months"] += 1
        y["rows"] += m["rows"]
        y["fatalities"] += m["fatalities"]
    live: List[str] = []
    try:
        from pythia.tools.source_coverage import months_with_source_data, refresh_source_coverage

        refresh_source_coverage(con)
        live = sorted(months_with_source_data(con, "FATALITIES"))
    except Exception as exc:  # noqa: BLE001
        live = [f"error: {exc!r}"]
    in_table = [m["month"] for m in months]
    return {
        "since": since,
        "months": months,
        "by_year": by_year,
        "partial_rows_total": sum(m["partial_rows"] for m in months),
        "coverage_live_months": [m for m in live if m >= since],
        "months_not_live": [m for m in in_table if m not in set(live)],
    }


def verify_markdown(v: Dict[str, object]) -> str:
    lines = ["### acled_monthly_fatalities after the backfill", "",
             f"Rows written before their month ended: **{v['partial_rows_total']}** (must be 0).",
             f"Months in the table that source_coverage does not count as live: "
             f"{', '.join(v['months_not_live']) or 'none'}.", "",
             "| year | months | rows | fatalities |", "|---|---|---|---|"]
    for y, s in sorted(v["by_year"].items()):
        lines.append(f"| {y} | {s['months']} | {s['rows']} | {s['fatalities']:,} |")
    lines += ["", "| month | rows | countries | partial rows |", "|---|---|---|---|"]
    for m in v["months"]:
        lines.append(f"| {m['month']} | {m['rows']} | {m['countries']} | {m['partial_rows']} |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("snapshot")
    s1.add_argument("--db", required=True)
    s1.add_argument("--epoch", required=True)
    s1.add_argument("--out", required=True)
    s2 = sub.add_parser("verify")
    s2.add_argument("--db", required=True)
    s2.add_argument("--since", default="2018-01")
    s2.add_argument("--out", required=True)
    s3 = sub.add_parser("compare")
    s3.add_argument("--before", required=True)
    s3.add_argument("--after", required=True)
    s3.add_argument("--epoch", required=True)
    s3.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "snapshot":
            con = _connect(args.db)
            try:
                rows = snapshot_rows(con, args.epoch)
            finally:
                con.close()
            write_snapshot(rows, Path(args.out))
            print(f"[acled_history] {len(rows)} reference score rows for epoch {args.epoch} -> {args.out}")
        elif args.cmd == "verify":
            con = _connect(args.db)
            try:
                v = verify(con, args.since)
            finally:
                con.close()
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(v, indent=2), encoding="utf-8")
            md = verify_markdown(v)
            Path(args.out).with_suffix(".md").write_text(md, encoding="utf-8")
            print(md)
            if v["partial_rows_total"]:
                print(f"::warning title=Partial ACLED months::{v['partial_rows_total']} row(s) "
                      "were written before their month ended.")
        else:
            md = comparison_markdown(read_snapshot(Path(args.before)),
                                     read_snapshot(Path(args.after)), args.epoch)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(md, encoding="utf-8")
            print(md)
    except Exception as exc:  # noqa: BLE001 - evidence never fails the run
        print(f"::warning title=ACLED history report::{args.cmd} failed: {exc!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
