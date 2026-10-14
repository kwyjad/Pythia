# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""ACE/PA by country and ladder rung, and the questions the ladder revives.

For a window opening in ``--window`` (default: the month after next), and
for every country with a conflict displacement figure on any rung, this
reports the months in the twelve before the window that each rung answered
(IDMC, an extracted OCHA/UN figure, a DTM lower bound), whether the country
is a regular reporter under IDMC alone and under the ladder (a DTM-only
month never counts), and the rung that admits it.

It then counts REVIVED questions: existing ACE/PA questions with at least
one horizon-month that resolves today under the ladder and would not under
IDMC alone. That is the number the backfill was for.

Read-only. Always exits 0.

Usage:
    python -m scripts.ci.ace_pa_rung_table --db data/resolver.duckdb [--window 2026-12] [--out x.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional


def _default_window(today: date) -> str:
    y, m = today.year, today.month + 2
    while m > 12:
        y, m = y + 1, m - 12
    return f"{y:04d}-{m:02d}"


def rung_table(con, window: str) -> List[Dict[str, Any]]:
    from pythia.tools import base_rate_spd as brs  # noqa: PLC0415

    reported, _held = brs.conflict_displacement_series(con, before_ym=window)
    lower = brs.conflict_ladder_lower(con, before_ym=window)
    months = brs._window_months(window, brs.CONFLICT_REGULAR_WINDOW_MONTHS)
    rows: List[Dict[str, Any]] = []
    for iso in sorted(set(reported) | set(lower)):
        idmc = reported.get(iso, {})
        low = lower.get(iso, {})
        n = {"idmc": 0, "extracted": 0, "dtm_only": 0}
        for ym in months:
            rungs = {r.rung for r in low.get(ym, ())}
            if ym in idmc:
                n["idmc"] += 1
            elif brs.CONFLICT_RUNG_EXTRACTED in rungs:
                n["extracted"] += 1
            elif brs.CONFLICT_RUNG_DTM in rungs:
                n["dtm_only"] += 1
        rows.append({
            "iso3": iso,
            "months_idmc": n["idmc"],
            "months_extracted_only": n["extracted"],
            "months_dtm_only": n["dtm_only"],
            "months_counted": n["idmc"] + n["extracted"],
            "regular_idmc_alone": brs.conflict_regular_reporter(idmc, window),
            "admission_rung": brs.conflict_admission_rung(idmc, low, window),
        })
    rows.sort(key=lambda r: (r["admission_rung"] is None, -r["months_counted"], r["iso3"]))
    return rows


def revived_questions(con, today: Optional[date] = None) -> Dict[str, Any]:
    """ACE/PA questions with a horizon-month the ladder resolves and IDMC
    alone does not."""
    from pythia.tools import base_rate_spd as brs  # noqa: PLC0415
    from pythia.tools.compute_resolutions import horizon_to_calendar_month  # noqa: PLC0415

    today = today or date.today()
    reported, held = brs.conflict_displacement_series(con)
    lower = brs.conflict_ladder_lower(con)
    qs = con.execute(
        "SELECT question_id, upper(iso3), window_start_date FROM questions "
        "WHERE upper(hazard_code) = 'ACE' AND upper(metric) = 'PA' "
        "AND COALESCE(is_test, FALSE) = FALSE"
    ).fetchall()
    revived: List[str] = []
    months_revived = 0
    by_rung: Dict[str, int] = {}
    for qid, iso, ws in qs:
        hit = False
        for h in range(1, 7):
            try:
                ym = horizon_to_calendar_month(ws, h)
            except Exception:  # noqa: BLE001
                continue
            ym = str(ym)[:7]
            idmc_only = brs.resolve_conflict_month_detail(
                reported.get(iso, {}), ym, today, held=held.get(iso, set()))
            ladder = brs.resolve_conflict_month_detail(
                reported.get(iso, {}), ym, today, held=held.get(iso, set()),
                lower=lower.get(iso, {}).get(ym, ()))
            ok = (brs.CONFLICT_STATUS_REPORTED, brs.CONFLICT_STATUS_QUIET)
            if ladder.status in ok and idmc_only.status not in ok:
                hit = True
                months_revived += 1
                by_rung[ladder.rung or "zero"] = by_rung.get(ladder.rung or "zero", 0) + 1
        if hit:
            revived.append(str(qid))
    return {"n_questions": len(qs), "n_revived": len(revived), "horizon_months_revived": months_revived,
            "by_rung": by_rung, "revived": sorted(revived)}


def render(window: str, rows: List[Dict[str, Any]], revived: Dict[str, Any]) -> str:
    out = [f"## ACE/PA by ladder rung, window opening {window}", "",
           "| country | IDMC months | extracted only | DTM only | counted (of 12) | regular on IDMC alone | admitted on |",
           "|---|---:|---:|---:|---:|---|---|"]
    for r in rows:
        out.append(
            f"| {r['iso3']} | {r['months_idmc']} | {r['months_extracted_only']} | {r['months_dtm_only']} | "
            f"{r['months_counted']} | {'yes' if r['regular_idmc_alone'] else 'no'} | "
            f"{r['admission_rung'] or 'not admitted'} |"
        )
    adm = [r for r in rows if r["admission_rung"]]
    out += ["", f"Admitted: {len(adm)} ({sum(1 for r in adm if r['admission_rung'] == 'idmc')} on IDMC, "
            f"{sum(1 for r in adm if r['admission_rung'] != 'idmc')} on extracted figures).", "",
            f"Revived questions: {revived.get('n_revived', 0)} of {revived.get('n_questions', 0)} "
            f"ACE/PA questions, {revived.get('horizon_months_revived', 0)} horizon-months "
            f"(by rung: {revived.get('by_rung') or {}})."]
    return "\n".join(out) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--window", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    window = args.window or _default_window(date.today())
    path = args.db[len("duckdb:///"):] if args.db.startswith("duckdb:///") else args.db
    rows: List[Dict[str, Any]] = []
    revived: Dict[str, Any] = {}
    try:
        import duckdb  # noqa: PLC0415

        con = duckdb.connect(path, read_only=True)
        try:
            rows = rung_table(con, window)
            revived = revived_questions(con)
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001
        print(f"::warning title=ACE/PA rung table::{type(exc).__name__}: {exc}")
    text = render(window, rows, revived)
    print(text)
    if args.out:
        try:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps({"window": window, "rows": rows, "revived": revived},
                                                 indent=2), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
