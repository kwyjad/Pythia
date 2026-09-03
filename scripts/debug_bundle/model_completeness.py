# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""One row per question, model and month: did the forecast actually land.

In September three models wrote 2,694 rows apiece while Opus wrote 2,624
and gpt-5.6-luna 2,659, and every health view reported 5/5 models OK on
every question. The calls had succeeded; three responses failed to parse
after being billed in full, and a member that produced no forecast is a
member that was not in the ensemble.

Row counts alone cannot show that, because the shortfall is spread across
questions and months. So the grain here is the cell the aggregation
actually consumes — (question, model, month) — with the bucket count
beside the count that metric's scheme expects. A month with the wrong
number of buckets is as absent as a month with none: the SPD chain
rejects wrong-length vectors outright.
"""

from __future__ import annotations

from typing import Any

FIELDNAMES = [
    "question_id", "iso3", "hazard_code", "metric", "track", "model_name",
    "month_index", "row_exists", "n_buckets", "n_buckets_expected",
    "status", "verdict",
]


def _expected_buckets(metric: str) -> int | None:
    try:
        from pythia.buckets import n_buckets_for  # noqa: PLC0415

        # EVENT_OCCURRENCE is deliberately absent from BUCKET_SPECS: the
        # binary writer stores P(yes) in bucket 1 and is independent of the
        # SPD schemes, so "expected" is not a number here — and n_buckets_for
        # answers 0 for it, which is not the same as six.
        return int(n_buckets_for(metric)) or None
    except Exception:
        return None


def _num_horizons() -> int:
    try:
        from pythia.buckets import NUM_HORIZONS  # noqa: PLC0415

        return int(NUM_HORIZONS)
    except Exception:
        return 6


def _rows(con, sql: str, params: list[Any]) -> list[dict[str, Any]]:
    cur = con.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def collect(
    con,
    *,
    run_id: str | None,
    questions: list[dict[str, Any]],
    expected_models: list[str],
    track2_model: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return (per-cell rows, rollup). Never raises."""

    rollup: dict[str, Any] = {
        "n_question_months_short": 0,
        "n_cells_expected": 0,
        "n_cells_missing": 0,
        "by_model": {},
        "short_question_months": [],
    }
    if not run_id or not questions:
        return [], rollup

    qids = [str(q.get("question_id")) for q in questions if q.get("question_id")]
    if not qids:
        return [], rollup
    ph = ",".join("?" for _ in qids)
    try:
        present = _rows(
            con,
            f"""
            SELECT question_id, model_name, month_index,
                   COUNT(*) AS n_buckets,
                   MIN(COALESCE(status,'')) AS status
            FROM forecasts_raw
            WHERE run_id = ? AND question_id IN ({ph})
            GROUP BY question_id, model_name, month_index
            """,
            [run_id, *qids],
        )
    except Exception:
        return [], rollup

    by_key: dict[tuple[str, str, Any], dict[str, Any]] = {}
    models_seen: dict[str, set[str]] = {}
    for r in present:
        qid = str(r["question_id"] or "")
        model = str(r["model_name"] or "")
        by_key[(qid, model, r["month_index"])] = r
        models_seen.setdefault(qid, set()).add(model)

    n_months = _num_horizons()
    out: list[dict[str, Any]] = []
    for q in sorted(questions, key=lambda x: str(x.get("question_id") or "")):
        qid = str(q.get("question_id") or "")
        if not qid:
            continue
        metric = str(q.get("metric") or "")
        track = q.get("track")
        try:
            track_int = int(track) if track is not None else None
        except Exception:
            track_int = None
        expected = [track2_model] if track_int == 2 else list(expected_models)
        # A model that wrote for this question but is not in the configured
        # lineup (an aggregate row, a leftover from a swapped member) is
        # still reported: it is evidence about what ran.
        for extra in sorted(models_seen.get(qid, set()) - set(expected)):
            if extra and not extra.startswith("ensemble_"):
                expected.append(extra)
        n_expected = _expected_buckets(metric)
        for model in expected:
            for month in range(1, n_months + 1):
                cell = by_key.get((qid, model, month))
                n_buckets = int(cell["n_buckets"]) if cell else 0
                status = str(cell.get("status") or "") if cell else ""
                if cell is None:
                    verdict = "missing"
                elif status and status != "ok":
                    verdict = f"not_ok:{status}"
                elif n_expected is not None and n_buckets != n_expected:
                    verdict = "wrong_bucket_count"
                else:
                    verdict = "ok"
                out.append(
                    {
                        "question_id": qid,
                        "iso3": q.get("iso3") or "",
                        "hazard_code": q.get("hazard_code") or "",
                        "metric": metric,
                        "track": track_int if track_int is not None else "",
                        "model_name": model,
                        "month_index": month,
                        "row_exists": bool(cell),
                        "n_buckets": n_buckets,
                        "n_buckets_expected": n_expected if n_expected is not None else "",
                        "status": status,
                        "verdict": verdict,
                    }
                )

    rollup = _rollup(out, n_months)
    return out, rollup


def _rollup(rows: list[dict[str, Any]], n_months: int) -> dict[str, Any]:
    """Which question-months were aggregated from fewer members than expected."""

    per_qm: dict[tuple[str, int], dict[str, Any]] = {}
    by_model: dict[str, int] = {}
    n_expected_cells = 0
    n_missing_cells = 0
    for r in rows:
        n_expected_cells += 1
        key = (str(r["question_id"]), int(r["month_index"]))
        entry = per_qm.setdefault(
            key,
            {
                "question_id": r["question_id"],
                "iso3": r["iso3"],
                "hazard_code": r["hazard_code"],
                "metric": r["metric"],
                "month_index": r["month_index"],
                "n_expected": 0,
                "n_ok": 0,
                "missing_models": [],
            },
        )
        entry["n_expected"] += 1
        if r["verdict"] == "ok":
            entry["n_ok"] += 1
        else:
            entry["missing_models"].append(f"{r['model_name']}({r['verdict']})")
            by_model[str(r["model_name"])] = by_model.get(str(r["model_name"]), 0) + 1
            n_missing_cells += 1

    short = [e for e in per_qm.values() if e["n_ok"] < e["n_expected"]]
    short.sort(key=lambda e: (e["n_ok"] - e["n_expected"], str(e["question_id"])))
    return {
        "n_question_months_short": len(short),
        "n_question_months": len(per_qm),
        "n_cells_expected": n_expected_cells,
        "n_cells_missing": n_missing_cells,
        "by_model": by_model,
        "short_question_months": short,
    }
