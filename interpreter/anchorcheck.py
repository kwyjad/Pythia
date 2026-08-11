# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""What every base-rate anchor in a database actually rests on.

    python -m interpreter.anchorcheck --db "$PYTHIA_DB_URL" [--include-test]

The v3 report marked 138 of 185 anchors thin at a twelve-observation cutoff.
A flag that fires on three-quarters of rows has stopped discriminating, and
the demotion rule it drives is doing nothing. That is a finding about the
anchor builder, not a threshold that needs lowering, so this reports the
counts BEFORE anyone touches the cutoff.

Per (hazard, metric) it prints the anchor method, how many observations the
anchors rest on, and what share fall under the cutoff. The three causes worth
telling apart, which is what the columns are for:

* a window that is too short — every anchor for a pair lands on the same
  small number, and that number is the window length;
* counting events where months are wanted — the count is far below the months
  of record available;
* dropping quiet months instead of counting them as observed zeros — the
  count is small AND the anchor puts almost no weight on the zero bucket,
  because every observation it saw was an event.

Read-only, and ``main()`` returns 0 in every outcome: a diagnostic must never
be the thing that fails a pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from interpreter import config, names

LOGGER = logging.getLogger(__name__)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    pool = sorted(values)
    if len(pool) == 1:
        return pool[0]
    import math

    pos = q * (len(pool) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(pool) - 1)
    return pool[lo] + (pos - lo) * (pool[hi] - pool[lo])


def anchor_rows(con, *, include_test: bool) -> list[dict[str, Any]]:
    """One row per (question, best-available aggregate) with its anchor."""
    test_clause = "" if include_test else " AND COALESCE(is_test, FALSE) = FALSE"
    order = " ".join(
        f"WHEN model_name = '{m}' THEN {i}"
        for i, m in enumerate(names.AGGREGATE_PREFERENCE)
    )
    sql = f"""
        WITH ranked AS (
            SELECT question_id, run_id, hazard_code, metric, baserate_n_obs,
                   baserate_source, baserate_json,
                   ROW_NUMBER() OVER (
                       PARTITION BY question_id
                       ORDER BY CASE {order} ELSE 99 END, model_name
                   ) AS rn
            FROM forecast_deviation
            WHERE 1 = 1{test_clause}
        )
        SELECT question_id, run_id, hazard_code, metric, baserate_n_obs,
               baserate_source, baserate_json
        FROM ranked WHERE rn = 1
    """
    out: list[dict[str, Any]] = []
    for row in con.execute(sql).fetchall():
        detail: dict[str, Any] = {}
        probs: list[float] = []
        try:
            blob = json.loads(row[6]) if row[6] else {}
            detail = blob.get("detail") or {}
            probs = [float(p) for p in (blob.get("probs") or [])]
        except (TypeError, ValueError):
            pass
        out.append({
            "question_id": row[0],
            "run_id": row[1],
            "hazard_code": row[2],
            "metric": row[3],
            "n_obs": row[4],
            "source": row[5],
            "method": detail.get("method"),
            "window_months": detail.get("window_months"),
            # How much weight the anchor puts on "nothing recorded". An anchor
            # built only from months something happened cannot put weight
            # there, which is the signature of dropped quiet months.
            "p_zero": probs[0] if probs else None,
        })
    return out


def summarise(rows: list[dict[str, Any]], *, thin_below: int) -> list[dict[str, Any]]:
    """Per (hazard, metric): the counts a decision about the cutoff needs."""
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (str(row.get("hazard_code")), str(row.get("metric"))), []
        ).append(row)
    out: list[dict[str, Any]] = []
    for (hazard, metric), group in sorted(groups.items()):
        obs = [float(r["n_obs"]) for r in group if r.get("n_obs") is not None]
        zeros = [float(r["p_zero"]) for r in group if r.get("p_zero") is not None]
        thin = [r for r in group if r.get("n_obs") is not None
                and int(r["n_obs"]) < thin_below]
        methods = sorted({str(r.get("method") or "-") for r in group})
        windows = sorted({
            str(r.get("window_months")) for r in group
            if r.get("window_months") is not None
        })
        out.append({
            "hazard_code": hazard,
            "metric": metric,
            "n_questions": len(group),
            "n_with_anchor": len(obs),
            "min_obs": min(obs) if obs else None,
            "median_obs": _percentile(obs, 0.5),
            "max_obs": max(obs) if obs else None,
            "n_thin": len(thin),
            "share_thin": (len(thin) / len(group)) if group else 0.0,
            "median_p_zero": _percentile(zeros, 0.5),
            "methods": ",".join(methods),
            "window_months": ",".join(windows) or "-",
        })
    return out


def _table(summary: list[dict[str, Any]]) -> str:
    header = (
        f"{'hazard':<8}{'metric':<22}{'qs':>5}{'anch':>6}{'min':>5}"
        f"{'med':>6}{'max':>6}{'thin':>6}{'%thin':>7}{'p(0)':>7}  window  method"
    )
    lines = [header, "-" * (len(header) + 10)]
    for row in summary:
        def _num(value: Any, places: int = 0) -> str:
            return "-" if value is None else f"{float(value):.{places}f}"

        lines.append(
            f"{row['hazard_code']:<8}{row['metric']:<22}"
            f"{row['n_questions']:>5}{row['n_with_anchor']:>6}"
            f"{_num(row['min_obs']):>5}{_num(row['median_obs']):>6}"
            f"{_num(row['max_obs']):>6}"
            f"{row['n_thin']:>6}{row['share_thin'] * 100:>6.0f}%"
            f"{_num(row['median_p_zero'], 2):>7}"
            f"  {row['window_months']:<7} {row['methods']}"
        )
    return "\n".join(lines)


def _diagnosis(summary: list[dict[str, Any]], *, thin_below: int) -> list[str]:
    """The plain reading of the table, so nobody has to infer it."""
    notes: list[str] = []
    for row in summary:
        pair = f"{row['hazard_code']}/{row['metric']}"
        if row["n_with_anchor"] == 0:
            notes.append(f"{pair}: no anchor carries an observation count.")
            continue
        if row["max_obs"] is not None and row["max_obs"] < thin_below:
            windows = [
                int(w) for w in str(row["window_months"] or "").split(",")
                if w.strip().isdigit()
            ]
            # A declared window shorter than the cutoff is ARITHMETIC: no
            # anchor for this pair can ever clear the flag, whatever the data
            # says. A pair with no declared window (the occurrence x severity
            # anchors) is a short RECORD, which is a different finding and
            # needs a different fix.
            if windows and max(windows) < thin_below:
                notes.append(
                    f"{pair}: EVERY anchor is thin, and the window is "
                    f"{max(windows)} months against a cutoff of {thin_below} "
                    "observations. This is arithmetic, not data: the flag can "
                    "never clear, whatever the record holds. Lengthen the "
                    "window."
                )
            else:
                notes.append(
                    f"{pair}: EVERY anchor is thin, and the largest rests on "
                    f"{row['max_obs']:.0f} observations. There is no declared "
                    "window here, so this is a short RECORD rather than a "
                    "short window. Check what the count is measuring before "
                    "changing anything: for the occurrence-times-severity "
                    "anchors it is the number of reported impact months, not "
                    "the occurrence evidence behind them."
                )
        elif row["share_thin"] >= 0.5:
            notes.append(
                f"{pair}: {row['share_thin'] * 100:.0f}% thin, median "
                f"{row['median_obs']:.0f} observations."
            )
        pz = row["median_p_zero"]
        if pz is not None and pz < 0.05 and row["metric"] != "EVENT_OCCURRENCE":
            notes.append(
                f"{pair}: the median anchor puts {pz * 100:.1f}% weight on "
                "nothing recorded. An anchor built only from months when "
                "something was reported cannot put weight there, so quiet "
                "months are being dropped rather than counted as observed "
                "zeros, and the anchor overstates the usual level."
            )
    return notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="DuckDB URL or path")
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--json-out", default=None,
                        help="Also write the summary as JSON here")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[anchorcheck] %(message)s")
    thin_below = config.baserate_min_obs()
    try:
        from resolver.db import duckdb_io

        con = duckdb_io.get_db(args.db or duckdb_io.DEFAULT_DB_URL)
        try:
            rows = anchor_rows(con, include_test=args.include_test)
        finally:
            duckdb_io.close_db(con)
    except Exception as exc:  # noqa: BLE001 - a diagnostic never fails a run
        LOGGER.error("failed: %s", exc)
        return 0

    if not rows:
        print("[anchorcheck] no forecast_deviation rows in this database")
        return 0
    summary = summarise(rows, thin_below=thin_below)
    n_thin = sum(r["n_thin"] for r in summary)
    print(
        f"[anchorcheck] {len(rows)} anchors, {n_thin} thin at the current "
        f"cutoff of {thin_below} observations "
        f"({n_thin / len(rows) * 100:.0f}%)"
    )
    print(_table(summary))
    notes = _diagnosis(summary, thin_below=thin_below)
    if notes:
        print("\n[anchorcheck] what the table says:")
        for note in notes:
            print(f"  - {note}")
    if args.json_out:
        try:
            from pathlib import Path

            Path(args.json_out).write_text(
                json.dumps({"thin_below": thin_below, "summary": summary},
                           indent=1, default=str),
                encoding="utf-8",
            )
        except OSError as exc:
            LOGGER.warning("could not write %s: %s", args.json_out, exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
