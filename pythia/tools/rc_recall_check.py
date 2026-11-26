# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Does a regime-change flag catch the breaks that happen? Production vs shadow.

For each production HS run with RC assessments, and each assessed
(country, hazard) pair, whether or not it reached Track 1, this asks the
resolver whether a PATTERN BREAK happened in the RC window (the six months a
question opened by that run would forecast: the month after the run and the
five after it), and scores the RC flag against it, for production
(``hs_triage``) and the shadow arm (``hs_rc_shadow``) separately.

**The break definition** (``BREAK_CONFIG``, the one place its thresholds live):
the hazard's headline series is read for each window month; a month BREAKS
when its bucket sits ``bins_above_median`` (2) or more buckets above the
median bucket of the pair's base-rate distribution
(``pythia.tools.base_rate_spd.base_rate_spd`` as of the window's first month,
the same anchor the forecaster is shown and the climatology reference is
scored on). The pair breaks when any resolved month does.

* ACE: ``FATALITIES`` from ``acled_monthly_fatalities`` (all event types,
  complete months only; a month ACLED published, for a country ACLED covers,
  with no row is a zero, as the base rate counts it).
* FL, TC: ``PA`` from the PA resolution machine (``haz_resolutions``,
  RESOLVED_VALUE / RESOLVED_ZERO), which resolves FL/PA and TC/PA questions.
* DR: ``PHASE3PLUS_IN_NEED`` from ``facts_resolved`` (FEWS NET / IPC current
  situation), the series DR questions resolve on.

**Only resolved windows are scored.** A pair with a breaking month is a
break whatever else is missing. A pair with no break is scored ``no break``
only when at least ``min_resolved_months`` (3) of its window months resolved;
otherwise it is ``unresolvable`` and is left out of every denominator, never
read as "no break".

**Output**, per run and pooled, per hazard, per arm (production, shadow,
shadow_replay): pairs, resolvable, breaks, flagged (level >= 1), hits,
recall of breaks at level >= 1, precision at level >= 1, the Brier score of
the RC likelihood against the break indicator, and how many pairs each arm
would have sent to Track 1 (level >= 1, all pairs). Every rate is printed
beside its counts; the break counts are small. Written to ``rc_recall_check``
(DuckDB) and a markdown report.

**Replay** (``--replay``, capped by ``--replay-max-usd``, default 10): for
production runs with no live shadow rows, the stored pass-1 RC prompts in
``llm_calls`` are sent to the shadow model and stored ``is_replay = TRUE``.
A model trained after the run date may have seen the outcome, so replay
scores are optimistic; the report says so beside them.

    python -m pythia.tools.rc_recall_check --db-url duckdb:///data/resolver.duckdb \\
        --out diagnostics/rc_recall_check.md [--replay --replay-max-usd 10]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

BREAK_CONFIG: Dict[str, Any] = {
    # hazard -> the metric whose series decides a break
    "metric_by_hazard": {"ACE": "FATALITIES", "FL": "PA", "TC": "PA", "DR": "PHASE3PLUS_IN_NEED"},
    # a month breaks at this many buckets or more above the base-rate median
    "bins_above_median": 2,
    # a pair with no break needs this many resolved window months to count
    "min_resolved_months": 3,
    # the RC flag scored: level >= this
    "flag_level": 1,
    "window_months": 6,
}

ARMS = ("production", "shadow", "shadow_replay")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _has_table(con, name: str) -> bool:
    try:
        return bool(con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [name]
        ).fetchone()[0])
    except Exception:  # noqa: BLE001
        return False


def _has_column(con, table: str, column: str) -> bool:
    try:
        return bool(con.execute(
            "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = ? AND column_name = ?",
            [table, column],
        ).fetchone()[0])
    except Exception:  # noqa: BLE001
        return False


def _add_months(ym: str, n: int) -> str:
    y, m = int(ym[:4]), int(ym[5:7])
    idx = y * 12 + (m - 1) + n
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def run_window(run_date: Any) -> List[str]:
    """The six window months a run's questions forecast: the month after
    the run's month, and the five after it."""

    if isinstance(run_date, str):
        run_date = datetime.fromisoformat(run_date[:19])
    ym = f"{run_date.year:04d}-{run_date.month:02d}"
    first = _add_months(ym, 1)
    return [_add_months(first, i) for i in range(int(BREAK_CONFIG["window_months"]))]


def median_bucket(probs: List[float]) -> Optional[int]:
    """0-based bucket where the cumulative probability first reaches 0.5."""

    if not probs:
        return None
    total = 0.0
    for i, p in enumerate(probs):
        total += float(p)
        if total >= 0.5 - 1e-9:
            return i
    return len(probs) - 1


def bucket_of(value: float, metric: str) -> Optional[int]:
    from pythia.tools.base_rate_spd import _bucket_index_for_value

    return _bucket_index_for_value(value, metric)


# ---------------------------------------------------------------------------
# What actually happened
# ---------------------------------------------------------------------------


class Outcomes:
    """Reads the realised value of a (country, hazard, month), or None."""

    def __init__(self, con) -> None:
        self.con = con
        self._acled_live: Optional[set] = None
        self._acled_countries: Optional[set] = None

    def value(self, iso3: str, hazard: str, ym: str) -> Optional[float]:
        try:
            if hazard == "ACE":
                return self._acled(iso3, ym)
            if hazard in ("FL", "TC"):
                return self._machine(iso3, hazard, ym)
            if hazard == "DR":
                return self._phase3(iso3, ym)
        except Exception as exc:  # noqa: BLE001
            logger.debug("outcome read failed %s %s %s: %s", iso3, hazard, ym, exc)
        return None

    def _acled(self, iso3: str, ym: str) -> Optional[float]:
        if not _has_table(self.con, "acled_monthly_fatalities"):
            return None
        from pythia.tools.base_rate_spd import acled_complete_month_clause

        complete = acled_complete_month_clause(self.con)
        if self._acled_live is None:
            rows = self.con.execute(
                f"SELECT DISTINCT substr(CAST(month AS VARCHAR),1,7), iso3 "
                f"FROM acled_monthly_fatalities WHERE {complete}"
            ).fetchall()
            self._acled_live = {r[0] for r in rows}
            self._acled_countries = {(r[1] or "").upper() for r in rows}
        row = self.con.execute(
            f"""SELECT SUM(fatalities) FROM acled_monthly_fatalities
                WHERE iso3 = ? AND substr(CAST(month AS VARCHAR),1,7) = ? AND {complete}""",
            [iso3, ym],
        ).fetchone()
        if row and row[0] is not None:
            return float(row[0])
        if ym in (self._acled_live or set()) and iso3 in (self._acled_countries or set()):
            return 0.0
        return None

    def _machine(self, iso3: str, hazard: str, ym: str) -> Optional[float]:
        if not _has_table(self.con, "haz_resolutions"):
            return None
        row = self.con.execute(
            """SELECT status, value FROM haz_resolutions
               WHERE iso3 = ? AND hazard = ? AND year = ? AND month = ?""",
            [iso3, hazard, int(ym[:4]), int(ym[5:7])],
        ).fetchone()
        if not row:
            return None
        if row[0] == "RESOLVED_ZERO":
            return 0.0
        if row[0] == "RESOLVED_VALUE" and row[1] is not None:
            return float(row[1])
        return None

    def _phase3(self, iso3: str, ym: str) -> Optional[float]:
        if not _has_table(self.con, "facts_resolved"):
            return None
        row = self.con.execute(
            """SELECT MAX(value) FROM facts_resolved
               WHERE iso3 = ? AND LOWER(metric) = 'phase3plus_in_need' AND ym = ?""",
            [iso3, ym],
        ).fetchone()
        return float(row[0]) if row and row[0] is not None else None


def pair_outcome(con, outcomes: Outcomes, iso3: str, hazard: str, window: List[str]) -> Dict[str, Any]:
    """{'state': break | no_break | unresolvable, ...} for one pair's window."""

    from pythia.tools.base_rate_spd import base_rate_spd

    metric = BREAK_CONFIG["metric_by_hazard"].get(hazard)
    if not metric:
        return {"state": "unresolvable", "reason": "no break series for this hazard"}
    probs, source, _detail = base_rate_spd(con, iso3, hazard, metric, window[0])
    med = median_bucket(probs)
    if med is None:
        return {"state": "unresolvable", "reason": f"no base rate ({source})"}
    threshold = med + int(BREAK_CONFIG["bins_above_median"])
    resolved = 0
    breaking: List[str] = []
    for ym in window:
        v = outcomes.value(iso3, hazard, ym)
        if v is None:
            continue
        resolved += 1
        b = bucket_of(v, metric)
        if b is not None and b >= threshold:
            breaking.append(ym)
    if breaking:
        return {"state": "break", "months": breaking, "resolved": resolved, "median_bucket": med}
    if resolved >= int(BREAK_CONFIG["min_resolved_months"]):
        return {"state": "no_break", "resolved": resolved, "median_bucket": med}
    return {"state": "unresolvable", "resolved": resolved,
            "reason": f"{resolved} of {len(window)} months resolved"}


# ---------------------------------------------------------------------------
# What each arm said
# ---------------------------------------------------------------------------


def production_runs(con, include_test: bool = False) -> List[Tuple[str, Any]]:
    if not _has_table(con, "hs_runs"):
        return []
    test = "" if include_test or not _has_column(con, "hs_runs", "is_test") else \
        "WHERE COALESCE(is_test, FALSE) = FALSE"
    return [(r[0], r[1]) for r in con.execute(
        f"SELECT hs_run_id, generated_at FROM hs_runs {test} ORDER BY generated_at"
    ).fetchall() if r[0] and r[1] is not None]


def production_assessments(con, hs_run_id: str) -> List[Dict[str, Any]]:
    if not _has_table(con, "hs_triage"):
        return []
    skip = ""
    if _has_column(con, "hs_triage", "data_quality_json"):
        skip = ("AND COALESCE(json_extract_string(data_quality_json, '$.status'), '') "
                "<> 'seasonal_skip'")
    rows = con.execute(
        f"""SELECT iso3, hazard_code, regime_change_likelihood, regime_change_level
            FROM hs_triage WHERE run_id = ? AND hazard_code IN ('ACE','DR','FL','TC') {skip}""",
        [hs_run_id],
    ).fetchall()
    return [{"iso3": (r[0] or "").upper(), "hazard": (r[1] or "").upper(),
             "likelihood": r[2], "level": r[3]} for r in rows]


def shadow_assessments(con, hs_run_id: str, replay: bool) -> List[Dict[str, Any]]:
    if not _has_table(con, "hs_rc_shadow"):
        return []
    rows = con.execute(
        """SELECT iso3, hazard_code, likelihood, level FROM hs_rc_shadow
           WHERE hs_run_id = ? AND status = 'ok' AND COALESCE(is_replay, FALSE) = ?""",
        [hs_run_id, bool(replay)],
    ).fetchall()
    return [{"iso3": (r[0] or "").upper(), "hazard": (r[1] or "").upper(),
             "likelihood": r[2], "level": r[3]} for r in rows]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score(assessments: Iterable[Dict[str, Any]], outcomes: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    """Counts and rates for one arm (and one hazard, if pre-filtered)."""

    flag = int(BREAK_CONFIG["flag_level"])
    n_pairs = n_res = n_breaks = n_flagged = n_hits = n_track1 = 0
    brier_sum = 0.0
    n_brier = 0
    for a in assessments:
        n_pairs += 1
        lvl = a.get("level")
        flagged = lvl is not None and int(lvl) >= flag
        if flagged:
            n_track1 += 1
        out = outcomes.get((a["iso3"], a["hazard"])) or {"state": "unresolvable"}
        if out["state"] == "unresolvable":
            continue
        n_res += 1
        broke = out["state"] == "break"
        n_breaks += int(broke)
        n_flagged += int(flagged)
        n_hits += int(flagged and broke)
        if a.get("likelihood") is not None:
            brier_sum += (float(a["likelihood"]) - (1.0 if broke else 0.0)) ** 2
            n_brier += 1
    return {
        "n_pairs": n_pairs,
        "n_resolvable": n_res,
        "n_unresolvable": n_pairs - n_res,
        "n_breaks": n_breaks,
        "n_flagged": n_flagged,
        "n_hits": n_hits,
        "recall": round(n_hits / n_breaks, 3) if n_breaks else None,
        "precision": round(n_hits / n_flagged, 3) if n_flagged else None,
        "brier": round(brier_sum / n_brier, 4) if n_brier else None,
        "n_track1": n_track1,
    }


def compute(con, *, include_test: bool = False) -> List[Dict[str, Any]]:
    """Result rows: per run and pooled ('*'), per hazard and all ('*'), per arm."""

    outcomes_reader = Outcomes(con)
    pooled: Dict[Tuple[str, str], List[Tuple[Dict[str, Any], Dict]]] = {}
    rows: List[Dict[str, Any]] = []
    for hs_run_id, generated_at in production_runs(con, include_test=include_test):
        window = run_window(generated_at)
        arms = {
            "production": production_assessments(con, hs_run_id),
            "shadow": shadow_assessments(con, hs_run_id, replay=False),
            "shadow_replay": shadow_assessments(con, hs_run_id, replay=True),
        }
        if not arms["production"]:
            continue
        pairs = {(a["iso3"], a["hazard"]) for arm in arms.values() for a in arm}
        outcomes = {p: pair_outcome(con, outcomes_reader, p[0], p[1], window) for p in pairs}
        for arm, assessments in arms.items():
            if not assessments:
                continue
            for hz in ("*", "ACE", "DR", "FL", "TC"):
                subset = [a for a in assessments if hz == "*" or a["hazard"] == hz]
                if not subset:
                    continue
                rows.append({"hs_run_id": hs_run_id, "window_start": window[0], "hazard": hz,
                             "arm": arm, **score(subset, outcomes)})
                pooled.setdefault((arm, hz), []).extend((a, outcomes) for a in subset)
    for (arm, hz), items in sorted(pooled.items()):
        merged = {}
        assessments = []
        for i, (a, outs) in enumerate(items):
            key = (f"{i}", a["hazard"])
            merged[key] = outs.get((a["iso3"], a["hazard"])) or {"state": "unresolvable"}
            assessments.append({**a, "iso3": f"{i}"})
        rows.append({"hs_run_id": "*", "window_start": "*", "hazard": hz, "arm": arm,
                     **score(assessments, merged)})
    return rows


_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS rc_recall_check (
    hs_run_id TEXT, window_start TEXT, hazard TEXT, arm TEXT,
    n_pairs INTEGER, n_resolvable INTEGER, n_unresolvable INTEGER,
    n_breaks INTEGER, n_flagged INTEGER, n_hits INTEGER,
    recall DOUBLE, precision DOUBLE, brier DOUBLE, n_track1 INTEGER,
    is_replay BOOLEAN, break_config_json TEXT, computed_at TIMESTAMP
)
"""


def write(con, rows: List[Dict[str, Any]]) -> None:
    con.execute(_TABLE_DDL)
    con.execute("DELETE FROM rc_recall_check")
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    cfg = json.dumps(BREAK_CONFIG, sort_keys=True)
    for r in rows:
        con.execute(
            """INSERT INTO rc_recall_check VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [r["hs_run_id"], r["window_start"], r["hazard"], r["arm"], r["n_pairs"],
             r["n_resolvable"], r["n_unresolvable"], r["n_breaks"], r["n_flagged"], r["n_hits"],
             r["recall"], r["precision"], r["brier"], r["n_track1"],
             r["arm"] == "shadow_replay", cfg, now],
        )


def _rate(value: Optional[float], num: int, den: int) -> str:
    return "n/a (0)" if value is None else f"{value:.2f} ({num}/{den})"


def markdown(rows: List[Dict[str, Any]]) -> str:
    cfg = BREAK_CONFIG
    lines = [
        "# RC recall check: production against the shadow arm", "",
        f"A pair breaks when a window month resolves {cfg['bins_above_median']} or more buckets above "
        "the median of its base-rate distribution (ACE conflict deaths, FL/TC people affected from the "
        "resolution machine, DR Phase 3+). A pair with no break needs "
        f"{cfg['min_resolved_months']} resolved months to count; otherwise it is unresolvable and left "
        f"out of every rate. Flag = RC level {cfg['flag_level']} or above.", "",
        "**Replay rows are optimistic**: the shadow model may have been trained after the run and seen "
        "the outcome.", "",
    ]
    def table(sel: List[Dict[str, Any]]) -> List[str]:
        out = ["| run | hazard | arm | pairs | resolvable | breaks | recall | precision | Brier | to Track 1 |",
               "|---|---|---|--:|--:|--:|---|---|--:|--:|"]
        for r in sel:
            out.append(
                f"| {r['hs_run_id']} | {r['hazard']} | {r['arm']} | {r['n_pairs']} | {r['n_resolvable']} "
                f"| {r['n_breaks']} | {_rate(r['recall'], r['n_hits'], r['n_breaks'])} "
                f"| {_rate(r['precision'], r['n_hits'], r['n_flagged'])} "
                f"| {'' if r['brier'] is None else r['brier']} | {r['n_track1']} |"
            )
        return out
    pooled = [r for r in rows if r["hs_run_id"] == "*"]
    lines += ["## Pooled", ""] + (table(pooled) if pooled else ["No runs scored yet."]) + [""]
    per_run = [r for r in rows if r["hs_run_id"] != "*" and r["hazard"] == "*"]
    lines += ["## Per run (all hazards)", ""] + (table(per_run) if per_run else ["None."]) + [""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def replay(con, *, max_usd: float = 10.0, include_test: bool = False) -> Dict[str, Any]:
    """Send stored pass-1 RC prompts of runs with no live shadow rows to the
    shadow model; store is_replay = TRUE. Bounded by ``max_usd``."""

    from horizon_scanner import rc_shadow

    report = {"runs": 0, "calls": 0, "skipped_cap": 0, "spent_usd": 0.0}
    if not _has_table(con, "llm_calls"):
        return report
    os.environ["PYTHIA_RC_SHADOW_MAX_USD"] = str(max_usd)
    rc_shadow.reset()
    spec = rc_shadow.shadow_spec()
    for hs_run_id, _gen in production_runs(con, include_test=include_test):
        if shadow_assessments(con, hs_run_id, replay=False) or shadow_assessments(con, hs_run_id, replay=True):
            continue
        prompts = con.execute(
            """SELECT iso3, hazard_code, prompt_text FROM llm_calls
               WHERE hs_run_id = ? AND call_type = 'rc_pass_1' AND prompt_text IS NOT NULL
                 AND COALESCE(error_text, '') = ''""",
            [hs_run_id],
        ).fetchall()
        if not prompts:
            continue
        report["runs"] += 1
        for iso3, label, prompt in prompts:
            hz = (label or "").upper().replace("RC_", "").split("_PASS_")[0]
            est = rc_shadow.estimate_cost(prompt, spec.model_id)
            if not rc_shadow._reserve(est):
                report["skipped_cap"] += 1
                continue
            from forecaster.providers import call_chat_ms, estimate_cost_usd
            import asyncio

            try:
                text, usage, error = asyncio.run(call_chat_ms(
                    spec, prompt, temperature=0.0, prompt_key="hs.regime_change.shadow_replay",
                    component="HorizonScanner", run_id=hs_run_id, log_call=False))
            except Exception as exc:  # noqa: BLE001
                text, usage, error = "", {}, f"provider call error: {exc}"
            usage = dict(usage or {})
            cost = float(usage.get("cost_usd") or estimate_cost_usd(spec.model_id, usage) or 0.0)
            rc_shadow._settle(est, cost)
            result = {"status": "error", "error_text": str(error)[:300]} if error else rc_shadow.parse_answer(text)
            rc_shadow.write_row(con, hs_run_id=hs_run_id, iso3=iso3 or "", hazard_code=hz,
                                provider=spec.provider, model_id=spec.model_id, result=result,
                                cost_usd=cost, source="replay", is_replay=True)
            report["calls"] += 1
    report["spent_usd"] = round(rc_shadow.stats()["spent"], 4)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db-url", default=os.getenv("PYTHIA_DB_URL", ""))
    ap.add_argument("--out", default="diagnostics/rc_recall_check.md")
    ap.add_argument("--include-test", action="store_true")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--replay-max-usd", type=float, default=10.0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    import duckdb

    path = args.db_url.replace("duckdb:///", "") if args.db_url else "data/resolver.duckdb"
    con = duckdb.connect(path)
    try:
        if args.replay:
            rep = replay(con, max_usd=args.replay_max_usd, include_test=args.include_test)
            print(f"[rc_recall] replay: {rep}", flush=True)
        rows = compute(con, include_test=args.include_test)
        write(con, rows)
    finally:
        con.close()
    text = markdown(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump({"break_config": BREAK_CONFIG, "rows": rows}, fh, indent=2, default=str)
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
