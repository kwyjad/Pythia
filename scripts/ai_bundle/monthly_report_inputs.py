# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""``monthly_report_inputs.json``: the figures the monthly performance report quotes.

A scheduled Claude session writes the monthly report from the scored bundle,
reading it through the Google Drive connector. The connector reads text files
and cannot open the 3 MB ``scores_flat.csv`` comfortably, and the report's
rule is "quote, do not recompute". So every figure the report needs beyond
``headline.json`` is computed here, once, with the bundle's own rules:

* read-only; every section degrades to ``{"stub": true, "reason": ...}``;
* intervals resample whole QUESTIONS (``error_attribution._cluster_*``,
  seeded with ``error_attribution.SEED``), 2,000 resamples;
* binary and SPD scores are never blended;
* one run per question (the context's forecast of record);
* Track 1 is never compared with Track 2.

It also writes ``case_studies_new/<question_id>.json``: the full records of
the worst and best NEW (question, horizon) pairs, capped at 2 MB in all.

Sections: ``round``, ``forecast_vintage``, ``new_this_round``,
``paired_comparisons``, ``crisiswatch_split``, ``gdacs_prompt_checks``,
``confounded_experiments``, ``case_candidates``, ``case_studies_new``.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence  # noqa: F401

from scripts.ai_bundle import error_attribution as _err
from scripts.ai_bundle.common import column_exists, table_exists, write_json

LOGGER = logging.getLogger(__name__)

FILE_NAME = "monthly_report_inputs.json"
CASE_DIR = "case_studies_new"
SCHEMA_VERSION = 1

#: Resamples for every interval here: enough for a 90% interval on a few dozen
#: questions, and the brief's figure.
BOOTSTRAP = 2000
#: Candidates per family and side, and how many of those get a full record.
N_CANDIDATES = 5
N_CASE_RECORDS = 3
CASE_BYTES_CAP = 2 * 1024 * 1024

#: The flags a forecast run is described by; one value across the whole scored
#: set means the method has not yet reached the scored questions.
VINTAGE_FLAGS = (
    "lineup_id", "base_rate_block_version", "rc_guidance", "advice_arm",
    "recalibration_mode", "input_partial_month",
)
#: Flag pairs where one is DEFINED from the other: identical arms are expected.
DERIVED_FLAG_PAIRS = frozenset({frozenset({"base_rate_block_version", "base_rate_block_group"})})

GDACS_HAZARDS = ("DR", "FL", "TC")
CLIMATOLOGY = "__ext_climatology"
PERSISTENCE = "__ext_persistence"
SIBYL = "sibyl"
BAYESMC = "ensemble_bayesmc_v2"
MEAN = "ensemble_mean_v2"


def _stub(reason: str) -> dict[str, Any]:
    return {"stub": True, "reason": reason}


def _family_score_type(family: str) -> str:
    """The score a family is ranked and compared on: RPS (stored as ``crps``)
    for SPD, Brier for binary."""
    return "brier" if family == "binary" else "crps"


# ---------------------------------------------------------------------------
# Per-question facts, gathered while the bundle streams its records
# ---------------------------------------------------------------------------


def question_facts(record: Mapping[str, Any]) -> dict[str, Any]:
    """The few fields of one question record this module needs.

    Called inside the bundle's record loop so the records themselves can be
    released; the GDACS block is parsed here from the STORED prompt, never
    rebuilt.
    """
    inj = record.get("inject_status") or {}
    cw = inj.get("crisiswatch") or {}
    gh = inj.get("gdacs_history") or {}
    block = None
    try:
        from forecaster.gdacs_history import parse_gdacs_prompt_block  # noqa: PLC0415

        parsed = parse_gdacs_prompt_block(record.get("spd_prompt"))
        if parsed is not None:
            block = {k: parsed.get(k) for k in (
                "total_months", "event_months", "history_unavailable", "repeated_months",
                "timeline_gaps", "format")}
    except Exception as exc:  # noqa: BLE001 - a parse never costs the bundle
        block = {"error": f"{type(exc).__name__}: {exc}"}
    resolutions = {}
    for r in (record.get("outcome") or {}).get("resolutions") or []:
        if r.get("horizon_m") is None:
            continue
        resolutions[int(r["horizon_m"])] = {
            k: r.get(k) for k in ("value", "source_desc", "resolution_reading",
                                  "observed_month", "scoring_class")
        }
    return {
        "has_prompt": record.get("spd_prompt") is not None,
        "gdacs_block": block,
        "gdacs_event_months_db": gh.get("event_months"),
        "crisiswatch": {
            "applicable": cw.get("applicable"), "available": cw.get("available"),
            "edition": cw.get("edition"), "edition_age_months": cw.get("edition_age_months"),
        },
        "resolutions": resolutions,
    }


# ---------------------------------------------------------------------------
# 1. round
# ---------------------------------------------------------------------------


def _git_sha() -> str | None:
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10, cwd=Path(__file__).resolve().parents[2])
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def latest_snapshot_date(con) -> date | None:
    """The newest ``source_snapshot_ym`` date among resolutions: the day the
    round's Resolver Update read its sources."""
    if not table_exists(con, "resolutions") or not column_exists(con, "resolutions", "source_snapshot_ym"):
        return None
    row = con.execute(
        "SELECT MAX(substr(source_snapshot_ym, 1, 10)) FROM resolutions "
        "WHERE source_snapshot_ym IS NOT NULL AND length(source_snapshot_ym) >= 10"
    ).fetchone()
    return _err._as_date(row[0]) if row and row[0] else None


def build_round(con, round_date: date | None, calibration_run_id: str | None) -> dict[str, Any]:
    derived = latest_snapshot_date(con)
    chosen = round_date or derived
    return {
        "round_date": chosen.isoformat() if chosen else None,
        "round_date_source": ("--round-date" if round_date else
                              "latest resolutions.source_snapshot_ym" if derived else None),
        "latest_source_snapshot_date": derived.isoformat() if derived else None,
        "calibration_run_id": calibration_run_id or os.environ.get("GITHUB_RUN_ID") or None,
        "git_sha": _git_sha(),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# 2. forecast_vintage
# ---------------------------------------------------------------------------


def run_date_from_id(run_id: Any) -> str | None:
    """``fc_<epoch>`` to the UTC date it was minted (the forecast date)."""
    s = str(run_id or "")
    if not s.startswith("fc_"):
        return None
    try:
        return datetime.fromtimestamp(int(s[3:]), tz=timezone.utc).date().isoformat()
    except (ValueError, OverflowError, OSError):
        return None


def _value(v: Any) -> str:
    return "null" if v is None else str(v)


def build_forecast_vintage(ctx: Any) -> dict[str, Any]:
    by_run: dict[str, list[str]] = defaultdict(list)
    for qid in ctx.bundle_qids:
        by_run[str(ctx.qmeta[qid].get("run_id"))].append(qid)
    runs = []
    for run, qids in sorted(by_run.items()):
        counts = {f: dict(sorted(Counter(_value(ctx.qmeta[q].get(f)) for q in qids).items()))
                  for f in VINTAGE_FLAGS}
        runs.append({"latest_run_id": None if run == "None" else run,
                     "run_date": run_date_from_id(run), "n_questions": len(qids),
                     "flags": counts})
    single = []
    for f in VINTAGE_FLAGS:
        vals = {_value(ctx.qmeta[q].get(f)) for q in ctx.bundle_qids}
        if len(vals) == 1:
            single.append({"flag": f, "only_value": next(iter(vals)),
                           "note": "not yet in scope: every scored question carries one value"})
    return {"runs": runs, "single_value_flags": single,
            "n_scored_questions": len(ctx.bundle_qids)}


# ---------------------------------------------------------------------------
# 3. new_this_round
# ---------------------------------------------------------------------------


def resolution_dates(con, qids: Sequence[str]) -> dict[tuple[str, int], dict[str, Any]]:
    """{(question_id, horizon_m): {"snapshot", "first_recorded", "first_value", "value"}}."""
    if not qids or not table_exists(con, "resolutions"):
        return {}
    snap = "substr(r.source_snapshot_ym, 1, 10)" if column_exists(
        con, "resolutions", "source_snapshot_ym") else "NULL"
    has_v = table_exists(con, "resolution_vintages") and column_exists(
        con, "resolution_vintages", "recorded_at")
    join = (" LEFT JOIN resolution_vintages v ON v.question_id = r.question_id "
            "AND v.horizon_m = r.horizon_m AND v.vintage = 'first'") if has_v else ""
    first = "CAST(v.recorded_at AS DATE), v.value" if has_v else "NULL, NULL"
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for qid, h, s, fr, fv, val in con.execute(
        f"SELECT r.question_id, r.horizon_m, {snap}, {first}, r.value FROM resolutions r{join} "
        "WHERE r.question_id IN (SELECT UNNEST(?::VARCHAR[]))", [list(qids)],
    ).fetchall():
        if h is None:
            continue
        out[(str(qid), int(h))] = {"snapshot": _err._as_date(s), "first_recorded": _err._as_date(fr),
                                   "first_value": fv, "value": val}
    return out


def classify_new(dates: Mapping[str, Any], round_date: date) -> tuple[str, bool]:
    """('new' | 'earlier' | 'undated', rewritten).

    The snapshot date of the source row the resolution was read from decides;
    a zero-default carries none, so its first recorded vintage stands in. A
    new resolution whose first vintage predates the round was read before and
    re-read now: ``rewritten``.
    """
    written = dates.get("snapshot") or dates.get("first_recorded")
    if written is None:
        return "undated", False
    if written < round_date:
        return "earlier", False
    first = dates.get("first_recorded")
    return "new", bool(first is not None and first < round_date)


def build_new_this_round(ctx: Any, round_date: date | None,
                         dates: Mapping[tuple[str, int], Mapping[str, Any]]) -> tuple[dict[str, Any], set]:
    if round_date is None:
        return _stub("no round date: no resolution carries a source_snapshot_ym and none was given"), set()
    bundle = set(ctx.bundle_qids)
    groups: dict[tuple, dict[str, Any]] = defaultdict(lambda: {
        "questions_new": set(), "horizon_months_new": 0, "horizon_months_earlier": 0,
        "horizon_months_undated": 0, "new_indicative": 0, "new_rewritten": 0})
    new_pairs: set[tuple[str, int]] = set()
    rewritten: list[dict[str, Any]] = []
    for (qid, h), d in sorted(dates.items()):
        if qid not in bundle:
            continue
        meta = ctx.qmeta[qid]
        g = groups[(meta["hazard_code"], meta["metric"], meta.get("track"))]
        kind, rew = classify_new(d, round_date)
        g[f"horizon_months_{kind}"] += 1
        if kind != "new":
            continue
        new_pairs.add((qid, h))
        g["questions_new"].add(qid)
        if (qid, h) in ctx.indicative:
            g["new_indicative"] += 1
        if rew:
            g["new_rewritten"] += 1
            rewritten.append({"question_id": qid, "horizon_m": h, "rewritten": True,
                              "first_recorded": str(d.get("first_recorded")),
                              "first_value": d.get("first_value"), "value": d.get("value")})
    rows = []
    for (hz, metric, track), g in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        rows.append({"hazard_code": hz, "metric": metric, "track": track,
                     "score_family": _err._family(metric),
                     "questions_new": len(g["questions_new"]),
                     **{k: g[k] for k in ("horizon_months_new", "horizon_months_earlier",
                                          "horizon_months_undated", "new_indicative",
                                          "new_rewritten")}})
    return {"round_date": round_date.isoformat(),
            "rule": ("new = the resolution's source_snapshot_ym (else, for a zero-default, its first "
                     "resolution_vintages reading) is on or after the round date; rewritten = new, "
                     "with a first reading before the round"),
            "groups": rows, "rewritten": rewritten}, new_pairs


# ---------------------------------------------------------------------------
# 4. paired_comparisons (Track 1 only)
# ---------------------------------------------------------------------------


def _paired(ctx: Any, qids: Iterable[str], a: Callable[[str], str | None],
            b: Callable[[str], str | None], score_type: str) -> dict[str, list[float]]:
    """{question_id: [a - b per horizon]} on the same (question, horizon),
    indicative months excluded."""
    by_q: dict[str, list[float]] = defaultdict(list)
    for qid in qids:
        ma, mb = a(qid), b(qid)
        if not ma or not mb or ma == mb:
            continue
        for h in range(1, 7):
            if (qid, h) in ctx.indicative:
                continue
            va = ctx.scores.get((qid, ma, h, score_type))
            vb = ctx.scores.get((qid, mb, h, score_type))
            if va is not None and vb is not None:
                by_q[qid].append(va - vb)
    return by_q


def _entry(ctx: Any, name: str, a_label: str, b_label: str, family: str, score_type: str,
           by_q: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    n = len([q for q, v in by_q.items() if v])
    d, lo, hi = _err._cluster_mean_ci(by_q, n_boot=BOOTSTRAP, seed=_err.SEED)
    tracks = sorted({str(ctx.qmeta[q].get("track")) for q in by_q})
    return {"comparison": name, "a": a_label, "b": b_label, "score_family": family,
            "score_type": score_type, "track": 1, "tracks_seen": tracks,
            "n_paired_questions": n, "n_pairs": sum(len(v) for v in by_q.values()),
            "difference": _err._r(d), "ci90_low": _err._r(lo), "ci90_high": _err._r(hi),
            "verdict": _err.verdict(n, n, lo, hi, a_label, b_label)}


def build_paired_comparisons(ctx: Any) -> dict[str, Any]:
    t1 = [q for q in ctx.bundle_qids if str(ctx.qmeta[q].get("track")) == "1"]
    spd = [q for q in t1 if _err._family(ctx.qmeta[q]["metric"]) == "spd"]
    binary = [q for q in t1 if _err._family(ctx.qmeta[q]["metric"]) == "binary"]
    primary = lambda q: ctx.primary.get(q)  # noqa: E731
    const = lambda m: (lambda _q: m)  # noqa: E731
    out = [
        _entry(ctx, "bayesmc_minus_mean", BAYESMC, MEAN, "spd", "crps",
               _paired(ctx, spd, const(BAYESMC), const(MEAN), "crps")),
        _entry(ctx, "bayesmc_minus_mean", BAYESMC, MEAN, "binary", "brier",
               _paired(ctx, binary, const(BAYESMC), const(MEAN), "brier")),
    ]
    spd_set = set(spd)
    members = sorted({m for (q, m, _h, _st) in ctx.scores if q in spd_set and _err._is_member(m)
                      and m != "track2_flash"})
    for m in members:
        out.append(_entry(ctx, "member_minus_primary", m, "primary aggregate", "spd", "crps",
                          _paired(ctx, spd, const(m), primary, "crps")))
    for st in ("crps", "brier"):
        out.append(_entry(ctx, "sibyl_minus_primary", SIBYL, "primary aggregate", "spd", st,
                          _paired(ctx, spd, const(SIBYL), primary, st)))
    return {"scope": "Track 1 only; same (question, horizon), latest run, indicative months excluded; "
                     "difference = a - b, lower is better; 90% interval resampling questions",
            "bootstrap": BOOTSTRAP, "seed": _err.SEED, "comparisons": out}


# ---------------------------------------------------------------------------
# 5. crisiswatch_split
# ---------------------------------------------------------------------------


def crisiswatch_class(cw: Mapping[str, Any] | None) -> str:
    cw = cw or {}
    if not cw.get("available"):
        return "absent"
    age = cw.get("edition_age_months")
    if age is None:
        return "other"
    if age <= 1:
        return "current"
    if age >= 3:
        return "stale"
    return "other"


def build_crisiswatch_split(ctx: Any, facts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    qids = [q for q in ctx.bundle_qids
            if ctx.qmeta[q]["hazard_code"] == "ACE" and ctx.qmeta[q]["metric"] == "FATALITIES"]
    if not qids:
        return _stub("no scored ACE/FATALITIES question")
    tracks: dict[str, Any] = {}
    for track in sorted({str(ctx.qmeta[q].get("track")) for q in qids}):
        classes: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(dict)
        runs: dict[str, set[str]] = defaultdict(set)
        for qid in qids:
            if str(ctx.qmeta[qid].get("track")) != track:
                continue
            prim = ctx.primary.get(qid)
            pairs = []
            for h in range(1, 7):
                if (qid, h) in ctx.indicative:
                    continue
                m = ctx.scores.get((qid, prim, h, "crps")) if prim else None
                c = ctx.scores.get((qid, CLIMATOLOGY, h, "crps"))
                if m is not None and c is not None:
                    pairs.append((m, c))
            if not pairs:
                continue
            cls = crisiswatch_class((facts.get(qid) or {}).get("crisiswatch"))
            classes[cls][qid] = pairs
            runs[cls].add(str(ctx.qmeta[qid].get("run_id")))
        rows = []
        for cls in ("current", "stale", "other", "absent"):
            by_q = classes.get(cls) or {}
            skill, lo, hi = _err._cluster_skill_ci(by_q, n_boot=BOOTSTRAP, seed=_err.SEED)
            rows.append({"class": cls, "n_questions": len(by_q),
                         "latest_run_ids": sorted(runs.get(cls, set())),
                         "skill_vs_climatology": _err._r(skill),
                         "ci90_low": _err._r(lo), "ci90_high": _err._r(hi),
                         "verdict": ("too few" if len(by_q) < _err.MIN_N or lo is None else
                                     "skill" if lo > 0 else "worse than climatology" if hi < 0
                                     else "no clear difference")})
        present = [r for r in rows if r["n_questions"]]
        disjoint = [
            [a["class"], b["class"]]
            for i, a in enumerate(present) for b in present[i + 1:]
            if not set(a["latest_run_ids"]) & set(b["latest_run_ids"])
        ]
        tracks[track] = {
            "classes": rows,
            # Two classes drawn from different forecast runs differ by run
            # (lineup, prompt version, month) as much as by edition age.
            "confounded_with_run": bool(disjoint),
            "class_pairs_with_no_shared_run": disjoint,
        }
    return {"metric": "paired RPS skill against climatology (1 - sum(primary)/sum(climatology))",
            "classes_defined": {"current": "edition age 1 month or less", "stale": "3 months or more",
                                "other": "2 months, or age unknown", "absent": "no edition shown"},
            "by_track": tracks}


# ---------------------------------------------------------------------------
# 6. gdacs_prompt_checks
# ---------------------------------------------------------------------------


def build_gdacs_prompt_checks(ctx: Any, facts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    from forecaster.gdacs_history import MIN_HISTORY_MONTHS  # noqa: PLC0415

    out = {}
    for hz in GDACS_HAZARDS:
        qids = [q for q in ctx.bundle_qids if ctx.qmeta[q]["hazard_code"] == hz]
        c = Counter()
        for qid in qids:
            f = facts.get(qid) or {}
            c["prompts"] += 1 if f.get("has_prompt") else 0
            b = f.get("gdacs_block")
            if not b or b.get("error"):
                if b and b.get("error"):
                    c["parse_errors"] += 1
                continue
            c["with_block"] += 1
            if b.get("repeated_months"):
                c["repeated_months"] += 1
            if b.get("history_unavailable"):
                c["unavailable"] += 1
                continue
            if b.get("event_months") == 0 and (f.get("gdacs_event_months_db") or 0) > 0:
                c["zero_where_db_has_events"] += 1
            tm = b.get("total_months")
            if tm is not None and tm < MIN_HISTORY_MONTHS:
                c["rate_under_min_months"] += 1
        out[hz] = {k: c.get(k, 0) for k in (
            "prompts", "with_block", "repeated_months", "zero_where_db_has_events",
            "unavailable", "rate_under_min_months", "parse_errors")} | {"questions": len(qids)}
    return {"source": "parsed from each question's stored prompt (forecaster.gdacs_history."
                      "parse_gdacs_prompt_block), never rebuilt; zero_where_db_has_events compares "
                      "with inject_status.gdacs_history.event_months",
            "min_history_months": MIN_HISTORY_MONTHS, "by_hazard": out}


# ---------------------------------------------------------------------------
# 7. confounded_experiments
# ---------------------------------------------------------------------------


def build_confounded_experiments(ctx: Any) -> dict[str, Any]:
    """Pairs of flags whose arms hold identical question sets, per (hazard,
    metric, track), on the questions ``experiments.csv`` compares (primary and
    climatology both scored)."""
    groups: dict[tuple, set[str]] = defaultdict(set)
    bundle = set(ctx.bundle_qids)
    for (qid, model, h, st) in ctx.scores:
        if qid not in bundle or st not in ("brier", "crps") or model != ctx.primary.get(qid):
            continue
        if (qid, CLIMATOLOGY, h, st) in ctx.scores:
            meta = ctx.qmeta[qid]
            groups[(meta["hazard_code"], meta["metric"], meta.get("track"))].add(qid)
    rows = []
    for (hz, metric, track), qids in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        parts: dict[str, dict[str, frozenset]] = {}
        for flag in _err.EXPERIMENT_FLAGS:
            arms: dict[str, set[str]] = defaultdict(set)
            for q in qids:
                arms[str(ctx.qmeta[q].get(flag))].add(q)
            if len(arms) >= 2:
                parts[flag] = {a: frozenset(s) for a, s in arms.items()}
        flags = sorted(parts)
        for i, fa in enumerate(flags):
            for fb in flags[i + 1:]:
                sa, sb = set(parts[fa].values()), set(parts[fb].values())
                if sa != sb:
                    continue
                inv = {s: arm for arm, s in parts[fb].items()}
                rows.append({"hazard_code": hz, "metric": metric, "track": track,
                             "flag_a": fa, "flag_b": fb, "n_questions": len(qids),
                             "arms": {arm: inv[s] for arm, s in sorted(parts[fa].items())},
                             "derived": frozenset({fa, fb}) in DERIVED_FLAG_PAIRS})
    return {"rule": "two flags whose arms partition the compared questions identically cannot be "
                    "told apart; a 'derived' pair is defined one from the other",
            "pairs": rows}


# ---------------------------------------------------------------------------
# 8. case_candidates
# ---------------------------------------------------------------------------


def build_case_candidates(ctx: Any, facts: Mapping[str, Mapping[str, Any]],
                          new_pairs: set) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for qid, h in sorted(new_pairs):
        meta = ctx.qmeta.get(qid) or {}
        family = _err._family(meta.get("metric"))
        st = _family_score_type(family)
        prim = ctx.primary.get(qid)
        p = ctx.scores.get((qid, prim, h, st)) if prim else None
        c = ctx.scores.get((qid, CLIMATOLOGY, h, st))
        if p is None or c is None:
            continue
        res = ((facts.get(qid) or {}).get("resolutions") or {}).get(h) or {}
        by_family[family].append({
            "question_id": qid, "horizon_m": h, "hazard_code": meta.get("hazard_code"),
            "metric": meta.get("metric"), "track": meta.get("track"), "score_type": st,
            "primary_model": prim, "primary": _err._r(p), "climatology": _err._r(c),
            "persistence": _err._r(ctx.scores.get((qid, PERSISTENCE, h, st))),
            "sibyl": _err._r(ctx.scores.get((qid, SIBYL, h, st))),
            "excess_over_climatology": _err._r(p - c),
            "resolved_value": res.get("value"), "observed_month": res.get("observed_month"),
            "source_desc": res.get("source_desc"), "resolution_reading": res.get("resolution_reading"),
            "scoring_class": res.get("scoring_class"),
        })
    out = {}
    for family, rows in sorted(by_family.items()):
        rows.sort(key=lambda r: (r["excess_over_climatology"], r["question_id"], r["horizon_m"]))
        out[family] = {"score_type": _family_score_type(family), "n_new_pairs_ranked": len(rows),
                       "worst": list(reversed(rows[-N_CANDIDATES:])), "best": rows[:N_CANDIDATES]}
    return {"rule": "new (question, horizon) pairs ranked by the primary aggregate's score minus "
                    "climatology's on the same pair (lower is better)", "by_family": out}


def write_case_studies(staging: Path, candidates: Mapping[str, Any], *,
                       cap_bytes: int = CASE_BYTES_CAP) -> dict[str, Any]:
    """The full records of the 3 worst and 3 best candidates per family,
    deduplicated, largest dropped first until the set fits ``cap_bytes``."""
    chosen: list[str] = []
    for fam in (candidates.get("by_family") or {}).values():
        for side in ("worst", "best"):
            for row in (fam.get(side) or [])[:N_CASE_RECORDS]:
                if row["question_id"] not in chosen:
                    chosen.append(row["question_id"])
    sizes = {}
    missing = []
    for qid in chosen:
        src = staging / "questions" / f"{qid}.json"
        if src.is_file():
            sizes[qid] = src.stat().st_size
        else:
            missing.append(qid)
    keep = dict(sizes)
    dropped = []
    while keep and sum(keep.values()) > cap_bytes:
        big = max(keep, key=lambda q: (keep[q], q))
        dropped.append({"question_id": big, "bytes": keep.pop(big)})
    out_dir = staging / CASE_DIR
    if keep:
        out_dir.mkdir(parents=True, exist_ok=True)
    for qid in keep:
        (out_dir / f"{qid}.json").write_bytes((staging / "questions" / f"{qid}.json").read_bytes())
    return {"dir": CASE_DIR, "written": [{"question_id": q, "bytes": keep[q]} for q in chosen if q in keep],
            "dropped_for_size": dropped, "missing_record": missing,
            "cap_bytes": cap_bytes, "total_bytes": sum(keep.values())}


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _safe(name: str, fn: Callable[[], Any], status: dict[str, Any]) -> Any:
    try:
        value = fn()
        status[name] = "stub" if isinstance(value, dict) and value.get("stub") else "ok"
        return value
    except Exception as exc:  # noqa: BLE001 - one section never costs the file
        LOGGER.warning("monthly_report_inputs %s failed: %s", name, exc)
        status[name] = "stub"
        return _stub(f"{type(exc).__name__}: {exc}")


def build(con, ctx: Any, facts: Mapping[str, Mapping[str, Any]], staging: Path, *,
          round_date: date | None = None, calibration_run_id: str | None = None,
          ctx_error: str | None = None) -> dict[str, Any]:
    """Every section, each degraded on its own; writes the JSON and the case
    records into ``staging`` and returns the document."""
    status: dict[str, str] = {}
    doc: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    doc["round"] = _safe("round", lambda: build_round(con, round_date, calibration_run_id), status)
    rd = _err._as_date((doc["round"] or {}).get("round_date"))
    if ctx is None:
        reason = f"error-attribution context unavailable: {ctx_error or 'unknown'}"
        for name in ("forecast_vintage", "new_this_round", "paired_comparisons", "crisiswatch_split",
                     "gdacs_prompt_checks", "confounded_experiments", "case_candidates",
                     "case_studies_new"):
            doc[name] = _stub(reason)
            status[name] = "stub"
    else:
        doc["forecast_vintage"] = _safe("forecast_vintage", lambda: build_forecast_vintage(ctx), status)
        new_pairs: set = set()

        def _new() -> dict[str, Any]:
            nonlocal new_pairs
            section, new_pairs = build_new_this_round(ctx, rd, resolution_dates(con, ctx.bundle_qids))
            return section

        doc["new_this_round"] = _safe("new_this_round", _new, status)
        doc["paired_comparisons"] = _safe("paired_comparisons", lambda: build_paired_comparisons(ctx), status)
        doc["crisiswatch_split"] = _safe("crisiswatch_split",
                                         lambda: build_crisiswatch_split(ctx, facts), status)
        doc["gdacs_prompt_checks"] = _safe("gdacs_prompt_checks",
                                           lambda: build_gdacs_prompt_checks(ctx, facts), status)
        doc["confounded_experiments"] = _safe("confounded_experiments",
                                              lambda: build_confounded_experiments(ctx), status)
        doc["case_candidates"] = (
            _safe("case_candidates", lambda: build_case_candidates(ctx, facts, new_pairs), status)
            if status.get("new_this_round") == "ok"
            else _stub("new_this_round unavailable, so no pair is known to be new")
        )
        status.setdefault("case_candidates", "stub")
        doc["case_studies_new"] = (
            _safe("case_studies_new", lambda: write_case_studies(staging, doc["case_candidates"]), status)
            if status.get("case_candidates") == "ok"
            else _stub("no case candidates")
        )
        status.setdefault("case_studies_new", "stub")
    doc["section_status"] = status
    write_json(staging / FILE_NAME, doc)
    return doc


__all__ = [
    "FILE_NAME", "CASE_DIR", "question_facts", "build", "build_round", "build_forecast_vintage",
    "build_new_this_round", "build_paired_comparisons", "build_crisiswatch_split",
    "build_gdacs_prompt_checks", "build_confounded_experiments", "build_case_candidates",
    "write_case_studies", "classify_new", "crisiswatch_class", "run_date_from_id",
    "resolution_dates", "latest_snapshot_date",
]
