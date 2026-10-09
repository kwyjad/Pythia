# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl post-mortems: notes on resolved questions, and lessons drawn from them.

``python -m sibyl.postmortem --db-url ...`` runs monthly in
``compute_calibration_pythia.yml`` after ``sibyl.advice``, at medium effort,
under a hard cap of ``SIBYL_POSTMORTEM_CAP_USD`` ($5).

Notes
-----
Each newly resolved question (Sibyl's record as ``sibyl.advice`` reads it:
the latest evidence-backed ok forecast, joined to its resolutions) gets ONE
note, written once per (question, Sibyl run) to ``sibyl_postmortem_notes``:
what happened, how the forecast compared, what would have helped, and one
general lesson the case suggests.

Lessons
-------
Per (hazard, metric) class, once ``SIBYL_POSTMORTEM_MIN_NOTES`` (8) notes
exist and a note has arrived since the last version, the model is asked for
lessons over the class's notes. A lesson is kept only when it cites
``SIBYL_LESSON_MIN_CASES`` (3) or more of the notes given and names no
country and no dated event: the code checks country names, ISO3 codes, and
years, and the prompt forbids the rest. The kept lessons, at most
``SIBYL_LESSONS_MAX_CHARS`` (6,000) characters, are a new version in
``sibyl_lessons``; rejected ones are stored beside them with the reason.

Failure types (Oct 2026, review Part 2)
---------------------------------------
Each note carries up to three labels from ``FAILURE_TYPES`` (most important
first), each with the ledger id or short quote the trial recorded at the
time that supports it. Information that did not exist at forecast time is
``unforeseeable``, never a fault. Unknown labels are dropped and kept in
``failure_types_raw``; a note with no valid label is ``unlabelled``.
``failure_rates`` counts, per class and pooled, the distinct questions
carrying each label; a share is shown from
``SIBYL_FAILURE_RATE_MIN_QUESTIONS`` (10) labelled questions. Notes written
before the labels (``prompt_version`` NULL) are re-labelled, oldest first,
inside the same cap. The rates reach the pooled advice row's findings and
the dashboard; they never go into a trial's prompt.

Use
---
``lessons_block`` gives a question the class's newest lessons plus up to
``SIBYL_MAX_ANALOGUES`` (4) notes on past questions, same country and hazard
first. ``sibyl.run`` shows it only in the track-record arm, and never in
backtest: a note written after the as-of date is leakage.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from sibyl import config as _cfg

logger = logging.getLogger(__name__)

ModelCall = Callable[[str], Tuple[str, Dict[str, Any], str]]

LESSONS_HEADING = "=== LESSONS FROM YOUR RESOLVED QUESTIONS ==="

#: The note prompt's version, stamped on every note it writes.
PROMPT_VERSION = "pm_v2"

#: Failure types (Oct 2026): the single source of the labels a note may carry,
#: in the order the note prompt lists them.
FAILURE_TYPES: Dict[str, str] = {
    "resolver_misread": "Forecast a quantity other than the one the source records: wrong series, wrong window, wrong unit",
    "stale_or_wrong_fact": "Leaned on a figure that was out of date or wrong at the time",
    "double_counted": "Added a rise already in the reference, or counted one event more than once",
    "coverage_as_signal": "Read heavy or thin news coverage as evidence of level",
    "statement_as_commitment": "Took an announced plan or date as likely to hold",
    "wrong_scale_of_event": "Modelled only the extreme version of an event, or only the mild one",
    "rigid_reference": "Kept the reference after the mechanism behind it had changed",
    "retreat_to_reference": "Found and reasoned the evidence correctly, then stayed near the prior",
    "spike_carried_forward": "Carried a spike to month 6 that faded",
    "absence_as_evidence": "Read 'the search found nothing' as 'nothing happened'",
    "missed_dated_event": "Missed a dated event inside the window that was knowable",
    "zero_misjudged": "p_zero badly wrong with the positive part sound",
    "tails_too_thin": "Outcome beyond the stated 0.05 to 0.95 range with the centre sound",
    "thin_research": "Too few or too poor sources to support the forecast",
    "reference_fault": "The mechanical reference itself was wrong, and the trial followed it",
    "unforeseeable": "The outcome turned on information that did not exist at forecast time",
    "no_fault": "The forecast was sound and the outcome fell inside its central range",
}
UNLABELLED = "unlabelled"
MAX_LABELS = 3


def _failure_type_lines() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in FAILURE_TYPES.items())


NOTE_PROMPT = """You forecast the question below some months ago and the outcome is now known. Write a short post-mortem.

QUESTION: {wording}
Class: {hazard_code} / {metric}. Country: {country}.
Your forecast (raw pool of your research trials, before the reference was mixed in):
{forecast}
Outcome by window month: {outcomes}
Your reference (prior) median for month 1: {reference_median}

=== EACH RESOLVED MONTH ===
Bucket vectors run from the lowest bucket (zero) to the highest. "Inside" is stated by the code, not judged.
{months}

=== WHAT YOUR TRIALS RECORDED AT THE TIME ===
{trials}

=== FAILURE TYPES ===
{failure_types}

Label the forecast with up to {max_labels} failure types from the list above, most important first. A label must rest on what the trials recorded at the time: for each label give a ledger id (such as E3) or a short quote from the record above. Information that did not exist at forecast time is "unforeseeable", never a fault. If the forecast was sound and the outcome fell inside its central range, say "no_fault".

Answer ONLY with a JSON object:
{{"what_happened": "<one or two sentences>",
  "forecast_vs_outcome": "<one sentence: too high, too low, too wide, too narrow, about right>",
  "what_would_have_helped": "<one or two sentences: which evidence or reasoning would have moved you the right way>",
  "general_lesson": "<one sentence that would help on a DIFFERENT question of this class; name no country, place, year or event>",
  "failure_types": ["<label>", "..."],
  "label_evidence": {{"<label>": "<ledger id or short quote>"}}}}"""

RELABEL_PROMPT = """Below is a post-mortem you wrote on one of your resolved forecasts, with what your trials recorded at the time. Label it.

QUESTION: {wording}
Class: {hazard_code} / {metric}. Country: {country}.
Outcome by window month: {outcomes}
Your post-mortem: {note}

=== EACH RESOLVED MONTH ===
{months}

=== WHAT YOUR TRIALS RECORDED AT THE TIME ===
{trials}

=== FAILURE TYPES ===
{failure_types}

Choose up to {max_labels} failure types from the list above, most important first. A label must rest on what the trials recorded at the time: for each give a ledger id (such as E3) or a short quote. Information that did not exist at forecast time is "unforeseeable", never a fault.

Answer ONLY with a JSON object:
{{"failure_types": ["<label>", "..."], "label_evidence": {{"<label>": "<ledger id or short quote>"}}}}"""

LESSONS_PROMPT = """Below are post-mortem notes on {n} of your resolved forecasts of {hazard_code} / {metric} questions. Each note has an id.

{notes}

Each note carries the failure types it was labelled with; a lesson may address a type that recurs.

Draw at most six lessons that would improve FUTURE forecasts of this class. A lesson must:
- rest on at least {min_cases} of the notes above, cited by id;
- be general: name no country, region, place, year, month or specific event;
- say what to do differently, in one or two sentences.
Leave out anything only one or two notes support.

Answer ONLY with a JSON object:
{{"lessons": [{{"lesson": "...", "cases": ["<note id>", "..."]}}]}}"""


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _default_call(prompt: str) -> Tuple[str, Dict[str, Any], str]:
    from forecaster.providers import call_anthropic, estimate_cost_usd  # noqa: PLC0415

    effort = _cfg.POSTMORTEM_EFFORT
    result = call_anthropic(
        prompt, _cfg.MODEL, 1.0, purpose="sibyl_postmortem",
        thinking_level=effort if effort not in ("", "off", "none") else None,
    )
    usage = dict(result.usage or {})
    if not usage.get("cost_usd"):
        usage["cost_usd"] = estimate_cost_usd(_cfg.MODEL, usage)
    return result.text or "", usage, result.error or ""


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", s)
    try:
        obj = json.loads(s)
    except ValueError:
        m = re.search(r"\{.*\}", s, re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            return None
    return obj if isinstance(obj, dict) else None


@dataclass
class Budget:
    cap_usd: float
    spent_usd: float = 0.0

    def can_spend(self) -> bool:
        return self.spent_usd < self.cap_usd


# ---------------------------------------------------------------------------
# Lessons gate (pure)
# ---------------------------------------------------------------------------


def _country_terms() -> List[str]:
    path = Path(__file__).resolve().parents[1] / "resolver" / "data" / "countries.csv"
    terms: List[str] = []
    try:
        with path.open(encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                name = (row.get("country_name") or "").strip()
                if len(name) >= 4:
                    terms.append(name)
    except OSError:
        pass
    return terms


_YEAR = re.compile(r"\b(19|20)\d{2}\b")


def lesson_problem(
    lesson: str,
    cases: Sequence[str],
    note_ids: Sequence[str],
    *,
    iso3s: Sequence[str] = (),
    country_terms: Optional[Sequence[str]] = None,
    min_cases: Optional[int] = None,
) -> Optional[str]:
    """Why a lesson is refused, or None when it is kept."""
    min_cases = _cfg.LESSON_MIN_CASES if min_cases is None else min_cases
    text = (lesson or "").strip()
    if not text:
        return "empty"
    cited = {str(c) for c in cases or []} & {str(n) for n in note_ids}
    if len(cited) < min_cases:
        return f"rests on {len(cited)} cited note(s), fewer than {min_cases}"
    if _YEAR.search(text):
        return "names a year"
    for code in iso3s:
        if code and re.search(rf"\b{re.escape(code)}\b", text):
            return f"names a country code ({code})"
    low = text.lower()
    for name in (country_terms if country_terms is not None else _country_terms()):
        if re.search(rf"\b{re.escape(name.lower())}\b", low):
            return f"names a country ({name})"
    return None


def gate_lessons(
    proposed: Sequence[Dict[str, Any]],
    note_ids: Sequence[str],
    *,
    iso3s: Sequence[str] = (),
    country_terms: Optional[Sequence[str]] = None,
    max_chars: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    """(kept, rejected, text). Text lists kept lessons, cut by whole lessons."""
    max_chars = _cfg.LESSONS_MAX_CHARS if max_chars is None else max_chars
    terms = list(country_terms) if country_terms is not None else _country_terms()
    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for item in proposed or []:
        if not isinstance(item, dict):
            continue
        lesson = str(item.get("lesson") or "").strip()
        cases = [str(c) for c in (item.get("cases") or [])]
        why = lesson_problem(lesson, cases, note_ids, iso3s=iso3s, country_terms=terms)
        if why:
            rejected.append({"lesson": lesson, "cases": cases, "reason": why})
            continue
        n = len({c for c in cases if c in set(note_ids)})
        kept.append({"lesson": lesson, "cases": cases, "n_cases": n})
    lines: List[str] = []
    final: List[Dict[str, Any]] = []
    for item in kept:
        line = f"- {item['lesson']} (seen in {item['n_cases']} resolved questions)"
        if len("\n".join(lines + [line])) > max_chars:
            rejected.append({**item, "reason": "over the length limit"})
            continue
        lines.append(line)
        final.append(item)
    return final, rejected, "\n".join(lines)


def validate_labels(parsed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The labels a note may carry, checked against ``FAILURE_TYPES``.

    Valid labels are kept in order, at most ``MAX_LABELS``, each with its
    evidence where one was given; anything else is kept in
    ``failure_types_raw``. No valid label: status ``unlabelled``.
    """
    raw = (parsed or {}).get("failure_types") or []
    if isinstance(raw, str):
        raw = [raw]
    evidence = (parsed or {}).get("label_evidence") or {}
    if not isinstance(evidence, dict):
        evidence = {}
    labels: List[str] = []
    unknown: List[str] = []
    for item in raw if isinstance(raw, list) else []:
        label = str(item or "").strip().lower()
        if label in FAILURE_TYPES:
            if label not in labels and len(labels) < MAX_LABELS:
                labels.append(label)
        elif label:
            unknown.append(str(item))
    return {
        "status": "labelled" if labels else UNLABELLED,
        "failure_types": labels,
        "label_evidence": {
            k: str(evidence.get(k) or "")[:300] for k in labels if evidence.get(k)
        },
        "failure_types_raw": unknown,
    }


def failure_rates(con, *, min_questions: Optional[int] = None) -> Dict[str, Any]:
    """Per class and pooled: distinct questions carrying each label.

    The denominator is the distinct questions with a labelled note (a
    question with several notes counts once, from its newest labelled
    note). Counts are always given; a share only from *min_questions*.
    """
    min_q = _cfg.FAILURE_RATE_MIN_QUESTIONS if min_questions is None else int(min_questions)
    out: Dict[str, Any] = {"min_questions": min_q, "labels": list(FAILURE_TYPES),
                           "classes": {}, "pooled": None}
    try:
        rows = con.execute(
            "SELECT question_id, hazard_code, metric, failure_types_json FROM "
            "sibyl_postmortem_notes WHERE failure_types_json IS NOT NULL "
            "ORDER BY created_at DESC, sibyl_run_id DESC"
        ).fetchall()
    except Exception:  # noqa: BLE001 - an older DB has no column
        return out
    per_q: Dict[str, Tuple[str, List[str]]] = {}
    unlabelled: Dict[str, str] = {}
    for qid, hz, metric, ftj in rows:
        try:
            ft = json.loads(ftj) or {}
        except (TypeError, ValueError):
            continue
        cls = f"{str(hz).upper()}/{str(metric).upper()}"
        labels = [x for x in (ft.get("failure_types") or []) if x in FAILURE_TYPES]
        if labels:
            per_q.setdefault(str(qid), (cls, labels))
        else:
            unlabelled.setdefault(str(qid), cls)

    def _summary(items: List[Tuple[str, List[str]]], n_unl: int) -> Dict[str, Any]:
        n = len(items)
        counts = {k: sum(1 for _, ls in items if k in ls) for k in FAILURE_TYPES}
        ok = n >= min_q
        return {
            "n_labelled_questions": n,
            "n_unlabelled_questions": n_unl,
            "status": "ok" if ok else "not_yet",
            "counts": counts,
            "shares": {k: (c / n if ok and n else None) for k, c in counts.items()},
        }

    classes = sorted({c for c, _ in per_q.values()} | set(unlabelled.values()))
    for cls in classes:
        items = [v for v in per_q.values() if v[0] == cls]
        n_unl = sum(1 for q, c in unlabelled.items() if c == cls and q not in per_q)
        out["classes"][cls] = _summary(items, n_unl)
    out["pooled"] = _summary(list(per_q.values()),
                             sum(1 for q in unlabelled if q not in per_q))
    return out


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


def _ensure(con) -> None:
    from pythia.db.schema import ensure_sibyl_measurement_tables  # noqa: PLC0415

    ensure_sibyl_measurement_tables(con)


def _question_info(con, qid: str) -> Dict[str, Any]:
    try:
        row = con.execute(
            "SELECT wording, iso3 FROM questions WHERE question_id = ?", [qid]
        ).fetchone()
    except Exception:  # noqa: BLE001
        row = None
    return {"wording": (row[0] if row else "") or "", "iso3": (row[1] if row else "") or ""}


def _fmt_vec(vec: Any) -> str:
    try:
        return "[" + ", ".join(f"{float(x):.2f}" for x in vec) + "]"
    except (TypeError, ValueError):
        return "(none)"


def _forecast_context(con, qid: str, srid: str) -> Dict[str, Any]:
    """What the note prompt shows about one stored forecast.

    ``raw`` (month 1 and 6 quantiles of the raw pool), ``reference_median``,
    the per-month vectors (``reference``, ``raw_vectors``, ``raw_quantiles``,
    ``final``), the metric, and per trial its lane, plan findings,
    reconciliation with the reference and ledger items.
    """
    try:
        row = con.execute(
            "SELECT raw_by_month_json, reference_json, trials_json, metric FROM sibyl_forecasts "
            "WHERE question_id = ? AND sibyl_run_id = ? LIMIT 1", [qid, srid],
        ).fetchone()
    except Exception:  # noqa: BLE001
        row = None
    out: Dict[str, Any] = {"raw": {}, "reference_median": None, "evidence": [],
                           "reference": {}, "raw_vectors": {}, "raw_quantiles": {},
                           "final": {}, "metric": None, "trials": []}
    if not row:
        return out
    out["metric"] = str(row[3] or "").upper() or None
    try:
        final_row = con.execute(
            "SELECT final_by_month_json FROM sibyl_forecasts "
            "WHERE question_id = ? AND sibyl_run_id = ? LIMIT 1", [qid, srid],
        ).fetchone()
        out["final"] = json.loads(final_row[0]) if final_row and final_row[0] else {}
    except Exception:  # noqa: BLE001 - an older DB has no column
        pass
    try:
        raw = json.loads(row[0]) if row[0] else {}
        out["raw"] = {k: v for k, v in (raw.get("quantiles") or {}).items() if k in ("1", "6")}
        out["raw_vectors"] = raw.get("vectors") or {}
        out["raw_quantiles"] = raw.get("quantiles") or {}
    except (TypeError, ValueError, AttributeError):
        pass
    try:
        from sibyl.aggregate import dist_from_vector  # noqa: PLC0415
        from sibyl.trials import month_median  # noqa: PLC0415

        ref = json.loads(row[1]) if row[1] else {}
        out["reference"] = ref.get("by_month") or {}
        vec = out["reference"].get("1")
        if vec and out["metric"]:
            d = dist_from_vector(vec, out["metric"])
            out["reference_median"] = round(month_median(d.p_zero, d.qpos), 1)
    except Exception:  # noqa: BLE001
        pass
    try:
        for t in json.loads(row[2]) if row[2] else []:
            t = t or {}
            trace = t.get("belief_trace") or []
            last = (trace[-1] or {}).get("belief") if trace else {}
            last = last or {}
            plan = last.get("plan") or {}
            items = []
            for k, it in enumerate(t.get("ledger") or [], start=1):
                items.append({
                    "id": it.get("id") or f"E{k}", "date": it.get("date"),
                    "tier": it.get("tier"), "kind": it.get("kind"),
                    "quote": str(it.get("quote") or "")[:200],
                    "direction": it.get("direction"),
                })
                out["evidence"].append(
                    f"[{it.get('date') or 'undated'}] {it.get('direction') or ''}: "
                    f"{str(it.get('quote') or '')[:200]}"
                )
            out["trials"].append({
                "lane": t.get("lane") or str(t.get("perspective") or "").split(",")[0],
                "plan": {slot: str((v or {}).get("finding") or "")[:200]
                         for slot, v in plan.items() if isinstance(v, dict)},
                "reconciliation": str(last.get("baserate_reconciliation") or "")[:400],
                "ledger": items,
            })
    except (TypeError, ValueError, AttributeError):
        pass
    out["evidence"] = out["evidence"][:20]
    return out


def month_lines(ctx: Dict[str, Any], outcomes: Sequence[float],
                horizons: Sequence[Optional[int]]) -> List[str]:
    """One line per resolved month: the vectors, the outcome and its bucket,
    and whether the outcome fell inside the raw pool's 0.05 to 0.95 range
    (stated by the code)."""
    from sibyl.score_variants import _bucket  # noqa: PLC0415

    lines: List[str] = []
    metric = ctx.get("metric")
    for y, h in zip(outcomes, horizons or [None] * len(outcomes)):
        key = str(h) if h is not None else "1"
        q = ctx.get("raw_quantiles", {}).get(key) or {}
        lo, hi = q.get("0.05"), q.get("0.95")
        if lo is not None and hi is not None:
            inside = "inside" if float(lo) <= float(y) <= float(hi) else (
                "BELOW" if float(y) < float(lo) else "ABOVE")
            rng = f"raw 0.05-0.95 range {float(lo):g} to {float(hi):g}: outcome {inside}"
        else:
            rng = "raw 0.05-0.95 range not stored"
        bucket = _bucket(float(y), metric) if metric else None
        lines.append(
            f"Month {h if h is not None else '?'}: outcome {float(y):g} "
            f"(bucket {bucket + 1 if bucket is not None else '?'}); {rng}; "
            f"reference {_fmt_vec(ctx.get('reference', {}).get(key))}; "
            f"raw pool {_fmt_vec(ctx.get('raw_vectors', {}).get(key))}; "
            f"published {_fmt_vec(ctx.get('final', {}).get(key))}"
        )
    return lines


def trial_text(trials: Sequence[Dict[str, Any]], max_chars: int) -> Tuple[str, int]:
    """The trials block inside *max_chars*: whole ledger items are dropped
    from the end (lowest tier first is not attempted: the order is the
    trial's own) and the number dropped is said."""
    blocks: List[List[str]] = []
    for t in trials:
        head = [f"Trial, lane {t.get('lane') or '?'}:"]
        for slot, finding in (t.get("plan") or {}).items():
            if finding:
                head.append(f"  plan {slot}: {finding}")
        if t.get("reconciliation"):
            head.append(f"  reconciliation with the reference: {t['reconciliation']}")
        items = [
            f"  [{it['id']}] {it.get('date') or 'undated'} tier {it.get('tier') or '?'} "
            f"{it.get('kind') or ''} {it.get('direction') or ''}: {it.get('quote') or ''}"
            for it in t.get("ledger") or []
        ]
        blocks.append(head + items)
    n_heads = [len([ln for ln in b if not ln.startswith("  [")]) for b in blocks]
    dropped = 0

    def _join() -> str:
        return "\n".join(ln for b in blocks for ln in b)

    while len(_join()) > max_chars:
        # Drop the last ledger line of the longest trial block.
        idx = max(range(len(blocks)), key=lambda i: len(blocks[i]) - n_heads[i], default=None)
        if idx is None or len(blocks[idx]) <= n_heads[idx]:
            break
        blocks[idx].pop()
        dropped += 1
    text = _join() or "(none recorded)"
    if dropped:
        text += f"\n({dropped} ledger item(s) left out for length)"
    return text, dropped


def _prompt_parts(ctx: Dict[str, Any], outcomes: Sequence[float],
                  horizons: Sequence[Optional[int]], fixed_chars: int) -> Tuple[str, str]:
    months = "\n".join(month_lines(ctx, outcomes, horizons)) or "(none)"
    budget = max(1000, int(_cfg.POSTMORTEM_PROMPT_MAX_CHARS) - fixed_chars - len(months))
    trials, _ = trial_text(ctx.get("trials") or [], budget)
    return months, trials


def write_notes(
    con,
    *,
    call: Optional[ModelCall] = None,
    budget: Optional[Budget] = None,
    log: Optional[Callable[..., None]] = None,
) -> int:
    """A note on every resolved Sibyl question that has none. Returns notes written."""
    from sibyl.advice import load_records  # noqa: PLC0415

    _ensure(con)
    call = call or _default_call
    budget = budget or Budget(_cfg.POSTMORTEM_CAP_USD)
    done = {
        (str(q), str(s))
        for q, s in con.execute(
            "SELECT question_id, sibyl_run_id FROM sibyl_postmortem_notes"
        ).fetchall()
    }
    written = 0
    for r in load_records(con):
        key = (r.question_id, str(r.sibyl_run_id))
        if key in done:
            continue
        if not budget.can_spend():
            logger.warning("sibyl.postmortem: cap $%.2f reached; notes stop here", budget.cap_usd)
            break
        info = _question_info(con, r.question_id)
        ctx = _forecast_context(con, r.question_id, str(r.sibyl_run_id))
        outcomes = ", ".join(
            f"month {h}: {y:g}" for y, h in zip(r.outcomes, r.outcome_horizons or [None] * len(r.outcomes))
        )
        horizons = r.outcome_horizons or [None] * len(r.outcomes)
        fields = dict(
            wording=info["wording"] or "(no wording stored)",
            hazard_code=r.hazard_code, metric=r.metric, country=info["iso3"] or "?",
            forecast=json.dumps(ctx["raw"] or {"month_1": r.quantiles}, default=str),
            outcomes=outcomes, reference_median=ctx["reference_median"],
            failure_types=_failure_type_lines(), max_labels=MAX_LABELS,
        )
        fixed = len(NOTE_PROMPT.format(months="", trials="", **fields))
        months, trials = _prompt_parts(ctx, r.outcomes, horizons, fixed)
        prompt = NOTE_PROMPT.format(months=months, trials=trials, **fields)
        text, usage, error = call(prompt)
        cost = float((usage or {}).get("cost_usd") or 0.0)
        budget.spent_usd += cost
        if log:
            log(prompt=prompt, response=text, usage=usage or {}, error=error,
                question_id=r.question_id, iso3=info["iso3"], hazard_code=r.hazard_code,
                metric=r.metric, call_type="sibyl_postmortem_note")
        note = _parse_json(text) if not error else None
        if not note:
            logger.warning("sibyl.postmortem: no usable note for %s (%s)", r.question_id,
                           error or "unparseable")
            continue
        labels = validate_labels(note)
        for k in ("failure_types", "label_evidence"):
            note.pop(k, None)
        con.execute(
            """
            INSERT INTO sibyl_postmortem_notes
                (question_id, sibyl_run_id, iso3, hazard_code, metric, note_json, model,
                 cost_usd, created_at, failure_types_json, prompt_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
            """,
            [r.question_id, str(r.sibyl_run_id), info["iso3"], r.hazard_code, r.metric,
             json.dumps(note), _cfg.MODEL, cost, json.dumps(labels), PROMPT_VERSION],
        )
        written += 1
    return written


def relabel_notes(
    con,
    *,
    call: Optional[ModelCall] = None,
    budget: Optional[Budget] = None,
    log: Optional[Callable[..., None]] = None,
) -> int:
    """Label notes written before the labels existed, oldest first, inside the cap.

    A note counts as unlabelled-by-age when ``prompt_version`` is NULL; a
    ``pm_v2`` note that came back unlabelled is not asked again (it was).
    """
    from sibyl.advice import load_records  # noqa: PLC0415

    _ensure(con)
    call = call or _default_call
    budget = budget or Budget(_cfg.POSTMORTEM_CAP_USD)
    pending = con.execute(
        "SELECT question_id, sibyl_run_id, hazard_code, metric, note_json "
        "FROM sibyl_postmortem_notes WHERE prompt_version IS NULL "
        "ORDER BY created_at, question_id"
    ).fetchall()
    if not pending:
        return 0
    records = {r.question_id: r for r in load_records(con)}
    done = 0
    for qid, srid, hz, metric, nj in pending:
        r = records.get(str(qid))
        if r is None:
            continue
        if not budget.can_spend():
            logger.warning("sibyl.postmortem: cap reached; re-labelling stops here")
            break
        info = _question_info(con, str(qid))
        ctx = _forecast_context(con, str(qid), str(srid))
        horizons = r.outcome_horizons or [None] * len(r.outcomes)
        outcomes = ", ".join(f"month {h}: {y:g}" for y, h in zip(r.outcomes, horizons))
        fields = dict(
            wording=info["wording"] or "(no wording stored)", hazard_code=hz,
            metric=metric, country=info["iso3"] or "?", outcomes=outcomes,
            note=nj or "{}", failure_types=_failure_type_lines(), max_labels=MAX_LABELS,
        )
        fixed = len(RELABEL_PROMPT.format(months="", trials="", **fields))
        months, trials = _prompt_parts(ctx, r.outcomes, horizons, fixed)
        prompt = RELABEL_PROMPT.format(months=months, trials=trials, **fields)
        text, usage, error = call(prompt)
        cost = float((usage or {}).get("cost_usd") or 0.0)
        budget.spent_usd += cost
        if log:
            log(prompt=prompt, response=text, usage=usage or {}, error=error,
                question_id=str(qid), iso3=info["iso3"], hazard_code=hz, metric=metric,
                call_type="sibyl_postmortem_relabel")
        parsed = _parse_json(text) if not error else None
        if parsed is None:
            continue
        con.execute(
            "UPDATE sibyl_postmortem_notes SET failure_types_json = ?, prompt_version = ? "
            "WHERE question_id = ? AND sibyl_run_id = ?",
            [json.dumps(validate_labels(parsed)), PROMPT_VERSION, qid, srid],
        )
        done += 1
    return done


def attach_failure_rates(con) -> bool:
    """Write ``failure_rates`` into the newest pooled advice row's findings."""
    try:
        row = con.execute(
            "SELECT as_of_month, findings_json FROM sibyl_calibration_advice "
            "WHERE hazard_code = '*' AND metric = '*' ORDER BY as_of_month DESC LIMIT 1"
        ).fetchone()
    except Exception:  # noqa: BLE001 - no table yet
        return False
    if not row:
        return False
    try:
        findings = json.loads(row[1]) if row[1] else {}
    except (TypeError, ValueError):
        findings = {}
    findings["failure_types"] = failure_rates(con)
    con.execute(
        "UPDATE sibyl_calibration_advice SET findings_json = ? "
        "WHERE as_of_month = ? AND hazard_code = '*' AND metric = '*'",
        [json.dumps(findings, default=str), row[0]],
    )
    return True


def write_lessons(
    con,
    *,
    as_of_month: Optional[str] = None,
    call: Optional[ModelCall] = None,
    budget: Optional[Budget] = None,
    log: Optional[Callable[..., None]] = None,
) -> List[Dict[str, Any]]:
    """A new lessons version for every class with enough notes and a new one."""
    _ensure(con)
    call = call or _default_call
    budget = budget or Budget(_cfg.POSTMORTEM_CAP_USD)
    as_of_month = as_of_month or date.today().strftime("%Y-%m")
    out: List[Dict[str, Any]] = []
    classes = con.execute(
        "SELECT hazard_code, metric, COUNT(*), MAX(created_at) FROM sibyl_postmortem_notes "
        "GROUP BY 1, 2 ORDER BY 1, 2"
    ).fetchall()
    for hz, metric, n, newest in classes:
        if int(n) < _cfg.POSTMORTEM_MIN_NOTES:
            continue
        last = con.execute(
            "SELECT MAX(version), MAX(created_at), MAX(n_notes) FROM sibyl_lessons "
            "WHERE hazard_code = ? AND metric = ?", [hz, metric],
        ).fetchone()
        if last and last[0] is not None and int(last[2] or 0) >= int(n):
            continue  # no note since the last version
        if not budget.can_spend():
            logger.warning("sibyl.postmortem: cap reached; lessons for %s/%s wait", hz, metric)
            break
        notes = con.execute(
            "SELECT question_id, iso3, note_json, failure_types_json FROM sibyl_postmortem_notes "
            "WHERE hazard_code = ? AND metric = ? ORDER BY created_at, question_id",
            [hz, metric],
        ).fetchall()
        ids = [str(q) for q, _, _, _ in notes]
        iso3s = sorted({str(i) for _, i, _, _ in notes if i})
        blocks = []
        for qid, _, nj, ftj in notes:
            try:
                nd = json.loads(nj)
            except (TypeError, ValueError):
                nd = {}
            try:
                labels = (json.loads(ftj) or {}).get("failure_types") or [] if ftj else []
            except (TypeError, ValueError):
                labels = []
            blocks.append(
                f"[{qid}] happened: {nd.get('what_happened', '')} | "
                f"forecast: {nd.get('forecast_vs_outcome', '')} | "
                f"would have helped: {nd.get('what_would_have_helped', '')} | "
                f"lesson: {nd.get('general_lesson', '')} | "
                f"failure types: {', '.join(labels) if labels else 'none'}"
            )
        prompt = LESSONS_PROMPT.format(
            n=len(notes), hazard_code=hz, metric=metric, notes="\n".join(blocks),
            min_cases=_cfg.LESSON_MIN_CASES,
        )
        text, usage, error = call(prompt)
        cost = float((usage or {}).get("cost_usd") or 0.0)
        budget.spent_usd += cost
        if log:
            log(prompt=prompt, response=text, usage=usage or {}, error=error,
                question_id="", iso3="", hazard_code=hz, metric=metric,
                call_type="sibyl_postmortem_lessons")
        parsed = _parse_json(text) if not error else None
        if parsed is None:
            logger.warning("sibyl.postmortem: no usable lessons for %s/%s", hz, metric)
            continue
        kept, rejected, lessons_text = gate_lessons(parsed.get("lessons") or [], ids, iso3s=iso3s)
        version = int(last[0] or 0) + 1 if last and last[0] is not None else 1
        con.execute(
            """
            INSERT INTO sibyl_lessons
                (hazard_code, metric, version, as_of_month, lessons_text, lessons_json,
                 n_notes, rejected_json, model, cost_usd, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [hz, metric, version, as_of_month, lessons_text, json.dumps(kept), int(n),
             json.dumps(rejected), _cfg.MODEL, cost],
        )
        out.append({"hazard_code": hz, "metric": metric, "version": version,
                    "n_kept": len(kept), "n_rejected": len(rejected)})
    return out


# ---------------------------------------------------------------------------
# Reading for a prompt
# ---------------------------------------------------------------------------


def load_lessons_text(con, hazard: str, metric: str, as_of: Any = None,
                      *, backtest: Optional[bool] = None) -> str:
    """The class's newest lessons ('' when none, in backtest, or on any failure)."""
    if _cfg.BACKTEST_MODE if backtest is None else backtest:
        return ""
    try:
        params: List[Any] = [hazard.upper(), metric.upper()]
        where = "hazard_code = ? AND metric = ? AND COALESCE(lessons_text, '') <> ''"
        if as_of is not None:
            where += " AND as_of_month <= ?"
            params.append(str(as_of)[:7])
        row = con.execute(
            f"SELECT lessons_text FROM sibyl_lessons WHERE {where} "
            "ORDER BY version DESC LIMIT 1", params,
        ).fetchone()
    except Exception:  # noqa: BLE001 - an older DB has no table
        return ""
    return (row[0] or "") if row else ""


def load_analogues(con, question: Any, *, k: Optional[int] = None,
                   backtest: Optional[bool] = None) -> List[Dict[str, Any]]:
    """Up to *k* notes on OTHER questions: same country and hazard first, then the class."""
    if _cfg.BACKTEST_MODE if backtest is None else backtest:
        return []
    k = _cfg.MAX_ANALOGUES if k is None else k
    if k <= 0:
        return []
    try:
        rows = con.execute(
            """
            SELECT question_id, iso3, note_json,
                   CASE WHEN iso3 = ? THEN 0 ELSE 1 END AS same
            FROM sibyl_postmortem_notes
            WHERE hazard_code = ? AND metric = ? AND question_id <> ?
            ORDER BY same, created_at DESC, question_id
            LIMIT ?
            """,
            [question.iso3, question.hazard_code.upper(), question.metric.upper(),
             question.question_id, int(k)],
        ).fetchall()
    except Exception:  # noqa: BLE001
        return []
    out = []
    for qid, iso3, nj, _same in rows:
        try:
            note = json.loads(nj)
        except (TypeError, ValueError):
            continue
        out.append({"question_id": qid, "iso3": iso3, "note": note})
    return out


def render_lessons_block(lessons_text: str, analogues: Sequence[Dict[str, Any]]) -> str:
    """The prompt block, or '' when there is nothing to show."""
    parts: List[str] = []
    if (lessons_text or "").strip():
        parts.append(
            "Lessons drawn from post-mortems of your resolved forecasts of this class. "
            "Each rests on several cases; treat them as tendencies to check, not rules.\n"
            + lessons_text.strip()
        )
    if analogues:
        lines = []
        for a in analogues:
            n = a.get("note") or {}
            lines.append(
                f"- {a['question_id']}: {n.get('what_happened', '')} "
                f"Forecast: {n.get('forecast_vs_outcome', '')} "
                f"Would have helped: {n.get('what_would_have_helped', '')}"
            )
        parts.append("Notes on past questions like this one:\n" + "\n".join(lines))
    if not parts:
        return ""
    return "\n\n" + LESSONS_HEADING + "\n" + "\n\n".join(parts)


def lessons_block_for(con, question: Any, as_of: Any = None) -> str:
    """Lessons plus analogues for *question*, rendered ('' when none)."""
    return render_lessons_block(
        load_lessons_text(con, question.hazard_code, question.metric, as_of),
        load_analogues(con, question),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _log_call(**kw: Any) -> None:
    from sibyl.cost import log_sibyl_call  # noqa: PLC0415

    log_sibyl_call(
        run_id=f"sibyl_postmortem_{date.today().strftime('%Y-%m')}",
        question_id=kw.get("question_id") or "",
        prompt_text=kw["prompt"], response_text=kw["response"], provider="anthropic",
        model_id=_cfg.MODEL, usage=kw["usage"], iso3=kw.get("iso3") or "",
        hazard_code=kw.get("hazard_code") or "", metric=kw.get("metric") or "",
        error_text=kw.get("error") or "", call_type=kw["call_type"],
    )


def run(con, *, as_of_month: Optional[str] = None, call: Optional[ModelCall] = None,
        log: Optional[Callable[..., None]] = _log_call) -> Dict[str, Any]:
    budget = Budget(_cfg.POSTMORTEM_CAP_USD)
    notes = write_notes(con, call=call, budget=budget, log=log)
    relabelled = relabel_notes(con, call=call, budget=budget, log=log)
    lessons = write_lessons(con, as_of_month=as_of_month, call=call, budget=budget, log=log)
    # The rates go to the pooled advice row (sibyl.advice ran before this
    # step) and to the dashboard; never into a trial's prompt.
    attach_failure_rates(con)
    return {"notes_written": notes, "notes_relabelled": relabelled, "lessons": lessons,
            "spent_usd": round(budget.spent_usd, 4), "cap_usd": budget.cap_usd}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sibyl post-mortems and lessons")
    parser.add_argument("--db-url", default=None)
    parser.add_argument("--as-of-month", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import os

    if args.db_url:
        os.environ["PYTHIA_DB_URL"] = args.db_url
    from pythia.db.schema import connect, ensure_schema  # noqa: PLC0415

    ensure_schema()
    con = connect(read_only=False)
    try:
        summary = run(con, as_of_month=args.as_of_month)
    except Exception as exc:  # noqa: BLE001 - never fails the calibration chain
        logger.exception("sibyl.postmortem failed: %s", exc)
        return 0
    finally:
        con.close()
    print(f"sibyl_postmortem {json.dumps(summary, default=str)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
