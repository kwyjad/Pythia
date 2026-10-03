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

NOTE_PROMPT = """You forecast the question below some months ago and the outcome is now known. Write a short post-mortem.

QUESTION: {wording}
Class: {hazard_code} / {metric}. Country: {country}.
Your forecast (raw pool of your research trials, before the reference was mixed in):
{forecast}
Outcome by window month: {outcomes}
Your reference (prior) median for month 1: {reference_median}
What your trials recorded as evidence (excerpt):
{evidence}

Answer ONLY with a JSON object:
{{"what_happened": "<one or two sentences>",
  "forecast_vs_outcome": "<one sentence: too high, too low, too wide, too narrow, about right>",
  "what_would_have_helped": "<one or two sentences: which evidence or reasoning would have moved you the right way>",
  "general_lesson": "<one sentence that would help on a DIFFERENT question of this class; name no country, place, year or event>"}}"""

LESSONS_PROMPT = """Below are post-mortem notes on {n} of your resolved forecasts of {hazard_code} / {metric} questions. Each note has an id.

{notes}

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


def _forecast_context(con, qid: str, srid: str) -> Dict[str, Any]:
    try:
        row = con.execute(
            "SELECT raw_by_month_json, reference_json, trials_json FROM sibyl_forecasts "
            "WHERE question_id = ? AND sibyl_run_id = ? LIMIT 1", [qid, srid],
        ).fetchone()
    except Exception:  # noqa: BLE001
        row = None
    out: Dict[str, Any] = {"raw": {}, "reference_median": None, "evidence": []}
    if not row:
        return out
    try:
        raw = json.loads(row[0]) if row[0] else {}
        out["raw"] = {k: v for k, v in (raw.get("quantiles") or {}).items() if k in ("1", "6")}
    except (TypeError, ValueError):
        pass
    try:
        from sibyl.aggregate import dist_from_vector  # noqa: PLC0415
        from sibyl.trials import month_median  # noqa: PLC0415

        ref = json.loads(row[1]) if row[1] else {}
        vec = (ref.get("by_month") or {}).get("1")
        if vec:
            metric = con.execute(
                "SELECT metric FROM sibyl_forecasts WHERE question_id = ? LIMIT 1", [qid]
            ).fetchone()[0]
            d = dist_from_vector(vec, str(metric).upper())
            out["reference_median"] = round(month_median(d.p_zero, d.qpos), 1)
    except Exception:  # noqa: BLE001
        pass
    try:
        for t in json.loads(row[2]) if row[2] else []:
            for it in (t or {}).get("ledger") or []:
                out["evidence"].append(
                    f"[{it.get('date') or 'undated'}] {it.get('direction') or ''}: "
                    f"{str(it.get('quote') or '')[:200]}"
                )
    except (TypeError, ValueError):
        pass
    out["evidence"] = out["evidence"][:20]
    return out


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
        prompt = NOTE_PROMPT.format(
            wording=info["wording"] or "(no wording stored)",
            hazard_code=r.hazard_code, metric=r.metric, country=info["iso3"] or "?",
            forecast=json.dumps(ctx["raw"] or {"month_1": r.quantiles}, default=str),
            outcomes=outcomes, reference_median=ctx["reference_median"],
            evidence="\n".join(ctx["evidence"]) or "(none recorded)",
        )
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
        con.execute(
            """
            INSERT INTO sibyl_postmortem_notes
                (question_id, sibyl_run_id, iso3, hazard_code, metric, note_json, model,
                 cost_usd, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [r.question_id, str(r.sibyl_run_id), info["iso3"], r.hazard_code, r.metric,
             json.dumps(note), _cfg.MODEL, cost],
        )
        written += 1
    return written


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
            "SELECT question_id, iso3, note_json FROM sibyl_postmortem_notes "
            "WHERE hazard_code = ? AND metric = ? ORDER BY created_at, question_id",
            [hz, metric],
        ).fetchall()
        ids = [str(q) for q, _, _ in notes]
        iso3s = sorted({str(i) for _, i, _ in notes if i})
        blocks = []
        for qid, _, nj in notes:
            try:
                nd = json.loads(nj)
            except (TypeError, ValueError):
                nd = {}
            blocks.append(
                f"[{qid}] happened: {nd.get('what_happened', '')} | "
                f"forecast: {nd.get('forecast_vs_outcome', '')} | "
                f"would have helped: {nd.get('what_would_have_helped', '')} | "
                f"lesson: {nd.get('general_lesson', '')}"
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
    lessons = write_lessons(con, as_of_month=as_of_month, call=call, budget=budget, log=log)
    return {"notes_written": notes, "lessons": lessons, "spent_usd": round(budget.spent_usd, 4),
            "cap_usd": budget.cap_usd}


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
