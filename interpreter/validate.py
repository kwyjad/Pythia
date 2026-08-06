# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Phase 4 validation: runs after generation, before storage (plan §8).

Four checks, each reported separately in ``validation_json`` so failures are
inspectable:

1. **schema** — jsonschema shape validation (interpreter/schema.py) plus the
   kind-conditional requirements.
2. **referential** — every ``question_id`` the content cites exists in the
   pack; every figure reference (both ``figure_refs`` entries and inline
   ``{{fig:...}}`` placeholders) resolves against that entry's figure maps.
3. **numeric** — each referenced figure that can be re-derived from the DB
   with INDEPENDENT SQL (forecast_deviation, hs_triage, forecasts_raw — not
   the pack files) is recomputed and compared. A mismatch beyond tolerance
   is a pack bug and fails the check. Skipped (reported, never silently)
   when no DB connection or source table is available.
4. **prose** — the bare-numeral lint (digits in prose outside a placeholder
   fail, with a whitelist for calendar month/year references), lexicon
   band agreement (a lexicon word attached to a probability figure must sit
   in that figure's band), and a per-field length cap.

Failures set ``status='failed_validation'``; the report is still written so
it can be inspected. With PYTHIA_INTERPRETER_STRICT_VALIDATION=1 the runner
additionally suppresses the publication artifacts (--out-dir files); the
Phase 5 dashboard renders non-ok rows behind an explicit warning banner.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

from interpreter import lexicon, schema
from interpreter.render import _PLACEHOLDER  # the single placeholder regex

LOGGER = logging.getLogger(__name__)

# Relative tolerance for the numeric guard. Pack values ride through CSV/JSON
# string round-trips, so they are copies, not recomputations — anything past
# float formatting noise means the pack and the DB disagree.
NUMERIC_RTOL = 1e-6
NUMERIC_ATOL = 1e-9

# Per-field prose cap (the jsonschema maxLength is 2000; this keeps the lint
# self-contained if the schema cap ever moves).
PROSE_MAX_CHARS = 2000

# The prose fields linted, by container. (question_ids / figure_refs / enums
# are data, not prose.)
_ENTRY_PROSE_FIELDS = (
    "why_it_stands_out",
    "how_to_read_the_distribution",
    "what_the_model_was_reacting_to",
)
_ENTRY_PROSE_LISTS = ("impacts", "operational_challenges")
_PERFORMANCE_PROSE_FIELDS = (
    "plain_summary", "skill_statement", "track_comparison",
    "sibyl_comparison", "vs_system_average",
)
_TOP_PROSE_LISTS = ("changes_since_last_run", "blind_spots", "confidence_notes")

# Whitelisted digit uses in prose: a calendar year (1900–2099), optionally
# preceded by a month name or preceded by "early/mid/late". Everything else
# containing a digit is a bare numeral — the model tried to write a number
# instead of a placeholder.
_MONTHS = (
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
)
_YEAR_RE = re.compile(
    r"(?:(?:" + "|".join(_MONTHS) + r")\s+)?(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
_DIGIT_RE = re.compile(r"\d")

# Probability-typed figure keys the lexicon check can anchor a word against.
_PROBABILITY_KEYS = ("p_modal_bucket", "p_top_two_buckets", "p_event_mean",
                     "rc_score", "triage_score")


@dataclass
class CheckResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    skipped: bool = False
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"passed": self.passed, "errors": self.errors}
        if self.skipped:
            out["skipped"] = True
        if self.detail:
            out["detail"] = self.detail
        return out


@dataclass
class ValidationReport:
    checks: dict[str, CheckResult]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": {name: c.as_dict() for name, c in self.checks.items()},
        }


# ---------------------------------------------------------------------------
# Content walking helpers
# ---------------------------------------------------------------------------


def _prose_fields(content: dict[str, Any]) -> list[tuple[str, list[str], str]]:
    """Every prose (path, question_ids-context, text) triple in the content."""
    out: list[tuple[str, list[str], str]] = []
    if content.get("headline"):
        out.append(("headline", [], str(content["headline"])))
    for i, entry in enumerate(content.get("attention") or []):
        qids = [str(q) for q in entry.get("question_ids") or []]
        for name in _ENTRY_PROSE_FIELDS:
            if entry.get(name):
                out.append((f"attention[{i}].{name}", qids, str(entry[name])))
        for name in _ENTRY_PROSE_LISTS:
            for j, item in enumerate(entry.get(name) or []):
                out.append((f"attention[{i}].{name}[{j}]", qids, str(item)))
    performance = content.get("performance") or {}
    for name in _PERFORMANCE_PROSE_FIELDS:
        if performance.get(name):
            out.append((f"performance.{name}", [], str(performance[name])))
    for side in ("best_calls", "worst_calls"):
        for i, call in enumerate(performance.get(side) or []):
            qids = [str(q) for q in call.get("question_ids") or []]
            for name in ("what_was_right", "what_went_wrong"):
                if call.get(name):
                    out.append((f"performance.{side}[{i}].{name}", qids, str(call[name])))
    for name in _TOP_PROSE_LISTS:
        for i, item in enumerate(content.get(name) or []):
            out.append((f"{name}[{i}]", [], str(item)))
    return out


def _cited_question_ids(content: dict[str, Any]) -> list[tuple[str, str]]:
    """Every (path, question_id) citation in the content."""
    out: list[tuple[str, str]] = []
    for i, entry in enumerate(content.get("attention") or []):
        for q in entry.get("question_ids") or []:
            out.append((f"attention[{i}]", str(q)))
    performance = content.get("performance") or {}
    for side in ("best_calls", "worst_calls"):
        for i, call in enumerate(performance.get(side) or []):
            for q in call.get("question_ids") or []:
                out.append((f"performance.{side}[{i}]", str(q)))
    return out


def _referenced_figures(
    content: dict[str, Any],
) -> list[tuple[str, list[str], str]]:
    """Every (path, question_ids-context, figure key) reference: explicit
    ``figure_refs`` entries AND inline placeholders in prose."""
    out: list[tuple[str, list[str], str]] = []
    for i, entry in enumerate(content.get("attention") or []):
        qids = [str(q) for q in entry.get("question_ids") or []]
        for key in entry.get("figure_refs") or []:
            out.append((f"attention[{i}].figure_refs", qids, str(key)))
    performance = content.get("performance") or {}
    for side in ("best_calls", "worst_calls"):
        for i, call in enumerate(performance.get(side) or []):
            qids = [str(q) for q in call.get("question_ids") or []]
            for key in call.get("figure_refs") or []:
                out.append((f"performance.{side}[{i}].figure_refs", qids, str(key)))
    for path, qids, text in _prose_fields(content):
        for match in _PLACEHOLDER.finditer(text):
            out.append((path, qids, match.group(1)))
    return out


def _lookup_figure(
    key: str,
    qids: list[str],
    per_question: dict[str, dict[str, Any]],
    global_figures: dict[str, Any],
) -> tuple[bool, Any]:
    for qid in qids:
        figs = per_question.get(qid)
        if figs and key in figs:
            return True, figs[key]
    if key in global_figures:
        return True, global_figures[key]
    return False, None


# ---------------------------------------------------------------------------
# Check 2: referential
# ---------------------------------------------------------------------------


def check_referential(
    content: dict[str, Any],
    *,
    valid_question_ids: set[str],
    per_question: dict[str, dict[str, Any]],
    global_figures: dict[str, Any],
) -> CheckResult:
    errors: list[str] = []
    for path, qid in _cited_question_ids(content):
        if qid not in valid_question_ids:
            errors.append(f"{path}: question_id {qid!r} is not in the pack")
    for path, qids, key in _referenced_figures(content):
        found, _ = _lookup_figure(key, qids, per_question, global_figures)
        if not found:
            errors.append(
                f"{path}: figure {key!r} does not resolve "
                f"(context qids: {', '.join(qids) or 'global'})"
            )
    return CheckResult(passed=not errors, errors=errors)


# ---------------------------------------------------------------------------
# Check 3: numeric guard (independent SQL)
# ---------------------------------------------------------------------------


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=NUMERIC_RTOL, abs_tol=NUMERIC_ATOL)


def _table_exists(con, name: str) -> bool:
    try:
        con.execute(f"SELECT 1 FROM {name} LIMIT 0")
        return True
    except Exception:  # noqa: BLE001
        return False


def _db_deviation_values(con, run_id: str | None, qid: str) -> list[dict[str, Any]]:
    params: list[Any] = [qid]
    run_clause = ""
    if run_id:
        run_clause = " AND run_id = ?"
        params.append(run_id)
    rows = con.execute(
        "SELECT js_vs_baserate, log_ev_ratio, eiv_nominal, eiv_per_100k "
        f"FROM forecast_deviation WHERE question_id = ?{run_clause}",
        params,
    ).fetchall()
    return [
        {"js_vs_baserate": r[0], "log_ev_ratio": r[1],
         "eiv_nominal": r[2], "eiv_per_100k": r[3]}
        for r in rows
    ]


def _db_triage_values(con, qid: str) -> list[dict[str, Any]]:
    rows = con.execute(
        """
        SELECT t.regime_change_level, t.regime_change_score, t.triage_score
        FROM questions q
        JOIN hs_triage t ON t.run_id = q.hs_run_id
            AND upper(t.iso3) = upper(q.iso3)
            AND upper(t.hazard_code) = upper(q.hazard_code)
        WHERE q.question_id = ?
        """,
        [qid],
    ).fetchall()
    return [
        {"rc_level": r[0], "rc_score": r[1], "triage_score": r[2]}
        for r in rows
    ]


_DEVIATION_KEYS = ("js_vs_baserate", "log_ev_ratio", "eiv_nominal", "eiv_per_100k")
_TRIAGE_KEYS = ("rc_level", "rc_score", "triage_score")


def check_numeric(
    content: dict[str, Any],
    *,
    con,
    run_id: str | None,
    per_question: dict[str, dict[str, Any]],
    global_figures: dict[str, Any],
) -> CheckResult:
    """Re-derive referenced figures from the DB and compare to pack values.

    Only DB-checkable keys are compared; per-figure outcomes are counted in
    the detail. With no connection or no source tables the check is SKIPPED
    (reported as such) — a missing DB must not read as a validated pack.
    """
    if con is None:
        return CheckResult(passed=True, skipped=True,
                           detail={"reason": "no DB connection provided"})
    have_dev = _table_exists(con, "forecast_deviation")
    have_triage = _table_exists(con, "questions") and _table_exists(con, "hs_triage")
    if not have_dev and not have_triage:
        return CheckResult(passed=True, skipped=True,
                           detail={"reason": "no source tables in this DB"})

    errors: list[str] = []
    n_checked = 0
    n_unverifiable = 0
    seen: set[tuple[str, str]] = set()

    for path, qids, key in _referenced_figures(content):
        found, pack_value = _lookup_figure(key, qids, per_question, global_figures)
        if not found:
            continue  # the referential check owns unresolvable refs
        qid = next((q for q in qids if key in (per_question.get(q) or {})), None)
        if qid is None:
            continue  # global figures have no per-question DB row to check
        if (qid, key) in seen:
            continue
        seen.add((qid, key))
        try:
            pack_float = float(pack_value)
        except (TypeError, ValueError):
            continue  # labels/sources are covered by the referential check

        candidates: list[Any] = []
        if key in _DEVIATION_KEYS and have_dev:
            candidates = [
                row[key] for row in _db_deviation_values(con, run_id, qid)
                if row.get(key) is not None
            ]
        elif key in _TRIAGE_KEYS and have_triage:
            candidates = [
                row[key] for row in _db_triage_values(con, qid)
                if row.get(key) is not None
            ]
        else:
            n_unverifiable += 1
            continue

        n_checked += 1
        if not candidates:
            errors.append(
                f"{path}: figure {key!r} for {qid} has no DB row to verify against"
            )
            continue
        if not any(_close(pack_float, float(c)) for c in candidates):
            errors.append(
                f"{path}: figure {key!r} for {qid} is {pack_float!r} in the "
                f"pack but the DB says {sorted(float(c) for c in candidates)!r} "
                "— the pack and the DB disagree (pack bug)"
            )

    return CheckResult(
        passed=not errors, errors=errors,
        detail={"n_checked": n_checked, "n_unverifiable_keys": n_unverifiable},
    )


# ---------------------------------------------------------------------------
# Check 4: prose lint
# ---------------------------------------------------------------------------


def _strip_placeholders(text: str) -> str:
    return _PLACEHOLDER.sub(" ", text or "")


def _strip_whitelisted(text: str) -> str:
    return _YEAR_RE.sub(" ", text)


def find_bare_numerals(text: str) -> list[str]:
    """Digit runs in prose after removing placeholders and the calendar
    whitelist. Any survivor is a bare numeral."""
    cleaned = _strip_whitelisted(_strip_placeholders(text))
    return re.findall(r"[\d][\d,.–—%kKmM-]*", cleaned) if _DIGIT_RE.search(cleaned) else []


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?;])\s+", text or "") if s.strip()]


def _lexicon_words_in(text: str) -> list[str]:
    """Lexicon phrases in a sentence, matched longest-first and CONSUMED so
    'very unlikely' does not also count as 'unlikely' (or 'likely'), while a
    separate standalone 'likely' elsewhere in the sentence still counts."""
    remaining = (text or "").lower()
    found: list[str] = []
    for word in sorted(lexicon.WORDS, key=len, reverse=True):
        pattern = r"\b" + re.escape(word) + r"\b"
        hits = re.findall(pattern, remaining)
        if hits:
            found.extend([word] * len(hits))
            remaining = re.sub(pattern, " ", remaining)
    return found


def check_prose(
    content: dict[str, Any],
    *,
    per_question: dict[str, dict[str, Any]],
    global_figures: dict[str, Any],
) -> CheckResult:
    errors: list[str] = []
    for path, qids, text in _prose_fields(content):
        if len(text) > PROSE_MAX_CHARS:
            errors.append(f"{path}: {len(text)} chars exceeds the {PROSE_MAX_CHARS} cap")
        numerals = find_bare_numerals(text)
        if numerals:
            errors.append(
                f"{path}: bare numeral(s) {numerals!r} in prose — figures "
                "must be {{fig:...}} placeholders"
            )
        # Lexicon band agreement, sentence-scoped and deliberately
        # conservative: only a sentence with exactly ONE lexicon word and at
        # least one resolvable probability placeholder is checked — an
        # ambiguous sentence is skipped, a checkable mismatch fails.
        for sentence in _sentences(text):
            words = _lexicon_words_in(sentence)
            if len(words) != 1:
                continue
            prob_values = []
            for match in _PLACEHOLDER.finditer(sentence):
                key = match.group(1)
                if key not in _PROBABILITY_KEYS:
                    continue
                found, value = _lookup_figure(key, qids, per_question, global_figures)
                if found:
                    try:
                        prob_values.append((key, float(value)))
                    except (TypeError, ValueError):
                        pass
            for key, p in prob_values:
                if not lexicon.word_matches(words[0], p):
                    errors.append(
                        f"{path}: {words[0]!r} does not match "
                        f"{{{{fig:{key}}}}} = {p:.3f} "
                        f"(band says {lexicon.word_for(p)!r})"
                    )
    return CheckResult(passed=not errors, errors=errors)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def validate_interpretation(
    content: dict[str, Any],
    *,
    kind: str,
    valid_question_ids: set[str],
    per_question: dict[str, dict[str, Any]],
    global_figures: dict[str, Any],
    con=None,
    run_id: str | None = None,
) -> ValidationReport:
    """Run all four checks. Never raises — a validator crash is reported as
    a failed check, not an unhandled error (the report must still store)."""
    checks: dict[str, CheckResult] = {}

    schema_errors = schema.validate_output(content, kind=kind)
    checks["schema"] = CheckResult(passed=not schema_errors, errors=schema_errors)

    for name, fn in (
        ("referential", lambda: check_referential(
            content, valid_question_ids=valid_question_ids,
            per_question=per_question, global_figures=global_figures)),
        ("numeric", lambda: check_numeric(
            content, con=con, run_id=run_id,
            per_question=per_question, global_figures=global_figures)),
        ("prose", lambda: check_prose(
            content, per_question=per_question, global_figures=global_figures)),
    ):
        try:
            checks[name] = fn()
        except Exception as exc:  # noqa: BLE001 - a crashed check is a failed check
            LOGGER.warning("[interpreter] %s check crashed: %s", name, exc)
            checks[name] = CheckResult(passed=False, errors=[f"check crashed: {exc}"])

    return ValidationReport(checks=checks)
