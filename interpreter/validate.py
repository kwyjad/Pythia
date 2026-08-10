# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Phase 4 validation: runs after generation, before storage (plan §8).

Seven checks, each reported separately in ``validation_json`` so failures are
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
5. **style** — the house-style rules that can be checked mechanically: no em
   or en dashes, no banned words, no "not X, but Y", and no bare codes where
   the reader needs a name.
6. **categories** — the pack decides which forecasts the report covers and
   under which heading; the model copies that decision and may not change it.
7. **proper_nouns** — a name in prose must appear in the pack's evidence.
   Reports rather than fails (see ``check_proper_nouns``), but the violations
   are quoted back to the model in the correction pass.

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
    "planning_sentence",
    "spd_shape",
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
    if content.get("run_summary"):
        out.append(("run_summary", [], str(content["run_summary"])))
    for i, entry in enumerate(content.get("attention") or []):
        qids = [str(q) for q in entry.get("question_ids") or []]
        for name in _ENTRY_PROSE_FIELDS:
            if entry.get(name):
                out.append((f"attention[{i}].{name}", qids, str(entry[name])))
        for name in _ENTRY_PROSE_LISTS:
            for j, item in enumerate(entry.get(name) or []):
                out.append((f"attention[{i}].{name}[{j}]", qids, str(item)))
        # The decision point's action is prose and is linted like any other:
        # its sibling fields (deadline_month, basis) are data the runner
        # derives, and carry digits by design.
        action = (entry.get("decision_point") or {}).get("action")
        if action:
            out.append((f"attention[{i}].decision_point.action", qids, str(action)))
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


def _normalise_figure_key(key: str) -> str:
    """``fig:eiv_nominal`` and ``{{fig:eiv_nominal}}`` both mean ``eiv_nominal``.

    The figure maps are keyed on the bare name, but a model asked to write
    ``{{fig:x}}`` in prose reasonably writes ``fig:x`` in figure_refs too. The
    prefix carries no meaning, so accepting it costs nothing; rejecting it
    failed a whole report over punctuation.
    """
    text = str(key or "").strip()
    if text.startswith("{{") and text.endswith("}}"):
        text = text[2:-2].strip()
    if text.startswith("fig:"):
        text = text[4:]
    return text


def _lookup_figure(
    key: str,
    qids: list[str],
    per_question: dict[str, dict[str, Any]],
    global_figures: dict[str, Any],
) -> tuple[bool, Any]:
    key = _normalise_figure_key(key)
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
    out: list[dict[str, Any]] = []
    for r in rows:
        row = {"js_vs_baserate": r[0], "log_ev_ratio": r[1],
               "eiv_nominal": r[2], "eiv_per_100k": r[3]}
        # ev_multiple is the readable form of log_ev_ratio and is what the
        # report actually leads with, so it must be verifiable. Without it the
        # guard reported "0 checked, 9 unverifiable" on a whole live report:
        # honest, and no protection at all.
        if r[1] is not None:
            try:
                row["ev_multiple"] = math.exp(float(r[1]))
            except (TypeError, ValueError, OverflowError):
                pass
        out.append(row)
    return out


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


_DEVIATION_KEYS = ("js_vs_baserate", "log_ev_ratio", "ev_multiple",
                   "eiv_nominal", "eiv_per_100k")
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
# Check 5: house style
# ---------------------------------------------------------------------------

# Words and phrases that mark a text as machine-written to a reader who reads
# a lot of them. The ban is absolute: there is always another word.
BANNED_PHRASES = (
    "delve", "tapestry", "treasure trove", "unleash", "game-changer",
    "game changer", "revolutionary", "landscape", "utilize", "leverage",
    "pivotal", "intricate", "load-bearing", "the distinction matters",
    "table stakes", "on the other hand",
)

# Two constructions the reader asked to be rid of, matched loosely enough to
# catch the variants ("it's not X, it's Y", "not a failure, but a delay").
_NOT_BUT = re.compile(r"\bnot\s+(?:a|an|the)?\s*[^,.;]{1,40},\s*but\b", re.IGNORECASE)

# Em dash and en dash. Full stops, commas and semi-colons only.
_DASHES = ("—", "–")

# A code leaking into prose. Three capitals alone are NOT enough to condemn a
# token: FAO, IOM, WHO and ICRC are ordinary prose to a humanitarian reader,
# and flagging them failed a correct report. Only a token that is genuinely an
# ISO3 country code counts, because that is the one a reader cannot decode
# and the one the entry heading is supposed to have spelled out.
_CAPS_IN_PROSE = re.compile(r"(?<![A-Za-z])[A-Z]{3}(?![A-Za-z])")
_METRIC_CODES = ("EVENT_OCCURRENCE", "PHASE3PLUS_IN_NEED", "FATALITIES")
_HAZARD_SLASH = re.compile(r"\b[A-Z]{2}/[A-Z_]{2,}")

# A few ISO3 codes are also words or common acronyms in humanitarian prose.
# "CAR" is the Central African Republic's code and an English noun; "AND",
# "ARE", "CAN", "ALL", "ONE", "TON", "USA", "GBR" read as prose or as names a
# reader already knows. Allowing them risks missing a real code; refusing them
# guarantees false positives on correct writing, and a validator that cries
# wolf gets ignored.
_ALLOWED_ISO3 = {"AND", "ARE", "CAN", "ALL", "ONE", "TON", "USA", "COD", "CAR"}


def _iso3_codes() -> frozenset[str]:
    """The ISO3 set, from the same country table the report names things by."""
    try:
        from interpreter import names as _names

        return frozenset(_names.iso3_codes()) - _ALLOWED_ISO3
    except Exception:  # noqa: BLE001 - no table means no code check, not a crash
        return frozenset()


def find_banned_phrases(text: str) -> list[str]:
    lowered = (text or "").lower()
    hits = [p for p in BANNED_PHRASES if p in lowered]
    if _NOT_BUT.search(text or ""):
        hits.append("not X, but Y")
    return hits


def find_codes_in_prose(text: str) -> list[str]:
    """Codes the reader would have to look up. Placeholders are stripped first
    (a figure key is not prose), and known acronyms are allowed."""
    cleaned = _strip_placeholders(text or "")
    hits = [m for m in _METRIC_CODES if m in cleaned]
    hits.extend(_HAZARD_SLASH.findall(cleaned))
    codes = _iso3_codes()
    hits.extend(m for m in _CAPS_IN_PROSE.findall(cleaned) if m in codes)
    return hits


def check_style(content: dict[str, Any]) -> CheckResult:
    """The house-style rules that can be checked mechanically.

    Deliberately narrow. Rhythm, variety and the habit of joining clauses with
    "and" are asked for in the prompt and judged by a reader; a lint that tried
    to score them would fail honest prose. What is here is unambiguous: a
    banned word is banned, an em dash is an em dash, and a country code in a
    sentence is a code the reader has to decode.
    """
    errors: list[str] = []
    for path, _qids, text in _prose_fields(content):
        for dash in _DASHES:
            if dash in text:
                errors.append(
                    f"{path}: contains {dash!r} — use a full stop, comma or semi-colon"
                )
                break
        banned = find_banned_phrases(text)
        if banned:
            errors.append(f"{path}: banned phrase(s) {banned!r}")
        codes = find_codes_in_prose(text)
        if codes:
            errors.append(
                f"{path}: code(s) {sorted(set(codes))!r} in prose — "
                "write the country, hazard and metric out in words"
            )
    return CheckResult(passed=not errors, errors=errors)


# ---------------------------------------------------------------------------
# Check 7: proper nouns (task 8)
# ---------------------------------------------------------------------------

# The existing guards protect NUMBERS. They do not protect NAMES, and the
# report's footer tells the reader that every figure in it is machine-derived,
# which invites them to trust the rest of it too. The August report named
# "Typhoon Maysak" in its Vietnam entry. That string appears nowhere in the
# pack. The model supplied it.
#
# So: a proper noun in prose must appear somewhere in the evidence the pack
# actually carried. Named storms, named operations, named agreements and
# named places are the risk cases, and all of them are things a reader will
# take as fact.
#
# This check REPORTS rather than fails (the brief's instruction): the
# violations land in validation_json and in the correction pass's complaints,
# so the model is asked to remove the name, but a false positive on an unusual
# spelling cannot stop a report being published.

# Capitalised runs, allowing the small joining words a name can contain
# ("Horn of Africa", "Lake Chad Basin"). Sentence-initial words are excluded
# by the caller, because a capital there carries no information.
_PROPER_RUN = re.compile(
    r"\b[A-Z][A-Za-z'’-]+(?:\s+(?:of|the|and|de|du|da|el|al)\s+[A-Z][A-Za-z'’-]+"
    r"|\s+[A-Z][A-Za-z'’-]+)*"
)
_SENTENCE_START = re.compile(r"(?:^|(?<=[.!?;:])\s+|(?<=^)|(?<=\n))")

# Words that are capitalised in ordinary writing and name nothing a reader
# could be misled by: the calendar, the report's own vocabulary, and the
# system's own parts.
_PROPER_ALLOWED = {
    *(m.capitalize() for m in _MONTHS),
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    "Sunday",
    "Fred", "Sibyl", "Track", "Brier", "Phase", "Level", "Deep", "Research",
    "North", "South", "East", "West", "Northern", "Southern", "Eastern",
    "Western", "Central", "Horn", "Sahel", "Africa", "Asia", "Europe",
    "America", "Americas", "Pacific", "Atlantic", "Indian", "Caribbean",
    "January", "The", "A", "An", "This", "That", "These", "Those", "It",
    "Its", "There", "Their", "They", "We", "Our", "If", "In", "On", "At",
    "By", "For", "From", "With", "But", "And", "Or", "So", "As", "Where",
    "When", "While", "Both", "Neither", "Each", "Every", "No", "Not",
    "Government", "Ministry", "United", "Nations", "Red", "Cross", "Crescent",
}


def _known_names() -> set[str]:
    """Country and hazard names the report is REQUIRED to print.

    They come from the system's own tables, so they are never the model's
    invention even when they do not appear in a pack's evidence text.
    """
    out: set[str] = set()
    try:
        from interpreter import names as _names

        for code in _names.iso3_codes():
            for word in _names.country_name(code).replace("-", " ").split():
                if word[:1].isupper():
                    out.add(word)
        out.update(_names.HAZARD_NAMES.values())
    except Exception:  # noqa: BLE001 - a missing table means fewer allowances
        pass
    return out


def find_unsupported_proper_nouns(text: str, evidence_lower: str) -> list[str]:
    """Proper nouns in one prose field that the pack's evidence does not carry."""
    cleaned = _strip_placeholders(text or "")
    allowed = _PROPER_ALLOWED | _known_names()
    hits: list[str] = []
    # Only look at what follows a sentence boundary's first word: the first
    # word of a sentence is capitalised by grammar, not by naming.
    for sentence in _sentences(cleaned):
        body = sentence.strip()
        # Drop the first token so a sentence-initial capital never counts.
        first_space = body.find(" ")
        body = body[first_space + 1:] if first_space > 0 else ""
        for match in _PROPER_RUN.finditer(body):
            phrase = match.group(0).strip()
            if len(phrase) < 4:
                continue
            words = [w for w in phrase.replace("-", " ").split() if w[:1].isupper()]
            if all(w in allowed for w in words):
                continue
            if phrase.lower() in evidence_lower:
                continue
            # A multiword name whose distinctive half is present is supported:
            # "Typhoon Yagi" is fine when the pack says "Yagi".
            distinctive = [w for w in words if w not in allowed]
            if distinctive and all(w.lower() in evidence_lower for w in distinctive):
                continue
            hits.append(phrase)
    return hits


def check_proper_nouns(
    content: dict[str, Any], *, evidence_text: str | None
) -> CheckResult:
    """Names in prose must come from the pack, not from the model's memory.

    Non-blocking by design: reported in ``validation_json`` and quoted back to
    the model in the correction pass. A name the model invented is a serious
    defect, and a validator that failed the whole report on an unusual
    spelling would get switched off within two months.
    """
    if not evidence_text:
        return CheckResult(
            passed=True, skipped=True,
            detail={"reason": "pack carried no evidence text to check against"},
        )
    evidence_lower = evidence_text.lower()
    violations: list[str] = []
    for path, _qids, text in _prose_fields(content):
        for phrase in find_unsupported_proper_nouns(text, evidence_lower):
            violations.append(
                f"{path}: {phrase!r} appears nowhere in the pack — remove the "
                "name or describe the thing without naming it"
            )
    if violations:
        LOGGER.warning(
            "[interpreter] %d proper noun(s) in prose are not supported by the "
            "pack: %s", len(violations), violations[:5],
        )
    # passed stays True: this check informs, it does not block.
    return CheckResult(
        passed=True, errors=[],
        detail={"n_violations": len(violations), "violations": violations},
    )


def check_categories(
    content: dict[str, Any],
    *,
    pack_categories: dict[str, tuple[str | None, str | None]],
) -> CheckResult:
    """The pack decides which forecasts are covered and under which heading.

    The model copies that decision across. If it promotes, demotes or invents
    an entry, the report says a country is worsening on the model's authority
    rather than the system's, which is exactly the claim the interpreter is
    not allowed to make. Skipped when the pack carries no categories (a scored
    interpretation has no attention list at all).
    """
    if not pack_categories:
        return CheckResult(
            passed=True, errors=[], skipped=True,
            detail={"reason": "pack carries no categorised rows"},
        )
    errors: list[str] = []
    seen: set[str] = set()
    for i, entry in enumerate(content.get("attention") or []):
        qids = [str(q) for q in entry.get("question_ids") or []]
        matched = [q for q in qids if q in pack_categories]
        if not matched:
            errors.append(
                f"attention[{i}]: cites no question the pack put in a category"
            )
            continue
        qid = matched[0]
        seen.add(qid)
        want_cat, want_family = pack_categories[qid]
        got_cat = entry.get("category")
        got_family = entry.get("hazard_family")
        if got_cat != want_cat:
            errors.append(
                f"attention[{i}] ({qid}): category {got_cat!r} but the pack "
                f"says {want_cat!r}"
            )
        if got_family != want_family:
            errors.append(
                f"attention[{i}] ({qid}): hazard_family {got_family!r} but the "
                f"pack says {want_family!r}"
            )
    missing = sorted(set(pack_categories) - seen)
    if missing:
        errors.append(
            f"attention: the pack categorised {len(missing)} question(s) the "
            f"report never covers: {missing[:5]}"
        )
    return CheckResult(
        passed=not errors, errors=errors,
        detail={"n_pack_categorised": len(pack_categories), "n_covered": len(seen)},
    )


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
    pack_categories: dict[str, tuple[str | None, str | None]] | None = None,
    evidence_text: str | None = None,
    require_performance: bool = True,
    require_attention: bool = True,
) -> ValidationReport:
    """Run every check. Never raises — a validator crash is reported as
    a failed check, not an unhandled error (the report must still store)."""
    checks: dict[str, CheckResult] = {}

    schema_errors = schema.validate_output(
        content, kind=kind, require_performance=require_performance,
        require_attention=require_attention,
    )
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
        ("style", lambda: check_style(content)),
        ("categories", lambda: check_categories(
            content, pack_categories=pack_categories or {})),
        ("proper_nouns", lambda: check_proper_nouns(
            content, evidence_text=evidence_text)),
    ):
        try:
            checks[name] = fn()
        except Exception as exc:  # noqa: BLE001 - a crashed check is a failed check
            LOGGER.warning("[interpreter] %s check crashed: %s", name, exc)
            checks[name] = CheckResult(passed=False, errors=[f"check crashed: {exc}"])

    return ValidationReport(checks=checks)
