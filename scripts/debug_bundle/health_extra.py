# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Two health lines the forecast pipeline review needs and the table lacked.

**Prompt Cache.** On 13 November 2026 the executive summary showed prompt
caching on with a 0.0% hit rate for Google (6.1M prompt tokens) and 0.1% for
OpenAI, and the health table said nothing. A flag being on proves nothing;
only a cache read does. WARN for any provider with caching enabled, more than
``MIN_PROMPT_TOKENS`` prompt tokens and a hit rate below ``MIN_HIT_RATE_PCT``.

**Adversarial Sync.** CLAUDE.md's "CRITICAL sync constraint": adversarial
checks are generated for RC level 1 and above and injected into the SPD
prompt for RC level 1 and above. If the two thresholds drift, checks are paid
for and never shown, or a block is shown with no check behind it. This counts
questions at RC 1+, checks generated, checks injected (SPD prompts carrying
the block), checks withheld by the adversarial experiment, and the two
mismatches; FAIL when either mismatch is above zero.

Pure functions over rows the bundle already reads, plus one guarded DB reader.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

MIN_PROMPT_TOKENS = 1_000_000
MIN_HIT_RATE_PCT = 5.0
ADVERSARIAL_MARKER = "ADVERSARIAL EVIDENCE CHECK"


def _int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def prompt_cache_check(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The Prompt Cache health line from ``prompt_cache_report`` rows."""
    by_provider: dict[str, dict[str, Any]] = {}
    for r in rows:
        prov = str(r.get("provider") or "?")
        entry = by_provider.setdefault(prov, {"prompt_tokens": 0, "hit_tokens": 0, "enabled": False})
        entry["prompt_tokens"] += _int(r.get("prompt_tokens"))
        entry["hit_tokens"] += _int(r.get("cache_read_input_tokens")) + _int(r.get("cached_tokens"))
        if str(r.get("prompt_cache_enabled") or "").strip() in ("1", "true", "True"):
            entry["enabled"] = True
    low = []
    parts = []
    for prov, e in sorted(by_provider.items()):
        if not e["prompt_tokens"]:
            continue
        rate = 100.0 * e["hit_tokens"] / e["prompt_tokens"]
        parts.append(f"{prov} {rate:.1f}% of {e['prompt_tokens']:,}")
        if e["enabled"] and e["prompt_tokens"] > MIN_PROMPT_TOKENS and rate < MIN_HIT_RATE_PCT:
            low.append(f"{prov} {rate:.1f}% of {e['prompt_tokens']:,} prompt tokens")
    if not parts:
        return {"subsystem": "Prompt Cache", "status": "N/A", "detail": "no prompt tokens in this run"}
    if low:
        return {"subsystem": "Prompt Cache", "status": "WARN",
                "detail": "caching on, hit rate under 5%: " + "; ".join(low)
                          + " (a flag being on proves nothing; only a cache read does)"}
    return {"subsystem": "Prompt Cache", "status": "OK", "detail": "; ".join(parts)}


def adversarial_counts(*, rc1_spd_questions: Mapping[str, str], generated_pairs: set[str],
                       injected: set[str], withheld: set[str],
                       spd_question_pairs: Mapping[str, str]) -> dict[str, Any]:
    """``rc1_spd_questions``: question id -> ``ISO3|HZ`` for SPD questions at RC 1+;
    ``spd_question_pairs``: the same for every SPD question of the run."""
    with_check = {q for q, pair in rc1_spd_questions.items() if pair in generated_pairs}
    missed = sorted(with_check - injected - withheld)
    phantom = sorted(q for q in injected if spd_question_pairs.get(q) not in generated_pairs)
    return {
        "questions_rc1_plus": len(rc1_spd_questions),
        "checks_generated": len(generated_pairs),
        "rc1_plus_with_a_check": len(with_check),
        "injected": len(injected),
        "withheld": len(withheld & with_check),
        "generated_not_injected_not_withheld": missed,
        "injected_without_check": phantom,
    }


def adversarial_check(counts: Mapping[str, Any]) -> dict[str, Any]:
    missed = counts.get("generated_not_injected_not_withheld") or []
    phantom = counts.get("injected_without_check") or []
    detail = (f"{counts.get('questions_rc1_plus', 0)} SPD questions at RC 1+; "
              f"{counts.get('checks_generated', 0)} checks generated; "
              f"{counts.get('injected', 0)} injected; {counts.get('withheld', 0)} withheld by the experiment; "
              f"{len(missed)} generated, not injected, not withheld; {len(phantom)} injected with no check")
    if missed:
        detail += " (" + ", ".join(missed[:5]) + ")"
    if phantom:
        detail += " [no check: " + ", ".join(phantom[:5]) + "]"
    if not counts.get("questions_rc1_plus") and not counts.get("checks_generated"):
        return {"subsystem": "Adversarial Sync", "status": "N/A", "detail": detail}
    status = "FAIL" if (missed or phantom) else "OK"
    return {"subsystem": "Adversarial Sync", "status": status, "detail": detail}


def collect_adversarial(con: Any, hs_run_id: str | None, fc_run_id: str | None) -> dict[str, Any]:
    """Counts for the Adversarial Sync line, from the DB. Raises on a missing table."""
    if not hs_run_id or not fc_run_id:
        raise RuntimeError("no hs_run_id or forecaster run id")
    generated = {f"{r[0]}|{r[1]}" for r in con.execute(
        "SELECT DISTINCT upper(iso3), upper(hazard_code) FROM hs_adversarial_checks WHERE hs_run_id = ?",
        [hs_run_id]).fetchall()}
    # hs_triage keys its HS run on ``run_id``; hs_adversarial_checks on ``hs_run_id``.
    rows = con.execute(
        """
        SELECT DISTINCT fe.question_id, upper(q.iso3), upper(q.hazard_code),
               COALESCE(t.regime_change_level, 0)
        FROM forecasts_ensemble fe
        JOIN questions q ON q.question_id = fe.question_id
        LEFT JOIN hs_triage t ON t.run_id = ? AND upper(t.iso3) = upper(q.iso3)
                              AND upper(t.hazard_code) = upper(q.hazard_code)
        WHERE fe.run_id = ? AND upper(q.metric) <> 'EVENT_OCCURRENCE'
        """, [hs_run_id, fc_run_id]).fetchall()
    spd = {str(r[0]): f"{r[1]}|{r[2]}" for r in rows}
    rc1 = {str(r[0]): f"{r[1]}|{r[2]}" for r in rows if int(r[3] or 0) >= 1}
    injected = {str(r[0]) for r in con.execute(
        "SELECT DISTINCT question_id FROM llm_calls WHERE run_id = ? AND phase = 'spd_v2' "
        "AND prompt_text LIKE ?", [fc_run_id, f"%{ADVERSARIAL_MARKER}%"]).fetchall() if r[0]}
    withheld: set[str] = set()
    try:
        withheld = {str(r[0]) for r in con.execute(
            "SELECT DISTINCT question_id FROM forecasts_ensemble WHERE run_id = ? AND adversarial_arm = 'withheld'",
            [fc_run_id]).fetchall()}
    except Exception:  # noqa: BLE001 - a DB before the experiment has no column
        withheld = set()
    return adversarial_counts(rc1_spd_questions=rc1, generated_pairs=generated, injected=injected,
                              withheld=withheld, spd_question_pairs=spd)
