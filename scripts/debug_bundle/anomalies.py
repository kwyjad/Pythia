# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Everything that failed or looks wrong, in one machine-readable list.

This is the first file a reader opens and the top of the executive
summary. Its job is to make a bad run announce itself, because the twelve
files that came before it made a bad run look exactly like a good one: the
September bundle reported "LLM Calls: OK, 0 errors" for a cycle that had
lost four provider batches, three member forecasts and five months of
CrisisWatch.

Each entry names its severity, the subsystem, what is wrong in one line,
and the bundle file holding the evidence. The evidence pointer is the part
that matters: a finding with nowhere to look is a rumour.
"""

from __future__ import annotations

from typing import Any

FAIL = "fail"
WARN = "warn"
INFO = "info"

_SEVERITY_ORDER = {FAIL: 0, WARN: 1, INFO: 2}

# Below this share of prompt tokens read from cache, on a provider that
# reported any cache activity at all, the caching is not working.
CACHE_HIT_WARN_PCT = 5.0
BATCH_FALLBACK_WARN_PCT = 20.0

# Sources whose emptiness is the intended state, so it is not a finding.
#: Connector rows whose empty table is the intended state. Empty since the
#: legacy ``ipc_phases`` table was dropped (Sept 2026); kept so a future
#: deliberately-silent source has somewhere to be declared.
SILENT_SOURCES: frozenset[str] = frozenset()


def _entry(severity: str, subsystem: str, description: str, evidence: str, **extra: Any) -> dict[str, Any]:
    row = {
        "severity": severity,
        "subsystem": subsystem,
        "description": description,
        "evidence_file": evidence,
    }
    row.update(extra)
    return row


def build(
    *,
    health_checks: list[dict[str, Any]] | None = None,
    batch_lifecycle: dict[str, Any] | None = None,
    prompt_cache_summary: dict[str, Any] | None = None,
    connector_rows: list[dict[str, Any]] | None = None,
    crisiswatch: dict[str, Any] | None = None,
    completeness_rollup: dict[str, Any] | None = None,
    workflow_logs_index: dict[str, Any] | None = None,
    retry_rows: list[dict[str, Any]] | None = None,
    breaker: dict[str, Any] | None = None,
    collector_errors: list[dict[str, Any]] | None = None,
    file_names: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Assemble the anomaly list. Never raises."""

    names = file_names or {}

    def f(key: str, default: str) -> str:
        return names.get(key, default)

    out: list[dict[str, Any]] = []

    # --- the existing health checks, translated ---
    for check in health_checks or []:
        status = str(check.get("status") or "").upper()
        if status == "OK":
            continue
        out.append(
            _entry(
                FAIL if status == "FAIL" else WARN,
                str(check.get("subsystem") or "pipeline"),
                str(check.get("detail") or ""),
                f("health_report", "health_report.json"),
            )
        )

    # --- batches ---
    bl = batch_lifecycle or {}
    totals = bl.get("totals") or {}
    for batch in bl.get("batches") or []:
        if batch.get("yielded_nothing"):
            out.append(
                _entry(
                    FAIL,
                    "batch",
                    f"{batch.get('provider')}/{batch.get('phase')} batch {batch.get('batch_id')} "
                    f"returned no results (provider state {batch.get('provider_state')}); "
                    f"{batch.get('n_requests')} requests fell through to synchronous full-price calls",
                    f("batch_lifecycle", "batch_lifecycle.json"),
                    batch_id=batch.get("batch_id"),
                )
            )
    if float(totals.get("fallback_pct_of_requests") or 0.0) > BATCH_FALLBACK_WARN_PCT:
        out.append(
            _entry(
                WARN,
                "batch",
                f"{totals['fallback_pct_of_requests']}% of batched requests fell back to "
                f"synchronous calls (${totals.get('lost_discount_usd', 0)} of discount lost)",
                f("batch_lifecycle", "batch_lifecycle.json"),
            )
        )

    # --- prompt cache ---
    pcs = prompt_cache_summary or {}
    if pcs.get("prompt_tokens"):
        flags = pcs.get("flags") or {}
        enabled = str(flags.get("PYTHIA_PROMPT_CACHE_ENABLED") or "").strip() in ("1", "true", "True")
        rate = float(pcs.get("cache_hit_rate_pct") or 0.0)
        if enabled and rate < CACHE_HIT_WARN_PCT:
            out.append(
                _entry(
                    WARN,
                    "prompt_cache",
                    f"prompt caching is enabled but only {rate}% of "
                    f"{pcs['prompt_tokens']:,} prompt tokens were read from cache — "
                    "the flag being on is not evidence the cache worked",
                    f("prompt_cache_report", "prompt_cache_report.csv"),
                )
            )

    # --- connectors ---
    for row in connector_rows or []:
        status = str(row.get("status") or "")
        if str(row.get("source") or "") in SILENT_SOURCES:
            # An empty table that is the intended state must not warn: a
            # warning about it teaches readers to skim this list.
            continue
        if status == "stale":
            out.append(
                _entry(
                    FAIL,
                    "connector",
                    f"{row.get('source')} is {row.get('age_days_at_run_start')} days old "
                    f"(threshold {row.get('stale_threshold_days')}d); prompt staleness label: "
                    f"{row.get('prompt_staleness_warning')}",
                    f("connector_freshness", "connector_freshness.csv"),
                    source=row.get("source"),
                )
            )
        elif status == "warn":
            out.append(
                _entry(
                    WARN,
                    "connector",
                    f"{row.get('source')} is {row.get('age_days_at_run_start')} days old "
                    f"(warn threshold {row.get('warn_threshold_days')}d)",
                    f("connector_freshness", "connector_freshness.csv"),
                    source=row.get("source"),
                )
            )
        elif status == "empty":
            out.append(
                _entry(
                    WARN,
                    "connector",
                    f"{row.get('source')} table {row.get('table')} holds no rows",
                    f("connector_freshness", "connector_freshness.csv"),
                    source=row.get("source"),
                )
            )

    # --- CrisisWatch's ACE inject, called out on its own ---
    cw = crisiswatch or {}
    if cw.get("available"):
        if int(cw.get("n_editions") or 0) > 1 and cw.get("edition_span"):
            out.append(
                _entry(
                    WARN,
                    "crisiswatch",
                    f"the table holds {cw['n_editions']} editions spanning {cw['edition_span']} — "
                    "per-country rows are a mix of editions, and the months between never landed",
                    f("connector_freshness", "connector_freshness.csv"),
                )
            )
        missing = int(cw.get("ace_countries_without_crisiswatch_row") or 0)
        if missing:
            out.append(
                _entry(
                    WARN,
                    "crisiswatch",
                    f"{missing} of {cw.get('ace_countries_forecast')} countries with ACE questions "
                    f"had no CrisisWatch row, so {cw.get('n_ace_questions_without_crisiswatch')} "
                    "ACE questions were forecast with no conflict arrow in their prompts",
                    f("connector_freshness", "connector_freshness.csv"),
                )
            )
        if int(cw.get("n_rows") or 0) and not int(cw.get("n_countries_with_arrow") or 0):
            out.append(
                _entry(
                    FAIL,
                    "crisiswatch",
                    "no country carries an arrow — the Global Overview parse produced nothing",
                    f("connector_freshness", "connector_freshness.csv"),
                )
            )

    # --- partial ensembles ---
    roll = completeness_rollup or {}
    if int(roll.get("n_cells_missing") or 0):
        by_model = ", ".join(
            f"{m}: {c}"
            for m, c in sorted((roll.get("by_model") or {}).items(), key=lambda kv: (-kv[1], kv[0]))
        )
        share = 0.0
        if roll.get("n_cells_expected"):
            share = roll["n_cells_missing"] / roll["n_cells_expected"]
        out.append(
            _entry(
                FAIL if share > 0.10 else WARN,
                "ensemble",
                f"{roll['n_cells_missing']} of {roll.get('n_cells_expected')} "
                f"(question, model, month) forecasts are missing or unusable ({by_model}); "
                f"{roll.get('n_question_months_short')} question-months were aggregated "
                "from fewer members than expected",
                f("model_completeness", "model_completeness.csv"),
            )
        )

    # --- retries and breakers ---
    for row in retry_rows or []:
        if int(row.get("n_billing") or 0):
            out.append(
                _entry(
                    FAIL,
                    "provider",
                    f"{row.get('provider')}/{row.get('model_id')} returned "
                    f"{row['n_billing']} billing/quota error(s) in {row.get('phase')}",
                    f("retry_report", "retry_report.csv"),
                )
            )
        if int(row.get("n_auth") or 0):
            out.append(
                _entry(
                    FAIL,
                    "provider",
                    f"{row.get('provider')}/{row.get('model_id')} returned "
                    f"{row['n_auth']} auth error(s) in {row.get('phase')} — a credential is wrong or missing",
                    f("retry_report", "retry_report.csv"),
                )
            )
    br = breaker or {}
    if int(br.get("brave_breaker_short_circuits") or 0):
        out.append(
            _entry(
                FAIL,
                "grounding",
                f"the Brave circuit breaker was tripped for "
                f"{br['brave_breaker_short_circuits']} call(s) — those hazards got no grounding evidence",
                f("retry_report", "retry_report.csv"),
            )
        )
    elif int(br.get("no_backend_calls") or 0):
        out.append(
            _entry(
                WARN,
                "grounding",
                f"{br['no_backend_calls']} grounding call(s) reached no backend at all",
                f("retry_report", "retry_report.csv"),
            )
        )

    # --- the bundle's own failures ---
    for problem in (workflow_logs_index or {}).get("problems") or []:
        out.append(_entry(INFO, "bundle", str(problem), "workflow_logs/INDEX.json"))
    for run in (workflow_logs_index or {}).get("runs") or []:
        if run.get("ok") is False:
            out.append(
                _entry(
                    INFO,
                    "bundle",
                    f"workflow log for run {run.get('run_id')} could not be fetched: {run.get('error')}",
                    str(run.get("file") or "workflow_logs/INDEX.json"),
                )
            )
    for err in collector_errors or []:
        out.append(
            _entry(
                WARN,
                "bundle",
                f"collector {err.get('collector')} failed: {err.get('error')}",
                str(err.get("file") or "BUNDLE_MANIFEST.json"),
            )
        )

    out.sort(key=lambda e: (_SEVERITY_ORDER.get(e["severity"], 3), e["subsystem"], e["description"]))
    return out


def counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    out = {FAIL: 0, WARN: 0, INFO: 0}
    for entry in entries:
        key = str(entry.get("severity") or INFO)
        out[key] = out.get(key, 0) + 1
    return out
