# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""ANALYST_GUIDE.md generation for AI analysis bundles.

The guide is written FOR the consuming AI. Anything that can drift with the
codebase (bucket definitions, model lineup) is rendered from the live source
of truth at build time — never hand-copied into prose.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

LOGGER = logging.getLogger(__name__)


def _bucket_tables_md() -> str:
    """Render bucket definitions from pythia.buckets (the single source)."""
    try:
        from pythia.buckets import BUCKET_SPECS, labels_for, thresholds_for
    except Exception:  # noqa: BLE001
        return "_Bucket definitions unavailable in this environment._\n"

    lines: list[str] = []
    for metric in BUCKET_SPECS:
        labels = labels_for(metric)
        thresholds = thresholds_for(metric)
        lines.append(f"**{metric}** ({len(labels)} buckets):")
        lines.append("")
        lines.append("| bucket_index | label | lower bound (incl.) |")
        lines.append("|---|---|---|")
        for i, label in enumerate(labels, start=1):
            lower = thresholds[i - 1] if i - 1 < len(thresholds) else ""
            lines.append(f"| {i} | {label} | {lower} |")
        lines.append("")
    lines.append(
        "**EVENT_OCCURRENCE (binary)** is not in BUCKET_SPECS: bucket_1 = "
        "P(event occurs, i.e. a GDACS Orange/Red alert month), bucket_2 = "
        "1 − P(yes), remaining buckets 0."
    )
    return "\n".join(lines) + "\n"


def _model_lineup_md() -> str:
    """Best-effort render of the current SPD ensemble lineup."""
    try:
        from pythia.llm_profiles import get_ensemble_resolved

        members = get_ensemble_resolved()
        rows = []
        for m in members:
            if isinstance(m, Mapping):
                provider = m.get("provider", "?")
                model_id = m.get("model_id", "?")
            else:
                provider = getattr(m, "provider", "?")
                model_id = getattr(m, "model_id", "?")
            rows.append(f"- `{model_id}` ({provider})")
        if rows:
            return (
                "Current Track-1 SPD ensemble (from config at bundle-build "
                "time — forecasts in this bundle may pre-date a lineup "
                "change; trust the per-question `members[]` model names):\n"
                + "\n".join(rows)
                + "\n"
            )
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Model lineup unavailable: %s", exc)
    return "_Model lineup unavailable in this environment — see run_config or the per-question `members[]` names._\n"


_PIPELINE_OVERVIEW = """\
Pythia is an end-to-end AI forecasting system for humanitarian crises. Each
monthly cycle:

1. **Horizon Scanner (HS)** — per country (≈122) and hazard (ACE=armed
   conflict, DR=drought/food insecurity, FL=flood, TC=tropical cyclone):
   a *regime-change (RC)* assessment (is this departing from its base
   rate? score = likelihood × magnitude, levels L0–L3) and a *triage*
   assessment (overall risk tier), each preceded by its own web-grounding
   search pass. RC level ≥ 1 also triggers an *adversarial check*
   (devil's-advocate counter-evidence search + synthesis).
2. **Question generation** — epoch-suffixed question IDs (e.g.
   `SOM_ACE_FATALITIES_2026-08`) with a 6-month forecast window
   (`window_start_date` = month 1, `target_month` = month 6).
3. **Forecaster** — Track 1 (RC > 0): a multi-model SPD ensemble; each
   member returns a Sub-national Probability Distribution over impact
   buckets per window month plus a structured reasoning trace. Aggregated
   as `ensemble_mean_v2` and `ensemble_bayesmc_v2` (calibration-weighted).
   Track 2 (RC 0 + priority tier): a single model, stored as
   `track2_flash`. Binary EVENT_OCCURRENCE questions get per-month
   P(event) instead of an SPD.
4. **Sibyl** — an independent deep-research track (agentic web research,
   Claude Opus + Brave) re-forecasts the ~10 highest-volatility questions;
   stored under `model_name='sibyl'` and scored head-to-head.
5. **Resolution** — realized values from the Resolver DB (ACLED, IFRC,
   IDMC, GDACS, FEWS NET…), per horizon month (1–6).
6. **Scoring** — Brier / log loss / RPS per (question, horizon, model).
7. **Calibration** — per-model weights + LLM-written advice per
   (hazard, metric), consumed by the next cycle's ensemble.

This bundle is built at step 7, after a scoring round, and covers only
**scored** questions — every record links forecast-time reasoning to a
realized outcome.
"""

_IDENTIFIERS = """\
- `question_id` — epoch-suffixed (`{ISO3}_{HAZARD}_{METRIC}_{YYYY-MM}`); the
  primary join key across every file in this bundle.
- `hs_run_id` — the Horizon Scanner run that generated the question (joins
  RC/triage/grounding/adversarial context).
- `run_id` — the forecaster run whose SPDs were scored (joins prompts, raw
  responses, forecasts).
- `horizon_m` — 1..6; horizon 1 is the `window_start_date` month.
- `model_name` — a specific model id (e.g. `gemini-3.5-flash`) for ensemble
  members; stable aggregate names `ensemble_mean_v2`, `ensemble_bayesmc_v2`,
  `track2_flash`, `sibyl` for aggregations. (Runs before July 2026 used
  generic family labels like "Gemini Flash".)
"""

_SCORE_SEMANTICS = """\
All scores: **lower is better**.

- `score_family='spd'` (PA, FATALITIES, PHASE3PLUS_IN_NEED): multiclass
  Brier (range 0–2), log loss, and normalized RPS stored under
  `score_type='crps'` (range 0–1).
- `score_family='binary'` (EVENT_OCCURRENCE): single Brier (range 0–1).
  Binary questions are scored under the *mean* ensemble only.

**HARD RULE: never average `binary` and `spd` scores together.** They are on
different scales; a blended mean is meaningless. Group by `score_family`
first in every aggregation. `rollups.csv` already keeps them apart.

Baseline intuition for SPD Brier: a uniform forecast over K buckets scores
(K−1)/K (≈0.83 for 6 buckets); 0 is a perfect confident forecast; 2 is a
maximally wrong confident forecast. For binary Brier: 0.25 = always saying
50%; rare events forecast well score near 0.
"""

_RESOLUTION_SEMANTICS = """\
- A missing (question, horizon) row in `resolutions` means **unresolved /
  unknown**, NOT zero. Horizons are left unresolved when no source data
  exists (e.g. IFRC never reported that month).
- Zero-defaulting applies only where absence genuinely means zero:
  FATALITIES for ACE (ACLED continuous coverage) and EVENT_OCCURRENCE
  (GDACS satellite-global) — and only for months/countries inside the
  source's observed coverage. These rows carry `source_desc='zero_default'`.
- `source_desc` (present for resolutions computed after July 2026) names the
  winning source, e.g. `facts_resolved:IFRC:2026-03`. NULL on older rows.
- `observed_month` is the calendar month the horizon resolves against.
"""

_REASONING_TRACE = """\
Each Track-1 member's `reasoning_trace` (self-reported, from its JSON
output) has:

- `prior`: `{spd: [K probs], rationale}` — the base-rate-anchored starting
  distribution before evidence.
- `updates[]`: sequential evidence updates, each with `signal`, `direction`
  (UP/DOWN), `magnitude` (SMALL/MODERATE/LARGE), `months_affected`,
  `delta` ([K]), `post_update_spd` ([K]).
- `point_estimate` / `point_estimate_bucket`.
- `rc_assessment`: whether the model accepted/rebutted the HS regime-change
  view (accepted | rebutted | partially_accepted).

`trace_quality` per member is **recomputed at bundle-build time** from the
stored trace (deterministic checks: delta arithmetic, magnitude
consistency). Caveat: the `prior_quality` component is checked WITHOUT the
original base-rate context here, so it returns a neutral 0.7 ("no base rate
to compare") — treat `delta_arithmetic` and `magnitude_consistency` as the
meaningful components. Track 2 traces are reduced (prior + rc_assessment
only; empty updates are expected, not a defect).
"""

_LLM_CALLS_VOCAB = """\
The `spd_prompt` in each question record is taken from `llm_calls`
(phase `spd_v2` / `binary_v2`). Vocabulary you may meet if you dig into
`llm_calls` yourself:

- Forecaster rows: `phase` ∈ spd_v2, binary_v2, scenario_v2,
  forecast_web_research; keyed by `run_id` + `question_id`.
- HS rows: `phase` is ALWAYS 'hs_triage' (a costs-page grouping
  constraint). The call's purpose lives in `call_type` for runs after July
  2026: `rc_pass_{n}`, `rc_grounding`, `triage_pass_{n}`,
  `triage_grounding`, `adversarial_search`, `adversarial_synthesis`.
  Older rows have `call_type='chat'` — for those, parse the synthetic
  `hazard_code`: `RC_{HZ}_PASS_{N}`, `GROUNDING_{HZ}`,
  `TRIAGE_GROUNDING_{HZ}`, `TRIAGE_{HZ}_PASS_{N}`, `ADVERSARIAL_{HZ}`,
  `ADVERSARIAL_SYNTH_{HZ}`. HS rows have empty `run_id` and NULL
  `question_id` — join on (`hs_run_id`, `iso3`).
- Grounding rows' `response_text` is a compact summary (≤15 URLs, trimmed
  snippets). The FULL grounding evidence is inline in each question
  record's `grounding` section (from `hs_hazard_tail_packs`).
- Sibyl rows: `phase='sibyl'`, `call_type='sibyl_trial{i}_step{n}'` — full
  step prompts/responses.

**Sent-prompt guarantee**: for forecaster runs after July 2026 the logged
prompt is byte-exact what the model received (including any appended web
evidence or retry instruction). For earlier runs, the logged prompt is the
pre-evidence base prompt; evidence appendices and retry substitutions were
not captured (in production those paths were disabled, so the base prompt
almost always WAS the sent prompt).
"""

_ANALYSIS_PROMPTS = """\
Suggested analyses (the reason this bundle exists):

1. Compare `case_studies/` best vs worst reasoning traces for systematic
   differences: prior anchoring vs the realized bucket, update magnitudes
   (over/under-reaction to signals), dampener signals mentioned in
   `grounding` but ignored in `updates`, rc_assessment agreement vs outcome.
2. Correlate recomputed `trace_quality` with per-member Brier — do
   arithmetically sloppy traces predict bad forecasts?
3. Check calibration-weight movement (`calibration_weights.csv`) against
   per-member skill in `rollups.csv` — is the weighting loop rewarding the
   right members?
4. Look for hazard/metric pockets in `rollups.csv` where ALL models are
   poor — those need better data injects or prompt guidance, not better
   weighting; cross-reference what the `grounding` evidence contained.
5. For binary questions: compare forecast P(event) against GDACS base rates
   in the prompt — are models ignoring the provided base rate?
6. Where Sibyl covered a question, compare its trial belief traces with the
   standard members' traces on the same question — which evidence did the
   deep-research track surface that the injects missed?
"""

_BLIND_SPOTS = """\
Honest limits of this bundle — do not chase these:

- `question_research` in the DB is placeholder-only; question-level web
  research was retired. The prompt itself (with its structured-data
  injects) IS the input record.
- Parse-failure raw texts (a member that returned unparseable JSON) are
  retained only as `llm_calls.response_text`; members with no llm_calls row
  (e.g. provider errors before logging) appear only via their error status
  in `members[]`.
- Sibyl covers only the top-volatility subset of questions; absence of a
  `sibyl` section is normal.
- Resolutions before July 2026 have no `source_desc`.
- HS-side prompts (RC/triage) are not inlined per question record (they are
  per country-hazard, not per question); the RC/triage *outputs* and full
  grounding evidence are. If you need the HS prompt text, it is in
  `llm_calls` in the DB itself.
- Aggregate rows (`ensemble_*`) have no reasoning trace of their own; they
  are arithmetic over member SPDs (BayesMC uses calibration weights in
  `weights_applied`).
"""


def build_analyst_guide(context: Mapping[str, Any]) -> str:
    """Assemble the full ANALYST_GUIDE.md."""
    n_questions = context.get("n_questions", "?")
    months_back = context.get("months_back", "?")
    parts = [
        "# ANALYST GUIDE — Pythia Scored-Forecast Analysis Bundle",
        "",
        "You are reading a self-contained analysis package produced after a "
        "Pythia scoring/calibration round. It contains, for every scored "
        f"forecast question ({n_questions} questions, window: last "
        f"{months_back} months of question epochs): the inputs and reasoning "
        "that produced each forecast, and the realized outcome and scores. "
        "Your job is to find patterns that improve forecast performance.",
        "",
        "## Reading order",
        "",
        "1. `digest.md` — run-level summary and anomaly list (start here).",
        "2. `questions_index.csv` — one row per question; pick items of "
        "interest by score, hazard, or trace quality.",
        "3. `questions/{question_id}.json` — the full reasoning→outcome "
        "record for one question.",
        "4. `case_studies/` — pre-selected best/worst questions per score "
        "family (same record shape, plus Sibyl trial traces when available).",
        "5. Flat tables: `scores_flat.csv`, `forecast_vs_outcome.csv`, "
        "`rollups.csv`, `calibration_weights.csv`, `calibration_advice.md`.",
        "",
        "`briefing/` holds condensed, size-capped versions of the digest and "
        "case studies for chat-upload contexts. If you can read the whole "
        "bundle, prefer the full files.",
        "",
        "## The system",
        "",
        _PIPELINE_OVERVIEW,
        "## Identifiers and join keys",
        "",
        _IDENTIFIERS,
        "## Impact buckets",
        "",
        _bucket_tables_md(),
        "",
        "## Model lineup",
        "",
        _model_lineup_md(),
        "## Score semantics",
        "",
        _SCORE_SEMANTICS,
        "## Resolution semantics",
        "",
        _RESOLUTION_SEMANTICS,
        "## Reasoning traces",
        "",
        _REASONING_TRACE,
        "## Prompts and llm_calls vocabulary",
        "",
        _LLM_CALLS_VOCAB,
        "## Suggested analyses",
        "",
        _ANALYSIS_PROMPTS,
        "## Known blind spots",
        "",
        _BLIND_SPOTS,
    ]
    return "\n".join(parts)


def build_question_record_schema_md() -> str:
    """Field-by-field schema of questions/{id}.json, embedded in the guide
    consumers can request separately if needed."""
    return _QUESTION_RECORD_SCHEMA


_QUESTION_RECORD_SCHEMA = """\
### questions/{question_id}.json schema

- `question`: the questions-table row (wording, window_start_date,
  target_month, track, metric, pythia_metadata_json).
- `regime_change`: HS RC output — score/level/direction/window plus
  `rationale_bullets` and `trigger_signals` from the RC LLM.
- `triage`: tier, triage_score, need_full_spd, drivers, data_quality.
- `grounding`: the FULL web-grounding evidence packs (rc + triage) the HS
  stage collected for this country-hazard: report markdown, source URLs,
  recent signals. This evidence also reached the SPD prompt.
- `adversarial`: counter-evidence check (RC L1+ only): net_assessment,
  summary, structured payload, sources.
- `spd_prompt`: the exact prompt sent to the ensemble (stored once).
  `spd_prompt_source` names the llm_calls row family it came from.
- `members[]`: per ensemble member — model_name/provider, full raw
  `response_text`, parsed `spd_json`, `reasoning_trace`,
  `human_explanation`, recomputed `trace_quality`, cost/tokens/status, and
  `sent_prompt_override` when that member's prompt differed from
  `spd_prompt`.
- `ensemble`: per aggregate model — {month_index: {bucket_index: prob}},
  `ev_value` per month, weights_profile.
- `weights_applied`: calibration_weights rows for (hazard, metric) at the
  latest as_of_month (what the NEXT run consumes; the run being analyzed
  used the vintage in `run_config.json` if present).
- `outcome`: resolutions per horizon (value, observed_month, source_desc)
  plus `unresolved_horizons` (absent ≠ zero!).
- `scores`: [{horizon_m, model_name, score_type, value}].
- `sibyl`: status, divergences, cost; `trials` (full belief traces) present
  in case_studies/ or when built with --include-sibyl-trials=all.
"""
