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
"""

_SCORED_BUNDLE_POSITION = """\
This bundle is built at step 7, after a scoring round, and covers only
**scored** questions — every record links forecast-time reasoning to a
realized outcome.
"""

_CURRENT_BUNDLE_POSITION = """\
This bundle is built between steps 4 and 5 — the run's forecasts (and
Sibyl's, when present) exist, but none of its outcomes do yet.
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

_SKILL_SEMANTICS = """\
Five **reference forecasters** are scored beside the real models (rows in
`scores` and `rollups.csv` under `run_id IS NULL`):

- `__ext_climatology` — a base-rate SPD built by `pythia.tools.base_rate_spd`
  from the series that resolves the question, using only complete months
  strictly before the question window. This is "what you would have said with
  no model". It is NOT, in general, what the prompt showed: see the table
  below.
- `__ext_uniform` — flat across buckets (0.5 for binary questions). The
  floor: any model losing to uniform is actively destroying information.
- `__ext_persistence` — "next month looks like last month": the last complete
  value observed strictly before the window (ACE/FATALITIES from
  `acled_monthly_fatalities`, a live month with no row counting as 0;
  DR/PHASE3PLUS_IN_NEED from the latest `phase3plus_in_need` row), placed in
  its bucket and SMOOTHED: 90% of the mass on that bucket and the other 10%
  spread evenly over all K buckets (`PERSISTENCE_SMOOTHING = 0.1`). A pure
  one-hot vector gives an infinite log loss whenever the outcome leaves the
  bucket; the smoothing is a fixed constant, never tuned on outcomes.
- `__ext_level_volatility` (ACE/FATALITIES only) — the LEVEL is the last
  complete month the forecaster could have read at forecast time (a month
  counts once it ended 14 days before the forecast and the table holds it
  complete); the SPREAD is how far a monthly count moved, in buckets, over
  the same number of months as separates the level from the target month,
  measured on the country's last 36 complete months (quiet months zero),
  pooled with countries in the same activity band when the country gives
  fewer than 12 pairs; every bucket is floored at 0.005. Under
  `PYTHIA_PRIOR_ANCHOR_SPD=1` this is the distribution the prompt shows and
  tells members to copy as their prior, so its score is the score of a
  member that changed nothing. The audit row in `baseline_scored_forecasts`
  names the level month, its value, the gap and whether pairs were pooled.
- `__ext_level_transition` (ACE/FATALITIES only, scored, never shown to a
  model) — the level-and-volatility recipe with one change: a bucket move
  counts only when the month it started from sits in the level's bucket
  (own country first, pooled with the same activity band below 12 pairs,
  under the same start-bucket rule). The plain recipe pools every move, so a
  country at zero inherits the downward moves of months that started higher
  and they pile onto bucket 0 at the edge. A horizon with no pair from the
  level's bucket has no row. The audit source starts `level_transition:`.

What the prompt showed, against how each reference is built:

| hazard / metric | what the SPD prompt shows as base rate | climatology | persistence | level + volatility | level + transition |
|---|---|---|---|---|---|
| ACE / FATALITIES | a 6-month trajectory from `acled_monthly_fatalities` (last complete month, 3-month average, trend); with `PYTHIA_PRIOR_ANCHOR_SPD=1` also the level + volatility distribution for months 1 and 6 (`forecasts_raw.base_rate_block_version = prior_anchor_v1`) | empirical buckets over the last 36 complete months, quiet months as zero | last complete month before the window | as above | as above, moves only from the level's bucket (never shown) |
| ACE / PA | a 6-month IDMC displacement trajectory from `facts_deltas` | empirical buckets over the last 36 months of IDMC flows, quiet months as zero | — | — | — |
| DR / PHASE3PLUS_IN_NEED | up to 36 months of FEWS NET / IPC Phase 3+ values, gaps shown as null | empirical buckets over the last 36 Phase 3+ values | latest Phase 3+ value | — | — |
| FL, TC / PA | a seasonal profile of reported PA, GDACS alert history and (flood, cyclone) the PA machine's base-rate block | GDACS occurrence rate for the month × the distribution of reported PA magnitudes | — | — | — |
| FL, DR, TC / EVENT_OCCURRENCE | GDACS alert history counted in calendar months over the source's own window | per-calendar-month event rate from GDACS `event_occurrence` rows | — | — | — |

For ACE/FATALITIES before `prior_anchor_v1`, the prompt never showed a
distribution at all: it showed six months, while climatology uses thirty-six.
A member that "ignored the base rate" there ignored a number it was never
given.

**Two experiments.** `advice_experiment.csv` compares questions whose
members were shown calibration advice with questions shown none
(`PYTHIA_ADVICE_EXPERIMENT_SHARE`; the arm is a hash of the question id and
is recorded on `forecasts_raw.advice_arm` / `forecasts_ensemble.advice_arm`):
primary aggregate mean Brier per arm over distinct questions, the difference
(advice minus no advice, negative = advice helped) and a bootstrap 90%
interval. `recalibration_effect.csv` compares a member's corrected forecast
with its uncorrected one on the same (question, horizon)
(`PYTHIA_FAMILY_RECALIBRATION_MODE`): `<model>` against `<model>__raw` where
the correction was applied, `<model>__recal` against `<model>` where it was
only shadowed, plus an unweighted member mean rebuilt both ways, each with a
paired bootstrap 90% interval. `<model>__raw` and `<model>__recal` rows appear
in the score tables as models; they are copies of a member, never members,
and never vote, carry calibration weight or receive advice. Each member row
says what was done to it in `forecasts_raw.recalibration_json`.

**One run per question.** A question forecast in several runs has score
and forecast rows for each. `rollups.csv`, `forecast_vs_outcome.csv`, the
digest and every `questions/*.json` score list use the LATEST run only;
`scores_flat.csv` keeps every run and says which is latest
(`is_latest_run`), and `questions_index.csv` carries `n_runs`, `is_rerun`
and `latest_run_id`.

**RPS beside Brier.** For SPD metrics the digest reports RPS (stored as
`score_type='crps'`) next to Brier. Brier ignores bucket ORDER — one bucket
off and five buckets off score alike — and RPS does not.

**Bucket edges.** `forecast_vs_outcome.csv` carries `nearest_boundary`, the
closest FINITE interior bucket boundary to the resolved value, and flags
`bucket_edge = True` only when the value lies within 5% of it. A small data
revision would move that outcome to the neighbouring bucket; weigh such
"misses" accordingly. A value of 0 is never an edge case, and binary rows
carry neither column. The file also carries the question's `track`.

**Reference vectors.** `forecast_vs_outcome.csv` also carries rows for the
`__ext_` reference forecasters, with the exact vector each one was scored
on (from `baseline_scored_forecasts`), and every row carries its full
`probs` vector as JSON. For a binary question the vector is
`[P(yes), P(no)]`.

**Cost.** `rollups.csv` carries `cost_per_question_usd`: for a member, its
mean forecast-phase spend per question; for an aggregate row, the mean
total spend of the questions it covered. Reference forecasters cost
nothing and carry no figure.

**Coverage.** The manifest's `resolved_questions` counts resolved questions
by horizon and by calendar month, so a mean over "all scored questions" can
be read against how many months it actually rests on.

Skill, wherever you compute it:

    skill = 1 − (model_score / climatology_score)

Positive = beat the base rate. Zero = matched it. Negative = lost to it.
Compute per (score_family, score_type), NEVER across them, and on PAIRED
scores only: the same (question_id, horizon_m) scored by both the model and
`__ext_climatology`. `rollups.csv` has one row per (hazard, metric,
score_family, track, model, score_type) and carries the paired figures:
`n_paired`, `paired_model_mean`, `climatology_mean` (over the same pairs) and
`skill_vs_climatology`. `mean_value`, `median_value`, `n_samples` and
`n_questions_scored` describe everything the model scored; `n_questions` is
the paired question count wherever the group has a climatology reference.
A ratio of a model's mean over its own questions to climatology's mean over
every question in the hazard and metric compares two different sets of
questions, and once read Track 2 drought event skill as +0.80 where the
paired figure was about +0.74.

**Sharpness.** The digest's sharpness table gives, per (hazard, metric,
track), the primary aggregate's mean largest bucket probability and its
mean probability on the bucket that happened. A forecast can be sharp and
wrong; the pair of numbers says which.

**The Track 1 vs Track 2 trap**: Track 1 questions are high regime-change,
Track 2 questions are quiet — disjoint populations, and Track 2's are
easier. A flat comparison of raw scores will "show" the cheap single model
beating the expensive ensemble, and it will be teaching you something
false. Compare each track's skill against ITS OWN climatology baseline,
never raw scores across tracks. (Sibyl vs Track 1 is different: that
comparison is already restricted to Sibyl's covered set.)

Where a (hazard, metric) pair has no climatology row, the pair has no
prompt-time base rate — skill cannot be computed there and the columns stay
empty; do not substitute uniform as the denominator.
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
- `acled_snapshot_date` (ACE/FATALITIES) is the date of the ACLED pull the
  figure came from. ACLED revises counts for weeks, and every resolution run
  REPLACES the row, so `resolution_vintages` keeps the first resolution and
  those taken ~60 and ~90 days after month end (`vintage` = first / d60 /
  d90, with the actual `days_after_month_end`). Compare them to see whether
  an outcome moved after it was scored.
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
- Sibyl covers a subset of questions (25 a run since October 2026: each
  hazard's three most volatile, then the most volatile of the rest, at most
  ten per hazard; ten by plain volatility before that); absence of a `sibyl`
  section is normal.
- Resolutions before July 2026 have no `source_desc`.
- HS-side prompts (RC/triage) are not inlined per question record (they are
  per country-hazard, not per question); the RC/triage *outputs* and full
  grounding evidence are. If you need the HS prompt text, it is in
  `llm_calls` in the DB itself.
- Aggregate rows (`ensemble_*`) have no reasoning trace of their own; they
  are arithmetic over member SPDs (BayesMC uses calibration weights in
  `weights_applied`).
"""


_ERROR_ATTRIBUTION = """\
These files split a score into its parts. Every one keeps the bundle's three
rules: binary and SPD scores are never blended; skill is computed only on
(question, horizon) pairs both sides scored, within one track; one run per
question (the latest). Intervals are 90% and resample QUESTIONS (horizons of
one question share an outcome history), seeded so a rebuild gives the same
numbers: 4000 resamples, except `skill_history.csv` (1000, it has thousands of
cells). A verdict reads "too few" below 10 questions. A file whose only column
is `stub_reason` (or a JSON `{"stub": true, "reason": ...}`) is a section that
could not be computed; the manifest's `error_attribution.files` lists every
file with its status, row count and reason.

**`input_partial_month`** (every question, on `questions_index.csv`,
`forecast_vs_outcome.csv`, `rollups.csv`, `skill_history.csv`,
`trace_stages*.csv`, and each record with `input_partial_month_basis`): true
where the conflict prompt's "last month" was written before that month ended.
Before 30 Sept 2026 the ACLED writer stored the month in progress, so the
1 August 2026 run read a July row written on 15 July (a median 28% of the
settled count) and the 1 September run an August row written on 28 August.
Reconstructed from the scheduled-ingest calendar (the 15th until 3 Aug 2026,
the 28th since), because a rewritten row keeps no history; a manual ingest in
between is not seen. False for non-ACE questions (no ACLED trajectory in the
prompt), null where the forecast date is unknown. Rollups and history are
split on it: never pool a partial-input forecast with a complete one.

**`headline.json`** — the figures a report quotes. Per (hazard, metric,
track) and score type (`brier`; `crps` = RPS for SPD): paired questions and
pairs, the primary aggregate's mean (bayesmc, else mean, else track2_flash),
each reference's mean on the same pairs, wins and losses (questions where
the primary's mean score is below / above the reference's), skill against
`__ext_climatology` and `__ext_level_volatility` with its interval, and
`warning` below 10 paired questions. Also `questions_resolved_per_horizon`.
The digest's first table is generated from this file and nothing else.

**`trace_stages.csv`** — one row per (question, resolved horizon, member)
for Track 1 and Track 2 SPD questions (Track 2's member is `track2_flash`).
Three distributions, each a JSON vector: `shown_spd`, the base rate the prompt
showed (`__ext_level_volatility` from `baseline_scored_forecasts` when the
member saw the prior-anchor block, `base_rate_block_version` set; otherwise
the `__ext_climatology` anchor; `shown_source` says which, or `absent`);
`prior_spd`, the member's declared prior from its reasoning trace (one vector,
compared with every horizon); `final_spd`, what it wrote for that month. For
each: Brier, log, RPS against the outcome, max bucket probability, entropy in
bits, expected bucket (1-based). Then `js_distance_prior_vs_shown` (sqrt of
the natural-log Jensen-Shannon divergence), the expected-bucket shift and
entropy change from prior to final, and the RC level, direction and the
member's `rc_assessment`, with the prompt versions, advice arm, recalibration
mode and lineup. **`trace_stages_summary.csv`** per (hazard, metric, track,
`base_rate_block_version`, `rc_guidance`, `input_partial_month`, score type):
mean score of shown, prior and final, with intervals on `prior_minus_shown`
(the cost of the starting point) and `final_minus_prior` (the cost of the
adjustments). Negative is better. This is the question "did the error come
from the start or from the adjustments?". The prior is what the member SAID
its prior was; a member that wrote an anchored prior and then ignored it
reads as a good start and bad adjustments.

**`update_value.csv`** — one row per (trace update, resolved horizon it
touched): the update's `attribution_id` (the attribution bundle's join key,
same recipe), `signal_class` (the attribution bundle's taxonomy,
`signal_taxonomy.csv`), mass moved, the probability on the realised bucket
before and after that update alone, and the RPS change it caused
(`post_update_spd` against the SPD before it). `months_affected` is the
trace's own wording; an unreadable one counts as all six months.
**`update_value_summary.csv`** per signal class and hazard: updates, the share
that moved toward the outcome, mean RPS change with its interval. This is
CLAIMED attribution: the update is the model's own account of what moved it,
written after the fact. It says whether the adjustments a model reports
helped, not whether the evidence caused them.

**`rc_outcomes.csv`** — one row per (scored SPD question, resolved horizon)
where a last value exists (ACE/FATALITIES, DR/PHASE3PLUS_IN_NEED; PA has
none): the HS regime-change level, score and direction, the last value
before the window (`base_rate_spd.last_observed_value`, read as the table
stands now; `input_partial_month` marks where the prompt saw a partial count
instead), the outcome, the signed bucket move and whether it matched the
flagged direction (null for mixed/unclear). **`rc_outcomes_summary.csv`** by
metric, RC level and direction: mean absolute move, share moving 2+ buckets,
share matching direction. RC 0 rows are the control.

**`unasked_outcomes.csv`** — cells across the horizon scanner's country list
(`horizon_scanner/hs_country_list.txt`) and the four forecast hazards, in the
scored months, with NO question whose window covered them, where ACE
all-types deaths reached bucket 5 (100+) or rose two buckets above the last
month before the epoch; a GDACS Orange or Red FL/DR/TC event occurred; or IPC
Phase 3+ rose a bucket. With the value, the source, and the HS tier, triage
score and RC level recorded for the cell, or `not assessed`.

**`experiments.csv`** — for each flag holding more than one value
(`lineup_id`, `base_rate_block_version`, `rc_guidance`, `advice_arm`,
`recalibration_mode`, `input_partial_month`), within (hazard, metric, track)
and score type: each arm against the most common one. Arms hold different
questions, so each question is first paired with climatology on its own
horizons (primary minus `__ext_climatology`) and the arms are compared on
that excess: question difficulty drops out. `correction` rows are genuinely
paired: a member's corrected forecast against its own `__raw` one. Verdict:
`too few`, `no clear difference`, or `arm X better` (lower score is better).
`rollups.csv` carries the same split columns plus `correction` (`raw`,
`corrected`, `shadow_corrected`, `none`), and its paired skill is computed
within them.

**`skill_history.csv`** — every resolved (question, horizon) in the DATABASE,
not only this bundle's window: paired skill against `__ext_climatology`,
`__ext_persistence` and `__ext_level_volatility` (where scored), for the
primary aggregate (`forecaster = primary`) and each member, per observed
month, hazard, metric, track, lineup and prompt versions, with n and an
interval. The digest shows the last six months, pooled across lineups and
versions.

**`tail_outcomes.csv`** — every resolved SPD outcome in the top two buckets
and every binary event that occurred, with the probability each member,
aggregate, Sibyl and reference forecaster gave it (`forecaster_kind`).
**`binary_reliability.csv`** per hazard, track and forecaster: bins 0-5%,
5-20%, 20-50%, 50-80%, 80-100%, with n, mean forecast and observed rate.

**`inject_health.csv`** — one row per question and inject (ENSO, GDACS
history, CrisisWatch, ACLED CAST, ViEWS, base rate, the ACLED trajectory):
present, the observation or vintage, age in days or months, stale, and the
reason when absent. Stale means: ENSO carried forward or over 100 days old;
CrisisWatch three or more editions old; a conflict-forecast vintage over 45
days; the ACLED trajectory when `input_partial_month`. The digest counts
stale and absent injects per hazard and lists CrisisWatch status for ACE.

**Record fields.** `base_rate_shown`: the structured figures behind the
prompt's base-rate block. For ACE: the six ACLED months before the forecast
month (value, month, the row's `updated_at` and `updated_after_forecast`, true
where the row was rewritten after the forecast, so the value read now may
differ from the value shown), the three-month means and trend computed by
the same function the prompt uses, and the level-and-volatility vector when
the prompt showed it (months 1 and 6 printed). For every question: the anchor
`forecast_deviation` reconstructs. `forecast_versions`: lineup, prompt
versions, advice arm, recalibration mode, forecast date. `inject_status` for
ACE always carries `crisiswatch` (edition, `edition_age_months`, `arrow`,
`alert`, `stale`, or `available: false` with the reason), `acled_cast` and
`views` (vintage, `age_days`, `stale`).
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
        "6. Error attribution: `headline.json` first, then `trace_stages_summary.csv` "
        "(start or adjustments?), `update_value_summary.csv`, `rc_outcomes_summary.csv`, "
        "`experiments.csv`, `skill_history.csv`, `tail_outcomes.csv`, "
        "`binary_reliability.csv`, `unasked_outcomes.csv`, `inject_health.csv`.",
        "",
        "`briefing/` holds condensed, size-capped versions of the digest and "
        "case studies for chat-upload contexts. If you can read the whole "
        "bundle, prefer the full files.",
        "",
        "## The system",
        "",
        _PIPELINE_OVERVIEW,
        _SCORED_BUNDLE_POSITION,
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
        "## Skill and the reference forecasters",
        "",
        _SKILL_SEMANTICS,
        "## Resolution semantics",
        "",
        _RESOLUTION_SEMANTICS,
        "## Reasoning traces",
        "",
        _REASONING_TRACE,
        "## Prompts and llm_calls vocabulary",
        "",
        _LLM_CALLS_VOCAB,
        "## Error attribution files",
        "",
        _ERROR_ATTRIBUTION,
        "## Suggested analyses",
        "",
        _ANALYSIS_PROMPTS,
        "## Known blind spots",
        "",
        _BLIND_SPOTS,
    ]
    return "\n".join(parts)


_DEVIATION_SEMANTICS = """\
The attention metrics (from the `forecast_deviation` table, computed by
`pythia/tools/compute_deviation.py` — all arithmetic is SQL/Python, none of
it model-generated):

- `js_vs_baserate` — Jensen-Shannon divergence (natural log, range 0 to
  ln 2 ≈ 0.693) between the published forecast SPD (averaged over the six
  window months) and the base-rate anchor the forecaster was shown at
  prompt time. 0 = the ensemble came back at the base rate; large = the
  ensemble moved far from it.
- `log_ev_ratio` — ln(EV_forecast / EV_baserate), signed so direction is
  legible: positive = the forecast expects MORE impact than the base rate,
  negative = less. Binary questions use ln(p_forecast / p_baserate).
- `eiv_nominal` — expected impact value over the window (surge blend
  `max + 0.1 × (sum − max)` over monthly EIVs, the same formula the
  dashboard risk index uses; bucket centroids from pythia.buckets). NULL
  for binary questions — event occurrence has no impact centroid.
- `eiv_per_100k` — population-normalised.
- `baserate_source` / the `baserate` block in each question record —
  provenance of the anchor. A question with NO deviation row has no
  prompt-time base rate; that is a statement about coverage, never invent
  an anchor for it.

`attention_index.csv` carries four 1-based rank columns (1 = most
attention-worthy; empty = not rankable for that ordering):

- `rank_deviation` — by `js_vs_baserate` descending.
- `rank_impact_nominal` — by `eiv_nominal` descending (SPD questions only).
- `rank_impact_per_capita` — by `eiv_per_100k` descending, restricted to
  rows whose `eiv_nominal` clears an absolute floor
  (`PYTHIA_INTERPRETER_PER_CAPITA_FLOOR`, default 10 000 — without it this
  ordering returns the same small island states every cycle).
- `rank_rc_disagreement` — by |rc_score − js_vs_baserate/ln 2| descending:
  large where the Horizon Scanner flagged regime change but the ensemble
  came back at the base rate, or the reverse. The most interesting list.

`attention_rank` is the blend the pack is ordered (and, under the token
budget, truncated) by: the mean of the available rank columns, missing
orderings excluded. Top-ranked questions are never truncated.
"""

_CURRENT_BLIND_SPOTS = """\
Honest limits of this pack — do not chase these:

- **No outcomes yet.** This pack describes a run whose window has not
  resolved; there are no scores or resolutions in it. Performance material
  lives in the scored-forecast bundle, built at the calibration terminus.
- Questions with no `forecast_deviation` row have no prompt-time base rate
  (see `blind_spots.json` → `no_baserate_questions`); their attention rank
  uses only the impact orderings.
- `deltas.json` matches runs on (iso3, hazard_code, metric) — question ids
  are epoch-suffixed and never match across months by design.
- Sibyl covers a subset (25 questions a run, spread across hazards); absence of a `sibyl` section
  in a record is normal. Sibyl rows in `forecast_deviation` exist only
  when the pack is built after the Sibyl stage.
- The PA resolution machine runs in shadow mode: its base rates inform
  FL/TC PA prompts, but PA ground truth used for scoring still comes from
  the legacy path with thin coverage (~4%) — treat PA "impacts" prose with
  corresponding humility.
- ACE inputs carry narrative-salience bias: heavily reported conflicts
  produce more signal for the models to react to, independent of severity.
- Blocked hazards (CU, DI, HW, ACO) are fully deactivated upstream — no
  questions exist for them; their absence is policy, not a gap.
"""


def build_current_run_guide(context: Mapping[str, Any]) -> str:
    """ANALYST_GUIDE.md for the current-run (pre-outcome) bundle."""
    n_questions = context.get("n_questions", "?")
    run_id = context.get("run_id", "?")
    parts = [
        "# ANALYST GUIDE — Pythia Current-Run Bundle",
        "",
        "You are reading the input pack for interpreting a Pythia forecast "
        f"run ({run_id}: {n_questions} questions) BEFORE its outcomes exist. "
        "It contains what the system forecast, how far each forecast moved "
        "from its base-rate anchor, what the models were reacting to, and "
        "what changed since the previous run. Your job is to explain what "
        "deserves attention and why — in plain language, without computing "
        "any number yourself: every figure you need is pre-computed here.",
        "",
        "## Reading order",
        "",
        "1. `MANIFEST.json` — run ids, window, counts, lineup, cost, and the "
        "`pack_tokens` / truncation record.",
        "2. `attention_index.csv` — one row per question with the deviation "
        "and impact metrics and the four rank columns.",
        "3. `deltas.json` — entries/exits vs the previous run, largest SPD "
        "movements, and how the previous run's flagged risks are tracking.",
        "4. `blind_spots.json` — what this run cannot see.",
        "5. `questions/{question_id}.json` — the full record for one "
        "question (present for the top-ranked questions; the low-ranked "
        "tail may be truncated under the token budget — the manifest says "
        "exactly which).",
        "",
        "## The system",
        "",
        _PIPELINE_OVERVIEW,
        _CURRENT_BUNDLE_POSITION,
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
        "## Deviation and attention metrics",
        "",
        _DEVIATION_SEMANTICS,
        "## Score semantics (for context — no scores in this pack)",
        "",
        _SCORE_SEMANTICS,
        "## Known blind spots",
        "",
        _CURRENT_BLIND_SPOTS,
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
  An RC-promoted hazard (RC level 1+) skipped triage: `tier` reads
  `rc_promoted`, `triage_score` is null and `triage_skipped` is true. Rows
  stored before Oct 2026 said `quiet` / 0.0 for these; the bundle corrects
  them from `data_quality.status`.
- `grounding`: the FULL web-grounding evidence packs (rc + triage) the HS
  stage collected for this country-hazard: report markdown, source URLs,
  recent signals. This evidence also reached the SPD prompt.
- `adversarial`: counter-evidence check (RC L1+ only): net_assessment,
  summary, structured payload, sources.
- `spd_prompt`: the exact prompt sent to the ensemble (stored once).
  `spd_prompt_source` names the llm_calls row family it came from. When it
  is null, `spd_prompt_missing_reason` says why (no call logged, or calls
  logged with an empty prompt); `questions_index.csv` flags it as
  `spd_prompt_missing`.
- `inject_status`: what the prompt was built on, recovered from the DB:
  the ENSO record current on the run date (phase, ONI, `observation_date`),
  the GDACS history window and event count (FL/DR/TC; recomputed from
  `facts_resolved` at bundle time, so `reconstructed_at_bundle_time` is
  true), the CrisisWatch edition and its age in months (ACE), and the
  base-rate source from `forecast_deviation`. Every sub-block says
  `available: false` with a `reason` rather than going missing.
- `lineup`: the members that forecast this question (`model_id`,
  `provider`, `effort`, `shadow`) and a `lineup_id` hashed from model ids
  and effort. Effort comes from the config at bundle time
  (`effort_source`), because the call log does not record it; the manifest
  lists every lineup seen with its question count.
- `resolution_series`: in words, the series that resolves this question.
- `cost_usd`: forecast-phase spend for the latest run, per member and
  `__total__`.
- `members[]`: per ensemble member — model_name/provider, full raw
  `response_text`, parsed `spd_json`, `reasoning_trace`,
  `human_explanation`, recomputed `trace_quality`, cost/tokens/status, and
  `sent_prompt_override` when that member's prompt differed from
  `spd_prompt`.
- `ensemble`: per aggregate model — {month_index: {bucket_index: prob}},
  `ev_value` per month, weights_profile.
- `weights_applied`: calibration_weights rows for (hazard, metric) at the
  latest as_of_month (what the NEXT run consumes; the run being analyzed
  used the vintage in `run_config.json` if present). When no weights exist,
  `calibration_status.csv` (and the manifest's `calibration_status`) says
  why: each (hazard, metric) needs 20 resolved questions with member Brier
  scores, and the file gives the count so far. A new version of a model
  family inherits its predecessor's record as a prior (`inherited_from`).
- `outcome`: resolutions per horizon (value, observed_month, source_desc)
  plus `unresolved_horizons` (absent ≠ zero!).
- `scores`: [{horizon_m, model_name, score_type, value}].
- `sibyl`: status, divergences, cost; `trials` (full belief traces) present
  in case_studies/ or when built with --include-sibyl-trials=all.
"""


# ---------------------------------------------------------------------------
# Forecast attribution bundle
# ---------------------------------------------------------------------------

_ATTRIBUTION_CLAIMED = """\
**Everything in `attribution/signal_ledger.parquet` is CLAIMED attribution.**
The deltas are what a model said moved it, written after the fact in the
same response that carried the forecast. A model can be confidently wrong
about its own reasoning: the stated prior may be back-fitted to the answer,
the signals may be a narrative assembled to justify a number already chosen,
and the per-bucket deltas are arithmetic the model was asked to produce, not
a measurement of anything. Measured influence needs ablation (re-running the
forecast with an input removed and observing the change), and this bundle
does not do that. Anyone who reads the ledger as causal evidence is building
on sand. Read it as testimony: consistent, queryable, and worth comparing
across models and months, but testimony.
"""

_ATTRIBUTION_FILES = """\
- `attribution/signal_ledger.parquet` (+ `signal_ledger_sample.csv`, the
  first 500 rows) — one row per (run, question, model, update). `update_index
  = -1` is the model's stated prior (`is_prior_row = true`). A model with no
  parseable trace gets exactly one row with `signal_class = 'no_trace'` and
  null delta columns: absence is visible in the table, never inferred from a
  missing row. `mass_moved_l1` is half the L1 norm of the delta, so it reads
  as the share of probability mass relocated by that signal. `direction` is
  the change in expected bucket index (positive = toward higher severity).
  `delta_sums_to_zero` and `post_spd_reconciles` are the same two checks
  `forecaster/trace_validation.py` performs, at the same tolerances.
- `attribution/prior_anchoring.csv` — the stated prior beside the base-rate
  anchor the prompt carried (from `forecast_deviation.baserate_json`), with
  the Jensen-Shannon divergence and distance between them and the signed
  change in expected bucket index. This is where narrative salience shows
  first: a prior far from its anchor was not built from the anchor.
- `attribution/rc_assessment.csv` — the HS regime-change flag as supplied
  beside the model's `rc_assessment` verdict (accepted / partial / rebutted /
  absent) and the mass it moved on signals classed `rc_flag`. A model that
  reports acceptance and moves nothing is a finding.
- `attribution/trace_quality.csv` — `trace_validation.py` scores per model
  per question plus trace presence and update count. The `prior_quality`
  component runs WITHOUT the original base-rate summary and returns its
  neutral 0.7; trust `delta_arithmetic` and `magnitude_consistency`. The
  by-model and by-hazard roll-up is in `MANIFEST.json` — if one member never
  emits usable traces, everything downstream is skewed by its absence.
- `inputs/input_inventory.csv` — one row per question: what was actually on
  the table at forecast time (resolver history depth, base-rate anchor and
  its provenance, structured-inject row counts, CrisisWatch edition,
  evidence counts and ages, HS tier/score/RC level, and the mean ensemble's
  `js_vs_baserate` / `log_ev_ratio`). It exists so somebody can ask whether
  questions with an inject, or with heavy recent evidence, deviate further
  from base rate than those without. That is a comparison across the panel,
  not a causal claim.
- `inputs/evidence_items.jsonl.gz` — every evidence item the run carried,
  flattened from `question_research` (HS country pack, question-specific web
  research, merged) and from the HS grounding packs the SPD prompt injected,
  keyed by `evidence_id`. `inputs/evidence_to_signal.csv` links
  `attribution_id` to `evidence_id` by token containment between the signal
  text and the item's title plus text; `match_score` and `match_method` say
  how. **This linkage is a heuristic**: a shared vocabulary is not proof the
  model read that item, and links below the threshold in the manifest are
  not emitted at all.
- `inputs/base_rates.csv` — the anchor per question, with its source and
  observation count.
- `prompts/prompt_sections.csv`, `prompts/section_hashes.json` — each
  distinct prompt template this run used, split into its headed sections
  with sizes and sha256 hashes. Diff `section_hashes.json` against last
  month's to tell in seconds whether a shift in forecasts followed a prompt
  edit or followed the world. Text before the first heading is
  `unclassified`, so nothing is silently dropped.
- `prompts/token_share.csv` — per question, the share of prompt characters
  (and estimated tokens) each section took. Token share is a weak proxy for
  influence, but it is MEASURED rather than self-reported, and it is the
  only measured signal in this bundle.
- `contrasts/model_disagreement.csv` — pairwise JS divergence between
  member SPDs per question, decomposed into disagreement already present at
  the prior and disagreement introduced by updates. Two models landing on
  the same number by different routes is worth knowing.
- `contrasts/fred_vs_sibyl.csv` — the ensemble aggregate against Sibyl's
  pooled SPD, with the signal classes each cited. Sibyl emits no reasoning
  trace by design; its classes are inferred from its trial text.
- `contrasts/run_over_run.csv` — matched to the previous run on (iso3,
  hazard_code, metric): change in expected value, in the mean stated prior,
  and in the mix of signal classes cited.
- `hazard/<CODE>.md` — a plain-markdown brief per hazard: top classes by
  mass and by frequency, mean prior-anchoring distance, RC acceptance rate,
  evidence volume, and the five questions with the largest movement.
- `questions/<question_id>.json` — the shared question record (same shape
  as the scored bundle's) plus an `attribution` block: the ledger rows, the
  inventory row, the prompt section fingerprint and the evidence ids.
- `MANIFEST.json` — run ids, window, counts, lineup, taxonomy version,
  thresholds, per-file row counts, every collector that failed and why, and
  the `linked_bundles` block naming the operational bundle this was built
  beside.
- `LINKAGE.md` — the join contract with the other bundles.
"""

_ATTRIBUTION_TAXONOMY = """\
`signal_class` comes from `scripts/ai_bundle/signal_taxonomy.csv`: ordered
case-insensitive regular expressions with a priority; the highest-priority
match wins, and `signal_class_confidence` rises with how much of the text
the pattern actually matched (a six-letter hit is a weaker claim than a
whole phrase) and with a second hit for the same class. `other` is the
fallback and carries confidence 0.1. Classification is deterministic and
never uses a model call, because month-over-month comparison is worthless
if the classes drift with the classifier. The taxonomy is versioned
(`taxonomy_version` in the manifest); compare ledgers only within one
version.
"""


def build_attribution_guide(context: Mapping[str, Any]) -> str:
    """ANALYST_GUIDE.md for the forecast attribution bundle. The claimed-
    attribution warning is the opening paragraph, not a footnote."""
    run_id = context.get("run_id", "?")
    n_questions = context.get("n_questions", "?")
    n_rows = context.get("n_ledger_rows", "?")
    rollup = context.get("trace_quality_rollup") or {}
    by_model = rollup.get("by_model") or {}
    lineup_lines = []
    for model, stats in sorted(by_model.items()):
        lineup_lines.append(
            f"| {model} | {stats.get('n')} | {stats.get('share_with_trace')} | "
            f"{stats.get('mean_trace_quality')} | {stats.get('mean_updates')} |"
        )
    classes = [e.get("signal_class") for e in (context.get("taxonomy") or []) if e.get("signal_class")]
    parts = [
        "# ANALYST GUIDE — Pythia Forecast Attribution Bundle",
        "",
        _ATTRIBUTION_CLAIMED,
        f"This bundle covers forecaster run `{run_id}` ({n_questions} questions, "
        f"{n_rows} ledger rows). It answers one question the other bundles do not: "
        "why did the models produce these numbers, and what did they say influenced "
        "them most, per hazard. It is a forensic record with no token cap, built for "
        "querying, not a model input.",
        "",
        "## Reading order",
        "",
        "1. `MANIFEST.json` — counts, the trace-quality roll-up (which members emitted "
        "usable traces), collector failures.",
        "2. `hazard/<CODE>.md` — the human-readable brief per hazard.",
        "3. `attribution/signal_ledger_sample.csv`, then the parquet with a real query "
        "engine (DuckDB reads it directly).",
        "4. `attribution/prior_anchoring.csv` and `rc_assessment.csv`.",
        "5. `inputs/input_inventory.csv` for the panel comparison.",
        "6. `contrasts/` and `prompts/` for what changed between models and between runs.",
        "",
        "## The system",
        "",
        _PIPELINE_OVERVIEW,
        "## Files",
        "",
        _ATTRIBUTION_FILES,
        "## The signal taxonomy",
        "",
        _ATTRIBUTION_TAXONOMY,
        f"Classes (taxonomy version {context.get('taxonomy_version', '?')}): "
        + ", ".join(f"`{c}`" for c in classes),
        "",
        "## Trace quality in this run",
        "",
        "| model | calls | share with trace | mean trace quality | mean updates |",
        "|---|---|---|---|---|",
        *lineup_lines,
        "",
        "## Identifiers and join keys",
        "",
        _IDENTIFIERS,
        "- `attribution_id` = sha256(\"{run_id}|{question_id}|{model_name}|{update_index}\")[:16]. "
        "Stable and deterministic; the future resolutions bundle joins on it.",
        "- `evidence_id` = sha256(\"{url}|{title}\")[:16].",
        "",
        "## Impact buckets",
        "",
        _bucket_tables_md(),
        "",
        "## Model lineup",
        "",
        _model_lineup_md(),
        "## Reasoning traces",
        "",
        _REASONING_TRACE,
        "## Suggested analyses",
        "",
        "- Which signal classes carry the most mass per hazard, and does the mix "
        "differ by model family? (`signal_ledger`, group by hazard_code × model_family.)",
        "- How far do stated priors sit from their anchors, and does the distance "
        "predict `js_vs_baserate`? (`prior_anchoring` joined to `input_inventory`.)",
        "- Do models that report accepting the RC flag actually move mass on it? "
        "(`rc_assessment`: acceptance beside `rc_flag_mass_moved_l1`.)",
        "- Do questions with a CrisisWatch inject or heavy recent evidence deviate "
        "further from base rate? (`input_inventory` — a panel comparison, not a cause.)",
        "- Did a prompt section change between runs? (`section_hashes.json` diff.)",
        "- Where do two members agree on the answer and disagree on the route? "
        "(`model_disagreement`: small `jsd_final`, large `jsd_prior`.)",
        "",
        "## Known blind spots",
        "",
        "- Binary EVENT_OCCURRENCE calls and Sibyl carry no reasoning trace by "
        "design; they appear as `no_trace` rows.",
        "- Track-2 traces are reduced (prior + rc_assessment, empty updates).",
        "- `evidence_to_signal` is lexical. `question_research` is a placeholder in "
        "production runs, so most evidence comes from the HS grounding packs.",
        "- `prior_quality` in `trace_quality.csv` is the neutral 0.7 (no base-rate "
        "summary at bundle time).",
        "- The section parser splits on headed blocks; a prompt edit that changes a "
        "heading shows as a removed and an added section, not a changed one.",
    ]
    return "\n".join(parts)


def build_linkage_md(context: Mapping[str, Any]) -> str:
    """LINKAGE.md: the join contract between this bundle and the others."""
    return "\n".join(
        [
            "# LINKAGE — how this bundle joins the others",
            "",
            f"Built for forecaster run `{context.get('run_id', '?')}` "
            f"(HS run `{context.get('hs_run_id', '?')}`).",
            "",
            "## Keys",
            "",
            "| key | recipe | where it lives | joins to |",
            "|---|---|---|---|",
            "| `run_id` | the forecaster run (`fc_<epoch>`) | every table | operational debug bundle "
            "(`pythia-debug-bundle`), current-run bundle, scored bundle, `forecasts_raw`/`llm_calls` |",
            "| `hs_run_id` | the Horizon Scanner run | ledger, manifest | `hs_triage`, "
            "`hs_hazard_tail_packs`, the operational bundle's HS files |",
            "| `question_id` | `<ISO3>_<HAZARD>_<METRIC>_<YYYY-MM>` | every table | every bundle; "
            "`questions/<id>.json` here shares the scored bundle's record schema |",
            "| `attribution_id` | `sha256(\"{run_id}|{question_id}|{model_name}|{update_index}\")[:16]` "
            "| `signal_ledger`, `evidence_to_signal`, `questions/*.json` | the future "
            "resolutions bundle, which attaches outcomes to it |",
            "| `evidence_id` | `sha256(\"{url}|{title}\")[:16]` | `evidence_items`, "
            "`evidence_to_signal` | the operational bundle's evidence CSVs by url |",
            "| `(iso3, hazard_code, metric)` | `interpreter.persistence.match_key` | "
            "`run_over_run` | the current-run bundle's `deltas.json` |",
            "",
            "## The shared question record",
            "",
            "`questions/<question_id>.json` is produced by the same `build_question_record` "
            "the scored and current-run bundles use, extended with an `attribution` block "
            "(ledger rows, inventory row, prompt section fingerprint, evidence ids). Fields "
            "outside that block mean exactly what the scored bundle's guide says they mean.",
            "",
            "## The resolutions bundle (not yet built)",
            "",
            "A fourth bundle will attach outcomes — resolved values, realised buckets, "
            "per-horizon scores — to `attribution_id`. That is what will eventually let "
            "signal CLASSES be scored, rather than only forecasts: did the signals a model "
            "moved on turn out to be the ones that mattered? Until it exists, nothing in "
            "this bundle says whether a claimed attribution was right.",
            "",
            "## Intended next step: ablation",
            "",
            "Claimed attribution becomes measured attribution only under ablation: re-run "
            "a forecast with one input withheld and record the change in the SPD. That "
            "harness is out of scope for the bundle; `attribution_id` and the prompt "
            "section fingerprints are the hooks it will need.",
            "",
        ]
    )
