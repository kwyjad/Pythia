# HS Regime-Change Scoring

Horizon Scanner (HS) captures a dedicated **regime-change / out-of-pattern risk** assessment for every hazard. The model returns a `regime_change` object, which is normalized, scored, and persisted into `hs_triage`.

## Key Concept: Regime Change vs Triage Score

- **`triage_score`** captures the *overall risk level*, including ongoing/chronic situations. A country with severe but steady conflict has a HIGH triage_score.
- **`regime_change`** captures *only* the probability and magnitude of a *departure from the established pattern*. That same country with steady conflict has LOW regime_change likelihood, because the pattern is continuing as expected.

A regime change means a statistically significant departure from the historical base rate — not just elevated risk. Most country-hazard pairs should have low RC values.

## Fields

Each hazard includes a normalized regime-change object with these fields:

- `likelihood` (0.0–1.0): likelihood of a base-rate break in the next 1–6 months.
- `magnitude` (0.0–1.0): how large the out-of-pattern change could be if it occurs.
- `direction`: one of `up`, `down`, `mixed`, `unclear`.
- `window`: one of `month_1`..`month_6`, `month_1-2`, `month_3-4`, `month_5-6`.
- `rationale_bullets`: concise bullets tying the regime-change risk to evidence pack signals.
- `trigger_signals`: optional list of specific signals that would indicate a break.

The normalized object is serialized into `regime_change_json` in `hs_triage`.

## Score and Level

HS computes a regime-change score and severity level:

- **Score**: `likelihood * magnitude` (defaults to `0.0` if either value is missing).
- **Level** (likelihood-only thresholds, env-overridable):
  - Level 0: `likelihood < 0.15`
  - Level 1: `likelihood >= 0.15`
  - Level 2: `likelihood >= 0.35`
  - Level 3: `likelihood >= 0.55`

## Expected Distribution

Across a full run of ~120 countries × 6 hazards (~720 assessments), the expected distribution is:

- ~80% at likelihood ≤ 0.10 (base-rate normal, no regime change signal)
- ~10% at likelihood 0.10–0.30 (watch)
- ~7% at likelihood 0.30–0.55 (emerging)
- ~3% at likelihood ≥ 0.55 (strong signal)

A run-level distribution check logs warnings when too many assessments exceed expected thresholds.

## `need_full_spd` Override

By default, HS forces `need_full_spd = TRUE` when regime-change risk is **Level 1+** (likelihood >= 0.15), even if the triage tier would otherwise be quiet.

**Track assignment**: RC level > 0 → Track 1 (full ensemble), RC level 0 + priority tier → Track 2 (single Gemini Flash model), otherwise no SPD.

## Environment Overrides

All thresholds are environment-overridable:

- Level thresholds (likelihood-only):
  - `PYTHIA_HS_RC_LEVEL1_LIKELIHOOD` (default `0.15`)
  - `PYTHIA_HS_RC_LEVEL2_LIKELIHOOD` (default `0.35`)
  - `PYTHIA_HS_RC_LEVEL3_LIKELIHOOD` (default `0.55`)
- Distribution warning thresholds:
  - `PYTHIA_HS_RC_DIST_WARN_L1_FRAC` (default `0.25`) — warn if >25% of assessments are L1+
  - `PYTHIA_HS_RC_DIST_WARN_L2_FRAC` (default `0.15`) — warn if >15% of assessments are L2+
  - `PYTHIA_HS_RC_DIST_WARN_L3_FRAC` (default `0.08`) — warn if >8% of assessments are L3

## Shadow arm (December 2026)

A second regime-change assessment from a different model family runs beside
production, on the same evidence, so its recall can be measured before anyone
decides whether to use it. It is stored for comparison and never used.

**What runs.** For every hazard (ACE, DR, FL, TC) that production pass 1
assessed, the shadow model is sent the byte-identical RC prompt pass 1
received, grounding and evidence pack included; nothing is re-grounded or
re-fetched. The shadow model is the config role `rc_shadow` (GPT-6.1 Sol)
at medium effort (`role_params.rc_shadow.thinking`). Its answer goes through
the production RC parser and normaliser, and its level through the same
`PYTHIA_HS_RC_LEVEL*_LIKELIHOOD` thresholds. It is not pass 2: the pass
count, `PYTHIA_HS_RC_PASSES` and the pass merge are unchanged, and production
RC stays single-pass on Gemini Flash. Code: `horizon_scanner/rc_shadow.py`.

**What it can never change.** Nothing from the shadow is written to
`hs_triage.regime_change_json`, `need_full_spd`, a track or a question. A
test runs the RC path with the shadow answering likelihood 0.95 everywhere
and asserts `hs_triage`, levels and tracks are identical to a run with the
shadow off.

**How it runs.**
- Staged pipeline: the shadow calls go to their own provider batch family,
  `hs_rc_shadow`, through the OpenAI Batch API. The family is non-blocking
  (`llm_batch.NON_BLOCKING_FAMILIES`): the poller does not wait for it and the
  collect drain skips it, so a slow batch cannot delay an HS stage. The
  hs_rc_collect and hs_finalize stages record what has finished;
  fc_collect_finalize records the rest and cancels anything still running
  (stored `unfinished`). A request that was never submitted runs
  synchronously at hs_rc_collect, inside the cap.
- Sync path (`run_horizon_scanner.yml`): the calls run on a small background
  pool, which the run waits on, bounded (`PYTHIA_RC_SHADOW_WAIT_SEC`, 600),
  at its end.
- A shadow failure is logged and counted, never raised.

**Where it is stored.**
- `hs_rc_shadow`: one row per (run, country, hazard): status (`ok`, `error`,
  `parse_error`, `skipped_cap`, `unfinished`), likelihood, magnitude, score,
  level, direction, window, rationale and trigger signals, cost, source
  (`batch`, `sync`, `sync_fallback`, `replay`), `is_shadow`, `is_replay`,
  `is_test`.
- `llm_calls`: `is_shadow = TRUE`, `call_type = 'rc_shadow'`,
  `hazard_code = RC_<HZ>_SHADOW`, `phase = 'hs_triage'` (so the cost is HS
  spend and `run_comparison` shows it separately as `cost_rc_shadow_usd`).
- `rc_pass_detail__*.csv` labels these rows `pass_number = shadow`.

**Cost cap.** `PYTHIA_RC_SHADOW_MAX_USD` (default 15) per process. Once the
shadow has spent (or, at submit, is estimated to spend) that much, no further
shadow call is issued and each skipped one is stored `skipped_cap`.

**Environment.**

| Variable | Default | Effect |
|---|---|---|
| `PYTHIA_RC_SHADOW_ENABLED` | on in production, off in test mode and under pytest | Master switch |
| `PYTHIA_RC_SHADOW_MODEL` | role `rc_shadow` (`openai:gpt-6.1-sol`) | `provider:model_id` override |
| `PYTHIA_RC_SHADOW_THINKING` | `medium` (config `role_params`) | Effort override |
| `PYTHIA_RC_SHADOW_MAX_USD` | 15 | Per-run spend cap |
| `PYTHIA_RC_SHADOW_WAIT_SEC` | 600 | How long a stage waits for sync shadow calls |

**Diagnostics.** The debug bundle's `rc_shadow_comparison__<run>.json/.md`
(lifted into the forecast pipeline review) and `stage_health`'s `rc_shadow`
section give: level counts for production and shadow, the share at
likelihood <= 0.10 beside the expected ~80%, a 4x4 level agreement matrix,
the "would have forecast" list (shadow level 1+, production 0, with both
likelihoods and the shadow's rationale) and the reverse, and the shadow's
cost, calls, failures and cap skips.

**The recall check.** `pythia/tools/rc_recall_check.py`, run by
`rc_recall_check.yml` on the 12th (after the Resolver Update) and by hand.
For each production HS run and every assessed pair, Track 1 or not, it reads
whether a pattern break happened in the RC window (the six months after the
run month): a month breaks when its value lands two or more buckets above the
median of the pair's base-rate distribution (`base_rate_spd`), on the series
each hazard resolves on (ACE conflict deaths from ACLED, FL/TC people affected
from the resolution machine, DR Phase 3+). A pair with no break needs three
resolved months; otherwise it is unresolvable and out of every denominator.
It reports, per run and pooled, by hazard and by arm: pairs, resolvable,
breaks, recall and precision at level 1+, the Brier score of the likelihood
against the break indicator, and how many pairs each arm would have sent to
Track 1, with counts beside every rate. Results go to `rc_recall_check` and
the `rc-recall-check` artifact (not the Drive archive, which is chain-based).
Thresholds live in one block, `BREAK_CONFIG`. `--replay` (dispatch input
`replay`, cap `replay_max_usd`, default $10) sends the stored pass-1 prompts
of runs with no shadow rows to the shadow model, stored `is_replay = TRUE`;
those scores are optimistic, because a model trained after the run date may
have seen the outcome, and the report says so.

**Out of scope.** The shadow never merges into production, and the RC
thresholds, Track 2 rules and triage are unchanged. Promotion is the owner's
decision after at least two live months of recall data.
