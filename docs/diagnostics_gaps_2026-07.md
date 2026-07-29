# Staged-pipeline diagnostics: coverage map and gap-closure plan (2026-07-29)

Scope: the four-stage Batch-API pipeline (`pythia_pipeline_stage.yml`) plus its
poller (`poll_llm_batches.yml`). Written while preparing a Somalia-only
test-mode smoke run; the coverage map below is derived from the workflow and
script sources, not from a completed run.

## 1. What each stage actually emits today

| Stage | `pythia-stage-diagnostics` contents | Other artifacts |
|---|---|---|
| S1 `hs_submit` | `hs_stdout.txt` | `pythia-batch-state`, `pythia-resolver-db-staged` |
| S2 `hs_rc_collect` | `hs_stdout.txt` | `pythia-batch-state`, `pythia-resolver-db-staged` |
| S3 `hs_finalize_fc_submit` | `hs_stdout.txt`, `forecaster_submit_stdout.txt`, `hs_triage_coverage__*.csv`, `hs_triage_failures__*.json` | `pythia-batch-state`, `pythia-resolver-db-staged` |
| S4 `fc_collect_finalize` | `forecaster_stdout.txt` | `pythia-health-report`, `pythia-llm-prompts`, `pythia-question-metrics`, `pythia-llm-calls-detail`, `pythia-spd-tables`, `pythia-debug-bundle`, `pythia-resolver-db` |

Each stage is a separate workflow run, so the repeated artifact name does not
collide — retrieval is per-run.

The asymmetry is the story: **S1–S3 emit raw stdout and a DB; essentially all
interpreted diagnostics live at S4.**

## 2. Blind spots

### G1 — Batch economics are recorded but never surfaced (highest value)

`llm_batches` carries `n_requests / n_succeeded / n_errored / n_expired /
n_fallback_sync` per batch, and `llm_batch_requests.status` carries
`fallback_sync` per request (written by `forecaster/cli.py:249` and
`horizon_scanner/hs_batch.py:256`). **Nothing reads these into any artifact or
step summary.** `emit_batch_state.py` serialises only *pending* batches.

Consequence: a pipeline in which every request silently fell through to the
synchronous path — full price, zero batch discount — produces artifacts
byte-indistinguishable from a fully-batched one. The entire economic premise of
the staged pipeline is unverified. This is the same failure shape as the
2026-07-28 audit finding C1 (prompt caching enabled in production but wired to
no call site, invisible because nothing consumed the telemetry).

### G2 — `post_run_diagnostics.py` runs only at S4

It has three sections (table counts, forecast breakdown, prompt-cache stats).
The prompt-cache zero-reads `::warning::` therefore never evaluates the HS
stages, so RC/triage cache effectiveness is never checked at all.

### G3 — The health report is gated behind the last stage succeeding

`Dump Pythia v2 debug artifacts` is `if: success()` at S4. If the pipeline
stalls at S2 — a documented and recently-fixed failure mode — the run produces
**no health report, no executive summary, no grounding health, no cost
attribution** for work already paid for.

### G4 — Grounding health surfaces hours after the money is spent

RC grounding runs at S1, triage grounding at S2, but `rc_grounding_health` /
`triage_grounding_health` only appear in the S4 debug bundle. A Brave
circuit-breaker trip at S1 (which blocks ungrounded hazards from forecasting)
is not visible until the pipeline ends.

### G5 — No per-stage cost attribution

`llm_calls` accumulates in the travelling DB, but only S4 emits
`llm_calls_detail`. Per-stage spend (S1 RC grounding + RC, S2 triage grounding +
triage, S3 adversarial + submit, S4 SPD collect) is never broken out.

### G6 — The poller emits no artifact

`poll_llm_batches.py` makes every dispatch decision, including the
dispatch-attempt cap and the stall `::error`. Its reasoning exists only in job
logs, which expire and are not collected. "Why did this pipeline stall?" has no
durable evidence trail.

### G7 — Test-mode stamping is never asserted

Given the documented boolean-input coercion hazard (the poller dispatches
stages 2..N over REST, where a boolean can arrive as the string `"false"`),
nothing in the artifacts confirms that `is_test` was stamped consistently across
`hs_runs` / `questions` / `forecasts_*` / `llm_calls`.

### G8 — Silent-gap ergonomics

Every diagnostics upload uses `if-no-files-found: warn`. A diagnostics file that
was never written degrades to a warning inside a green run.

## 3. Plan

Ordered by value per unit of effort. Items 1–3 are the ones worth doing before
the next production cycle.

**1. `scripts/ci/batch_economics.py` (closes G1).** Read `llm_batches` +
`llm_batch_requests` for the pipeline id; emit `diagnostics/batch_economics.json`
and a Markdown table into `$GITHUB_STEP_SUMMARY`: per family and provider —
requests, succeeded, errored, expired, fallback_sync, and a realised-discount
estimate joined to `llm_calls.usage_json.service_tier == "batch"`. Run it at
every stage that collects, and add it to the stage-diagnostics upload path. Add
a `::warning::` when the fallback-sync rate exceeds a threshold (suggest 20%),
mirroring the prompt-cache zero-reads warning. Roughly one script plus four
three-line workflow edits.

**2. Run `post_run_diagnostics.py` at every stage (closes G2, part of G5).**
It is already DB-only and cheap. Add a `--stage` argument so its output is
labelled, and extend it with a per-stage cost rollup (`llm_calls` grouped by
`phase` / `call_type`, filtered to this stage's window). Emit to
`diagnostics/post_run__<stage>.json` and include in the stage-diagnostics
artifact.

**3. Emit a minimal health snapshot at every stage (closes G3, G4, G7).** Not
the full debug bundle — a small `diagnostics/stage_health__<stage>.json`
carrying: grounding health counters (including
`no_backend_by_reason`, so a breaker trip is countable at S1), Brave breaker
state, RC/triage level distribution once available, row counts for the tables
this stage wrote, and an `is_test` consistency assertion across the stamped
tables. Gate it `if: always()` so a failing stage still reports. The existing
`_evaluate_pipeline_health` helpers in `scripts/dump_pythia_debug_bundle.py` can
be factored out rather than reimplemented.

**4. Persist poller decisions (closes G6).** Have `poll_llm_batches.py` write
`diagnostics/poll_decision.json` per tick (pipeline id, per-batch provider
status, terminal/waiting verdict, dispatch attempt count, action taken) and
upload it as `pythia-poller-decisions`. Small, and it is the only way to
reconstruct a stall after logs expire.

**5. Tighten upload ergonomics (closes G8).** For files a stage is *expected* to
produce, switch `if-no-files-found: warn` to `error`. Keep `warn` only for
genuinely optional artifacts.

## 4. Note on the smoke run

The Somalia test-mode run this document was written for could not be dispatched
from the agent session: the GitHub App installation lacks `actions: write`
(`403 Resource not accessible by integration` on the workflow dispatch
endpoint), and no `gh` CLI is available in the environment. Dispatch must be
performed by a maintainer; retrieval and reporting can then proceed from the
session, which does hold Actions read access.
