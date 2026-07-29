# Handover — pipeline diagnostics, 2026-07-29

Written at the end of a session that ran a two-country test pipeline end to end and
built the per-stage diagnostics around it. Read this first in a new chat; it is
self-contained.

---

## Part 1 — Handover note

### Where things stand

`main` is green at `471a5a2`. Seven PRs landed today:

| PR | What |
|---|---|
| #817 | diagnostics coverage map + gap plan (`docs/diagnostics_gaps_2026-07.md`) |
| #818 | `batch_economics` + `post_run_diagnostics` at every stage |
| #819 | `stage_health` (grounding, RC levels, is_test, per-stage cost) |
| #820 | poller self-rescheduling; poller-decisions artifact; `if-no-files-found: error` |
| #821 | is_test breakdown readable from the job log |
| #822 | is_test stamping fix + empty-batch diagnosis |
| #823 | stale ACE/PA base-rate test that had `main` red since #816 |

### The reference run

Pipeline `hs_20260729T071137`, test mode, `PYTHIA_HS_ONLY_COUNTRIES=IRN,SOM`.
Completed S1→S4 plus Sibyl. 80 LLM calls, **$3.2856**, 565,008 tokens.
Useful run ids for comparison: HS `hs_20260729T071137`, forecaster `fc_1785315556`,
final stage Actions run `30442241491`, Sibyl run `30442501547`.

Its two findings drove #822:
- `hs_hazard_tail_packs` stamped `is_test=FALSE` in a test run (writer defaulted to
  `False`, no caller passed it).
- Both OpenAI batches returned zero results — 16/16 requests fell back to
  synchronous full-price calls, **67% of forecaster spend**. Root cause still
  unknown; #822 instruments it so the next occurrence names itself.

### New capabilities as of this handover

The owner has granted, and a **new chat will have** (this session did not):

1. **Artifact downloads** — `*.blob.core.windows.net` allowlisted. Actions artifact
   URLs are pre-signed, so `mcp__github__actions_get(download_workflow_run_artifact)`
   → `curl` now works. This session needed three manual uploads to diagnose one bug.
2. **Workflow dispatch** — can start runs directly instead of asking.

**First thing to do in the new chat: verify both.** One artifact download and one
dispatch, and report what actually happens rather than assuming. If dispatch still
403s, the fallback options are in `docs/diagnostics_gaps_2026-07.md`.

### Gotchas that cost time today — don't relearn them

- **The `*/15` poller cron is not honoured** (GitHub fires it ~hourly). #820 makes
  the poller chain itself. The chain is self-*sustaining* but not self-*starting*:
  after a deploy, an in-flight pipeline needs one manual poller dispatch to ignite.
- **4 of 5 jobs "skipped" in a stage run is correct** — one job per stage, gated
  `if: inputs.stage == ...`. Each stage is its own workflow run. Do not read it as a
  malfunction.
- **`only_countries` example text is `IRN,SOM`** — today's run used it verbatim and
  cost double. Type the country you actually want.
- **Diagnostics print to stdout as well as the step summary** on purpose: job logs
  are readable via the API, step summaries are not.
- **Never let a diagnostics step fail a stage.** The poller dispatches stages with
  `--ref main`, so stages 2..N of a live pipeline run whatever is on main.

---

## Part 2 — What to fix before the next test run

Ordered. P0 items are worth doing first; P2 is optional.

### P0-a. Close the test/trigger mismatch (the sweep)

**Why:** three incidents in one session came from one cause — *a test guards the
directory it imports from, not the directory it lives in*, and nothing connects
them. `test_idmc_history_conflict_pa.py` let `main` sit red for days.

**Measured gap:** `resolver-ci-fast.yml` runs all of `resolver/tests/` but triggers
only on `resolver/**`; `ci-lint.yml` runs on every PR but executes exactly one file.
So a PR touching only `forecaster/`, `horizon_scanner/` or `pythia/` runs **none** of
`resolver/tests/`. Twelve files there import non-resolver packages; two are wired
(#822, #823); **ten are not**:

| file | guards |
|---|---|
| `test_hs_prompt_imports.py` | horizon_scanner |
| `test_hs_rc_promoted_selection.py` | horizon_scanner |
| `test_hs_regime_change.py` | horizon_scanner, pythia |
| `test_hs_tail_trigger_query.py` | horizon_scanner |
| `test_rc_calibration.py` | horizon_scanner |
| `test_hs_triage_resilience_and_rerun_lists.py` | horizon_scanner, forecaster |
| `test_forecaster_llm_logging_inserts_row.py` | forecaster |
| `test_conflict_forecasts_table.py` | pythia |
| `test_eiv_scoring.py` | pythia |
| `test_risk_index_prefers_bayesmc.py` | pythia |

**All 53 tests in these files pass on `main` today** — verified. Wiring them is a
pure safety-net addition; it will not turn CI red.

**Do:**
1. Name each in the `paths:` trigger **and** pytest invocation of the workflow owning
   the code it guards — horizon_scanner → `horizon-scanner-ci.yml`, forecaster →
   `forecaster-ci.yml`. Files with two owners go in both; duplicate execution is
   cheap, a gap is not.
2. **Judgement call:** the three `pythia`-guarding tests. `forecaster-ci` triggers on
   `pythia/db/**`, `pythia/tools/**`, `pythia/tests/**` and a few named files, not all
   of `pythia/`. Check what each actually imports and add specific paths. **Do not
   blanket-add `pythia/**`** — that fires forecaster-ci on nearly every PR.
3. **Make it self-enforcing.** Wiring by hand is what failed three times. Add a check
   to `resolver/tests/test_ci_shell_sanity.py` (the repo's existing CI-config lint,
   already run on every PR by `ci-lint.yml` with `--noconftest`): for every
   `test_*.py` under `resolver/tests/` importing a top-level package other than
   `resolver`, assert the path appears in some workflow's `paths:` **and** its pytest
   call. Fail naming the file, the package, and the workflow to add it to. Keep it
   stdlib + `yaml` only, to survive `ci-lint`'s minimal install.
4. Add a CLAUDE.md known-failure-mode entry generalising the rule and citing all
   three incidents, so it reads as a pattern rather than three unrelated bugs.

### P0-b. Cost-weight the batch economics

**Why:** the reporter says *"32% of terminal requests took the synchronous
fallback"*. The true impact was **67% of spend**, because the OpenAI members are the
expensive ones. A request count understates a cost problem by 2× and buries the
signal. Reconstructing the real number took a manual `llm_calls_detail` analysis.

Also, `batch_economics`'s ledger section is **not run-scoped** — it reports
`llm_calls` cumulatively across the travelling DB, which produced a meaningless
"$1.37 of $65.26" today.

**Do:** pass `--hs-run-id` / `--fc-run-id` into `scripts/ci/batch_economics.py`
(both are already exported in every stage's env) and add a provider × service-tier
cost table scoped to the run — batched spend, sync spend, and **% of spend that lost
the discount**. Warn on the cost share, not just the request share. This is the
single highest-value diagnostic addition; it is the number that actually matters.

### P1. Give Sibyl the same per-stage diagnostics

**Why:** `run_sibyl.yml` emits **no** `batch_economics`, `stage_health` or
`post_run_diagnostics` — grep confirms only a single generic `upload-artifact`.
Sibyl ran 41 minutes on Claude Opus with a `$40` hard cap and its spend for the
reference run is unmeasured. It is plausibly the most expensive stage in the
pipeline and it is the only one with no cost telemetry.

**Do:** add `Record stage start` + `stage_health` (scoped to the Sibyl run) to
`run_sibyl.yml`, and include `sibyl` in the phase rollup. Same never-fail contract.

### P1. Put a batch line in the debug bundle health report

**Why:** `scripts/dump_pythia_debug_bundle.py` contains **zero** references to
`llm_batches` / `service_tier` / batches. Today's bundle reported *"LLM Calls: OK,
80 calls, 0 errors"* for a run that had lost the discount on two-thirds of its
spend. The bundle is the artifact a human reads first, and it said everything was
fine.

**Do:** add a Batch Economics row to the executive summary's Pipeline Status table —
batched vs fallback, and any batch with zero results — reusing the queries in
`scripts/ci/batch_economics.py`.

### P2. Cheap OpenAI batch probe before committing to a full run

**Why:** the OpenAI batch failure is uncaused and may recur. A full run costs real
money to discover it.

**Do:** dispatch a minimal probe first — `only_countries=SOM`,
`batch_providers=openai`, `test_mode=true` — and read the new empty-batch diagnosis.
If OpenAI batching is still broken, either fix it or drop `openai` from
`PYTHIA_BATCH_PROVIDERS` for the real run so the cost is at least predictable rather
than silently full-price.

---

## Verification

1. `python3 -m pytest resolver/tests/test_ci_shell_sanity.py -q --noconftest` — new
   guard passes; delete one wiring line and confirm it fails informatively.
2. `python3 -m pytest <the 10 files> -q` — all 53 still pass.
3. `python3 -m pytest scripts/ci/tests/ pythia/tests/test_llm_batch.py -q` — reporter
   regressions.
4. `yaml.safe_load` on every edited workflow, plus the upload-artifact / `[[ ]]` /
   `echo -e` lint.
5. **End to end:** a fresh `SOM`-only test pipeline. Success looks like
   `stage_health` reporting `is_test: consistent` with no warning, and
   `batch_economics` either showing all providers batched or naming the failing
   provider with its recorded cause and cost share.

## Deliberately not doing

- Moving tests between directories — several need `resolver/tests/conftest.py`
  fixtures; wiring gets the same protection at far lower risk.
- Guessing the OpenAI batch root cause. #822 instruments it; the next occurrence
  names itself.
- Back-filling the 10 mis-stamped `hs_hazard_tail_packs` rows — they belong to a
  test run that will not be published.
