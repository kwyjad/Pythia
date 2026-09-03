# The four bundles

Pythia produces four analysis and diagnostic bundles per forecast cycle. Three are built today; the fourth is planned. This page says what each answers, where each is produced, and how they join.

## At a glance

| Bundle | Artifact | Built by | Answers |
|---|---|---|---|
| Operational debug bundle | `pythia-debug-bundle` (+ `pythia-debug-bundle-workflow-logs` when split) | `run_sibyl.yml`, after the interpreter report | Did the pipeline run, and what went wrong: anomalies, batch lifecycle, provider objects, workflow logs, environment, per-question metrics, every LLM call |
| Current-run bundle (interpreter input pack) | `pythia-current-run-bundle` | `run_sibyl.yml`, after `compute_deviation` | What this run forecast and how far it moved from base rates, for the plain-language report; token-capped |
| Scored-forecast bundle | `pythia-ai-analysis-bundle` | `compute_calibration_pythia.yml`, after calibration advice | Reasoning beside outcome for every scored question: which forecasts were wrong and what the models said at the time |
| Forecast attribution bundle | `pythia-forecast-attribution-bundle` | `run_sibyl.yml`, after the operational bundle | Where probability mass moved and what each model said moved it, per hazard; prompt section fingerprints; input inventory |
| Resolutions bundle | not yet built | planned | Outcomes attached to `attribution_id`, so signal classes can be scored rather than only forecasts |

There is a fifth, unrelated bundle for the ingestion side: `resolver-debug-bundle-<run_id>`, built by `resolver_update.yml`. It is documented in `CLAUDE.md` and in the README's artifact list, and it joins none of the four above.

## Where they are produced

Since September 2026 everything a forecast cycle produces at its end is produced in one job, in one order. The `run_sibyl.yml` job runs: forecast deviation, the current-run bundle, the interpreter report and PDF, the operational debug bundle with the LLM prompt snapshot and its sibling uploads, the forecast attribution bundle, and only then the canonical DB upload and the publish dispatch. Every diagnostic step is `if: always()` with `continue-on-error: true`, so a bundle bug can never cost a run its DB upload.

Two things had to move with the debug bundle. Its environment and DB-signature collectors read the job they run in, so the forecaster stage (`fc_collect_finalize` in the staged pipeline, `build-report` in the legacy synchronous workflow) now writes `diagnostics/stage_context.json` at its end and uploads it as `pythia-stage-context`; the Sibyl job downloads it into the bundle as `stage_context/`, with a stub saying so when it is missing. And the staged pipeline passes `pipeline_id`, `hs_run_id` and `forecaster_run_id` to Sibyl when it dispatches it; on the `workflow_run` path those are resolved from the canonical DB by `scripts/ci/sibyl_bundle_context.py`. The shell-sanity test holds `dump_pythia_debug_bundle` to exactly one invoking workflow.

The scored bundle stays at the calibration terminus, because that is where outcomes exist.

## Join keys

| Key | Recipe | Present in |
|---|---|---|
| `run_id` | the forecaster run, `fc_<epoch>` | all four; `forecasts_raw`, `forecasts_ensemble`, `llm_calls`, `forecast_deviation` |
| `hs_run_id` | the Horizon Scanner run | all four; `hs_triage`, `hs_hazard_tail_packs`, `hs_runs` |
| `question_id` | `<ISO3>_<HAZARD>_<METRIC>_<YYYY-MM>` (epoch-suffixed) | all four; never matches across runs by design |
| `(iso3, hazard_code, metric)` | `interpreter.persistence.match_key` | the current-run bundle's deltas and the attribution bundle's `run_over_run.csv`, for run-to-run matching |
| `attribution_id` | `sha256("{run_id}|{question_id}|{model_name}|{update_index}")[:16]` | the attribution bundle's signal ledger, `evidence_to_signal.csv` and question records; the resolutions bundle will attach outcomes to it |
| `evidence_id` | `sha256("{url}|{title}")[:16]` | the attribution bundle's evidence items and links |
| `questions/<question_id>.json` | one record shape, `build_question_record` in `scripts/ai_bundle/build_scored_forecast_bundle.py` | the scored, current-run and attribution bundles; the attribution bundle adds an `attribution` block |

The `attribution_id` recipe is pinned by a test with hardcoded values and must not change once merged.

## What the attribution bundle is, and is not

The signal ledger is **claimed** attribution: what a model said moved it, written after the fact. A model can be confidently wrong about its own reasoning. Measured influence needs ablation, which no bundle does yet. The bundle's `ANALYST_GUIDE.md` says this in its opening paragraph; `LINKAGE.md` names the ablation harness as the intended next step.

Signal classes come from `scripts/ai_bundle/signal_taxonomy.csv`, a versioned list of regular expressions. Classification is deterministic and never uses a model call, so ledgers built under one taxonomy version can be compared month over month.

## Building locally

```bash
# Operational debug bundle (needs an HS run id; forecaster run id optional)
python -m scripts.dump_pythia_debug_bundle --db duckdb:///data/resolver.duckdb \
  --hs-run-id hs_20260901T040916 --forecaster-run-id fc_1788237725

# Current-run bundle
python -m scripts.ai_bundle.build_current_run_bundle --db duckdb:///data/resolver.duckdb --out-dir ai_bundle

# Scored-forecast bundle
python -m scripts.ai_bundle.build_scored_forecast_bundle --db duckdb:///data/resolver.duckdb --out-dir ai_bundle

# Forecast attribution bundle
python -m scripts.ai_bundle.build_forecast_attribution_bundle --db duckdb:///data/resolver.duckdb --out-dir ai_bundle
```

All four exclude `is_test` rows by default and accept `--include-test`. None writes to the database.
