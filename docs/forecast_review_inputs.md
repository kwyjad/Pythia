# The forecast pipeline review's inputs in Drive

The forecast pipeline review follows the template in Drive at
`Fred_archive/Performance_Reports/template/forecast_review_template.md`. That
copy is the one the scheduled session reads, and Kevin edits it by hand. Its
step A2 already tells the session to read `forecast_review/` beside the chain
zip; this note records what the archive puts there.

## Where it is

Every forecast chain gets `forecast_review/` beside its zip in
`Fred_archive/runs/<YYYY>/<YYYY-MM-DD>__forecast__<fc_run_id>__gh<publish_id>/`.
The leg writes it after the zip; `scripts/ci/drive_archive.py`
(`extract_forecast_review`) lifts the files and
`scripts/ci/forecast_review_inputs.py` builds the inputs file.

- `forecast_review_inputs.json` goes in first. Read it first.
- `SKIPPED.md` names every listed file the folder does not hold, and why: not
  in this chain, over 1 MB, would take the folder past 10 MB, or never copied
  because its figures are in the inputs file (`llm_calls_detail`,
  `spd_tables`, `provider_batch_objects`, the workflow logs).
- Lifted files keep a prefix naming where they came from: `debug__<file>`
  (the debug bundle, with its `__<fc_run_id>` suffix), `attribution__<file>`
  (the forecast attribution bundle), `current__<file>` (the current-run
  bundle), and `<run folder slug>__<file>` for stage files
  (`pipeline_stage__stage_health__fc_collect_finalize.json`,
  `sibyl__sibyl_run.log`).
- `workflow_logs_index.json`: one row per run in the chain, each step with its
  conclusion, `continue_on_error` (read from the workflow file at the run's
  commit; null when that could not be read), and counts of Python tracebacks
  and `##[error]` lines, with the first of each. A `::error` the shell echoed as
  script source is not counted. The logs themselves are not stored.
- `prompt_commits.json`: written into the chain tree by the archive job before
  the inputs file is built. The commits between the previous run's commit
  (`previous_run.meta.commit`) and this run's that touched
  `forecaster/prompts.py`, `forecaster/hazard_prompts.py`, `pythia/prompts/`,
  `forecaster/history_loaders.py`, `forecaster/binary_prompts.py` or
  `forecaster/gdacs_history.py`, from the GitHub compare API; `commits` is
  null with a `reason` when the lookup failed. It is read into
  `evidence.prompt_section_changes.commits`, not lifted.

On 13 November 2026 the folder came to 3.3 MB in 58 files, none over 430 KB.

## Keys

Every count names its file in a `source` field beside it. A derived figure says
"derived" in its source. A key that could not be built is
`{"stub": true, "reason": "...", "fallback_file": "..."}`: say so in Appendix 1
and open the fallback.

| Key | Template section | Built from |
|---|---|---|
| `meta` | header | `manifest.json` chain ids; commit from the debug bundle's `env_and_config`; forecast window from `question_metrics`; `INPUTS_VERSION` |
| `chain` | 2 | each run's `api_run.json` and the `CYCLE_SUMMARY.md` row (`did_work` null means the run wrote no chain diagnostics); chain start to release; batch queue median and max (`batch_lifecycle`); poller ticks and ticks that dispatched a stage (the chain's own Poll LLM Batches run folders where present, else the debug bundle's logs, which stop at 40), and first submit to last batch ended per provider (`batch_lifecycle`); `manifest.json` missing; steps that succeeded and logged a traceback or `##[error]` (`workflow_logs_index.json`) |
| `issues` | 3 | debug bundle `anomalies`, every non-OK health check, every run's chain-diagnostics `issues.json`; each marked `known` against `resolver/config/known_issues.yml` or the three acknowledged external issues; `narrow_ok_checks`: an OK whose detail says disabled, skipped, absent, not run or off |
| `inputs` | 4 | `connector_freshness` (sources not fresh), `crisiswatch_inject` (latest edition, countries in it, age, ACE question countries on an older edition, ACE questions with no row, editions missing), per inject the questions that should carry it and those whose country holds it (`question_metrics` x `data_inject_inventory`; base rates from the attribution `base_rates.csv`; calibration advice by arm from `llm_calls_detail`), `gdacs_prompt_check`, `food_security_health` |
| `horizon_scanner` | 5 | `rc_triage_summary` (pairs expected, screened out, assessed, levels by hazard), `rc_pass_detail`, `rc_grounding_health`, distribution warnings from `hs_stdout.txt`; `rc_pairs` (every pair's level) so the next run can name changes and two-level jumps |
| `triage_questions` | 6 | tiers and skip patterns, triage grounding, `hs_triage_failures`, `hs_triage_coverage`, the ACE/PA and Phase 3+ eligibility lines (source, asked, dropped and named), questions by hazard, metric and track, with and without a forecast; `question_keys` so the next run can name entries and exits |
| `evidence` | 7 | the RC sync constraint: SPD questions at RC 1+, checks generated (`grounding_detail` adversarial searches), injected (SPD prompts in `llm_calls_detail` carrying `ADVERSARIAL EVIDENCE CHECK`), withheld (`experiment_arms`), and the two mismatch counts; prompt section changes against the previous run, with the commits between the two runs' commits that touched a prompt file (`prompt_commits.json`); prompt cache by provider (`prompt_cache_report`); grounding by stage |
| `ensemble` | 8 | per member: calls, errors, extra attempts, parse failures, SPD and binary questions ok, median trace quality; expected size and members present; cells landed; members outside the profile; family recalibration status; questions whose BayesMC and mean month-1 expected bucket differ by more than 0.10 (`spd_tables`); member spread above the 90th percentile (`model_disagreement.csv`); priors copied from the anchor; binary and scenario counts |
| `track2` | 9 | `question_metrics` track 2, `trace_quality.csv` for `track2_flash` (SPD only: binary returns no trace) |
| `experiments` | 10 | advice, RC-shift, adversarial and Sibyl variant arms: design share, eligible, arm counts, unstamped rows, and whether the treated arm sits inside the exact 5th to 95th percentile of Binomial(eligible, share). `experiment_arms__*.json` first (written by `fc_collect_finalize` from the 13 Dec 2026 run; an arm it marks off is a stub with its reason), else the advice arm from `llm_calls_detail`, the RC-shift arm from `rc_shift_arms.csv`, the adversarial arm from the `[adversarial-arm]` stdout line |
| `sibyl` | 11 | `stage_health__sibyl.json` (run, caps, cost, tools, depth), the eligible pool from `sibyl_run.log`, `sibyl.json`, the five largest differences from the ensemble with the classes Sibyl claimed moved it (`fred_vs_sibyl.csv`), the cost gate ratio |
| `outputs` | 12 | the Interpreter line, this run's row of `gate_recalibration.md`, `anchor_depth.json`, bundles and sizes from `manifest.json`, the Sibyl upload and the Publish conclusion |
| `cost` | 13 | `cost_summary` by phase, per question (derived over the question count) and its ratio to the previous run, per question by track, batch totals, cache saving, cost of failed calls (derived from `llm_calls_detail`), `run_comparison` |
| `run_over_run` | 14 | the 25 largest moves by \|ln((ev_now+1)/(ev_before+1))\| from `run_over_run.csv`, each with whether its template's prompt sections changed and which `input_inventory.csv` fields changed since the previous run; `inventory` for the next run; `deltas.json` largest movements |
| `known_issues` | 16.3 | `known_issues.yml`: each entry's `review_by` and whether it is overdue at the chain date |
| `previous_run` | all | the same keys from the previous production forecast chain's `forecast_review_inputs.json`, read from Drive (the newest older folder whose `meta.is_test` is false); its own `previous_run` is left out. A stub when none exists |

### Prompt section changes

The attribution bundle hashes each prompt section of one example question per
template. A section whose length differs between that template's questions on
this run carries question data and is not compared. A heading that names a
country, a date or a number, or a template with one question, is listed under
`likely_data_changes`, apart from `changes`. Read `changes` as edits to the
prompt; check `likely_data_changes` against the commits before calling any of
them an edit.

## Backfill

A chain archived before this folder existed gets it from a dispatch of
**Archive Run to Drive** with `publish_run_id` and `forecast_review_only`. It
reads the chain zip back from Drive, writes the workflow-log index from the
Actions API (logs are kept 90 days), finds the previous chain's inputs in
Drive, and uploads only `forecast_review/`. A folder already there is left
alone unless `forecast_review_replace` is also set, which removes that folder
alone once its replacement is built. Dispatch the 13 October chain
(`38355281121`) before the 13 November one (`43538181703`), so November finds
its `previous_run`.
