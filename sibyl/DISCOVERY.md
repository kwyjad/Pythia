# Sibyl — Discovery Map

Findings from the pre-implementation discovery pass (July 2026). Each numbered
section answers one discovery question from the Sibyl spec; decisions derived
from a finding are marked **Decision:**.

## 1. Volatility scores

There is **no first-class "volatility" score** in the codebase. The closest
per-question signals live in `hs_triage` (one row per `run_id, iso3,
hazard_code`):

- `regime_change_score DOUBLE` = likelihood × magnitude, clamped [0, 1]
  (`horizon_scanner/regime_change.py`). RC measures *expected departure from
  historical base rates* — i.e. exactly "how volatile is this question's
  underlying process right now".
- `triage_score DOUBLE` = overall risk level (not volatility).

Questions join to triage via `questions.hs_run_id = hs_triage.run_id AND
questions.iso3 = hs_triage.iso3 AND questions.hazard_code =
hs_triage.hazard_code`.

**Decision:** `volatility := regime_change_score` (primary key, descending),
then `question_id` for determinism. This is documented in
`sibyl/select_questions.py`.

**Superseded (Oct 2026): the selection rule.** Until October 2026 Sibyl took a
plain top 10 by volatility with `triage_score` as tiebreak. Two things were
wrong with that. The tiebreak was dead: every row with RC >= 0.1 is tier
`rc_promoted` and carries a placeholder `triage_score` of 0. And the mix
swung by month with no floor under any hazard: the 1 October run chose six
drought, two conflict and two flood questions and no cyclone, so Sibyl's
scored record could say nothing about cyclones. Selection is now
floor-then-fill (`floor_then_fill`): 25 questions (`SIBYL_N_QUESTIONS`), each
hazard first takes its three most volatile (`SIBYL_MIN_PER_HAZARD`), then the
most volatile remaining candidate whose hazard holds fewer than ten
(`SIBYL_MAX_PER_HAZARD`) until 25 are chosen. Ties go to the hazard holding
fewer picks, then `question_id`. Run order is floor picks first, then fill
picks, each by falling volatility, so a budget or time cut removes fill picks
first. Each `sibyl_forecasts` row records its `selection_pass`. Replayed on
the 1 Aug, 15 Sep and 1 Oct 2026 runs the rule gives 25 questions in 22-24
countries: conflict 6-10, drought 6-10, flood 3-6, cyclone 3-6
(`tests/fixtures/sibyl_candidate_pools.json` pins it). The pool limits how
selective 25 can be: eligible questions with RC >= 0.1 numbered 18, 29 and 26
in those runs, so in August 8 of the 25 picks fell below 0.1.

**Hazard/metric scope.** Question generation
(`scripts/create_questions_from_triage.py`) emits, for the active hazards:
ACE → FATALITIES + PA; FL → PA + EVENT_OCCURRENCE; TC → PA +
EVENT_OCCURRENCE; DR → PHASE3PLUS_IN_NEED + EVENT_OCCURRENCE. There are **no
DR/PA questions** — the "people affected" analogue for drought in this
codebase is `PHASE3PLUS_IN_NEED` (IPC Phase 3+ population).

**Decision:** Sibyl's eligible (hazard, metric) pairs, per the spec's "ACE
fatalities; DR/FL/TC affected" scope, are:
`{(ACE, FATALITIES), (FL, PA), (TC, PA), (DR, PHASE3PLUS_IN_NEED)}`.
EVENT_OCCURRENCE (binary) is excluded everywhere. ACE/PA is excluded (the
spec names ACE *fatalities* only). The set is a config constant
(`sibyl/config.py: ELIGIBLE_HAZARD_METRICS`) so it can be widened later.

## 2. SPD serialization

The native SPD representation is **bucketed probabilities**, not quantiles:

- Buckets are defined in `pythia/buckets.py` `BUCKET_SPECS` per metric:
  PA = 6 buckets (0, 1–<10k, 10k–<50k, 50k–<250k, 250k–<500k, ≥500k),
  FATALITIES = 7, PHASE3PLUS_IN_NEED = 6. Every metric leads with a
  dedicated "0" bucket. Helpers: `thresholds_for`, `labels_for`,
  `n_buckets_for` — never re-declare literals.
- Storage: one row per (month, bucket) in **two** tables
  (`pythia/db/schema.py`):
  - `forecasts_raw(run_id, question_id, model_name, month_index 1..6,
    bucket_index 1..K, probability, ok, elapsed_ms, cost_usd, tokens…,
    status, spd_json, human_explanation, horizon_m, class_bin, p, is_test,
    reasoning_trace_json)` — this is what **`compute_scores._load_spd` reads**
    (it scores every `DISTINCT model_name` found here).
  - `forecasts_ensemble(run_id, question_id, iso3, hazard_code, metric,
    model_name, month_index, bucket_index, probability, ev_value,
    weights_profile, created_at, status, human_explanation, is_test,
    reasoning_trace_json)` — this is what the dashboard question page and
    risk index read.
- Keying: `(run_id, question_id, model_name, month_index, bucket_index)`.
  Aggregates use reserved model_names (`ensemble_bayesmc_v2`,
  `ensemble_mean_v2`, `track2_flash`).
- Month-anchoring convention (critical): `month_index` 1 = the question's
  `window_start_date` month. Writers map labels via
  `_month_index_for_label(label, anchor_month)`
  (`forecaster/month_utils.py`); never positional.
- The reference writer is `forecaster/cli.py::_write_spd_outputs`
  (DELETE-then-INSERT per (run_id, question_id, model_name)).

**Decision:** Sibyl emits the identical representation under
`model_name = 'sibyl'` (config: `SIBYL_MODEL_NAME`) into both tables, with
`weights_profile = 'sibyl'` in `forecasts_ensemble` as the track marker.
Because `compute_scores` enumerates `DISTINCT model_name` from
`forecasts_raw` and resolutions/scoring are keyed by question+horizon, Sibyl
gets scored head-to-head with zero scoring changes. The pooled CDF is
discretized onto bucket boundaries from `thresholds_for(metric)` to produce
the bucket vector. `'sibyl'` is added to `AGGREGATE_MODEL_NAMES` in
`pythia/tools/compute_calibration_pythia.py` so it is scored but **excluded
from the ensemble-member weight softmax** (it is an aggregate of its own
trials, not a member of the standard ensemble).

Full trial-level provenance (quantiles, belief traces, evidence, costs,
divergences) goes to a new dedicated table `sibyl_forecasts` plus a run-level
`sibyl_runs` table (see §4/§7 of the implementation).

## 3. Resolver DB base rates

The forecaster already builds per-question base rates from `facts_resolved`
(Resolver DB) — reused wholesale:

- **Natural hazards (FL/TC + DR):**
  `forecaster/cli.py::_build_natural_hazard_seasonal_profile(iso3, hazard)`
  → `{type: "seasonal_profile", months: {1..12: {min, max, mean, median,
  n_observations}}, years_of_data, data_range, source}`. So per-calendar-month
  climatology **does** exist, with dispersion (min/max/median), not just a
  mean. DR/PHASE3PLUS_IN_NEED uses
  `_load_fewsnet_phase3_history(iso3)` → `{type: "fewsnet_phase3",
  last_6m_values, recent_mean, recent_max, trend, coverage_pct}` (null-aware
  monthly series).
- **Conflict (ACE):** `_build_conflict_base_rate(iso3, hazard)` →
  `{type: "conflict_trajectory", fatalities: {last_month, trailing_3m_avg,
  trend_pct, trend_direction}, displacements: {...}}` — ACLED recent-months
  framing (autocorrelated recency, not climatology), exactly what the spec
  asks for.
- Dispatch: `_build_history_summary(iso3, hazard_code, metric)`; prompt
  rendering: `forecaster/history_loaders.py::_format_base_rate_for_prompt`.

**Decision:** `sibyl/base_rates.py` calls `_build_history_summary` +
`_format_base_rate_for_prompt` and wraps the result in Sibyl's outside-view
framing (anchor-not-target, right-skew widening instruction when only means
are available, seasonal-adjustment instruction with the target calendar
months). No new Resolver queries are written.

## 4. DuckDB access layer

`pythia/db/schema.py::connect(read_only: bool = False)` returns a pooled
connection resolved from `PYTHIA_DB_URL`; `ensure_schema(con)` is idempotent.
All pipeline writers (`forecaster/cli.py`, `forecaster/llm_logging.py`) use
`connect()` + explicit `con.close()`. Test-mode stamping comes from
`pythia/test_mode.py::is_test_mode()` → `is_test` column.

**Decision:** Sibyl uses `connect()`/`ensure_schema()` exclusively; new
tables are added to `pythia/db/schema.py` (the authoritative schema file).

## 5. Brave search wrapper

`pythia/web_research/backends/brave_search.py::fetch_via_brave_search(query,
*, recency_days, include_structural, timeout_sec, max_results, hazard_code,
country_name)` → `EvidencePack` (`pythia/web_research/types.py`). It is wired
to the circuit breaker (`brave_circuit_breaker.py`: trips after 3 consecutive
failures; `is_tripped()` short-circuits), rate-limited
(`PYTHIA_BRAVE_MAX_RPS`), retries 429s, and reports `cost_usd` ($0.005/query)
in `pack.debug["usage"]`.

Limitation found: `freshness` is derived from `recency_days` via
`_map_freshness` (pd/pw/pm/py) — a window ending *now*. Backtest date-capping
needs Brave's date-range form (`YYYY-MM-DDtoYYYY-MM-DD`).

**Decision:** add one optional kwarg `freshness_override: str | None = None`
to `fetch_via_brave_search` (passed through to the API instead of the mapped
recency value). This keeps a single search path — no second Brave client.
`sibyl/tools.py` builds the date-range string from `asOf`; `sibyl/leakage.py`
post-filters results by date and blocked domains.

## 6. Cost tracking

Single ledger: the **`llm_calls`** table. Writer:
`forecaster/llm_logging.py::log_forecaster_llm_call(...)` (async) — computes
cost from `pythia/model_costs.json` via
`forecaster/providers.py::resolve_price_per_1m`/`estimate_cost_usd`, records
`phase`, `provider`, `model_id`, tokens, `cost_usd`, `iso3`, `hazard_code`,
`is_test`. Brave grounding calls are logged as `provider='brave'`,
`model_id='brave-web-search'` with explicit `cost_usd`.

Dashboard cost surface: `/v1/costs/*` (`pythia/api/routes/costs.py` →
`resolver/query/costs.py`) reads `llm_calls`, groups `by_model` and
`by_phase`; `known_phases = {web_search, hs, research, forecast, scenario,
other}` (line ~521).

**Decision:** every Sibyl Opus call and Brave query is logged through
`log_forecaster_llm_call` with `phase='sibyl'`; `'sibyl'` is added to
`known_phases` so it is itemised in the by-phase pivot. Opus-vs-Brave
itemisation falls out of the existing `by_model` grouping
(`claude-opus-4-8` vs `brave-web-search`). The run-level running total for
the budget guard is kept in-process by `sibyl/cost.py` (authoritative for
the cap) and persisted to `sibyl_runs` / `sibyl_forecasts`.

**Cost table gap fixed:** `pythia/model_costs.json` had no
`claude-opus-4-8` entry (missing entries silently log $0). Added
`[0.005, 0.025]` per 1K tokens ($5/$25 per MTok). (The cost table was
later converted to per-1M rates — the entry is now `[5.00, 25.00]`.)

**Provider gap fixed:** `providers.py::call_anthropic` always sends
`temperature`, but `claude-opus-4-8` (like Opus 4.7) rejects sampling
params with HTTP 400. Added a no-temperature model guard
(`_ANTHROPIC_NO_TEMPERATURE_PREFIXES`). Trial diversity comes
from prompt variation (per-trial perspective seeds), not temperature.

## 7. Dashboard integration points

**FastAPI** (`pythia/api/`): route modules in `pythia/api/routes/*.py`, each
`router = APIRouter()`, registered in `app.py` (~line 478-541). Shared
helpers in `pythia/api/core.py` (`_con`, `_execute`, `_test_filter`,
`_table_exists`). Route modules must never import `app.py`.

- Run-summary view: `GET /v1/diagnostics/run_summary`
  (`routes/diagnostics.py::diagnostics_run_summary`, ~line 1030) — gains a
  `sibyl` block (coverage, cost, `budget_capped`, skipped count) read from
  `sibyl_runs`/`sibyl_forecasts` when the tables exist.
- PA KPI / risk index: `GET /v1/risk_index` (`routes/risk_index.py`) selects
  rows from `forecasts_ensemble` via a chosen-model CTE preferring
  `ensemble_bayesmc_v2` > `ensemble_mean_v2` — gains an optional
  `model=sibyl` query param that overrides the preference, enabling the
  frontend Sibyl toggle.
- Performance/scores: `GET /v1/performance/scores` groups `scores` by
  `model_name` — Sibyl rows appear automatically once `compute_scores` runs
  (no change needed).
- Costs: `/v1/costs/*` — Sibyl appears via `phase='sibyl'` (see §6).
- New module: `pythia/api/routes/sibyl.py` — `/v1/sibyl/summary`,
  `/v1/sibyl/questions` (sortable JS-divergence table),
  `/v1/sibyl/question_detail` (trials, belief traces, pooled + standard SPD
  for overlay).

**Next.js** (`web/src/`): pages under `web/src/app/*/page.tsx`; API helper
`web/src/lib/api.ts::apiGet` (base URL `NEXT_PUBLIC_PYTHIA_API_BASE`); nav in
`web/src/components/Nav.tsx` (desktop ~46-116 AND mobile ~155-241 lists).
Question detail SPD rendering: `web/src/app/questions/[questionId]/SpdPanel.tsx`
merges `forecast.ensemble_spd` + `raw_spd` sources by `model_name` — a
`'sibyl'` model_name automatically becomes a selectable source there.

- New page: `web/src/app/sibyl/page.tsx` + `SibylClient.tsx` — per-question
  overlay of Sibyl pooled SPD vs standard SPD, K trial distributions,
  JS divergence (track-vs-track prominent + sortable, inter-trial secondary),
  expandable belief-state traces and evidence lists.
- Run summary: `web/src/components/RunSummaryView.tsx` gains a Sibyl block.
- PA KPI view: `web/src/components/RiskIndexPanel.tsx` gains a
  standard/Sibyl source toggle (passes `model=sibyl` to `/risk_index`).

## 8. fred_overview.md

`docs/fred_overview.md`, rendered on the About page. **Must run
`bash scripts/snapshot_overview.sh` before editing** and commit the snapshot
alongside (per CLAUDE.md). Prompt files similarly require
`bash scripts/snapshot_prompts.sh` before editing.

## 9. GitHub Actions

- The standard forecasting pipeline is a **single workflow**:
  `run_horizon_scanner.yml` ("Horizon Scanner Triage") — HS → create
  questions → forecaster, all in one job. It uploads the canonical
  `pythia-resolver-db` artifact at the end.
- Downstream chaining pattern: `on.workflow_run: {workflows: ["<name>"],
  types: [completed]}` + job-level
  `if: github.event_name == 'workflow_dispatch' ||
  github.event.workflow_run.conclusion == 'success'`; DB obtained via
  `gh run download ${{ github.event.workflow_run.id }} -n pythia-resolver-db`
  (Path A) with canonical-discovery fallback (Path B, as in
  `compute_calibration_pythia.yml`); shared
  `concurrency: {group: pythia-resolver-db, cancel-in-progress: false}`;
  re-upload of `pythia-resolver-db` at the end.
- Gating on "a run exists": row-count check pattern (HS_QUESTION_COUNT) — an
  inline duckdb query; Sibyl gates on hs_triage + eligible questions existing
  for the latest HS run.
- Secrets: `ANTHROPIC_API_KEY`, `BRAVE_SEARCH_API_KEY` (exact names used at
  job-level env in run_horizon_scanner.yml), `GITHUB_TOKEN` for `gh`.
- Deps install: `pip install -r python_library_requirements.txt` +
  `pip install duckdb`, Python 3.11 (`actions/setup-python@v5`).
- Publish note: `publish_latest_data.yml` fires on HS Triage completion (in
  parallel with Sibyl). To make Sibyl outputs visible on the dashboard
  without waiting for the next publish, `run_sibyl.yml` explicitly
  dispatches `publish_latest_data.yml` with its own `run_id` after uploading
  the artifact (same pattern as `compute_calibration_pythia.yml`; requires
  `permissions: actions: write`).

**Decision:** new `.github/workflows/run_sibyl.yml`, `workflow_run` on
"Horizon Scanner Triage", budget caps passed as env
(`SIBYL_RUN_HARD_CAP_USD`, `SIBYL_BUDGET_USD_PER_QUESTION`).

## 10. JS-divergence utility

`pythia/tools/generate_calibration_advice.py::_js_divergence(p, q)` —
Jensen–Shannon divergence over probability vectors (clips at 1e-12,
normalizes, natural log). Used there for month-1 vs month-6 SPD flatness
checks.

**Decision:** Sibyl imports `_js_divergence` from
`pythia.tools.generate_calibration_advice` (no copy). Track-vs-track JSD is
computed per month over bucket vectors (Sibyl vs `ensemble_bayesmc_v2`,
falling back to `ensemble_mean_v2`), averaged across months; inter-trial
disagreement is the mean pairwise JSD of the K trial bucket vectors.

## Files to modify (beyond the new `sibyl/` package)

| File | Change |
|---|---|
| `pythia/db/schema.py` | `sibyl_runs` + `sibyl_forecasts` tables |
| `pythia/model_costs.json` | `claude-opus-4-8` cost entry |
| `forecaster/providers.py` | no-temperature guard for Opus 4.7/4.8-family models |
| `pythia/web_research/backends/brave_search.py` | optional `freshness_override` kwarg |
| `pythia/tools/compute_calibration_pythia.py` | add `'sibyl'` to `AGGREGATE_MODEL_NAMES` |
| `resolver/query/costs.py` | add `'sibyl'` to `known_phases` |
| `pythia/api/app.py` | register sibyl router |
| `pythia/api/routes/sibyl.py` | new route module |
| `pythia/api/routes/diagnostics.py` | `sibyl` block in run_summary |
| `pythia/api/routes/risk_index.py` | optional `model` param |
| `web/src/components/Nav.tsx` | Sibyl nav entry (desktop + mobile) |
| `web/src/components/RunSummaryView.tsx` | Sibyl coverage block |
| `web/src/components/RiskIndexPanel.tsx` | Sibyl toggle for PA KPI view |
| `web/src/app/sibyl/*` | new dashboard page |
| `web/src/lib/types.ts` | Sibyl response types |
| `docs/fred_overview.md` | Sibyl section (after snapshot) |
| `.github/workflows/run_sibyl.yml` | new workflow |
| `CLAUDE.md`, `README.md` | documentation |

## Calibration advice (Oct 2026)

`sibyl/calibration.py::calibrate` stays an identity pass-through: six scored
questions cannot fit a statistical correction. What was added instead is an
advice loop built only on Sibyl's own record.

**Decision: the record is defined by `sibyl_forecasts`, never by `scores`
alone.** Status `ok`, production Sibyl runs only, the latest run per
question, joined to `resolutions`. The reason is a fault found on the way:
`compute_scores` stamped a score's `is_test` from the question, and question
ids are epoch-keyed, so a production question also forecast by a same-epoch
test run carried that run's scores as production rows (on the 2026-10-02
release: 9 Sibyl questions in `scores`, 3 of them test-only). Fixed in
`compute_scores` (a score is test when its forecast is) and in the two
latest-run helpers.

**Decision: the unit is the question.** A Sibyl question's six months share
one forecast. Counts are of distinct questions and intervals come from a
bootstrap over whole questions (2,000 draws, fixed seed, numpy).

**Decision: gate twice.** A class gets its own advice at 20 distinct scored
questions (`SIBYL_ADVICE_MIN_QUESTIONS`), the pooled row across the four
classes stands in at 20 otherwise, and a finding becomes an instruction
only when its 90% interval excludes the calibrated value. Perspective bias
and paired skill against `ensemble_mean_v2` / `__ext_climatology` are
findings only and never reach a prompt (independence).

**Decision: a held-out arm.** `SIBYL_ADVICE_EXPERIMENT_SHARE` (0.5) of
questions get no advice, by a hash of `"sibyl:" + question_id` (a split
independent of the standard track's). `sibyl_forecasts.advice_arm` and
`advice_as_of_month` record what each forecast saw, and `base_rate_json`
the outside view at forecast time (the anchor-departure diagnostic needs it).

**Decision: own table.** `sibyl_calibration_advice`, keyed
`(as_of_month, hazard_code, metric)`, pooled row `*`/`*`, not the standard
`calibration_advice` (whose non-shared rows are deleted monthly). In
backtest mode `load_advice` returns nothing: advice learned after the as-of
date is leakage.


## 2026-10-03 — Part 1: an honest record

A review of the 2 October 2026 release found that all six scored Sibyl
forecasts came from the 15 July run (`sibyl_1784113515141`), in which the
shared Brave circuit breaker tripped on the first three calls and all 216
searches failed. Sibyl stored ten forecasts with status `ok` anyway. Later
runs read little: 78 of 120 trials read no page.

**Decision: an evidence gate.** A trial counts searches that returned a
source (`n_search_ok`) and documents read (`n_docs_read`); it has evidence
at `SIBYL_MIN_SEARCH_OK` (1) and `SIBYL_MIN_DOCS_READ` (0). Only trials that
finished AND have evidence are pooled; below `SIBYL_MIN_VALID_TRIALS` (2)
the question is stored `failed` / `no evidence`, its trials kept, nothing
written to `forecasts_raw` or `forecasts_ensemble`. The thresholds are low
on purpose: Opus 5.5 averages three searches a trial, and Part 3 raises them
once the agent is made to research in depth.

**Decision: old forecasts are flagged, not deleted.**
`sibyl_forecasts.evidence_ok` is written for every new row and backfilled
for old ones (`sibyl/evidence.py`: evidence when at least two trials made a
search whose tool call succeeded), at the start of every run and of every
advice generation. The July run's ten forecasts read FALSE. Their rows and
scores stay; the advice loop, the head-to-head comparison, the Sibyl API's
figures and the interpreter's second opinion leave them out, and the Sibyl
page shows them with a "no evidence" badge.

**Decision: Sibyl owns its breaker reset.** The breaker is a process-wide
singleton shared with HS grounding. `run_sibyl` resets it at the start, and
a search that finds it tripped waits `SIBYL_BREAKER_COOLDOWN_SEC` (60),
resets it and retries once, at most `SIBYL_BREAKER_MAX_RESETS` (5) times a
run. `sibyl_runs` records searches made, searches failed (an empty HTTP 200
answer is not a failure), breaker trips and documents read;
`scripts/ci/stage_health.py` marks a run degraded above 20% failed.

**Decision (owner, 2026-10-03): resolution sources are open in live runs.**
ACLED, IFRC GO, IDMC, GDACS, FEWS NET and IPC are blocked only in backtest
mode (`sibyl.leakage.is_blocked_for`). In a live run the outcome does not
exist yet, so the resolving source's latest figures are the best evidence
of where a series stands; forecast products (ACLED CAST, VIEWS) may be read
too. This reverses the "always blocked" rule of July 2026.

**Smaller fixes.** Every written vector is floored at `SIBYL_BUCKET_FLOOR`
(0.005) and renormalised (ten of 258 buckets were exactly zero, and
`compute_scores` floors at 1e-9: about 20.7 nats if one occurs). A trial
whose later step fails on every attempt keeps its last valid belief
(`degraded: model_step_failed`). A seed with no anchor no longer claims a
base rate. Flood, cyclone and drought PA questions no longer say "as
resolved by EM-DAT"; they name IFRC GO and IDMC and say an unrecorded month
does not resolve. The standard calibration advice's month-position check
reads ensemble members only (Sibyl wrote one vector to all six months) and
uses the metric's real bucket count.

## 2026-10-03 — Part 2: start from the reference, pool with it

**Decision (owner): Sibyl gets its own prior.** `sibyl/reference.py` builds
one bucket vector per window month; the ensemble's anchor
(`PYTHIA_PRIOR_ANCHOR_SPD`, `forecaster/prompts.py`) is untouched.

* ACE/FATALITIES: `base_rate_spd.reference_pool_spds`, 0.75 x the bucket
  shares of the last 12 complete months (`conflictology_spds`) + 0.25 x
  `level_transition_spds`; the 12-month vector alone for a horizon with no
  transition vector. A mechanical ACLED backtest (8,371 country-forecasts,
  Mar 2021 - Dec 2025, production timing) gave Brier 0.390 for the 12-month
  shares, 0.384 for the pool, 0.470 for level_volatility. The backtest
  script was not supplied with the brief, so it is not committed.
* FL/PA, TC/PA: `_seasonal_pa` now returns `probs_by_month`, one vector per
  forecast calendar month from that month's event rate and PA records, with
  the pooled severity shares standing in below three records. The pooled
  return value is unchanged for existing callers.
* DR/PHASE3PLUS_IN_NEED: `SIBYL_DR_PERSISTENCE_WEIGHT` (0.5) x persistence
  of the last observed figure + the rest x the 36-month history vector, the
  same for all months. A starting value with no backtest behind it.

`score_baselines` scores `__ext_conflictology12` and `__ext_ref_pool` on
every ACE/FATALITIES question (both tracks), so the anchors can be compared.

**Decision: elicit two horizons with an explicit zero.** A trial states, for
month 1 and month 6, `p_zero` (for FL/TC: zero or no record) and the 0.05,
0.25, 0.5, 0.75, 0.95 quantiles given a positive value. Each month: mass
`p_zero` at zero, the rest on a monotone curve through the positive
quantiles in log space from half a unit to 5 x q0.95; bucket edges are
evaluated half a unit low so a quantile of exactly 100 cannot flip a bucket.
Trials are linearly pooled per month; months 2-5 are linear mixtures of 1
and 6. The belief is seeded from the reference vectors.

**Decision: publish a pool with the reference.** Per month, `SIBYL_REFERENCE_WEIGHT`
(0.5) x reference + the rest x the pooled trials, floored at 0.005. The
six month rows now differ. `sibyl_forecasts` gains `reference_json`,
`raw_by_month_json` (the pooled trials before the reference, vectors and
quantiles) and `final_by_month_json`; `bucket_probs_json` keeps the final
month-1 vector and `pooled_quantiles_json` the raw month-1 quantiles at the
old seven levels, and each trial keeps a legacy `quantiles` field. The
advice loop compares each resolved month with the RAW quantiles of that
month: it speaks to the agent about its own distribution. The identity
`calibrate` hook is no longer called (it took the old pooled CDF); it was a
pass-through and stays one.

**Prompt.** The outside-view block is replaced by the reference block (the
last 12 values with buckets, the ACLED incompleteness note, the month-1 and
month-6 vectors, and stay/rise/fall read off those vectors) and seven rules
on weighing evidence. No wording about Bayesian updating: tests on
forecasting prompts found it lowers accuracy.

## 2026-10-03 — Part 3: documents, ReliefWeb, a plan, and a submit gate

**What the runs showed.** 43 of the 57 pages Sibyl read in four production runs were Wikipedia. `fetch_url` kept 6,000 characters of stripped HTML and could not read a PDF, though the situation reports and appeals that carry the figures are PDFs. reliefweb.int answered all 45 page fetches in the September runs with HTTP 202. The prompt invited a submit "as soon as further research would not materially change" the belief, and most trials took that invitation early.

**What changed.**
- `sibyl/reader.py` reads a document properly. For HTML it keeps the main content, renders tables as `a | b` rows, and drops navigation and boilerplate. For a PDF it keeps the first two pages plus the pages that score highest on the country and the question's terms, up to 40 pages. Every document is capped at 80,000 characters.
- `sibyl/extract.py` handles long documents. Anything over 6,000 characters goes to role `sibyl_extraction` (Haiku 4.5) along with the agent's `extraction_request`. The model returns at most 500 words, with figures quoted word for word. If extraction fails, the agent sees the first 6,000 characters instead. Extraction is its own cost kind and is logged per call.
- `reliefweb_search` uses the ReliefWeb API, filtered on the question's primary country. A ReliefWeb report link is read through the API, together with its PDF attachment. The tool needs `RELIEFWEB_APPNAME`, which now goes to the Run Sibyl step. The secret already existed for the resolution machine. Dates are capped only in backtest.
- `brave_search` has two lanes. `news` covers the last 120 days. `reference` covers the last ten years, for base rates and past seasons. Both accept optional language and country hints.
- A step may carry up to three tool calls. A submit sent beside tool calls is dropped. `MAX_STEPS` moved from 10 to 12.
- The belief now carries a six-slot plan: resolver, nowcast, drivers, calendar, reversion, disconfirm.
  - A submit is refused until every slot is done or failed and three documents have been read.
  - The refusal names what is missing.
  - The resolver slot may be marked failed only after two steps that failed to find it.
  - The step limit still ends a trial.
- The evidence gate rose to three successful searches and two documents read.
- The early-submit sentence is gone.
- Each class has a resolver card in `sibyl/resolver_cards/`, shown under HOW THIS RESOLVES. Update the cards whenever resolution changes.

**Unverified.** The sandbox cannot reach api.reliefweb.int. The `url_alias` lookup for a report link that the search did not list has not been tested against the live API. If it fails, the agent is told so and can read the report through a search result instead.

**Tests.** `tests/test_sibyl_documents.py` (30 cases), with fixtures `tests/fixtures/sibyl/report.{pdf,html}` built by `build_fixtures.py`. The smoke test now runs a full three-step trial with a plan.

## 2026-10-03 — Part 4: keep the history

**What was wrong.** A step's prompt carried the belief state and the LAST tool result only. A figure read at step 2 survived to step 6 only if the model had copied it into `evidence_higher` or `evidence_lower`, which are free-text lists with no source, date or tier.

**What changed.**
- **Append-only transcript** (`sibyl/transcript.py`). Every earlier step stays in the prompt: the model's JSON as it returned it, the ledger ids the code assigned, and each tool result. An entry is rendered once and reused byte for byte. The prompt is built from five segments: static head, question block, trial perspective plus starting belief, transcript, and a short tail.
- **Caching.** Cache breakpoints sit on the question block, the trial segment and the end of the transcript, using the existing `cache_segments` argument. The legacy single-segment template is gone.
- **Evidence ledger** (`sibyl/ledger.py`). The model returns `ledger_add` each step. The code assigns ids and drops repeats, and the final ledger is stored with the trial.
- **Size guard.** Above `SIBYL_TRANSCRIPT_MAX_CHARS` (400,000 characters), the oldest tool results are replaced by a stub that keeps the URL.
- **Delta logging.** `llm_calls.prompt_text` holds the first step of a trial whole. Later steps are stored as a prefix hash and length plus the new tail.

The JSON action loop stays, so Part 7 can drive `call_openai` through the same code.

**Tests.** `tests/test_sibyl_history.py` (15 cases) covers:
- the prefix property;
- a step-2 figure still present at step 6;
- reconstruction of each logged prompt from the prefix and its tail;
- breakpoint placement;
- ledger cleaning, numbering and storage;
- the size guard, alone and inside a trial.

## 2026-10-03 — Part 5: lanes, parallel and extra trials, outlier guard, controls

Three trials with five loosely worded perspective seeds gave little diversity
where it matters, which is in what a trial reads first. Each trial now takes a
research lane:

- Lane A starts with the resolver and nowcast slots.
- Lane B starts with drivers and calendar.
- Lane C starts with reversion and disconfirm.
- Lane D starts with local-language sources.
- Lane E starts with reference-class material.

Every lane fills every slot. The lane text is still stored as `perspective`, so
the advice loop's per-seed bias table becomes a per-lane table.

How trials run:

- A question's trials run in threads. DuckDB is written only on the main
  thread: each trial buffers its `llm_calls` rows and the batch writes them in
  trial order.
- The budget is checked before each question and each batch, never between
  trials already running. The hard-cap test now expects a batch of three to
  finish.

Extra trials:

- After lanes A to C, lanes D and E run when either rule fires.
  - `disagreement`: the largest pairwise month-1 JSD is above 0.10.
  - `departure`: the pool's month-1 JSD from the reference is above 0.25.
- The rule and both measures are stored (`extra_trials_rule`,
  `trial_checks_json`).
- The outlier guard leaves out a trial whose month-1 median sits more than 1.5
  orders of magnitude from the others' median, while two trials remain. The
  dropped trial stays in `trials_json`.

Selection:

- 20 questions by floor-then-fill, with flood and cyclone held at 3.
- Plus 5 controls: ACE and DR questions with no RC flag, three and two, drawn
  by a hash of the run and question id.
- A control runs one trial on lane A and gets no extras. It is what Sibyl does
  where nothing is flagged as moving, which the volatility-ranked picks cannot
  show.
- Run order is floor, controls, fill.
- Replayed on the three fixture pools, the 20 come out ACE/DR/FL/TC 10/4/3/3,
  6/8/3/3 and 4/10/3/3.
- The fixture carries no RC level, so the control draw is tested on synthetic
  pools rather than replayed.

Tests: `tests/test_sibyl_lanes.py`.

## 2026-10-03 — Part 6: measuring what Sibyl read, what its research added, and what it got wrong

Evidence record:

- Every tool result a trial saw is a `sibyl_evidence` row: run, question,
  trial, step, call, tool, query or URL, search lane, retrieval time, HTTP
  status, ok, SHA-256, the text shown, and a document's pre-extraction text
  capped at 40,000 characters.
- Rows are built on the trial and written on the main thread, as the
  `llm_calls` rows are.
- It is third-party text, so `build_release_db` drops it from the public
  copy. A test holds that nothing in `pythia/api` or `web/src` reads it.
- Stage health reports its rows and characters for the run and in total.

Variant scores (`sibyl/score_variants.py`):

- `__ext_sibyl_raw` (the trials' pool, floored as the published vector is) and
  `__ext_sibyl_ref` (the reference) are scored by month into `scores`, through
  the `score_baselines` write path.
- The FL/TC two-part scores go to `sibyl_variant_scores`. "No record" is read
  as a month the resolver reached for another question of the class and not
  this one. That is a proxy, and it is labelled as one.
- The pool weight is chosen from {0.25, 0.5, 0.75} by log score once 20
  questions carry both scores. It is shrunk toward 0.5 by a 20-question prior,
  so at exactly 20 questions it cannot leave 0.5. `SIBYL_REFERENCE_WEIGHT_MODE`
  is `fitted` by default; backtest always uses the fixed weight.

Post-mortems (`sibyl/postmortem.py`):

- A note on each newly resolved question.
- Per class, from 8 notes, a lessons version. A lesson needs 3 cited cases and
  may name no country or year. Kept lessons are capped at 6,000 characters.
- $5 a month, medium effort.
- Lessons and up to 4 analogue notes are shown only in the track-record arm,
  never in backtest.

Advice:

- Findings report selected questions and controls apart.
- FL/TC get no zero-gap instruction; the two-part scores measure that instead.

Process measures on `sibyl_runs`: resolver slot done, documents per trial,
dated figures in the ledger, forecasts at the floor, and mean month-1 JSD of
the raw pool from the reference.

API and page:

- `sibyl_comparison.variants`: Sibyl against its reference, the 12-month
  conflictology and its raw pool, and raw against reference, each split by
  selection pass.
- The question view overlays the reference on each month's vector.
- The run panel shows the process measures.

Not done:

- No accuracy is measured in backtest.
- No threshold was lowered: on today's data no weight is fitted and no lesson
  can exist.

Tests: `tests/test_sibyl_measurement.py`, `tests/test_sibyl_postmortem.py`.

## 2026-10-03 — Part 7: the GPT Sol shadow arm

Question: does one trial from a second model family make the pool better?
Production stays all Claude; this part measures and switches nothing on.

How it runs (`sibyl/shadow.py`):

- After every question's production trials, each question that produced a
  forecast (controls excluded) gets one more lane C trial, same loop, same
  prompt, on `SIBYL_SHADOW_MODEL` (alias `gpt`, `openai:gpt-6-sol`) through
  `call_openai` at effort `high`, until `SIBYL_SHADOW_UNTIL` (2027-04).
- Running the whole phase after production, rather than per question, is
  what makes "production first" hold across the run: a shadow trial can
  never spend budget or time a later question needed.
- None starts within $2 of the hard cap or 20 minutes of the time cap.
- Every dollar a shadow trial spends is cost kind `shadow`. Its searches do
  not enter the run's tool counters (snapshotted before the phase).

The series: the evidence-valid production trials minus the Claude lane C
trial, plus the shadow trial, then the production steps (outlier guard,
linear pool by month, reference mix at the run's weight, floor). The
evidence gate applies to the shadow trial; an invalid one gives no series.
Where Claude's lane C trial had no evidence, nothing is removed and the
shadow trial is added: it still fills lane C.

Stored in `sibyl_forecasts.shadow_json`; never in `forecasts_raw` or
`forecasts_ensemble`, and the shadow trial never in `trials_json`.

Scores: `__ext_sibyl_shadow` in `scores`; the single shadow trial and the
single Claude lane C trial in `sibyl_variant_scores`. The comparison
(shadow minus `sibyl`, per score type, 90% interval resampling questions)
says "not yet" with no number below 20 questions. It sits in the pooled
advice row's findings, `/v1/sibyl/summary` and the Sibyl page.

Missing key: `run_sibyl.yml` passes the existing `OPENAI_API_KEY` secret to
the Run Sibyl step. Without it the run logs a warning, records
`shadow_status = no_key`, and production is untouched.

Not done:

- The shadow trial's evidence rows go to `sibyl_evidence` (role `shadow`),
  but process measures describe production only.
- Only OpenAI is wired as a shadow provider; another provider reports
  `unsupported_provider`.

Tests: `tests/test_sibyl_shadow.py`, a case in `tests/test_api_sibyl_routes.py`.

## 2026-10-09 — Review Part 1: a higher document gate, and research depth measured

From the October 2026 review of AI forecasting practice (FutureSearch, Preseen,
the Metaculus bot tournaments, the literature).

Evidence: before the rebuild a trial averaged three searches and most read no
page. FutureSearch's typical forecasting run makes 10 to 20 tool calls and
reads 5 to 20 pages. Sibyl's submit gate asked for three documents.

Changed:

- `SIBYL_SUBMIT_MIN_DOCS` 3 to 5. The step limit still ends a trial; one it
  ends with the gate unmet carries `submit_gate_unmet`, counted on
  `sibyl_runs.n_submit_gate_unmet`.
- A document counts once in a trial. The count was `+1` per successful
  `fetch_url`, so a second read of the same page counted again; now a repeat
  URL, or a text whose SHA-256 was already read under another URL, adds
  nothing. A failed fetch never counted and still does not.
  `sibyl_runs.n_docs_read` is now the trials' count, not the tool counter's.
- New process measures on `sibyl_runs`: `median_docs_per_trial`,
  `share_trials_under_doc_gate`, `steps_per_trial`, `tool_calls_per_trial`,
  `share_docs_wikipedia`. They show in `/v1/sibyl/summary` and on the "How
  this run researched" card. `scripts/ci/stage_health.py` warns, never
  fails, when the median is under 5 or the Wikipedia share over one half.

Not done: the evidence gate (`SIBYL_MIN_SEARCH_OK` 3, `SIBYL_MIN_DOCS_READ`
2) is unchanged. It decides whether a forecast is valid, and raising it would
fail more questions.

Cost: more reads a trial means more steps and more extraction calls. A step
of Opus 5.5 costs about $0.03-0.05 at October's rate and an extraction about
$0.005, so two more documents and one more step a trial is about $0.10 a
question, about $2.50 a run of 25 (more on questions that call for extra
trials).

Tests: `tests/test_sibyl_research_depth.py`; `tests/test_sibyl_documents.py`,
`tests/test_sibyl_smoke.py` and `tests/sibyl_test_utils.py` moved to five
documents.

## 2026-10-09 — Review Part 2: failure types in post-mortems

Evidence: FutureSearch's audit of its worst forecasts found a short list of
errors that repeat. Sibyl's notes were free text, so they could not be
counted.

Changed (`sibyl/postmortem.py`):

- `FAILURE_TYPES`: seventeen labels with one-line definitions, the single
  source for the prompt, validation, rates and the dashboard.
- The note prompt (`pm_v2`) now carries each trial's plan findings, its
  `baserate_reconciliation` and its ledger items (id, date, tier, kind,
  quote, direction), inside `SIBYL_POSTMORTEM_PROMPT_MAX_CHARS` (20,000;
  whole ledger items dropped and counted). For each resolved month it shows
  the reference, raw-pool and published vectors and the outcome bucket; the
  code, not the model, says whether the outcome fell inside the raw pool's
  0.05 to 0.95 range.
- The note returns up to three `failure_types` with `label_evidence`. The
  model is told a label must rest on what the trial recorded then, and that
  information that did not exist at forecast time is `unforeseeable`.
  Validation drops unknown labels into `failure_types_raw`; a note with no
  valid label is `unlabelled`.
- New columns on `sibyl_postmortem_notes`: `failure_types_json`,
  `prompt_version`. Notes from before (`prompt_version` NULL) are
  re-labelled, oldest first, inside `SIBYL_POSTMORTEM_CAP_USD`. A `pm_v2`
  note that came back unlabelled is not asked again.
- `failure_rates(con)`: per class and pooled, distinct questions per label
  over questions with a labelled note (newest note per question). Counts
  always; a share from `SIBYL_FAILURE_RATE_MIN_QUESTIONS` (10).
- Shown in the pooled `sibyl_calibration_advice` row's
  `findings.failure_types`, `/v1/sibyl/calibration` and a table on the
  Calibration tab. The lessons prompt sees each note's labels; the rates
  never reach a trial's prompt.

Not done: the labels are not used to choose analogues or to weight lessons.

Cost: the note prompt is longer (up to 20,000 characters, about 5,000
tokens), at medium effort; within the existing $5 monthly cap. Re-labelling
the existing notes is a one-off inside the same cap. No change to the cost
of a Sibyl run.

Tests: `tests/test_sibyl_failure_types.py`, a case in
`tests/test_api_sibyl_routes.py`.
