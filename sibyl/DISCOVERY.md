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
with `triage_score` as tiebreak, then `question_id` for determinism. This is
documented in `sibyl/select_questions.py`.

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
`forecaster/providers.py::resolve_price_per_1k`/`estimate_cost_usd`, records
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
`[0.005, 0.025]` per 1K tokens ($5/$25 per MTok).

**Provider gap fixed:** `providers.py::call_anthropic` always sends
`temperature`, but `claude-opus-4-8` (like Opus 4.7) rejects sampling
params with HTTP 400. Added a no-temperature model guard (mirrors the
existing `_KIMI_FIXED_TEMPERATURE_MODELS` pattern). Trial diversity comes
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
