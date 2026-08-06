# Interpreter module — Phase 0 discovery report

Answers to the five discovery questions in the interpreter implementation plan
(2026-08-06). Each answer names the files it rests on so later phases can
re-verify without re-deriving.

## 1. Base-rate coverage at prompt time

Three independent base-rate injectors exist, and they do not overlap. Routing
is `_build_history_summary` (`forecaster/cli.py:984`), rendered by
`_format_base_rate_for_prompt` (`forecaster/history_loaders.py:344`).

| Pair | Loader | Data source | Quantitative anchor feasible? |
|---|---|---|---|
| ACE/FATALITIES | `_build_conflict_base_rate` (`cli.py:688`) | `acled_monthly_fatalities`, 6 most recent complete months | Yes — monthly fatalities series |
| ACE/PA | same call, displacement side | IDMC flow rows in `facts_deltas` (`series_semantics='new'`, idmc source/metric filters) | Yes — monthly displacement series |
| DR/PHASE3PLUS_IN_NEED | `_load_fewsnet_phase3_history` (`cli.py:852`) | `facts_resolved` `phase3plus_in_need`, 36 months | Yes — richest, a monthly stock series |
| FL/PA, TC/PA | `_build_natural_hazard_seasonal_profile` (`cli.py:558`) + GDACS occurrence block (`history_loaders.py:228`) | `facts_resolved` PA metrics via `_pa_metric_in_clause()` + `event_occurrence` rows | Yes — best-shaped: occurrence rate × conditional magnitude |
| FL/DR/TC EVENT_OCCURRENCE | `build_binary_base_rate` (`forecaster/binary_prompts.py:476`) | `facts_resolved` `event_occurrence`, seasonal % table | Yes — it IS a probability per month (not a bucket SPD) |
| CU/PA, DI/PA | none — terminal `no_base_rate` | — | **No. Question A1 cannot be answered for these; the report must say so.** |

Notes that shaped Phase 1:

- `_build_conflict_base_rate` returns one `conflict_trajectory` dict for both
  ACE metrics; the renderer anchors PA on the IDMC displacement series and
  FATALITIES on ACLED, and excludes the current partial calendar month.
- The PA resolution machine's block (`resolver/hazard_resolution/prompt_block.py`)
  is a fourth injector, eligible for exactly FL/PA and TC/PA (DR/PA is
  remapped to PHASE3PLUS_IN_NEED at question generation). Its
  `haz_base_rates_occurrence`/`haz_base_rates_severity` tables are the most
  directly SPD-convertible material in the repo, but the machine is in shadow
  mode and its coverage is still building — Phase 1's `base_rate_spd`
  therefore anchors on the GDACS occurrence rates + reported-PA magnitudes
  the prompt also shows, and switching the FL/TC PA anchor to the machine's
  base rates once they are established is a deliberate follow-up.
- `sibyl/base_rates.py:141` has a bug NOT to copy: its `conflict_trajectory`
  branch reads the fatalities series unconditionally, so ACE/PA is anchored
  on the wrong series there.
- EVENT_OCCURRENCE never reaches `_build_history_summary`; it is routed to
  the binary path before any SPD prompt is built.

## 2. Anthropic thinking control

Confirmed: **no Anthropic thinking knob is wired.** `ModelSpec.thinking`
reaches OpenAI (`reasoning_effort`) and Google (`thinkingConfig`) but the
Anthropic branches of `build_body_for_spec` (`forecaster/providers.py:1528`)
and `_call_provider_sync` (`:1568`) drop it. `build_anthropic_body`
(`:1134-1213`) emits exactly `{model, max_tokens, messages}` (+ optional
`cache_control`) for claude-opus-5 — temperature is stripped by the
`_ANTHROPIC_NO_TEMPERATURE_PREFIXES` guard. Sibyl calls `call_anthropic`
directly with no thinking either.

Change required (Phase 3, when the interpreter role is wired):

1. Emit inside `build_anthropic_body` (so batch/sync parity and the existing
   body-equality tests hold for free):
   `body["output_config"] = {"effort": <level>}` — the effort knob is nested
   in `output_config`, GA, no beta header, and `budget_tokens` is REMOVED on
   Opus 5 (400 if sent). Valid levels: low/medium/high/xhigh/max; `high` is
   already the default, so "explicit high" is a no-op and the real request is
   `xhigh`/`max`.
2. Thread `thinking_level` through the two Anthropic dispatch sites above.
3. Gate on a literal-prefix tuple in the `_ANTHROPIC_NO_TEMPERATURE_PREFIXES`
   style — `grounding_claude` is pinned to claude-haiku-4-5, which rejects
   `effort`; an ungated emit 400s that role on its first call.
4. Thinking tokens share `max_tokens`: raising effort raises truncation risk,
   so the interpreter call site should pass a generous `max_tokens_override`
   (the `stop_reason: max_tokens` detector at `providers.py:1059` already
   reports this failure mode correctly, and usage accounting already folds
   thinking into `output_tokens`).

## 3. The removed map

**The premise does not hold — no run-results map component was ever deleted.**
`RiskIndexMap.tsx` has existed continuously since 2025-12-30. What happened
instead:

- PR #555/#556 (2025-12-30) rewrote the map in place: the original d3-geo +
  GeoJSON implementation was replaced by a pre-projected static
  `/maps/world.svg` (`data-iso3` attributes, imperative DOM mutation), after
  a nine-PR firefight over projection/rendering bugs. Reviving the d3 version
  would re-inherit exactly those bugs and re-add a dependency.
- PR #744 (2026-04-08) made `ALL_METRICS_SUMMARY` the default Run Results
  view, which renders `RunSummaryView` instead of the map — so the page now
  *lands* with no map visible. That is the "removed map".

**Recommendation: extend `RiskIndexMap`, do not revive.** Cheapest path is
widening its props (all optional, existing call sites untouched):
`valuesByIso3`, `colorFor`, `legendItems`, `onCountryClick`, `showRcOverlay`.
Click handling is ~15 lines beside the existing hover listeners. The
choropleth machinery, tooltip, SVG asset and visibility normalisation are
already debugged against this exact asset. (If a third map ever appears,
extract a `useWorldSvg()` hook; not before.)

## 4. Publish and release assembly

- Release identity: tag `pythia-data-latest`, exactly two assets today
  (`resolver.duckdb`, `manifest.json`), uploaded by the single
  `gh release upload ... --clobber` step in `publish_latest_data.yml` (~:563).
- The API (`pythia/api/db_sync.py`) fetches two exact asset URLs
  (`https://github.com/{repo}/releases/download/pythia-data-latest/...`) and
  never enumerates assets — **extra assets on the release cannot break sync**.
  The regression guard downloads with `--pattern "resolver.duckdb"`, also
  indifferent. Manifest keys pass through `fetch_manifest` untouched and
  `/v1/version` returns the manifest verbatim plus extras, so a new manifest
  key reaches the browser with zero API changes.
- PDF mechanism (Phase 6): the producing workflow uploads a
  `pythia-interpreter-report` artifact (continue-on-error, same run as the
  canonical DB); `publish_latest_data.yml` gains one best-effort step that
  `gh run download`s that artifact from the already-resolved `run_id`, copies
  the PDF into `data_out/`, extends the existing upload command, and records
  the asset name in the manifest (e.g. `interpreter_report_asset`). Release
  assets are a flat namespace (directory prefix dropped) and versioned names
  accumulate — also upload a constant-name copy
  (`interpreter_report_latest.pdf`) so the dashboard button has a stable URL,
  with the manifest key as the primary discovery channel.

## 5. Existing overlap

- `generate_calibration_advice.py` is **fully deterministic — no LLM call**
  ("prompt-ready" means the output text is injected into forecaster prompts).
  It writes `calibration_advice` rows (`model_name='__shared__'`, per-model,
  and global `*/*` rows; `findings_json` carries the raw computed findings —
  tail coverage, bucket calibration, per-model Brier, horizon bias,
  RC-conditional, ViEWS benchmark, EIV accuracy, centroid drift, prior
  anchoring). The interpreter consumes `advice` + `findings_json` directly.
  Its `_js_divergence` (`:427`, natural log, range [0, ln 2]) is the repo's
  one JSD implementation; `sibyl/spd.py:91` re-imports it lazily and
  `compute_deviation` now does the same. (`scripts/compare_prompt_order_runs.py`
  has an incompatible base-2 variant — do not confuse them.)
- `AboutScores.tsx` copy is **inline JSX, not exported data** — the four
  ScoreCards (`title/family/range/technical/plain/interpret`) cannot be
  imported as a glossary source today. Phase 5 must first refactor the copy
  into an exported array (the `ScoreCardProps` shape is already exactly a
  glossary entry) and re-render both `AboutScores` and the interpreter
  glossary from it, so the two never drift. `PerformancePanel.tsx:90-122`
  holds tooltip strings; `model_names.ts` prettifies model ids but does NOT
  handle `__ext_*` (that lives in `PerformancePanel.displayModelName`, which
  has a generic `__ext_` branch — `__ext_climatology` renders as
  "climatology (external)" with no frontend change).
- `score_views.py` structure (the template `score_baselines.py` now follows):
  `__ext_views` sentinel, delete-then-insert keyed `run_id IS NULL`, scoring
  via `_brier/_log_score/_crps_like/_bucket_index` imported from
  `compute_scores`, an audit-trail table, and downstream consumption in
  `generate_calibration_advice._compute_views_benchmark`. Calibration
  excludes externals via `model_name NOT LIKE '__ext_%'`
  (`compute_calibration_pythia.py:116`), not via `AGGREGATE_MODEL_NAMES`.

## Phase 1 decisions taken on the back of this report

- `base_rate_spd` builds **empirical bucket distributions from the same
  tables/filters/exclusions as the prompt-time loaders**, using only history
  strictly before the question's window_start month (no outcome leakage), with
  a Jeffreys 0.5 pseudo-count so small samples never assign an exact zero.
  FL/TC PA uses the occurrence × conditional-severity mixture (GDACS seasonal
  rate × reported-PA magnitudes); binary pairs return `[p, 1-p]` with
  per-month rates in `detail.probs_by_month`.
- Pairs with no prompt-time base rate (CU/PA, DI/PA, non-GDACS
  EVENT_OCCURRENCE) return `([], "NONE", reason)` and produce **no**
  deviation row and **no** climatology score — an anchor is never invented.
- `eiv_nominal` uses the risk index's surge blend (`max + 0.1*(sum − max)`)
  with `centroids_for` seeds, so the interpreter's impact numbers agree with
  the dashboard's.
- Persistence is not implemented (infinite log loss when wrong; smoothing it
  is an argument generator). Deferred, per the plan.
