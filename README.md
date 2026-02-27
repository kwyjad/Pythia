# Pythia

Pythia is an end-to-end AI forecasting system for humanitarian questions. It scans countries for emerging hazards, turns triage into forecastable questions, runs an LLM ensemble to produce Subjective Probability Distributions (SPD), and stores forecasts, research, and diagnostics in a DuckDB-backed system of record.

## Table of contents
- [What is Pythia?](#what-is-pythia)
- [System at a glance](#system-at-a-glance)
- [Core components](#core-components)
- [Regime Change (RC): out-of-pattern detection](#regime-change-rc-out-of-pattern-detection)
- [Hazard Tail Packs (HTP): hazard-specific trigger evidence](#hazard-tail-packs-htp-hazard-specific-trigger-evidence)
- [Retriever web research (shared evidence packs)](#retriever-web-research-shared-evidence-packs)
- [Data model / DuckDB tables](#data-model--duckdb-tables)
- [Model management](#model-management)
- [Quickstart (local)](#quickstart-local)
- [Running in GitHub Actions](#running-in-github-actions)
- [Dashboard](#dashboard)
- [Downloads / exports](#downloads--exports)
- [Troubleshooting](#troubleshooting)
- [Cross-links](#cross-links)

## What is Pythia?
Pythia connects historical hazard data, web research, and LLM reasoning into a repeatable pipeline for early-warning forecasts. It produces triage signals, structured research briefs, and probabilistic forecasts across hazards, then serves those outputs through a DuckDB system of record, a FastAPI service, and a Next.js dashboard.

## System at a glance

```
Resolver facts/base rates → Horizon Scanner triage → Questions seeded
                           → Shared evidence packs → Research v2 briefs
                           → Forecaster SPD v2 ensemble + scenarios
                           → DuckDB → API → Dashboard + downloads
```

- **System of record**: DuckDB (`PYTHIA_DB_URL` / `app.db_url`) holds HS triage, questions, research, forecasts, and diagnostics.
- **End-to-end flow**: Resolver facts and base rates inform HS triage; HS produces triage + questions; retriever packs evidence; Research v2 drafts structured briefs; Forecaster SPD v2 writes ensembles/scenarios back to DuckDB.

## Core components
- **Resolver (facts + base rates)**: Resolver tables in DuckDB (`facts_resolved`, `facts_deltas`, `snapshots`) provide historical context. Schema is defined in [`pythia/db/schema.py`](pythia/db/schema.py).
- **Horizon Scanner (HS)**: `python -m horizon_scanner.horizon_scanner` triages countries and hazards, writes `hs_runs`/`hs_triage`, seeds `questions`/`question_research`, and generates country evidence packs.
- **Retriever web research (shared evidence packs)**: When enabled, the retriever builds evidence packs reused across HS, research, and SPD prompts. The shared retriever defaults to `gemini-2.5-flash-lite` when `PYTHIA_RETRIEVER_ENABLED=1` and `PYTHIA_RETRIEVER_MODEL_ID` is unset; see [`pythia/web_research/web_research.py`](pythia/web_research/web_research.py).
- **Research v2**: Prompts generate structured briefs that separate verified vs unverified sources, stored in `question_research`/`llm_calls`.
- **Forecaster SPD v2 ensemble**: `python -m forecaster.cli --mode pythia` runs SPD v2 prompts across the active ensemble and writes per-model outputs (`forecasts_raw`) and aggregated results (`forecasts_ensemble`).
- **Scenarios (priority-only)**: When an ensemble SPD is available, scenarios can be generated for priority questions and written back to DuckDB.
- **Dashboard + API**: FastAPI serves `/v1/*` endpoints for the dashboard, downloads, and diagnostics. The Next.js UI lives in `web/`.

## Regime Change (RC): out-of-pattern detection

**Definition:** Regime Change (RC) is the likelihood of a generating-process shift within the next 1–6 months that makes historical base rates less reliable. HS captures RC for every hazard and writes it to `hs_triage`.

**Stored fields in `hs_triage`:**
- `regime_change_likelihood` (probability)
- `regime_change_direction` (`up`, `down`, `mixed`, `unclear`)
- `regime_change_magnitude`
- `regime_change_score` (computed)
- `regime_change_level` (severity)
- `regime_change_window`
- `regime_change_json` (full normalized object)

**Scoring + levels (defaults; env-overridable):**
- `score = likelihood × magnitude`
- Level 0: `likelihood < 0.35` **or** `score < 0.20`
- Level 1: `likelihood ≥ 0.35` **and** `score ≥ 0.20`
- Level 2: `likelihood ≥ 0.60` **and** `magnitude ≥ 0.50` (score ≥ 0.30)
- Level 3: `likelihood ≥ 0.75` **and** `magnitude ≥ 0.60` (score ≥ 0.45)

**Behavioral implications:**
- **`need_full_spd` override**: HS forces `need_full_spd = TRUE` when RC is elevated (Level ≥2 or `score ≥ 0.30`) even if the triage tier is quiet. Thresholds are controlled by `PYTHIA_HS_RC_FORCE_LEVEL_MIN`/`PYTHIA_HS_RC_FORCE_SCORE_MIN`.
- **Research v2**: RC is surfaced in the research prompt, and RC-elevated hazards require at least one `regime_shift_signals` entry (or a rebuttal).
- **SPD v2**: RC guidance is embedded in the SPD prompt, and the model must include a sentence starting with `RC:` in `human_explanation` describing how RC affected the SPD.

See [docs/hs_regime_change.md](docs/hs_regime_change.md) for full details and env overrides.

## Hazard Tail Packs (HTP): hazard-specific trigger evidence

Hazard Tail Packs are hazard-scoped evidence bundles generated by HS when RC is elevated. They provide targeted trigger/counter-trigger evidence for downstream Research and SPD.

**Trigger rules:**
- Generated only for **RC Level ≥2** hazards.
- Limited to **max 2 hazards per country** per HS run (`PYTHIA_HS_HAZARD_TAIL_PACKS_MAX_PER_COUNTRY`, default `2`).
- Requires web research to be enabled (see below). Tail packs are **off by default** unless `PYTHIA_HS_HAZARD_TAIL_PACKS_ENABLED=1`.

**Storage table:** `hs_hazard_tail_packs` (one row per `hs_run_id × iso3 × hazard_code`), including:
- `rc_level`, `rc_score`, `rc_direction`, `rc_window`
- `query`, `report_markdown`, `sources_json`, `grounded`, `grounding_debug_json`
- `structural_context`, `recent_signals_json`, `created_at`

**Output format:**
- `recent_signals` bullets are emitted as `TRIGGER | <window> | <direction> | <signal>`, `DAMPENER | ...`, `BASELINE | ...`.

**Caching behavior:**
- Tail packs are reused per `(hs_run_id, iso3, hazard_code)`; HS skips regeneration if the row already exists.

**Phase A / Phase B behavior:**
- **Phase A (Research)**: When present, the tail pack is injected into Research v2 evidence (`hs_hazard_tail_pack`).
- **Phase B (SPD v2)**: For RC-elevated hazards, the tail pack is injected into the SPD prompt. Signals are capped at **12 bullets** (`PYTHIA_SPD_TAIL_PACKS_MAX_SIGNALS`, default `12`).

**Relevant env vars (defaults):**
- `PYTHIA_HS_HAZARD_TAIL_PACKS_ENABLED` (`0`)
- `PYTHIA_HS_HAZARD_TAIL_PACKS_MAX_PER_COUNTRY` (`2`)
- `PYTHIA_HS_HAZARD_TAIL_PACKS_MAX_SOURCES` (defaults to `PYTHIA_HS_EVIDENCE_MAX_SOURCES`)
- `PYTHIA_HS_HAZARD_TAIL_PACKS_MAX_SIGNALS` (defaults to `PYTHIA_HS_EVIDENCE_MAX_SIGNALS`)
- `PYTHIA_SPD_TAIL_PACKS_MAX_SIGNALS` (`12`)

See [docs/hs_hazard_tail_packs.md](docs/hs_hazard_tail_packs.md) for full details.

## Retriever web research (shared evidence packs)
- HS country evidence queries are tuned for **out-of-pattern/regime-change triggers** and **baseline continuation signals** (e.g., TAIL-UP/DOWN/BASELINE bullet formatting).
- Tail packs are a **second-stage, hazard-scoped follow-up**: HS first builds a country evidence pack, then requests a hazard-specific tail pack when RC is elevated.
- Web research is controlled by `PYTHIA_WEB_RESEARCH_ENABLED=1` plus either `PYTHIA_RETRIEVER_ENABLED=1` (shared retriever) or `PYTHIA_HS_RESEARCH_WEB_SEARCH_ENABLED=1` (HS/Research web search).

## Data model / DuckDB tables

Key tables (see [`pythia/db/schema.py`](pythia/db/schema.py) and [`SCHEMAS.md`](SCHEMAS.md) for canonical fields):

- **HS**: `hs_runs`, `hs_triage` (includes RC columns), `hs_country_reports`, `hs_hazard_tail_packs`
- **Questions + research**: `questions`, `question_research`
- **Forecasts**: `forecasts_raw`, `forecasts_ensemble`
- **Diagnostics**: `llm_calls`, `question_run_metrics`

## Model management

The forecast ensemble and purpose-specific models are configured in [`pythia/config.yaml`](pythia/config.yaml) under `llm.profiles`. Each profile contains a single `ensemble` list in `provider:model_id` format. To add, remove, or swap a model, edit this list.

```yaml
llm:
  profile: "prod"        # or "test"; override with PYTHIA_LLM_PROFILE

  profiles:
    test:
      ensemble:
        - google:gemini-2.5-flash-lite

    prod:
      ensemble:
        - openai:gpt-5.1
        - anthropic:claude-sonnet-4-6
        - google:gemini-3.1-pro-preview
        - google:gemini-3-flash-preview

      # Purpose-specific overrides (optional)
      hs_fallback: openai:gpt-5.1
      scenario_writer: google:gemini-3-flash-preview
```

### Changing models

- **Swap a model**: change the `provider:model_id` line in the ensemble list.
- **Add a model**: add a new line. Multiple models from the same provider are supported (e.g. two Google models).
- **Remove a model**: delete the line.
- **Add a new provider**: requires adding a `call_<provider>()` function and dispatch branch in `forecaster/providers.py`, plus an entry in `_PROVIDER_ENV_KEYS`.

### Model costs

Per-model cost rates are stored in [`pythia/model_costs.json`](pythia/model_costs.json) as `[input, output]` cost per 1,000 tokens in USD. When switching to a new model ID, add its cost entry to this file.

### Purpose-specific overrides

The `hs_fallback` and `scenario_writer` keys under a profile set the default model for HS triage fallback and scenario generation respectively. These can still be overridden at runtime via `PYTHIA_HS_FALLBACK_MODEL_SPECS` and `PYTHIA_SCENARIO_MODEL_ID` env vars.

### Env var overrides

- `PYTHIA_SPD_ENSEMBLE_SPECS`: overrides the entire SPD ensemble at runtime (comma-separated `provider:model_id` pairs).
- `PYTHIA_BLOCK_PROVIDERS`: comma-separated provider names to exclude (e.g. `xai`).
- `PYTHIA_SPD_GOOGLE_MODEL_ID`: overrides all Google model IDs in the SPD ensemble.

## Quickstart (local)

### 1) Clone + install
```bash
git clone <YOUR_REPO_URL>
cd Pythia
python -m venv .venv
source .venv/bin/activate
pip install -r python_library_requirements.txt
```

### 2) Configure the DB and profile
```bash
export PYTHIA_DB_URL="duckdb:///data/resolver.duckdb"
export PYTHIA_LLM_PROFILE="test"  # test or prod
```

### 3) Ensure the schema exists
```bash
python - <<'PY'
from pythia.db.schema import ensure_schema
ensure_schema()
print("schema ready")
PY
```

### 4) (Optional) Enable web research + tail packs
```bash
export PYTHIA_WEB_RESEARCH_ENABLED=1
export PYTHIA_RETRIEVER_ENABLED=1  # or PYTHIA_HS_RESEARCH_WEB_SEARCH_ENABLED=1
export PYTHIA_HS_HAZARD_TAIL_PACKS_ENABLED=1  # tail packs are off by default
```

### 5) Run Horizon Scanner
- Default list: [`horizon_scanner/hs_country_list.txt`](horizon_scanner/hs_country_list.txt)
```bash
PYTHIA_LLM_PROFILE=test python -m horizon_scanner.horizon_scanner
```

### 6) Run the forecaster (bounded)
```bash
PYTHIA_LLM_PROFILE=test python -m forecaster.cli --mode pythia --limit 20 --purpose local_smoke
```

### 7) Inspect outputs
- **DuckDB**: open the database at `PYTHIA_DB_URL` (`app.db_url` default is `duckdb:///data/resolver.duckdb`).
- **Debug bundle** (optional):
```bash
python -m scripts.dump_pythia_debug_bundle \
  --db "${PYTHIA_DB_URL}" \
  --hs-run-id "<HS_RUN_ID>" \
  --forecaster-run-id "<FORECASTER_RUN_ID>"
```
- Bundles default to `debug/pytia_debug_bundle__<run_id>.md` (note: filename uses `pytia_`).

## Running in GitHub Actions

### Workflows
- **Production pipeline**: [`run_horizon_scanner.yml`](.github/workflows/run_horizon_scanner.yml) runs HS + forecaster end-to-end.
- **Forecaster CI**: [`forecaster-ci.yml`](.github/workflows/forecaster-ci.yml) covers SPD unit tests and optional compare artifacts.

### Canonical DB artifacts + signature guardrails
- Workflow downloads the canonical DB artifact (`pythia-resolver-db` by default), appends new HS + forecaster rows, and uploads the updated DB as a new artifact.
- It validates a **signature** (`scripts/ci/db_signature.py`) and writes:
  - `diagnostics/db_signature_before.json`
  - `diagnostics/db_signature_after.json`
- If signature validation fails, the workflow rejects the candidate DB and tries the next run. A missing/failed signature is a hard-stop; confirm required tables exist or re-run with a reset DB artifact for bootstrap runs.

### Artifact outputs
- **Debug bundle**: `debug/pytia_debug_bundle__<run_id>.md` (artifact)
- **SPD compare JSON**: `debug/spd_compare_smoke` or `debug/spd_compare_tests`
- **Diagnostics**: `diagnostics/` (DB signatures, compare JSON, latency summaries)
- **HS triage coverage**: `diagnostics/hs_triage_coverage__<HS_RUN_ID>.csv` and `diagnostics/hs_triage_failures__<HS_RUN_ID>.json`

### DB migrations
```bash
python scripts/migrate_llm_calls_telemetry.py --db duckdb:///path/to/resolver.duckdb
```

### HS triage reruns
- After the HS run completes, stdout includes `HS_TRIAGE_RERUN_ISO3S=...`. To re-run just those countries:
```bash
PYTHIA_HS_ONLY_COUNTRIES="<ISO3S>" python -m horizon_scanner.horizon_scanner
```

## Dashboard

### Run the API locally
```bash
PYTHIA_DB_URL="duckdb:///data/resolver.duckdb" uvicorn pythia.api.app:app --reload --port 8000
```

### Run the web UI locally
```bash
cd web
npm install
NEXT_PUBLIC_PYTHIA_API_BASE=http://localhost:8000/v1 npm run dev
```

### Pages and RC visibility
- **Forecast Index (Overview)**: shows the Humanitarian Impact Forecast Index, with RC KPI counts and RC map markers (highest RC level per country).
- **Forecasts** (`/questions`): latest forecasts table includes RC score and triage fields when `latest_only=true`.
- **HS Triage** (`/hs-triage`): displays per-run triage with RC likelihood/direction/magnitude/score.
- **Question detail** (`/questions/[questionId]`): RC fields appear alongside research and SPD outputs.
- **Countries** (`/countries`): includes highest RC level/score per country from the latest HS run.
- **Downloads** (`/downloads`): links to forecast/triage exports with RC fields.

## Downloads / exports
- **Forecast SPD & EIV export**: `/v1/downloads/forecasts.csv` and `/v1/downloads/forecasts.xlsx` include RC probability/direction/magnitude/score columns per row.
- **Countries endpoint**: `/v1/countries` includes `highest_rc_level`/`highest_rc_score` (latest HS run).
- **Questions endpoint**: `/v1/questions?latest_only=true` includes RC fields (`regime_change_*`).

See [PUBLIC_APIS.md](PUBLIC_APIS.md) for canonical API contracts.

## Configuration

### Canonical config
- [`pythia/config.yaml`](pythia/config.yaml) is the authoritative configuration.
- There is no `pythia/config.py`; all runtime defaults live in the YAML + env vars.
- `app.db_url` defines DuckDB; `PYTHIA_DB_URL` overrides at runtime.
- `llm.profile` selects the default model bundle; override with `PYTHIA_LLM_PROFILE`.
- `llm.profiles.<name>.ensemble` defines the forecast ensemble; see [Model management](#model-management).
- [`pythia/model_costs.json`](pythia/model_costs.json) contains per-model cost rates.

### Key env vars
- **Provider keys**: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `EXA_API_KEY`, `PERPLEXITY_API_KEY`.
- **Concurrency**:
  - `PYTHIA_LLM_CONCURRENCY` (global LLM call cap)
  - `HS_MAX_WORKERS`
  - `FORECASTER_RESEARCH_MAX_WORKERS`
  - `FORECASTER_SPD_MAX_WORKERS`
- **Retriever / web research**:
  - `PYTHIA_WEB_RESEARCH_ENABLED=1`
  - `PYTHIA_WEB_RESEARCH_BACKEND=gemini|openai|claude|auto`
  - `PYTHIA_WEB_RESEARCH_RECENCY_DAYS`, `PYTHIA_WEB_RESEARCH_TIMEOUT_SEC`, `PYTHIA_WEB_RESEARCH_MAX_RESULTS`
  - **Shared retriever**: `PYTHIA_RETRIEVER_ENABLED=1` and optional `PYTHIA_RETRIEVER_MODEL_ID` (default `gemini-2.5-flash-lite`).
  - **HS/Research web search**: `PYTHIA_HS_RESEARCH_WEB_SEARCH_ENABLED=1` (needed if retriever is off).
- **SPD tuning / timeouts (Gemini 3)**:
  - `PYTHIA_GOOGLE_SPD_THINKING_LEVEL_FLASH`, `PYTHIA_GOOGLE_SPD_THINKING_LEVEL_PRO`
  - `PYTHIA_GOOGLE_SPD_TIMEOUT_FLASH_SEC`, `PYTHIA_GOOGLE_SPD_TIMEOUT_PRO_SEC`
  - `PYTHIA_GOOGLE_SPD_RETRIES`
- **SPD ensemble override**: `PYTHIA_SPD_ENSEMBLE_SPECS` (e.g. `openai:gpt-5.1,google:gemini-3-flash-preview`).
- **HS triage resilience**:
  - `PYTHIA_HS_FALLBACK_MODEL_SPECS` (defaults to `hs_fallback` from the active profile, then `openai:gpt-5.1`; keeps HS triage running when Gemini fails).
  - `PYTHIA_HS_ONLY_COUNTRIES` (comma-separated ISO3s/names to rerun HS triage for a subset).
  - `PYTHIA_PROVIDER_FAILURE_THRESHOLD`, `PYTHIA_PROVIDER_COOLDOWN_SECONDS`, `PYTHIA_PROVIDER_RESET_ON_SUCCESS`
  - `PYTHIA_LLM_RETRY_TIMEOUTS` (set `0` to opt out of timeout retries outside HS triage)
  - `PYTHIA_HS_LLM_MAX_ATTEMPTS` (defaults to 3 for HS triage retries)
  - `PYTHIA_HS_GEMINI_TIMEOUT_SEC` (defaults to 120s for HS triage)

## GitHub Secrets setup

| Secret | Used by | Required for |
| --- | --- | --- |
| `OPENAI_API_KEY` | Forecaster SPD ensemble | OpenAI models in ensemble |
| `GEMINI_API_KEY` | HS, retriever, forecaster | Gemini models (HS + SPD + retriever) |
| `ANTHROPIC_API_KEY` | Forecaster SPD ensemble | Anthropic models |
| `XAI_API_KEY` | Forecaster SPD ensemble | XAI models |
| `EXA_API_KEY` | Web research | Exa backend (optional) |
| `PERPLEXITY_API_KEY` | Web research | Perplexity backend (optional) |
| `GITHUB_TOKEN` | Actions | Artifact download + summary updates |

**Minimum set for full pipeline**: `GEMINI_API_KEY` + at least one of (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`). Missing keys disable their providers, and the run proceeds with a partial ensemble.

## Operational notes
- **Partial ensembles are expected**: providers can timeout; the ensemble uses available members and records `ensemble_meta`.
- **Timeouts cap tail risk**: per-provider timeouts are enforced; increase them only if needed.
- **Shared retriever**: evidence packs are cached and reused to stabilize sources and reduce cost.
- **HS triage always writes**: HS stores triage outputs even if no questions are produced.

## Known limitations and challenges
- **Gemini 3 grounding**: grounding support for Gemini 3 preview is subject to provider support and may lag until early 2026.
- **Retriever fallback**: if retriever calls fail, research can fall back to unverified sources.
- **IFRC Montandon sparsity**: natural hazard PA data may be sparse for some hazards/countries, reducing base-rate strength.
- **IDMC month-key issue**: occasionally inconsistent month keys are tolerated and cleaned downstream.
- **Occasional “200 but ungrounded”**: some web search calls can return success without verified sources.

## Troubleshooting
- **No questions generated**: HS may mark all tiers quiet. Check `hs_triage` in DuckDB and confirm your country list (`horizon_scanner/hs_country_list.txt`) and `hazards_allowed` in config.
- **No hazard tail packs**: confirm `PYTHIA_HS_HAZARD_TAIL_PACKS_ENABLED=1`, web research enabled (`PYTHIA_WEB_RESEARCH_ENABLED=1` plus retriever or HS web search), and RC Level ≥2 for the hazard. Tail packs are limited to 2 hazards per country.
- **RC fields missing in API**: ensure `hs_triage` has `regime_change_*` columns (`pythia/db/schema.py`) and that `latest_only=true` is set on `/v1/questions`.
- **No active models**: verify `PYTHIA_LLM_PROFILE`, `llm.profiles` ensemble config in `pythia/config.yaml`, and provider API keys.
- **Debug bundle too large for step summary**: artifacts still exist under `debug/` even if GitHub Step Summary truncates.
- **Slow runs**: Gemini tails can dominate latency. Tune `PYTHIA_LLM_CONCURRENCY`, `FORECASTER_*_MAX_WORKERS`, and SPD timeouts.
- **Interpreting question_run_metrics**: `question_run_metrics` (if present) records wall-clock vs compute vs queue time per question; see `scripts/dump_pythia_debug_bundle.py`.

## Cross-links
- Non-technical system overview: [`docs/fred_overview.md`](docs/fred_overview.md)
- Config: [`pythia/config.yaml`](pythia/config.yaml)
- Model costs: [`pythia/model_costs.json`](pythia/model_costs.json)
- Horizon Scanner: [`horizon_scanner/horizon_scanner.py`](horizon_scanner/horizon_scanner.py)
- Regime Change docs: [`docs/hs_regime_change.md`](docs/hs_regime_change.md)
- Hazard Tail Packs docs: [`docs/hs_hazard_tail_packs.md`](docs/hs_hazard_tail_packs.md)
- HS country list: [`horizon_scanner/hs_country_list.txt`](horizon_scanner/hs_country_list.txt)
- Forecaster CLI: [`forecaster/cli.py`](forecaster/cli.py)
- Schema: [`pythia/db/schema.py`](pythia/db/schema.py)
- Debug bundle script: [`scripts/dump_pythia_debug_bundle.py`](scripts/dump_pythia_debug_bundle.py)
- Workflows: [`run_horizon_scanner.yml`](.github/workflows/run_horizon_scanner.yml), [`forecaster-ci.yml`](.github/workflows/forecaster-ci.yml)
- Public API contracts: [`PUBLIC_APIS.md`](PUBLIC_APIS.md)

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for coding standards and workflow notes. Archived READMEs live in [docs/archive/README_INDEX.md](docs/archive/README_INDEX.md).

## License
This repository follows the licensing terms bundled with the codebase (see `LICENSE` if present or repository metadata).
