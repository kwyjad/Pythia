# CLAUDE.md

This file provides guidance for Claude Code when working with the Pythia codebase.

## Project overview

Pythia is an end-to-end AI forecasting system for humanitarian crises. It scans countries for emerging hazards, produces triage signals, generates research briefs, runs LLM ensembles for probabilistic forecasts (SPDs), scores and calibrates them, and serves outputs via a FastAPI API and Next.js dashboard.

**Core pipeline:**
```
Resolver facts/base rates -> Horizon Scanner triage -> Questions seeded
  -> Shared evidence packs -> Research v2 briefs -> Forecaster SPD ensemble
  -> DuckDB -> FastAPI API -> Next.js Dashboard + CSV/Excel exports
```

**Post-forecast pipeline:**
```
compute_resolutions (ground truth from Resolver) -> resolutions table
  -> compute_scores (Brier, log loss, CRPS per horizon) -> scores table
  -> compute_calibration (weights + advice per hazard/metric) -> calibration_weights + calibration_advice
```

## Repository layout

```
Pythia/
  horizon_scanner/     # Country/hazard triage via LLM (regime change, tail packs)
  forecaster/          # LLM ensemble for SPD forecasts (research + SPD phases)
  resolver/            # Fact ingestion, resolution, and ground truth DB
    connectors/        #   Connector protocol + registry (ACLED, IDMC)
    ingestion/         #   Source-specific fetch/normalise clients (acled_client, idmc/)
    transform/         #   Adapters, normalisation, source resolution
    tools/             #   Pipeline orchestrator (run_pipeline.py), precedence, deltas, enrichment
    db/                #   DuckDB schema + helpers
  pythia/
    api/               # FastAPI service (50+ endpoints)
    db/                # DuckDB schema definitions + migrations (schema.py is authoritative)
    tools/             # Post-forecast compute scripts (resolutions, scores, calibration)
    web_research/      # Shared retriever backends (Gemini, OpenAI, Claude, Exa, Perplexity)
    tests/             # Pythia-specific tests
  web/                 # Next.js 14 dashboard (TypeScript, Tailwind)
  tests/               # Cross-module integration tests
  docs/                # Component documentation
  scripts/             # Operational scripts
  .github/workflows/   # CI + scheduled pipelines
```

## Key files

- `pythia/db/schema.py` — Authoritative Pythia DuckDB schema (tables, migrations, bucket defs)
- `resolver/db/schema.sql` — Resolver DuckDB schema (facts_resolved, facts_deltas, snapshots)
- `pythia/config.yaml` — Central configuration (LLM profiles, hazards, bucket specs)
- `horizon_scanner/horizon_scanner.py` — HS main entrypoint (~1700 LOC)
- `horizon_scanner/regime_change.py` — RC scoring (score = likelihood x magnitude, 4 levels)
- `horizon_scanner/prompts.py` — HS triage prompt builder (includes RC calibration anchors)
- `forecaster/cli.py` — Forecaster main runner (~7100 LOC)
- `pythia/api/app.py` — FastAPI application (~3200 LOC)
- `pythia/tools/compute_resolutions.py` — Resolves forecasts against Resolver ground truth
- `pythia/tools/compute_scores.py` — Brier/log/CRPS scoring per horizon
- `pythia/tools/compute_calibration_pythia.py` — Calibration weights + LLM advice
- `resolver/connectors/protocol.py` — Connector protocol (21-column canonical schema contract)
- `resolver/connectors/__init__.py` — Connector registry (discover_connectors)
- `resolver/tools/run_pipeline.py` — Pipeline orchestrator (fetch -> validate -> enrich -> precedence -> deltas -> DuckDB)
- `resolver/tools/enrich.py` — Enrichment (registry lookups, ym derivation, defaults)
- `resolver/tools/precedence_config.yml` — Precedence tier policy

## Databases

Two DuckDB databases:

**Pythia DB** (`PYTHIA_DB_URL`): system of record
- `hs_runs`, `hs_triage` — Horizon Scanner outputs (triage scores, RC fields)
- `hs_hazard_tail_packs` — RC-triggered hazard evidence packs
- `questions`, `question_research` — Seeded questions + research briefs
- `forecasts_raw`, `forecasts_ensemble` — Per-model + aggregated SPDs
- `resolutions` — Ground truth values per (question_id, horizon_m)
- `scores` — Brier/log/CRPS per (question, horizon, model)
- `calibration_weights`, `calibration_advice` — Per hazard/metric weights
- `llm_calls` — Full telemetry (cost, tokens, latency, errors)

**Resolver DB** (`resolver/db/schema.sql`): fact ingestion
- `facts_resolved` — Precedence-resolved facts (unique on ym, iso3, hazard_code, metric, series_semantics)
- `facts_deltas` — Monthly flow changes
- `snapshots`, `manifests`, `meta_runs` — Pipeline metadata

## Resolver architecture

The Resolver was refactored in PR #610 to a connector-based architecture. Defunct legacy connectors (DTM, EM-DAT ingestion, GDACS, HDX, IPC, ODP, ReliefWeb, UNHCR, WFP, WHO, WorldPop) were removed.

**Connector protocol** (`resolver/connectors/protocol.py`): Every data source implements a `Connector` protocol with a `name` attribute and a `fetch_and_normalize()` method that returns a DataFrame with exactly 21 canonical columns (event_id, iso3, hazard_code, metric, value, as_of_date, publisher, etc.).

**Active connectors** (`resolver/connectors/__init__.py` REGISTRY):
- `acled` — ACLED conflict/fatalities data (wraps `resolver/ingestion/acled_client`)
- `idmc` — IDMC internal displacement data (wraps `resolver/ingestion/idmc/`)
- `ifrc_montandon` — IFRC Go connector (stubbed, not yet active)

**Pipeline orchestrator** (`resolver/tools/run_pipeline.py`):
```
discover_connectors() -> fetch_and_normalize() per connector
  -> validate_canonical() -> enrich() + derive_ym()
  -> precedence_engine (tiered source resolution) -> make_deltas()
  -> write to DuckDB (facts_resolved + facts_deltas)
```

**Precedence policy** (`resolver/tools/precedence_config.yml`):
- Tier 0: IFRC Montandon/ACLED (highest priority; IFRC Montandon is the active source for natural hazard PA: FL, DR, TC, HW)
- Tier 1: IDMC
- Tier 2: EM-DAT (historical read-only, no active connector; replaced by IFRC Montandon)

**Transform adapters** (`resolver/transform/adapters/`): ACLED and IDMC adapters normalize source-specific schemas to the common format used by the precedence engine.

Run the pipeline: `python -m resolver.tools.run_pipeline [--connectors acled idmc] [--db path/to/resolver.duckdb]`

## Per-horizon architecture

Forecasts cover a 6-month window. Each question has `window_start_date` and `target_month` (= month 6).

- **Resolutions**: `compute_resolutions` resolves each horizon independently against Resolver's `facts_resolved.created_at` for ordering.
- **Scoring**: `compute_scores` scores each (question, horizon_m) pair separately using Brier, log loss, and CRPS.
- **Calibration**: `compute_calibration_pythia` aggregates scores across horizons to produce per-model weights.

## Regime Change (RC) scoring

RC detects departures from historical base rates (distinct from triage_score which measures overall risk).

- `score = likelihood x magnitude`, clamped [0, 1]
- **Levels** (env-overridable): L0 (likelihood < 0.45 or score < 0.25), L1, L2 (likelihood >= 0.60, magnitude >= 0.50), L3 (likelihood >= 0.75, magnitude >= 0.60)
- L2+ forces `need_full_spd = TRUE` and triggers hazard tail pack generation
- Distribution check warns when too many assessments exceed expected proportions
- See `docs/hs_regime_change.md` for full details

## Testing

```bash
# Run all pythia tests (skip fastapi-dependent tests if not installed)
python3 -m pytest pythia/tests/ --ignore=pythia/tests/test_api_question_bundle.py -v

# Run cross-module tests (skip fastapi/openai-dependent tests)
python3 -m pytest tests/ --ignore-glob='tests/test_api_*' --ignore=tests/test_web_research.py -v

# Run RC calibration tests
python3 -m pytest resolver/tests/test_rc_calibration.py -v

# Run resolver tests
python3 -m pytest resolver/tests/ -v
```

**pytest config** (`pytest.ini`): 120s timeout, thread-based, log_cli enabled.
**Markers**: `@pytest.mark.db`, `@pytest.mark.allow_network`, `@pytest.mark.slow`, `@pytest.mark.nightly`.

Some test files require `fastapi` or `openai` which may not be installed locally — exclude them with `--ignore` as shown above.

## Common environment variables

| Variable | Purpose |
|----------|---------|
| `PYTHIA_DB_URL` | DuckDB connection URL |
| `PYTHIA_LLM_PROFILE` | LLM profile: `test` or `prod` |
| `PYTHIA_LLM_CONCURRENCY` | Max concurrent LLM calls |
| `PYTHIA_API_TOKEN` | Admin API auth token |
| `PYTHIA_WEB_RESEARCH_ENABLED` | Enable shared retriever (0/1) |
| `HS_MAX_WORKERS` | HS concurrent country workers |
| `FORECASTER_RESEARCH_MAX_WORKERS` | Research phase concurrency |
| `FORECASTER_SPD_MAX_WORKERS` | SPD phase concurrency |
| `PYTHIA_HS_RC_LEVEL*_*` | RC threshold overrides |
| `PYTHIA_HS_RC_DIST_WARN_*` | RC distribution warning thresholds |

Provider API keys: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`.

## Build and run

```bash
# Install dependencies
pip install -r python_library_requirements.txt

# Initialize DB schema
python3 -c "from pythia.db.schema import ensure_schema; ensure_schema()"

# Run Resolver pipeline (fetch facts from ACLED + IDMC -> DuckDB)
python3 -m resolver.tools.run_pipeline
# Or specific connectors: python3 -m resolver.tools.run_pipeline --connectors acled idmc

# Run Horizon Scanner
python3 -m horizon_scanner.horizon_scanner

# Run Forecaster
python3 -m forecaster.cli --mode pythia

# Post-forecast pipeline
python3 -m pythia.tools.compute_resolutions
python3 -m pythia.tools.compute_scores
python3 -m pythia.tools.compute_calibration_pythia

# API server
uvicorn pythia.api.app:app --reload --port 8000

# Dashboard
cd web && npm install && npm run dev
```

## Prompt editing

Before editing any of the 3 prompt source files (`forecaster/prompts.py`, `horizon_scanner/prompts.py`, `pythia/web_research/backends/gemini_grounding.py`), always run `bash scripts/snapshot_prompts.sh` first. This archives the current prompts before changes so the About page can show historical versions. Commit the snapshot alongside the prompt edits.

## Code conventions

- Copyright header on every file: `# Pythia / Copyright (c) 2025 Kevin Wyjad`
- License: Pythia Non-Commercial Public License v1.0
- DuckDB is the only database backend (no Postgres/SQLite)
- `pythia/db/schema.py` is authoritative for Pythia tables; `resolver/db/schema.sql` for Resolver tables
- Config loaded via `pythia.config.load()` which reads `pythia/config.yaml`
- LLM providers abstracted through `forecaster/providers.py` (OpenAI, Google, Anthropic, XAI)
- All LLM calls logged to `llm_calls` table with cost, tokens, latency, error tracking
- Env vars override config defaults; threshold env vars use `_env_float()` pattern
