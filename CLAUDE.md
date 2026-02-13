# CLAUDE.md — Pythia Developer Guide

## What is Pythia?

Pythia is a humanitarian early-warning and forecasting system. It ingests ground-truth data from humanitarian sources, triages emerging hazards by country, runs an LLM ensemble to produce probabilistic forecasts, and serves results through a FastAPI backend and Next.js dashboard.

## High-Level Data Flow

```
Resolver (data ingestion) → base rates & facts
        ↓
Horizon Scanner (triage) → country × hazard scores, questions
        ↓
Web Research (evidence packs) → grounded sources per question
        ↓
Forecaster (LLM ensemble) → SPD forecasts + scenarios
        ↓
DuckDB (system of record)
        ↓
FastAPI + Next.js Dashboard
```

## Repository Layout

```
Pythia/
├── pythia/                  # Core package (API, DB, config, tools)
│   ├── api/                 # FastAPI application
│   ├── db/                  # DuckDB schema and helpers
│   ├── config/              # Config loading
│   ├── pipeline/            # Pipeline orchestration (run.py)
│   ├── prompts/             # Prompt registry
│   ├── tools/               # Calibration & scoring scripts
│   ├── utils/               # ID generation, misc
│   ├── web_research/        # Evidence retrieval (multi-backend)
│   │   └── backends/        # gemini, openai, claude
│   ├── tests/               # Unit tests for pythia package
│   ├── config.yaml          # Central configuration
│   ├── buckets.py           # Forecast bucket specs
│   └── llm_profiles.py     # LLM model profiles
├── horizon_scanner/         # Phase A: country × hazard triage
├── forecaster/              # Phase B+C: research + SPD ensemble
├── resolver/                # Data ingestion & precedence engine
│   ├── ingestion/           # Source connectors (ACLED, IDMC, EM-DAT, etc.)
│   ├── transform/adapters/  # Data normalization
│   ├── tools/               # Export, precedence, snapshots, deltas
│   ├── cli/                 # CLI query interface
│   ├── api/                 # Resolver API layer
│   ├── config/              # Series semantics, schemas
│   └── tests/               # Resolver tests
├── web/                     # Next.js dashboard (localhost:3000)
│   └── src/                 # App, components, lib, styles
├── calibration/             # Calibration weights & advice JSON
├── scripts/                 # Operational scripts (debug bundles, CI, maps)
├── tests/                   # Top-level integration tests
├── tools/                   # Context pack, offline wheels
├── docs/                    # Extended documentation
├── .github/workflows/       # CI/CD pipelines
└── data/                    # Local data directory
```

## Key Entry Points

| Command | What it does |
|---------|-------------|
| `python -m horizon_scanner.horizon_scanner` | Run Horizon Scanner triage |
| `python -m forecaster.cli --mode pythia [--limit N]` | Run forecaster ensemble |
| `uvicorn pythia.api.app:app --reload --port 8000` | Start API server |
| `cd web && npm run dev` | Start dashboard on localhost:3000 |
| `python -m scripts.dump_pythia_debug_bundle --db <URL> --hs-run-id <ID>` | Generate debug bundle |

## Core Modules

### Horizon Scanner (`horizon_scanner/`)

Triages countries for emerging hazards. Main file: `horizon_scanner.py` (~69KB).

- Scores each country × hazard with a triage tier
- Computes **Regime Change (RC)**: `score = likelihood × magnitude`, levels 0-3
- Generates hazard tail packs for RC Level >= 2
- Seeds the `questions` table based on triage results
- Key env: `HS_MAX_WORKERS` (default 6)

RC-related files: `regime_change.py`, `hs_prompt.py`, `db_writer.py`.

### Forecaster (`forecaster/`)

Main file: `cli.py` (~266KB). Two phases:

1. **Research Phase** (`research.py`): Builds evidence briefs with verified sources, base rates, and RC signals.
2. **SPD Phase**: Multi-model ensemble (OpenAI, Google, Anthropic, xAI) aggregated via Bayesian Monte Carlo (`bayes_mc.py`).

Other key files:
- `prompts.py` — Research v2 and SPD v2 prompt templates
- `providers.py` — LLM provider abstraction with failure tracking
- `ensemble.py` / `aggregate.py` — Ensemble aggregation
- `GTMC1.py` — Strategic question tagging for binary questions
- `scenario_writer.py` — Scenario generation for priority questions
- `seen_guard.py` — Duplicate question handling

Concurrency env: `FORECASTER_RESEARCH_MAX_WORKERS`, `FORECASTER_SPD_MAX_WORKERS` (default 6 each).

### Resolver (`resolver/`)

Data ingestion layer with source precedence.

- **Connectors** (`ingestion/`): ACLED, IDMC, EM-DAT, UNHCR, IPC, ReliefWeb, DTM, WorldPop, GDELT, HDX
- **Precedence engine** (`tools/precedence_engine.py`): Tiered source selection
- **Exports** (`tools/export_facts.py`): Map to canonical schema
- **Deltas** (`tools/make_deltas.py`): Monthly "new" calculations
- **Snapshots** (`tools/freeze_snapshot.py`): Monthly immutable bundles
- DB tables (when `RESOLVER_DB_URL` set): `facts_resolved`, `facts_deltas`, `acled_monthly_fatalities`, `emdat_pa`, `snapshots`, `manifests`

### Pythia API (`pythia/api/`)

FastAPI service. Main file: `app.py`.

**Public endpoints** (no auth):
- `GET /v1/questions` — Question index
- `GET /v1/question_bundle` — Full question context + HS/forecast/research
- `GET /v1/forecasts/*` — Forecast queries
- `GET /v1/risk_index` — Humanitarian impact index
- `GET /v1/countries` — Country list
- `GET /v1/downloads/forecasts.csv|.xlsx` — SPD export
- `GET /v1/hs_runs`, `/v1/hs_triage/all` — HS run history
- `GET /v1/diagnostics/*` — KPI summaries
- `GET /v1/resolver/*` — Resolver connector status, country facts

**Admin** (requires `X-Pythia-Token`):
- `POST /v1/run` — Enqueue forecaster run

Auth: `auth.py` — Token-based via `X-Pythia-Token` header. Debug endpoints require `FRED_DEBUG_TOKEN`.

### Web Research (`pythia/web_research/`)

Multi-backend evidence retrieval. Main orchestrator: `web_research.py`.

- Backends: Gemini (default), OpenAI, Claude — with fallback chain
- `cache.py` — File-backed caching with TTL
- `budget.py` — Budget management and call caps
- `types.py` — `EvidencePack`, `EvidenceSource` types

Key env:
- `PYTHIA_WEB_RESEARCH_ENABLED=1`
- `PYTHIA_WEB_RESEARCH_BACKEND=gemini|openai|claude|auto`
- `PYTHIA_RETRIEVER_ENABLED=1`

### Database (`pythia/db/`)

DuckDB as system of record. Schema in `schema.py` / `schema.sql`.

Main tables: `hs_runs`, `hs_triage`, `hs_country_reports`, `hs_hazard_tail_packs`, `questions`, `question_research`, `forecasts_raw`, `forecasts_ensemble`, `llm_calls`, `question_run_metrics`.

### Calibration & Scoring (`pythia/tools/`, `calibration/`)

- `compute_resolutions.py` — Match forecasts to ground truth
- `compute_scores.py` — Score accuracy (PIN, PA metrics)
- `compute_calibration_pythia.py` — Aggregate calibration metrics
- `calibration_loop.py` — Continuous calibration loop
- `calibration/calibration_weights.json` — Per-model weights
- `calibration/calibration_advice.json` — Weekly prompt guidance

Bucket system: 5 income-based bins defined in `buckets.py` (`<5k`, `5k-<25k`, `25k-<100k`, `100k-<500k`, `>=500k`).

## Configuration

### `pythia/config.yaml`

Central config: DB URL, cron schedule, security tokens, allowed hazards (`FL`, `DR`, `TC`, `HW`, `ACE`, `DI`, `CU`, `EC`, `PHE`), LLM profiles (`test` = cheap, `prod` = frontier), forecaster ensemble specs, ranking config.

### Environment Variables (see `.env.template`)

**LLM providers**: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `OPENROUTER_API_KEY`

**Concurrency**: `LLM_MAX_CONCURRENCY`, `HS_MAX_WORKERS`, `FORECASTER_RESEARCH_MAX_WORKERS`, `FORECASTER_SPD_MAX_WORKERS`

**Database**: `PYTHIA_DB_URL` (default `duckdb:///data/resolver.duckdb`), `RESOLVER_DB_URL`, `RESOLVER_API_BACKEND` (`files` | `db`)

**Calibration**: `CALIBRATION_PATH`, `CALIB_ADVICE_PATH`, `CALIB_WEIGHTS_PATH`

**Forecasting knobs**: `FORECAST_TEMP`, `FORECAST_TOP_P`, `GTMC1_ACTIVATION_THRESHOLD`

**Research knobs**: `RESEARCH_TEMP`, `RESEARCH_TOP_P`, `MIN_ANCHOR_MATCH`

**Resolver connectors**: Various per-source keys/URLs (ACLED, DTM, HDX, WFP, ReliefWeb, etc.)

## Development Setup

```bash
# Install dependencies
pip install -r python_library_requirements.txt
# Or with extras:
pip install -e ".[db,test]"

# Initialize DB schema
python -c "from pythia.db.schema import ensure_schema; ensure_schema()"

# Use cheap models for dev
export PYTHIA_LLM_PROFILE=test

# Run API
uvicorn pythia.api.app:app --reload --port 8000

# Run dashboard
cd web && npm install && npm run dev
```

Makefile targets: `make dev-setup`, `make dev-setup-online`, `make dev-setup-offline`, `make test-db`, `make db.inspect`.

## Testing

Config in `pytest.ini`: timeout 120s, asyncio support, `--maxfail=1`.

```bash
# All tests
pytest tests/

# Resolver tests
pytest -q resolver/tests/

# Forecaster tests
pytest -q forecaster/tests/
```

**Test markers**:
- `@pytest.mark.allow_network` — Makes network calls
- `@pytest.mark.db` — Requires DuckDB
- `@pytest.mark.slow` — Long-running
- `@pytest.mark.nightly` — Overnight tests
- `@pytest.mark.legacy_freeze` — Legacy snapshot pipeline

Test directories: `tests/`, `pythia/tests/`, `forecaster/tests/`, `horizon_scanner/tests/`, `resolver/tests/`.

## CI/CD Workflows (`.github/workflows/`)

| Workflow | Purpose |
|----------|---------|
| `run_horizon_scanner.yml` | Production HS + forecaster pipeline |
| `forecaster-ci.yml` | SPD unit tests, artifact comparison |
| `resolver-ci.yml` | Resolver ingestion + validation |
| `resolver-ci-fast.yml` | Quick resolver smoke test |
| `compute_calibration_pythia.yml` | Run calibration |
| `compute_scores.yml` | Compute scores |
| `publish_latest_data.yml` | Publish artifacts |
| `web-ci.yml` | Dashboard CI |
| `ci-lint.yml` / `lint.yml` | Linting |

## Package Management

Poetry-based, Python 3.11+. Defined in `pyproject.toml`.

Core deps: `duckdb`, `fastapi`, `pydantic`, `openai`, `requests`, `numpy`, `python-dotenv`, `python-decouple`, `python-docx`, `openpyxl`.

Extras: `db` (duckdb), `test` (duckdb + httpx + pytest), `connectors` (dtmapi), `ingestion` (dtmapi + pandas + python-dateutil), `dev` (ipykernel).

## Hazard Codes

`FL` (Flood), `DR` (Drought), `TC` (Tropical Cyclone), `HW` (Heatwave), `ACE` (Armed Conflict Event), `DI` (Displacement), `CU` (Currency/Economic), `EC` (Economic), `PHE` (Public Health Emergency).

## Design Patterns

- **Database-first**: All outputs persisted to DuckDB; exports and API derive from DB tables.
- **Fallback/resilience**: Provider failure tracking with cooldown; partial ensembles proceed without failed providers; web research backend fallback chain.
- **Shared evidence packs**: Web research results cached and reused across questions via the retriever layer.
- **LLM telemetry**: Every LLM call logged to `llm_calls` table with cost, latency, and optional transcript.
- **Regime Change as first-class concept**: RC scoring flows from HS triage through research into SPD prompts; level >= 2 triggers hazard tail packs and special forecasting guidance.
