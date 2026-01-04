# Pythia

Pythia is an end-to-end AI forecasting system for humanitarian questions. It scans countries for emerging hazards, turns triage into forecastable questions, runs an LLM ensemble to produce Subjective Probability Distributions (SPD), and stores forecasts, research, and diagnostics in a DuckDB-backed system of record.

## System at a glance
- **System of record**: DuckDB (`PYTHIA_DB_URL` / `app.db_url`) holds all HS triage, questions, research, forecasts, and diagnostics.
- **Resolver → HS → Questions → Research → SPD/Scenario → DuckDB**: resolver facts and base rates flow into Horizon Scanner; HS produces triage and questions; shared retriever packs evidence; Research v2 drafts structured briefs; Forecaster writes SPD ensembles and scenarios back to DuckDB.

## Architecture
- **Resolver (facts + base rates)**: Resolver tables in DuckDB (`facts_resolved`, `facts_deltas`, `snapshots`) provide historical context for triage, research, and scoring. The schema is defined in [`pythia/db/schema.py`](pythia/db/schema.py).
- **Horizon Scanner (HS)**: `python -m horizon_scanner.horizon_scanner` triages countries and hazards, writes `hs_runs`, `hs_triage`, and scenario rows, and seeds `question_research` + `questions`.
- **Retriever web research (shared evidence packs)**: When enabled, the retriever builds evidence packs that are re-used across SPD and research prompts (reducing cost and stabilizing sources). Retriever defaults to `gemini-2.5-flash-lite` when `PYTHIA_RETRIEVER_ENABLED=1` and `PYTHIA_RETRIEVER_MODEL_ID` is unset; see [`pythia/web_research/web_research.py`](pythia/web_research/web_research.py).
- **Research v2**: Research prompts generate structured briefs that separate verified vs unverified sources, backed by `question_research` and `llm_calls`.
- **Forecaster SPD v2 ensemble**: `python -m forecaster.cli --mode pythia` runs SPD v2 prompts across the active ensemble and writes per-model outputs (`forecasts_raw`) and aggregated results (`forecasts_ensemble`). BayesMC aggregation is optional and produces fallback metadata.
- **Scenarios (priority-only)**: When an ensemble SPD is available, scenarios can be generated for priority questions and written back to DuckDB.
- **Logging & diagnostics**: `llm_calls`, `question_run_metrics`, and debug bundles capture per-question timing and provenance. Debug bundles are produced by [`scripts/dump_pythia_debug_bundle.py`](scripts/dump_pythia_debug_bundle.py).

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
- Set the DuckDB URL (or use the default in [`pythia/config.yaml`](pythia/config.yaml)).
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

### 4) Run Horizon Scanner on a short list
- Default list: [`horizon_scanner/hs_country_list.txt`](horizon_scanner/hs_country_list.txt)
- For a small smoke run, edit the list or pass a short list in a helper script.
```bash
PYTHIA_LLM_PROFILE=test python -m horizon_scanner.horizon_scanner
```

### 5) Run the forecaster (limit work)
```bash
PYTHIA_LLM_PROFILE=test python -m forecaster.cli --mode pythia --limit 20 --purpose local_smoke
```

### 6) Inspect outputs
- **DuckDB**: open the database at `PYTHIA_DB_URL` (`app.db_url` default is `duckdb:///data/resolver.duckdb`).
- **Debug bundle** (optional):
```bash
python -m scripts.dump_pythia_debug_bundle \
  --db "${PYTHIA_DB_URL}" \
  --hs-run-id "<HS_RUN_ID>" \
  --forecaster-run-id "<FORECASTER_RUN_ID>"
```
- Bundles default to `debug/pytia_debug_bundle__<run_id>.md` (note: filename uses `pytia_`).

## Starter runs

### 1-country smoke run
```bash
# Update horizon_scanner/hs_country_list.txt to a single ISO3, e.g. "KEN"
PYTHIA_LLM_PROFILE=test python -m horizon_scanner.horizon_scanner
PYTHIA_LLM_PROFILE=test python -m forecaster.cli --mode pythia --limit 5 --purpose smoke_1_country
```

### 10-country run (bounded)
```bash
# Trim hs_country_list.txt to 10 ISO3 codes
PYTHIA_LLM_PROFILE=test python -m horizon_scanner.horizon_scanner
PYTHIA_LLM_PROFILE=test python -m forecaster.cli --mode pythia --limit 50 --purpose smoke_10_countries
```

### Bounding work
- **Question count**: `--limit N`
- **HS triage**: lower tiers or reduce country list (`hs_country_list.txt`)
- **Scenarios**: scenario generation only runs when ensemble SPD exists; use priority filters upstream.

## Configuration

### Canonical config
- [`pythia/config.yaml`](pythia/config.yaml) is the authoritative configuration.
- There is no `pythia/config.py`; all runtime defaults live in the YAML + env vars.
- `app.db_url` defines DuckDB; `PYTHIA_DB_URL` overrides at runtime.
- `llm.profile` selects the default model bundle; override with `PYTHIA_LLM_PROFILE`.

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
- **SPD tuning / timeouts (Gemini 3)**:
  - `PYTHIA_GOOGLE_SPD_THINKING_LEVEL_FLASH`, `PYTHIA_GOOGLE_SPD_THINKING_LEVEL_PRO`
  - `PYTHIA_GOOGLE_SPD_TIMEOUT_FLASH_SEC`, `PYTHIA_GOOGLE_SPD_TIMEOUT_PRO_SEC`
  - `PYTHIA_GOOGLE_SPD_RETRIES`
- **SPD ensemble override**: `PYTHIA_SPD_ENSEMBLE_SPECS` (e.g. `openai:gpt-5.1,google:gemini-3-flash-preview`).
- **HS triage resilience**:
  - `PYTHIA_HS_FALLBACK_MODEL_SPECS` (default `openai:gpt-5.1`, required to keep HS triage running when Gemini fails).
  - `PYTHIA_HS_ONLY_COUNTRIES` (comma-separated ISO3s/names to rerun HS triage for a subset).
  - `PYTHIA_PROVIDER_FAILURE_THRESHOLD`, `PYTHIA_PROVIDER_COOLDOWN_SECONDS`, `PYTHIA_PROVIDER_RESET_ON_SUCCESS`
  - `PYTHIA_LLM_RETRY_TIMEOUTS` (set `0` to opt out of timeout retries outside HS triage)
  - `PYTHIA_HS_LLM_MAX_ATTEMPTS` (defaults to 3 for HS triage retries)
  - `PYTHIA_HS_GEMINI_TIMEOUT_SEC` (defaults to 120s for HS triage)

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
- Upgrade legacy `llm_calls` telemetry columns with:
```bash
python scripts/migrate_llm_calls_telemetry.py --db duckdb:///path/to/resolver.duckdb
```

### HS triage reruns
- After the HS run completes, stdout includes `HS_TRIAGE_RERUN_ISO3S=...`. To re-run just those countries:
```bash
PYTHIA_HS_ONLY_COUNTRIES="<ISO3S>" python -m horizon_scanner.horizon_scanner
```

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
- **EM-DAT sparsity**: historical data gaps can reduce base-rate strength.
- **IDMC month-key issue**: occasionally inconsistent month keys are tolerated and cleaned downstream.
- **Occasional “200 but ungrounded”**: some web search calls can return success without verified sources.

## Troubleshooting
- **No questions generated**: HS may mark all tiers quiet. Check `hs_triage` in DuckDB and confirm your country list (`horizon_scanner/hs_country_list.txt`) and `hazards_allowed` in config.
- **No active models**: verify `PYTHIA_LLM_PROFILE`, `forecaster.providers` config, and provider API keys.
- **Debug bundle too large for step summary**: artifacts still exist under `debug/` even if GitHub Step Summary truncates.
- **Slow runs**: Gemini tails can dominate latency. Tune `PYTHIA_LLM_CONCURRENCY`, `FORECASTER_*_MAX_WORKERS`, and SPD timeouts.
- **Interpreting question_run_metrics**: `question_run_metrics` (if present) records wall-clock vs compute vs queue time per question; see `scripts/dump_pythia_debug_bundle.py`.

## Cross-links
- Config: [`pythia/config.yaml`](pythia/config.yaml)
- Horizon Scanner: [`horizon_scanner/horizon_scanner.py`](horizon_scanner/horizon_scanner.py)
- HS country list: [`horizon_scanner/hs_country_list.txt`](horizon_scanner/hs_country_list.txt)
- Forecaster CLI: [`forecaster/cli.py`](forecaster/cli.py)
- Schema: [`pythia/db/schema.py`](pythia/db/schema.py)
- Debug bundle script: [`scripts/dump_pythia_debug_bundle.py`](scripts/dump_pythia_debug_bundle.py)
- Workflows: [`run_horizon_scanner.yml`](.github/workflows/run_horizon_scanner.yml), [`forecaster-ci.yml`](.github/workflows/forecaster-ci.yml)

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for coding standards and workflow notes. Archived READMEs live in [docs/archive/README_INDEX.md](docs/archive/README_INDEX.md).

## License
This repository follows the licensing terms bundled with the codebase (see `LICENSE` if present or repository metadata).
