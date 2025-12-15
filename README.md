# Pythia

Pythia is an end-to-end AI forecasting system for humanitarian questions. It scans countries for emerging hazards, turns triage into forecastable questions, runs an ensemble of LLM forecasters (SPD and scenario prompts plus BayesMC aggregation), and records forecasts, research, and diagnostics in a DuckDB-backed workspace.

## System at a glance
- Horizon Scanner (HS) triages countries and hazards, storing `hs_runs`, `hs_triage`, and scenario rows in DuckDB.
- Question builder turns triage outputs into structured questions and research prompts stored in `questions` and `question_research`.
- Forecaster loads active questions, fans out SPD/scenario prompts to the active LLM ensemble, and writes both per-model (`forecasts_raw`) and ensemble (`forecasts_ensemble`) rows.
- Optional BayesMC aggregation and dual-compare runs emit side-by-side SPD JSON for diagnostics.
- Resolver tables in DuckDB keep historical questions, forecasts, research, UI runs, and provenance so downstream scoring or dashboards can query one location.

## Components
- **Horizon Scanner (HS)**: `horizon_scanner/horizon_scanner.py` performs triage over a country list (from `hs_country_list.txt` or API callers), writes `hs_triage`, and logs runs in `hs_runs`/`hs_scenarios`.
- **Question generation**: HS triage plus `question_research` rows feed the question builder that populates the `questions` table with wording, hazard, metric, and target month windows.
- **Forecaster (SPD + scenarios)**: `forecaster/cli.py --mode pythia` loads active questions, runs SPD v2 prompts (with optional scenario infill), captures per-model results in `forecasts_raw`, and aggregates into `forecasts_ensemble`. Research ablations and GTMC1 checks remain logged for diagnostics.
- **Resolver (DuckDB)**: `pythia/db/schema.py` owns the DuckDB schema (`questions`, `forecasts_raw`, `forecasts_ensemble`, `question_context`, `llm_calls`, etc.) and ensures tables exist for local runs and CI workflows.
- **Calibrator / scoring**: `pythia/tools/calibration_loop.py` runs post-forecast calibration when enough resolved questions exist; scoring workflows consume DuckDB outputs.
- **UI / artifacts**: Pipeline runs update `ui_runs` for API/UI triggers and leave SPD compare JSON under `debug/spd_compare_smoke` (smoke) or `debug/spd_compare_tests` (unit-test captures).

## Quickstart (local)
1. **Install dependencies**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r python_library_requirements.txt
   ```
2. **Set config and secrets**
   - Edit `pythia/config.yaml` (or export `PYTHIA_LLM_PROFILE`) to choose the active LLM profile and DB URL.
   - Export provider API keys (for example `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`) and any Horizon Scanner model overrides.
3. **Ensure the DuckDB schema exists**
   ```bash
   python - <<'PY'
   from pythia.db.schema import ensure_schema
   ensure_schema()
   print("schema ready")
   PY
   ```
4. **Run Horizon Scanner triage** (writes HS tables and new questions)
   ```bash
   PYTHIA_LLM_PROFILE=test python -m horizon_scanner.horizon_scanner
   ```
5. **Run the forecaster on active questions**
   ```bash
   PYTHIA_LLM_PROFILE=test python -m forecaster.cli --mode pythia --limit 20 --purpose local_smoke
   ```
6. **Inspect outputs**
   - DuckDB lives at `PYTHIA_DB_URL` or `app.db_url` (default `duckdb:///data/resolver.duckdb`).
   - Forecast and HS diagnostics live under `debug/` and `logs/` when enabled.

## How to run
- **HS + question creation**: call `python -m horizon_scanner.horizon_scanner` directly or trigger the `run_horizon_scanner.yml` workflow. By default HS reads `hs_country_list.txt`; to override, import and call `horizon_scanner.horizon_scanner.main(["KEN","ETH"] )` from a small helper script.
- **Forecaster SPD/scenario runs**: invoke `python -m forecaster.cli --mode pythia` with `--limit` to bound question count. Use `PYTHIA_DEBUG_SPD=1` for verbose logs and `PYTHIA_SPD_HARD_FAIL=1` to surface SPD exceptions.
- **Resolver/DB workflows**: `pythia/db/schema.ensure_schema()` is idempotent; `pythia/db/util.write_llm_call` records all provider calls. When backfilling or scoring, point tools at the same DuckDB URL so tables (`questions`, `forecasts_*`, `hs_*`, `llm_calls`, `ui_runs`) stay in sync.

## Configuration
- The canonical config lives at `pythia/config.yaml`.
- `app.db_url` selects the DuckDB location; `PYTHIA_DB_URL` overrides it at runtime.
- `llm.profile` chooses the default model bundle; override with `PYTHIA_LLM_PROFILE` to swap between `test` and `prod` profiles.
- **Provider activation**: a provider becomes active only when `forecaster.providers` marks it `enabled`, a `model` id is present (from config or the current profile), **and** the expected API key env var is non-empty. Missing any of the three disables the provider for that run.
- Hazard, HS, and researcher prompt versions are configured in `pythia/config.yaml`; HS respects `HS_MAX_WORKERS`, `HS_TEMPERATURE`, and DuckDB URL settings.

## CI / Workflows
- **Forecaster CI — SPD** (`.github/workflows/forecaster-ci.yml`): runs SPD unit tests by default; optional workflow-dispatch inputs enable dual-compare or BayesMC write paths. Smoke runs upload compare JSON under `debug/spd_compare_smoke`, and test captures land in `debug/spd_compare_tests` artifacts.
- **Core pipeline workflow** (`run_horizon_scanner.yml`): executes HS and forecaster end-to-end against the configured DuckDB. It is not part of the fast unit-test gate but is the source of fresh questions and ensemble forecasts.
- **Resolver-focused workflows** (`resolver-ci.yml`, `resolver-ci-fast.yml`, `resolver-ci-nightly.yml`): keep DuckDB schema and ingestion/tests healthy; they do not download SPD artifacts.
- Smoke/diagnostic artifacts from CI live under `debug/` and `dist/diagnostics-*` in workflow artifacts for later inspection.

## Ensemble / BayesMC
- **SPD v2 ensemble**: default SPD prompts call all active providers and aggregate into `forecasts_ensemble`.
- **BayesMC aggregation**: when `PYTHIA_SPD_V2_USE_BAYESMC=1`, SPD v2 uses BayesMC to fuse member distributions and attaches `ensemble_meta` diagnostics.
- **Dual compare**: set `PYTHIA_SPD_V2_DUAL_RUN=1` to run both the default ensemble and BayesMC paths; compare JSON is written to `debug/spd_compare_smoke` (smoke) or collected from pytest into `debug/spd_compare_tests`.

## Debugging
- **No active models**: check `PYTHIA_LLM_PROFILE`, `forecaster.providers` config, and provider API key env vars; SPD will skip providers without all three.
- **Missing artifacts**: ensure `PYTHIA_SPD_COMPARE_DIR` points to a writable directory when dual-compare is enabled; CI smoke sets it to `debug/spd_compare_smoke`.
- **Secrets not wired**: HS and forecaster respect the same provider env vars; runs without keys fall back to inactive providers and may yield empty ensembles.
- **DuckDB path issues**: confirm `PYTHIA_DB_URL` or `app.db_url` points to a writable location; `ensure_schema()` is safe to rerun.

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for coding standards and workflow notes. Archived READMEs now live under [docs/archive/README_INDEX.md](docs/archive/README_INDEX.md).

## License
This repository follows the licensing terms bundled with the codebase (see `LICENSE` if present or repository metadata).
