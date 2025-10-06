# CODEMAP — Pythia Resolver

## Purpose & Scope
Pythia's Resolver (also called Resolver/Resolver) prepares "database-first, resolution-ready" facts for People in Need (PIN) and People Affected (PA) questions. The pipeline ingests situation reports and registries from humanitarian data sources, normalises them into canonical facts, applies precedence policy to pick a single authoritative figure for every `(country, hazard, month)` combination, and publishes those resolved totals together with monthly deltas and audit artefacts. This map explains how the resolver repository is organised, how data flows through the system, and how to operate the tooling end-to-end.

## High-Level Architecture
```mermaid
flowchart LR
    A[Connectors<br/>(resolver/ingestion/*_client.py)] --> B[Staging<br/>(resolver/staging/*.csv,*.parquet)]
    B --> C[Validation & Precedence Inputs<br/>(schema.yml, registries, review overrides)]
    C --> D[Resolved Facts<br/>(exports/resolved.csv,jsonl)]
    D --> E[Deltas (monthly new PIN/PA)<br/>(exports/deltas.csv)]
    E --> F[Snapshots<br/>(resolver/snapshots/YYYY-MM)]
    F --> G[Exports & Diagnostics<br/>(exports/facts.csv,resolved_diagnostics.csv)]
    G --> H{DB Dual-Write?}
    H -- yes --> I[(DuckDB/External DB)]
    H -- no --> J[File-backed Mode]
    I & J --> K[Query Layer<br/>(resolver/cli, resolver/api)]
```

## Key Modules & Responsibilities
- [`resolver/ingestion`](resolver/ingestion): Source-specific connectors, retry helpers, and offline stubs. Each client reads `resolver/ingestion/config/*.yml` for API parameters and writes staging CSV/Parquet under `resolver/staging/`. The orchestrator [`run_all_stubs.py`](resolver/ingestion/run_all_stubs.py) applies smoke defaults and per-source feature flags.
- [`resolver/tools`](resolver/tools): Operational scripts that transform staging data into canonical outputs. Highlights include [`export_facts.py`](resolver/tools/export_facts.py) for column mapping, [`validate_facts.py`](resolver/tools/validate_facts.py) for schema and registry checks, [`precedence_engine.py`](resolver/tools/precedence_engine.py) for tiered selection, [`make_deltas.py`](resolver/tools/make_deltas.py) for month-over-month "new" figures, [`freeze_snapshot.py`](resolver/tools/freeze_snapshot.py) for immutable bundles, [`check_sizes.py`](resolver/tools/check_sizes.py) for artifact thresholds, and [`write_repo_state.py`](resolver/tools/write_repo_state.py) for copying outputs into `resolver/state/`.
- [`resolver/data`](resolver/data): Canonical registries (`countries.csv`, `shocks.csv`) and lookup tables consumed by validators, CLI, and connectors.
- [`resolver/reference`](resolver/reference): Static reference data such as `avg_household_size.csv` and override manifests that support ReliefWeb parsing and denominator adjustments.
- [`resolver/review`](resolver/review): Manual review workflow helpers. [`make_review_queue.py`](resolver/review/make_review_queue.py) emits `review_queue.csv`, and [`apply_review_overrides.py`](resolver/review/apply_review_overrides.py) folds curator decisions back into resolved exports.
- [`resolver/cli`](resolver/cli): Command-line interface that answers a single forecast resolution query by combining registries and exported datasets. [`resolver_cli.py`](resolver/cli/resolver_cli.py) encapsulates cutoff selection rules and handles deltas vs. stock series.
- [`resolver/api`](resolver/api): FastAPI wrapper exposing `/health` and `/resolve` endpoints via [`app.py`](resolver/api/app.py). Designed to share the same loaders as the CLI.
- [`resolver/tests`](resolver/tests): Pytest suites covering connectors, schema validation, precedence logic, deltas, and documentation generators such as [`test_generate_schemas_md.py`](resolver/tests/test_generate_schemas_md.py).
- [`resolver/db`](resolver/db): DuckDB schema + helpers. [`duckdb_io.py`](resolver/db/duckdb_io.py) initialises the schema, exposes `get_db()`, and writes `facts_resolved`, `facts_deltas`, `snapshots`, and `manifests` tables.
- [`resolver/docs`](resolver/docs): Extended documentation (pipeline overview, policy, data dictionary, troubleshooting) that complements this codemap for deep dives.

## Entrypoints & Commands
| Command | Purpose | Key Environment/Inputs |
| --- | --- | --- |
| `python resolver/ingestion/run_all_stubs.py --retries 2` | Generate staging CSVs using offline fixtures (no network). | Applies smoke defaults (`RESOLVER_MAX_RESULTS`, `RESOLVER_WINDOW_DAYS`, etc.) unless overridden. |
| `python resolver/ingestion/<connector>_client.py` | Run a specific connector against live APIs. | Source-specific tokens such as `ACLED_REFRESH_TOKEN`, `DTM_API_PRIMARY_KEY`, `GO_API_TOKEN`, `RELIEFWEB_APPNAME`, toggles like `RESOLVER_SKIP_<SOURCE>`, `RESOLVER_DEBUG=1` for verbose logging. |
| `python resolver/tools/export_facts.py --in resolver/staging --out resolver/exports` | Map staging columns to canonical facts. | Optional `--config resolver/tools/export_config.yml` for custom mappings. |
| `python resolver/tools/validate_facts.py --facts resolver/exports/facts.csv` | Enforce schema, registry, and enum rules before precedence. | Requires registries in `resolver/data/` and schema definition `resolver/tools/schema.yml`. |
| `python resolver/tools/precedence_engine.py --facts resolver/exports/facts.csv --cutoff 2025-09-30` | Apply precedence tiers to produce `resolved.csv/jsonl` plus diagnostics. | Configuration in `resolver/tools/precedence_config.yml`; uses Istanbul timezone lag logic. |
| `python resolver/tools/make_deltas.py --resolved resolver/exports/resolved.csv --out resolver/exports/deltas.csv --lookback-months 24` | Compute monthly "new" deltas with rebasing detection. | Reads resolved exports and writes `deltas.csv` with provenance columns. |
| `python resolver/tools/freeze_snapshot.py --facts resolver/exports/facts.csv --resolved resolver/exports/resolved.csv --deltas resolver/exports/deltas.csv --month 2025-09` | Freeze immutable monthly bundles. | Writes `resolver/snapshots/<YYYY-MM>/` artifacts and, with `RESOLVER_DB_URL`, upserts DuckDB tables. |
| `python resolver/tools/write_repo_state.py --mode daily --id 2025-09-30` | Copy exports/review outputs into `resolver/state/daily/...` for archival. | `--retain-days` controls pruning; stages deletions via Git. |
| `python resolver/tools/check_sizes.py` | Warn/fail when exports or snapshots exceed configured size limits. | Thresholds via `RESOLVER_LIMIT_PARQUET_MB`, `RESOLVER_LIMIT_CSV_MB`, `RESOLVER_LIMIT_REPO_MB`. |
| `python resolver/tools/generate_schemas_md.py --in resolver/tools/schema.yml --out SCHEMAS.md --sort` | Regenerate schema reference documentation. | Requires `pyyaml`; fails if schema definitions are missing. |
| `python resolver/cli/resolver_cli.py --country "Philippines" --hazard "Tropical Cyclone" --cutoff 2025-09-30 --backend db` | Query the latest resolved fact (defaults to monthly "new" series). | Reads file exports by default; set `--backend db` or `RESOLVER_CLI_BACKEND=db` to use DuckDB. Optional `--series stock` for totals. |
| `uvicorn resolver.api.app:app --reload` | Serve the Resolver API locally. | Same data dependencies as the CLI; respects `RESOLVER_DEBUG` for verbose logs. |
| `pytest -q resolver/tests/test_ingestion_smoke_all_connectors.py` | Offline smoke test covering stubbed connectors and schema checks. | Install dependencies from `resolver/requirements*.txt`; other targeted tests live under `resolver/tests/`. |

## Data Contracts & Schemas
Schema authority lives in [`resolver/tools/schema.yml`](resolver/tools/schema.yml) and the generated [`SCHEMAS.md`](SCHEMAS.md). Canonical columns for facts, deltas, and staging datasets follow the Resolver data dictionary ([`resolver/docs/data_dictionary.md`](resolver/docs/data_dictionary.md)). PIN/PA exports must include `event_id`, location/hazard tuples, metric/unit pairs, timestamps (`as_of_date`, `publication_date`, `ingested_at`), and citation fields (`publisher`, `source_type`, `source_url`, `doc_title`, `definition_text`). Monthly deltas add lineage columns such as `series_semantics`, `value_new`, `rebase_flag`, and `delta_negative_clamped` to explain adjustments. Regenerate documentation whenever schemas change so downstream teams can rely on `SCHEMAS.md` as the single source of truth.

## Configuration & Secrets
- Environment variables are loaded from your shell or a local `.env`; copy `.env.template` and fill only the resolver-specific keys you need (API tokens, feature flags). Never commit secrets to the repository.
- Connector settings live under [`resolver/ingestion/config/`](resolver/ingestion/config) and can be customised per source (rate limits, date windows, indicator filters).
- Precedence logic uses [`resolver/tools/precedence_config.yml`](resolver/tools/precedence_config.yml) for tier mappings, metric preferences, conflict rules, and lag allowances.
- ReliefWeb PDF, denominator overrides, and review behaviour are toggled via dedicated env vars (e.g., `RELIEFWEB_ENABLE_PDF`, `WORLDPOP_PRODUCT`, `RESOLVER_INCLUDE_STUBS`).
- Store GitHub Actions secrets (tokens, API keys) at the workflow level; CI jobs read them from the runner environment instead of tracked files.

## Storage Layout
- **Staging:** `resolver/staging/` (ignored by Git) accumulates raw connector outputs per source.
- **Exports:** `resolver/exports/` holds canonical facts, resolved outputs, deltas, diagnostics, and parity reports; temporary files with `_working` or `cache_` prefixes remain ignored per [`.gitignore`](.gitignore).
- **Snapshots:** `resolver/snapshots/<YYYY-MM>/` contains monthly parquet bundles plus manifests for immutable grading.
- **State:** `resolver/state/` mirrors CI artefacts — `pr/<PR>`, `daily/<YYYY-MM-DD>`, and `monthly/<YYYY-MM>` folders produced by [`write_repo_state.py`](resolver/tools/write_repo_state.py).
- **Logs:** Connector logs stream to `resolver/logs/ingestion/` (ignored) with per-source appenders set up by `_runner_logging.py`.
- **Reference & Review:** `resolver/reference/` and `resolver/review/` keep lightweight CSVs and overrides that *are* tracked to preserve auditability.

## DB Integration Toggle
Set `RESOLVER_DB_URL` to enable dual-writing exports and snapshots into DuckDB (default `duckdb:///resolver/db/resolver.duckdb`). When present, `export_facts.py` appends to `facts_resolved` (and `facts_deltas` when available) and `freeze_snapshot.py` records resolved totals, deltas, manifests, and snapshot metadata transactionally. The CLI/API remain file-backed by default; opt into DuckDB with `--backend db`, `RESOLVER_CLI_BACKEND=db`, or `RESOLVER_API_BACKEND=db` (use `auto` to prefer DB when available).

## Runbooks
### First-time setup
1. Install dependencies:
   ```bash
   pip install -r resolver/requirements.txt
   pip install -r resolver/requirements-dev.txt
   ```
2. Copy `.env.template` → `.env` and populate only the resolver keys you need (tokens, feature flags).
3. Generate offline staging data to verify the stack:
   ```bash
   python resolver/ingestion/run_all_stubs.py --retries 1
   python resolver/tools/export_facts.py --in resolver/staging --out resolver/exports
   python resolver/tools/validate_facts.py --facts resolver/exports/facts.csv
   ```
4. Build resolved outputs and deltas:
   ```bash
   python resolver/tools/precedence_engine.py --facts resolver/exports/facts.csv --cutoff YYYY-MM-30
   python resolver/tools/make_deltas.py --resolved resolver/exports/resolved.csv --out resolver/exports/deltas.csv
   ```
5. Regenerate docs/tests: `python resolver/tools/generate_schemas_md.py --in resolver/tools/schema.yml --out SCHEMAS.md --sort` and `pytest -q resolver/tests/test_ingestion_smoke_all_connectors.py`.

### Daily run (live connectors)
1. Export credentials to the shell or CI secrets (`ACLED_REFRESH_TOKEN`, `DTM_API_PRIMARY_KEY`, `GO_API_TOKEN`, etc.).
2. Run live connectors (optionally in parallel) with `RESOLVER_DEBUG=1` for richer logs:
   ```bash
   python resolver/ingestion/acled_client.py
   python resolver/ingestion/unhcr_client.py
   python resolver/ingestion/ipc_client.py
   # ...other sources as required
   ```
3. Process outputs:
   ```bash
   python resolver/tools/export_facts.py --in resolver/staging --out resolver/exports
   python resolver/tools/validate_facts.py --facts resolver/exports/facts.csv
   python resolver/tools/precedence_engine.py --facts resolver/exports/facts.csv --cutoff <YYYY-MM-DD>
   python resolver/tools/make_deltas.py --resolved resolver/exports/resolved.csv --out resolver/exports/deltas.csv
    python resolver/tools/freeze_snapshot.py --facts resolver/exports/facts.csv --resolved resolver/exports/resolved.csv --deltas resolver/exports/deltas.csv --month <YYYY-MM>
   ```
4. Prepare review queue if curators are on call: `python resolver/review/make_review_queue.py`.
5. Archive outputs and enforce hygiene:
   ```bash
   python resolver/tools/write_repo_state.py --mode daily --id <YYYY-MM-DD> --retain-days 14
   python resolver/tools/check_sizes.py
   ```
6. Commit or push artefacts if running in an automated environment (CI handles this in `resolver-ci-nightly`).

### Debugging common failures
- **Network/SSL or rate limits:** Rerun the connector with `RESOLVER_DEBUG=1` and inspect `resolver/logs/ingestion/<source>.log`. Use feature flags like `RESOLVER_SKIP_<SOURCE>=1` or stub fallbacks (`RESOLVER_INCLUDE_STUBS=1`) to keep the pipeline moving while troubleshooting credentials. Adjust retry/backoff knobs in connector configs if the upstream API is throttling.
- **Empty or partial results:** Confirm API keys in `.env`, check per-source configs under `resolver/ingestion/config/`, and run the corresponding stub (`python resolver/ingestion/<source>_stub.py`) to isolate mapping issues. Some connectors honour hints such as `WORLDPOP_PRODUCT`, `IPC_DEFAULT_HAZARD`, or `RESOLVER_MAX_RESULTS`; ensure those values cover the requested scope.
- **Schema mismatches or validator errors:** Run `python resolver/tools/validate_facts.py --facts <path>` to see failing columns, then update `resolver/tools/schema.yml` (and connector output) accordingly. Regenerate `SCHEMAS.md` and rerun targeted tests like `pytest -q resolver/tests/test_staging_schema_all.py` before committing changes.

## Testing & CI
- Local smoke tests: `pytest -q resolver/tests/test_ingestion_smoke_all_connectors.py` (stubbed ingestion), `pytest -q resolver/tests/test_resolved_and_review.py` (export pipeline), and connector-specific suites under `resolver/tests/ingestion/`.
- Schema/documentation guardrails: `pytest -q resolver/tests/test_generate_schemas_md.py` ensures the Markdown generator stays deterministic.
- Continuous integration: `.github/workflows/resolver-ci.yml` installs resolver requirements, runs offline connector smoke tests, ReliefWeb PDF unit tests, and performs intra-repo Markdown link checking. Nightly workflows extend this with live runs and state archival.

## Glossary & Appendix
- **PIN / PA:** `metric=in_need` (People in Need) and `metric=affected` (People Affected) totals normalised to `unit=persons` or `persons_cases` for outbreaks.
- **Series semantics:** `series_semantics=stock` represents cumulative totals; `series_semantics=new` captures monthly deltas produced by [`make_deltas.py`](resolver/tools/make_deltas.py).
- **`ym` column:** Normalised `YYYY-MM` month derived from `as_of_date` in Europe/Istanbul time; used to align deltas and resolved records.
- **Precedence tiers:** Configured ordering of publishers/source types in [`precedence_config.yml`](resolver/tools/precedence_config.yml); higher tiers win ties, with diagnostics logged to `exports/resolved_diagnostics.csv`.
- **Snapshots:** Immutable parquet bundles under `resolver/snapshots/<YYYY-MM>/` considered the grading truth for cutoff queries.
- **Additional reading:** See [`resolver/docs/pipeline_overview.md`](resolver/docs/pipeline_overview.md) for expanded diagrams, [`resolver/docs/data_dictionary.md`](resolver/docs/data_dictionary.md) for field definitions, and [`resolver/docs/troubleshooting.md`](resolver/docs/troubleshooting.md) for escalations.
