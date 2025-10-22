README (updated)
[![resolver-ci-fast](https://github.com/oughtinc/Pythia/actions/workflows/resolver-ci-fast.yml/badge.svg?branch=main)](https://github.com/oughtinc/Pythia/actions/workflows/resolver-ci-fast.yml)

What is Forecaster?

Forecaster is an AI forecasting assistant. Its job is to read forecasting questions (like those on Metaculus), gather relevant evidence, and combine the judgments of different models into one final forecast. It is designed to run automatically on GitHub, saving its forecasts and learning from past performance over time.

## CI guardrails and repo context checks

Our GitHub Actions workflows now include extra safety rails to ensure they always run inside the official [`oughtinc/Pythia`](https://github.com/oughtinc/Pythia) repository:

- Each job prints a **Debug repo context** block so you can see the repository slug, branch, commit SHA, and working directory tree size directly in the logs.
- An **anti-drift** script in `scripts/ci/` fails fast if any legacy references to the old repository slug sneak back into the tree.
- Database test jobs are protected with repo guards and file-existence checks so they only execute inside `oughtinc/Pythia` when the relevant tests are present.
- Every resolver workflow proactively uninstalls any cached wheel and runs `pip install -e .` so tests import the workspace copy of `resolver`. Immediately before the pytest steps we install the `db`/`test` extras (falling back to `pip install duckdb pytest` if the extras are unavailable) and log a positive `import duckdb` probe so CI failures never stem from missing DuckDB wheels. After every pytest invocation—regardless of success—the shared composite action at `.github/actions/collect-diagnostics` emits a top-level `SUMMARY.md` (covering run metadata, sanitized environment details, pytest results, resolver logs, and DuckDB table counts) and stages those text snapshots under `dist/diagnostics-<job>-<run_id>-<attempt>/`. Workflows upload that directory so GitHub produces a single archive per job with the summary baked in.
- When you need to guard a step on the presence of a secret, assign it to an environment variable first (for example, `env.MY_TOKEN: ${{ secrets.MY_TOKEN }}`) and reference `env.MY_TOKEN` inside the `if:` expression. GitHub blocks direct `secrets.*` lookups in `if:` conditions at workflow-parse time.
- Quote any `name:` values that contain colons or `${{ ... }}` expressions and keep heredoc terminators aligned with their `cat <<'TAG'` lines. The dedicated **CI Lint** workflow uses [`eifinger/actionlint-action@v1`](https://github.com/eifinger/actionlint-action) alongside a repository-level `.yamllint.yml` on every push and pull request to block malformed workflows before they reach the main pipelines. Workflow YAMLs must start with `---`, use consistent indentation/colons/trailing-space hygiene, and avoid bare truthy strings; local codex/unit test harnesses simply skip `yamllint` when the binary is unavailable instead of attempting a network install, and `resolver/tests/test_ci_yamllint_optional.py` enforces that guard without trying to `pip install` anything.
- For `actions/upload-artifact@v4`, pass `with.path` as a newline-delimited scalar instead of a YAML list. In Bash `run:` blocks, prefer `printf` when you need escape sequences, use explicit `if/else` conditionals instead of `A && B || C`, and keep spaces around `[[ ... ]]` tests so ShellCheck stays happy.
- Database-backed resolver jobs install the `db`/`test` extras (including DuckDB) up front so pytest can exercise the DuckDB suites, while `resolver/tests/conftest.py` lazily imports those helpers. Fast-test jobs therefore run without DuckDB present and simply skip fixtures that rely on it, yet still benefit from the mandatory DuckDB install so the same code path is exercised in both matrices.
- Resolver-specific diagnostics can be enabled in CI by exporting `RESOLVER_DEBUG=1`. When set, the resolver’s DuckDB selectors emit concise debug lines that show the backend, cutoff, `ym`, `iso3`, `hazard_code`, and row-count summaries during DB reads. The shared DuckDB connector also logs the canonicalised database path (for URLs like `duckdb:///tmp/db.duckdb`, `duckdb:////abs/path.duckdb`, relative file paths, or `:memory:`) so you can confirm reads and writes are pointed at the same file. The connector now self-heals cached handles: cache entries are evicted when `.close()` is called, stale connections are re-opened automatically, and cache hygiene is enforced at test boundaries as well as process exit. Set `RESOLVER_DISABLE_CONN_CACHE=1` to bypass caching entirely, or `RESOLVER_CONN_CACHE_MODE=thread` to scope caches to the current thread when running multithreaded diagnostics. `get_shared_duckdb_conn(db_url)` returns `(conn, resolved_path)` so tests can both use the live connection and assert which canonical DuckDB file is active. Snapshot writes also emit a one-line canonicalisation summary per table when `RESOLVER_DEBUG=1`, showing how messy semantics inputs were normalised before persistence.
- Database-backed resolver tests run twice in CI (with cache enabled and disabled) so regressions caused by cache state changes are caught immediately in both `.github/workflows/resolver-ci.yml` and `.github/workflows/resolver-ci-fast.yml`.
  - Cache-enabled runs expect repeated `duckdb_io.get_db(url)` calls to return the same connection object; cache-disabled runs (`RESOLVER_DISABLE_CONN_CACHE=1`) intentionally return fresh handles, and tests should branch accordingly.

## Go Live: Initial resolver backfill

- Navigate to **Actions → Resolver — Initial Backfill** and trigger the workflow (override `months_back` or `only` connector filter if you need a narrower run).
- The workflow executes the live connectors, freezes `resolver/snapshots/<YYYY-MM>/` for each of the requested months, and dual-writes into DuckDB so downstream selectors stay in sync.
- The final job calls `python -m resolver.tools.build_llm_context --months 12 --outdir context/` and uploads `context/facts_last12.jsonl` plus `context/facts_last12.parquet` alongside the snapshot artifacts.
- After the run completes download the combined artifact, confirm each `manifest.json` row count, and hand the bundle to the Forecaster deployment.

You can inspect or regenerate the bundle locally with the same helper:

```bash
python -m resolver.tools.build_llm_context --dry-run
# or write artifacts
python -m resolver.tools.build_llm_context --months 12 --outdir context/
```

### Resolver monthly pipeline

- Workflow: `.github/workflows/resolver-monthly.yml` runs on the first of every month at 02:00 Europe/Istanbul (and on manual dispatch) to ingest live connectors, freeze `resolver/snapshots/<YYYY-MM>/`, and rebuild the 12‑month LLM context bundle.
- Artifacts are uploaded to the run and attached to a GitHub Release tagged `resolver-<YYYY-MM>` so the latest snapshot and `context/` bundle stay available without manual steps.

If you fork the project under a different slug, update the guarded repository string inside the workflows before enabling CI.

### Resolver pipeline CI workflows

- `resolver-smoke.yml` runs on pushes, pull requests, and manual dispatches. It executes the full resolver pipeline end-to-end using offline stubs: `resolver.ingestion.ifrc_go_stub` seeds `data/staging/<period>/raw`, `resolver.transform.normalize` canonicalizes every adapter-backed CSV discovered there, and `scripts/ci/smoke_assert.py` tallies rows under `data/staging/<period>/canonical/*.csv`. The job passes when the summed total meets `SMOKE_MIN_ROWS` (default `1`), which keeps the smoke run focused on stub coverage rather than optional DuckDB or snapshot outputs. Once the gate passes, `resolver.tools.load_and_derive` runs the `load-canonical`, `derive-deltas --allow-negatives 1`, and `export --format parquet` subcommands to produce header-safe snapshots, while DuckDB files, logs, and snapshot directories remain optional so smoke runs stay green when those artifacts are absent. Any Parquet outputs that do appear are uploaded as the `resolver-smoke-snapshots` artifact so PRs can still validate the wiring without any network access. See `docs/CI.md` for environment knobs and diagnostics details.
- `resolver-nightly.yml` is scheduled for 06:00 UTC and available on demand. It restricts `run_all_stubs.py --mode real` to the small-window connectors we keep secrets for (currently IFRC GO and UNHCR ODP), computes a 14-day ISO window via `RESOLVER_START_ISO`/`RESOLVER_END_ISO`, and reuses the same normalize → load/derive pipeline plus an artifact upload. Nightly logs include per-Parquet row counts, making it easy to confirm when live data flowed (non-zero rows) or when only headers were produced.

### Workflow YAML & CLI

Global CLI flags such as `--period` must appear **before** the subcommand when invoking `resolver.tools.load_and_derive`. Every workflow follows the same ordering to avoid argparse complaints:

```bash
python -m resolver.tools.load_and_derive --period "${PERIOD_LABEL}" load-canonical \
  --in "data/staging/${PERIOD_LABEL}/canonical" \
  --db "data/resolver.duckdb"

python -m resolver.tools.load_and_derive --period "${PERIOD_LABEL}" derive-deltas \
  --allow-negatives 1 \
  --db "data/resolver.duckdb"

python -m resolver.tools.load_and_derive --period "${PERIOD_LABEL}" export \
  --format parquet --out "data/snapshots/${PERIOD_LABEL}" \
  --db "data/resolver.duckdb"


You can reproduce the diagnostics locally by running the same helpers; for example:

```bash
scripts/ci/run_and_capture.sh demo-load "python -m resolver.tools.load_and_derive --help"
python scripts/ci/list_canonical.py --dir data/staging/ci-smoke/canonical --out .ci/diagnostics/canonical-listing.txt
python scripts/ci/db_counts.py --db data/resolver.duckdb --out .ci/diagnostics/duckdb-counts.txt
```

All workflows upload `.ci/diagnostics/**` and `.ci/exitcodes/**` so the aggregated `SUMMARY.md` produced by `.github/actions/collect-diagnostics` has consistent log attachments and exit breadcrumbs.

Workflow YAML is linted on every push and pull request by `.github/workflows/lint.yml`, which shells out to the repo helper:

```bash
python scripts/ci/validate_yaml.py .github/workflows/*.yml
```

Run the command locally before pushing when editing workflows to surface parser errors (line/column) without waiting for CI.

### Testing

- `pytest` runs happily in both single-process and multi-process modes. Passing `-n auto` asks [`pytest-xdist`](https://pypi.org/project/pytest-xdist/) to parallelise across every available core, but when that plugin is missing (for example inside Codex sandboxes that block network installs) the repo-level `conftest.py` prints a short note and silently drops the flag so tests fall back to a single worker.
- Commands such as `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -n auto resolver/ingestion/tests/test_run_all_stubs_enablement.py` therefore succeed even if `pytest-xdist` cannot be imported.
- **Runner exit policy:** the ingestion runner exits with code `0` whenever at least one connector runs _or_ every connector was intentionally skipped (for example via `enable: false` in config or a CI-only disable). Missing secrets, unexpected skips, or connector crashes still emit a non-zero exit, and the summary table now logs per-status counts plus a short explanation of the final decision.

### Fast-test fixtures

Fast tests no longer rely on ad-hoc exports in the repository tree. During collection the `fast_exports` session fixture calls `resolver.tests.fixtures.bootstrap_fast_exports.build_fast_exports()` to copy the minimal canonical CSV under `resolver/tests/fixtures/staging/minimal/`, run `resolver.tools.load_and_derive` offline, and mirror the Parquet exports to lightweight CSVs. The fixture sets `RESOLVER_DB_PATH`, `RESOLVER_SNAPSHOTS_DIR`, and `RESOLVER_TEST_DATA_DIR` so contract tests exercise both the DuckDB and files/csv backends with the same data. A companion `fast_staging_dir` fixture points schema tests at the generated canonical directory, which means the previously skipped codex contract and staging-schema suites now run end-to-end in `resolver-ci-fast` without reaching out to staging buckets.

### Resolver diagnostics

Set `RESOLVER_DIAG=1` to emit JSON-formatted diagnostics for DuckDB reads and writes. When enabled, the resolver logs the source file path, Python/DuckDB versions, cache resolution events, schema/index snapshots, input column samples, and post-write row counts before each upsert. CLI reads that fail to find data record the resolved database path and row-count summaries (total rows, keyed rows, and cutoff-filtered rows) to stderr so you can distinguish empty writes from URL mismatches, and they now emit targeted DEBUG counts (total rows, rows matching the `(ym, iso3, hazard_code)` key, and rows surviving the cutoff filter) immediately before the CLI exits. When an upsert detects a declared-key mismatch, diagnostics now include the discovered unique indexes **with canonical column order** so you can confirm DuckDB has materialised the expected `ux_*` definitions, and schema initialisation logs the entire unique-index inventory for both facts tables. The flag leaves default logging untouched when unset, making it safe to toggle on only for triage runs or CI retries.

### Ingestion diagnostics

![Example connector diagnostics summary table](docs/images/ingestion-summary.svg)

- Every live ingestion run writes structured connector telemetry to `diagnostics/ingestion/connectors_report.jsonl`. Each line captures the connector id, mode (`real` vs `stub`), status, retry counts, HTTP tallies, row totals, coverage windows, and the top ISO3/hazard samples pulled from the output CSV.
- After `run_all_stubs.py` finishes, `python -m scripts.ci.summarize_connectors --report diagnostics/ingestion/connectors_report.jsonl --out diagnostics/ingestion/summary.md --github-step-summary` renders a Markdown dashboard. GitHub Actions surfaces that table directly in the job summary so you can spot failed connectors, zero-row exports, or spikes in `4xx/5xx` responses without downloading logs.
- The workflow also publishes a `connector-diagnostics` artifact containing both `diagnostics/ingestion/summary.md` (human-readable) and the raw `diagnostics/ingestion/connectors_report.jsonl` (machine-readable). Use the Markdown for quick triage and feed the JSONL into ad-hoc notebooks when you need to chart retries, coverage drift, or rate-limit exhaustion across runs.
- Reasons such as `disabled: config`, `missing secret`, or upstream HTTP errors are normalised in the summary so repeated skips bubble to the top of the reason histogram. Coverage columns render the year-month window (`ym_min → ym_max`) alongside the earliest/latest `as_of` dates, making it easy to confirm a connector backfilled the intended period.

### CI diagnostics & what to share

Every resolver-facing workflow (`resolver-ci-fast`, `resolver-ci`, and `resolver-smoke`) now calls the reusable diagnostics collector so each job publishes a single archive named `diagnostics-<job>-<run_id>-<run_attempt>.zip` (GitHub zips the uploaded directory automatically). Inside that archive you will find:

- `SUMMARY.md`, a human-friendly digest with Git metadata, versions, sanitized environment variables, staging quick stats (rows, headers, and samples for each CSV), DuckDB table counts or "database not found" breadcrumbs, pytest results, resolver log tails, and exit-code breadcrumbs from the workflow (including `run_stubs` and `gate_rows` for smoke). Command tails for each `scripts/ci/run_and_capture.sh` step are inlined so you can see the final 120 lines of `pip freeze`, import probes, and pytest without unzipping the artifact, and a disk-usage snapshot caps the report.
- Supporting text snapshots (`versions.txt`, `env.txt`, `git.txt`, `staging.txt`, `snapshots.txt`, `duckdb.txt`, `pytest.txt`, and more) for deeper triage when needed.

In practice you will see `diagnostics-fast-tests-<run>-<attempt>.zip` from the fast matrix plus `diagnostics-pipeline-smoke-<run>-<attempt>.zip` from the stubbed pipeline. Download the archive and open `SUMMARY.md` for the quick triage path; the aggregate workflow still merges per-job summaries but writes the consolidated markdown straight to the GitHub Actions step summary so the “Artifacts” list remains predictable.

Smoke runs always place the staged diagnostics under `dist/diagnostics-pipeline-smoke-<run>-<attempt>/` before upload, and the bundle also contains `smoke-assert.json`. The totals in that JSON match the `SUMMARY.md` canonical row totals so the gating step, diagnostics report, and uploaded archive stay in lockstep.

### Working behind a proxy

GitHub Actions and the devcontainer bootstrap understand constrained networks. The `resolver-ci-fast` workflow’s DuckDB job first
tries an editable install with `poetry-core` preloaded and build isolation disabled so that proxy-friendly indexes (surfaced via
`PIP_INDEX_URL`/`PIP_EXTRA_INDEX_URL` or `HTTP_PROXY`/`HTTPS_PROXY` secrets) are respected. If that still fails, the job falls
back to `requirements*.txt` dependencies, adds the repo to `PYTHONPATH`, and continues running the database tests. Locally, the
`.devcontainer/postCreate.sh` script mirrors the same two-phase approach and automatically appends the workspace path to
`PYTHONPATH` whenever an editable install is blocked.

How it works (conceptually)

Understand the question

Forecaster reads a forecasting question (binary “yes/no,” numeric, or multiple-choice).

It builds a tailored prompt to guide models toward giving a probability or range.

Gather research

It can query external news/search APIs (AskNews, Serper) to pull in up-to-date information.

This gives context to the models so they are not just reasoning in a vacuum.

Ask multiple models

Several large language models (like GPT-4o, Claude, Gemini, Grok) are asked the same question.

Each model returns its best estimate (e.g., “There’s a 35% chance…”).

For strategic, political questions, Forecaster can also run a game-theoretic simulation (GTMC1) that models actors, their interests, and possible bargaining outcomes.

Combine forecasts

All these signals are combined using a Bayesian Monte Carlo method.

This ensures the final probability or distribution is mathematically consistent and reflects the strength of evidence from each source.

Calibrate and improve

Forecaster logs every forecast in a CSV (configurable via FORECASTS_CSV_PATH; default is forecast_logs/forecasts.csv).

When questions resolve, it compares its predictions to reality.

A calibration script then adjusts the weights of different models, giving more influence to those that proved more accurate in similar situations.

This creates a closed learning loop so Forecaster improves over time.

Autonomous operation

In GitHub, Forecaster runs on a schedule (for test questions, the tournament, and the Metaculus Cup).

It automatically commits its logs and calibration updates back to the repository, so no human intervention is needed.

The big picture

Think of Forecaster as a forecasting orchestra conductor:

The models are the musicians, each with their own instrument (style of reasoning).

GTMC1 is the percussion section for power politics.

Bayesian Monte Carlo is the conductor, blending them into harmony.

And the calibration loop is like rehearsals, helping everyone play in tune with reality over time.

## Architecture & How It Works

See [CODEMAP.md](CODEMAP.md) for resolver architecture, data flow diagrams, entrypoints, and runbooks.

Quick Start

Local run (one question / test mode)

(Recommended) use Python 3.11 and Poetry

poetry install

Environment variables (see “Configuration” below)
The only must-have for submission is METACULUS_TOKEN

export METACULUS_TOKEN=your_token_here

Run a single test question (no submit)

poetry run python run_forecaster.py --mode test_questions --limit 1

Run a single question by post ID (no submit)

poetry run python run_forecaster.py --pid 22427

Submit (be careful!)

poetry run python run_forecaster.py --mode test_questions --limit 1 --submit

Autonomous runs on GitHub

This repo includes workflows for:

test_bot_fresh.yaml (fresh research),

test_bot_cached.yaml (use cache),

run_bot_on_tournament.yaml,

run_bot_on_metaculus_cup.yaml,

> **Note:** Metaculus renames the Cup slug every season. Update the `TOURNAMENT_ID`
> in `.github/workflows/run_bot_on_metaculus_cup.yaml` to match the current Cup
> URL (presently `metaculus-cup-fall-2025`).

calibration_refresh.yml (periodic update of calibration weights).

Add secrets (see below), push the repo, and Actions will:

run forecasts,

write logs to the repo,

update calibration weights on a schedule,

commit/push changes automatically.

### Live Mode Secrets

Repository connectors only run against live APIs when the expected credentials are present. Add repository secrets under **Settings → Secrets and variables → Actions → New repository secret** for each entry in the table below.【F:README.md†L624-L624】 The same names can be exported locally (for example via `direnv` or `.env`) to reproduce CI runs.

| Connector | Requires secret? | Secret name(s) | Notes |
|---|---|---|---|
| ACLED | Yes | `ACLED_REFRESH_TOKEN` (or `ACLED_TOKEN` / `ACLED_USERNAME`+`ACLED_PASSWORD`) | Requests a 450‑day window at 1,000 rows per call and the runner skips when no ACLED credentials are set.【F:resolver/ingestion/config/acled.yml†L1-L5】【F:resolver/ingestion/run_all_stubs.py†L268-L277】 |
| IFRC GO | Optional | `GO_API_TOKEN` | Covers the last 45 days of Field Reports, Appeals, and Situation Reports; the token header only matters for authenticated portals or higher-rate bursts.【F:resolver/ingestion/config/ifrc_go.yml†L2-L17】【F:resolver/ingestion/ifrc_go_client.py†L5-L14】 |
| ReliefWeb | Optional | `RELIEFWEB_APPNAME` | Uses 0.6 s pauses between 100-row pages and falls back to the configured app name when the secret is missing.【F:resolver/ingestion/config/reliefweb.yml†L20-L25】【F:resolver/ingestion/reliefweb_client.py†L1211-L1221】 |
| UNHCR ODP | Yes | `UNHCR_ODP_USERNAME`, `UNHCR_ODP_PASSWORD`, `UNHCR_ODP_CLIENT_ID`, `UNHCR_ODP_CLIENT_SECRET` | Scrapes monthly Operational Data Portal widgets (max one page) and only runs when the full credential set is available.【F:resolver/ingestion/config/unhcr_odp.yml†L2-L11】【F:resolver/ingestion/run_all_stubs.py†L278-L291】 |
| All other live connectors | No | – | DTM, EM-DAT, GDACS, HDX, IPC, UNHCR population, WFP mVAM, WHO PHE, and WorldPop rely on configuration only; they skip when disabled in YAML but do not require secrets.【F:resolver/ingestion/config/dtm.yml†L2-L6】【F:resolver/ingestion/config/emdat.yml†L2-L9】【F:resolver/ingestion/config/gdacs.yml†L2-L8】【F:resolver/ingestion/config/hdx.yml†L1-L18】【F:resolver/ingestion/config/ipc.yml†L1-L5】【F:resolver/ingestion/config/unhcr.yml†L2-L19】【F:resolver/ingestion/config/wfp_mvam_sources.yml†L1-L3】【F:resolver/ingestion/config/who_phe.yml†L1-L9】【F:resolver/ingestion/config/worldpop.yml†L1-L14】 |

Resolver artifact hygiene

Nightly resolver runs can generate sizeable exports. The CI workflow calls
`resolver/tools/check_sizes.py` to warn when CSV/Parquet artifacts cross the
recommended 25 MB / 150 MB thresholds and to fail if the tracked repo contents
grow beyond ~2 GB. Temporary scratch files (`resolver/tmp/`,
`resolver/exports/*_working.*`, etc.) stay out of Git history via `.gitignore`.

### ReliefWeb connector field allowlist

The ReliefWeb API client now builds its `fields` parameter from
`resolver/ingestion/config/reliefweb_fields.yml`. That allowlist intentionally
omits the deprecated `type` include after ReliefWeb started returning HTTP 400
errors for it. When the API responds with "Unrecognized field" the connector
logs the response, prunes the unsupported entry, retries once with the safe
allowlist, and still emits the canonical header-only CSV when no rows survive.

### Staging output directory controls

The ingestion runner now writes every connector CSV beneath a single staging
root driven by environment variables. Set `RESOLVER_STAGING_DIR` to the parent
directory (for example, `data/staging`) and the runner will emit files under
`$RESOLVER_STAGING_DIR/<period>/raw/`. Provide `RESOLVER_PERIOD` for a fixed
label or rely on `RESOLVER_START_ISO`/`RESOLVER_END_ISO` to derive a quarter- or
date-span based folder (falling back to `unknown`). When unset, outputs continue
to land in `resolver/staging/` for backwards compatibility. Example:

```bash
export RESOLVER_STAGING_DIR=$(pwd)/data/staging
export RESOLVER_START_ISO=2025-07-01
export RESOLVER_END_ISO=2025-09-30
python resolver/ingestion/run_all_stubs.py --connector who_phe_client.py
# writes data/staging/2025Q3/raw/who_phe.csv
```

All ingestion connectors emit a header-only CSV when no rows survive filtering
so downstream stages always find the expected schema.

### Connector enablement & overrides

Each connector YAML under `resolver/ingestion/config/` should expose an
`enabled` flag (the runner still honours the legacy `enable` key for
backwards compatibility). When neither key is present, the connector is treated
as disabled. The effective precedence is:

1. A comma-separated `RESOLVER_FORCE_ENABLE` list of connector names. Entries
   must match the canonical name or filename stem — no wildcards are honoured.
2. The connector configuration flag (`enabled` or legacy `enable`).
3. CI guardrails such as `RESOLVER_SKIP_IFRCGO=1`.

When you run in a dedicated mode, the module-level lists become authoritative:

- `--mode real` (or `RESOLVER_INGESTION_MODE=real`) honours the `REAL` list. Any
  connector explicitly listed there runs even if its config file is missing or
  sets `enabled: false`. The planner records `gated_by=selected:list origin=real_list`
  and the summary table tags the row with
  `notes=selected:list` to make the override explicit.
- `--mode stubs` does the same for the `STUBS` list (`origin=stub_list`).

Connectors that originate from config discovery continue to respect the YAML
flag unless they are forced via environment or CLI arguments.

To temporarily bypass the YAML flag without editing files, set
`RESOLVER_FORCE_ENABLE` to the canonical connector name(s):

```bash
RESOLVER_FORCE_ENABLE=acled,reliefweb python resolver/ingestion/run_all_stubs.py
```

The planner emits both a structured decision line and a legacy
`enable=<bool> gated_by=<reason>` log for every connector. The structured entry
now includes an `origin=` field (`real_list`, `stub_list`, or `config`) so you
can distinguish authoritative picks from config-driven ones. Disabled
connectors log `enable=False gated_by=config`, appear in the summary as
`status=skipped notes=disabled: config`, and never launch their subprocess. A
forced run produces `enable=True gated_by=forced_by_env`, still records a
structured line with `forced_by=env`, and tags the summary with
`notes=forced_by_env`.

Real-mode runs are strict by default: when `--mode real` (or the matching env)
is active, any connector failure returns a non-zero exit code (the same outcome
as passing `--strict`). Other modes retain the existing behaviour unless you
explicitly add `--strict`.

### Canonical normalization

Use the canonical normalizer to transform raw staging CSVs into the 12-column schema shared by downstream Resolver tooling. The CLI accepts the raw directory, an output directory for canonical CSVs, the optional staging period label, and a comma-separated source list. Example:

```bash
python -m resolver.transform.normalize \n  --in data/staging/2025Q3/raw \n  --out data/staging/2025Q3/canonical \n  --period 2025Q3 \n  --sources ifrc_go
```

Each adapter snaps `as_of_date` to the end of the month, filters rows outside the optional `RESOLVER_START_ISO`/`RESOLVER_END_ISO` window, coerces numeric values, and writes exactly 12 headers (`event_id` … `source`). Empty results still emit a header-only CSV so follow-on jobs can rely on the canonical directory layout.

### Load, derive, and export in one step

Once canonical CSVs exist for a staging period, run the bundled loader CLI to populate DuckDB and emit matching Parquet snapshots:

```bash
python -m resolver.tools.load_and_derive \
  --period 2025Q3 \
  --staging-root data/staging \
  --snapshots-root data/snapshots \
  --db ${RESOLVER_DB_URL:-resolver/db/resolver.duckdb} \
  --allow-negatives 1
```

The command:

- Inserts the 12 canonical columns into `facts_raw` and routes `stock` rows into `facts_resolved` (with `new` rows landing in `facts_deltas`).
- Applies the source-priority resolver so overlapping monthly stock rows collapse to a single winner with provenance (`provenance_source`/`provenance_rank`).
- Derives monthly `value_new` deltas from the cumulative stock history for each `(iso3, hazard_code, metric, unit, source)` grouping—preserving negative swings when `--allow-negatives` is `1` (the default).
- Writes `facts_resolved.parquet` and `facts_deltas.parquet` to `data/snapshots/<period>/`, mirroring the DuckDB rows for the requested quarter.

The CLI logs row counts per table along the way so you can confirm non-empty loads before inspecting the exported artifacts.

Source precedence is controlled via [`resolver/ingestion/config/source_priority.yml`](resolver/ingestion/config/source_priority.yml).
Higher-ranked identifiers win ties, and the resolver records the chosen source and its 1-based rank in the
`provenance_source`/`provenance_rank` columns on `facts_resolved`.

### Resolver DuckDB dual-write

Set the `RESOLVER_DB_URL` environment variable to enable database-backed parity
for resolver exports. When the variable is present (for example,
`RESOLVER_DB_URL=duckdb:///resolver/db/resolver.duckdb`),
`resolver/tools/export_facts.py` mirrors its CSV/Parquet outputs into the
`facts_resolved` (and, when supplied, `facts_deltas`) tables inside DuckDB.
`resolver/tools/freeze_snapshot.py` then wraps snapshot freezes in a single
transaction, recording the resolved totals, monthly deltas, manifest metadata,
and a corresponding `snapshots` row. Leave `RESOLVER_DB_URL` unset to continue
operating in file-only mode; all tooling falls back automatically when the
variable is absent.

#### Development tips

- Review [resolver/docs/db_upsert_strategy.md](resolver/docs/db_upsert_strategy.md)
  for details on the DuckDB MERGE-first upsert strategy.
- Set `RESOLVER_DUCKDB_DISABLE_MERGE=1` to force the legacy delete+insert path,
  which is useful for CI stability or when diagnosing upserts locally.

The exporter writes the very same dataframe that is saved to `facts.csv` into
DuckDB to guarantee row-for-row parity. During these writes the DuckDB helper
aliases staging-era columns such as `series_semantics_out` to the canonical
`series_semantics` field (and lets the alias win whenever the canonical column
is blank) while silently dropping any unexpected debug columns so upserts stay
resilient when schemas drift.

Series semantics are normalised by
[`resolver.common.series_semantics.compute_series_semantics`](resolver/common/series_semantics.py),
which draws from [`resolver/config/series_semantics.yml`](resolver/config/series_semantics.yml).
Writers canonicalise the `series_semantics` column to `"new"` or `"stock"`,
mapping blanks, synonyms, or unexpected text to table-specific defaults, and
enforce key hygiene by rejecting malformed `ym` values (`YYYY-MM` only) while
uppercasing `iso3` and `hazard_code`. Legacy exports may still populate the
historical `series` column, and DuckDB readers automatically coalesce
`series_semantics` with `series` so backfilled rows remain queryable during the
migration window.

- DDL helpers wrap schema statements in a `SAVEPOINT` whenever an outer
  transaction is already active, releasing or rolling back cleanly before
  returning. When no transaction is open, they fall back to `BEGIN`/`COMMIT` to
  preserve the previous behaviour without tripping nested-transaction errors.
- Frame-level `series_semantics` canonicalisation coerces unknown values to
  blanks and fills them with the table default (`"new"` for deltas,
  `"stock"` for snapshots) before validation, so stray text never leaks past
  `_assert_semantics_required`.

### DuckDB setup for local development

Follow [resolver/docs/DUCKDB.md](resolver/docs/DUCKDB.md) to install DuckDB and
run the DuckDB-specific test suite locally. The same commands ensure
`python -c "import duckdb; print(duckdb.__version__)"` works before running
`pytest -q resolver/tests -k duckdb`.

> Running the fast test target inside Codex or on a fresh clone without the
> large fixture checkout? See
> [docs/testing-without-fixtures.md](docs/testing-without-fixtures.md) for the
> synthetic fixture helpers that keep parity/staging suites green locally.

### DB backend tests

You can exercise the DuckDB-backed contract test either completely offline or
through a corporate proxy.

#### Offline workflow (recommended for blocked networks)

```bash
# One-time on a machine with internet access (skip if wheels already tracked)
python scripts/download_db_wheels.py
git add tools/offline_wheels/*.whl
git commit -m "chore: refresh offline duckdb wheels"

# On the offline/proxied machine
scripts/install_db_extra_offline.sh
export RESOLVER_DB_URL='duckdb:///resolver.duckdb'
export RESOLVER_API_BACKEND='db'
pytest -q resolver/tests/test_db_query_contract.py
```

On Windows PowerShell use the `.ps1` installer and `set`-style environment
variables:

```powershell
# Refresh wheels on a machine with internet (skip if wheels already tracked)
python scripts/download_db_wheels.py
git add tools/offline_wheels/*.whl
git commit -m "chore: refresh offline duckdb wheels"

# Install and run tests offline
scripts/install_db_extra_offline.ps1
$env:RESOLVER_DB_URL = 'duckdb:///resolver.duckdb'
$env:RESOLVER_API_BACKEND = 'db'
pytest -q resolver/tests/test_db_query_contract.py
```

#### Proxy-based install (if allowed)

Configure proxy variables before calling pip:

```bash
export HTTPS_PROXY="http://user:pass@proxy.host:port"
export HTTP_PROXY="http://user:pass@proxy.host:port"
python -m pip install duckdb pytest
```

```powershell
$env:HTTPS_PROXY = 'http://user:pass@proxy.host:port'
$env:HTTP_PROXY = 'http://user:pass@proxy.host:port'
python -m pip install duckdb pytest
```

You can also persist the proxy settings via pip configuration files, e.g.
`~/.pip/pip.conf` on Unix-like systems or `%APPDATA%\pip\pip.ini` on Windows:

```
[global]
proxy = http://user:pass@proxy.host:port
```

#### Run DB parity test in a Dev Container

- Open the repository in VS Code and choose **Reopen in Container** (requires the Dev Containers extension).
- During container creation, `.devcontainer/postCreate.sh` runs and installs DuckDB with the same interpreter that VS Code uses. It tries the vendored wheels in [`tools/offline_wheels/`](tools/offline_wheels/README.md) first and falls back to the online extras if needed. The setup logs should print `duckdb installed: <version>`.
- Once the container is ready, run `make test-db` to execute the parity test.
- Expected outcome: the test runs (not skipped). If it still skips, open a terminal and run:
  ```
  make which-python
  python -c "import sys; print(sys.executable)"
  python -c "import duckdb; print(duckdb.__version__)"
  ```
  Ensure the interpreter path matches `/usr/local/bin/python`. If it does not, update `PY_BIN` in `.devcontainer/devcontainer.json`, rebuild the container, and reopen it.

#### Debugging DB parity failures

- Set `RESOLVER_LOG_LEVEL=DEBUG` before running the exporter or tests to surface structured logs. The exporter and DuckDB writer print dataframe schemas, dropped columns, and `series_semantics` distributions.
- When `resolver/tests/test_db_parity.py` fails, it emits a `DB PARITY DIAGNOSTICS` block showing schema summaries and anti-join counts. The test also writes CSV artifacts (`parity_mismatch_expected.csv`, `parity_mismatch_db.csv`, and `parity_mismatch_cell_diffs.csv` when applicable) that CI uploads automatically under the `db-parity-diagnostics` artifact.
- CI sanitises DuckDB matrix labels (replacing `*`, spaces, and other punctuation with underscores) before naming debug artifacts and appends the job name plus `run_attempt`. Uploads run with `overwrite: true`, so retries replace prior diagnostics instead of failing with a 409 conflict.
- Each workflow prints the merge commit SHA, PR head SHA (when available), and the repo file that `resolver.db.duckdb_io` was imported from. CI uploads the exact `duckdb_io.py` used for the run and records its SHA-256 hash so any drift between checkout and execution is visible in the logs.
- DuckDB 0.10.x and later are both supported in CI; `init_schema` first attempts to add primary-key constraints and gracefully falls back to creating unique indexes when older engines lack `ALTER TABLE ... ADD PRIMARY KEY` support.
- Inspect those CSVs locally or download them from the GitHub Actions job to understand which rows or columns diverged between the CSV export and DuckDB snapshot.
- DDL helpers wrap operations in nest-safe transactions: they attempt a `SAVEPOINT` when another transaction is already in progress and fall back to `BEGIN/COMMIT` otherwise. On any failure we roll back to the savepoint (or the full transaction) so DuckDB connections never remain in the aborted state after schema updates.

What Forecaster Does (Pipeline)

Select question(s)
Test set, single PID, tournament, or Metaculus Cup.

Research (research.py)
Uses AskNews / Serper to pull context. Results are summarized and logged.

Prompting (prompts.py)
Builds compact, question-type aware prompts (binary, MCQ, numeric).

LLM Ensemble (providers.py, ensemble.py)
Calls multiple LLMs asynchronously, with rate-limit guards and usage/cost capture.

Binary → extracts a single probability p(yes)

MCQ → extracts a probability vector across options

Numeric → extracts P10/P50/P90 quantiles

Game-Theoretic Signal (optional, binary) — GTMC1.py
A reproducible, Bueno de Mesquita/Scholz-style bargaining Monte Carlo using an actor table (position, capability, salience, risk). Output includes:

exceedance_ge_50 (preferred probability-like signal),

coalition rate, dispersion, rounds to converge, etc.

Aggregation (aggregate.py + bayes_mc.py)
Bayesian Monte Carlo fusion of all evidence (LLMs + GTMC1).

Binary: Beta-Binomial update → final p(yes).

MCQ: Dirichlet update → final probability vector.

Numeric: Mixture from quantiles → final P10/P50/P90.

Submission (cli.py)
Submits to Metaculus if --submit is set and METACULUS_TOKEN is present.
Enforces basic constraints (e.g., numeric quantile ordering).

Logging (io_logs.py)

Machine-readable master CSV: configurable (FORECASTS_CSV_PATH, default forecast_logs/forecasts.csv)

Human log per run: forecast_logs/runs/<run_id>.md (or .txt)
In GitHub Actions, logs auto-commit/push unless disabled.

Calibration Loop (update_calibration.py)
On a schedule (e.g., every 6–24h), reads the forecast CSV, filters resolved questions (and only forecasts made before resolution), computes per-model skill and writes:

calibration_weights.json (used next run to weight models)

data/calibration_advice.txt (human-readable notes)
These outputs are committed, so the loop closes autonomously.

Repository Layout (key files)
forecaster/
run_forecaster.py # Thin wrapper: calls cli.main()
cli.py # Orchestrates runs, submission, and final log commit
research.py # AskNews/Serper search + summarization
prompts.py # Prompt builders per question type
providers.py # Model registry, rate-limiters, cost estimators
ensemble.py # Async calls to each LLM + parsing to structured outputs
aggregate.py # Calls bayes_mc to fuse LLMs (+ GTMC1) into final forecast
bayes_mc.py # Bayesian Monte Carlo updaters (Binary, MCQ, Numeric)
GTMC1.py # Game-theoretic Monte Carlo with reproducibility guardrails
io_logs.py # Canonical logging to forecast_logs/ + auto-commit
seen_guard.py # Duplicate-protection state & helpers
update_calibration.py # Builds class-conditional weights from resolutions
init.py # Package export list

forecast_logs/
forecasts.csv # If using the default path (or use root-level forecasts.csv via env)
runs/ # Human-readable per-run logs (.md or .txt)
state/ # Optional: duplicate-protection state if SEEN_GUARD_PATH points here

data/
calibration_advice.txt # Friendly notes from the calibration job (auto-created)

gtmc_logs/
... # Optional run-by-run GTMC CSVs/metadata (if enabled)

Configuration

Set these as environment variables (locally or in GitHub Actions). Sensible defaults are used where possible.

Required (for submission)

METACULUS_TOKEN — your Metaculus API token.

Recommended

FORECASTS_CSV_PATH # default: forecast_logs/forecasts.csv (you can set it to "forecasts.csv" at repo root)

CALIB_WEIGHTS_PATH=calibration_weights.json

CALIB_ADVICE_PATH=data/calibration_advice.txt

HUMAN_LOG_EXT=md (or txt)

LOGS_BASE_DIR=forecast_logs (default)

Duplicate protection (persist across CI runs)

SEEN_GUARD_PATH=forecast_logs/state/seen_forecasts.jsonl

Git commit behavior (logs & calibration outputs)

In GitHub Actions:

DISABLE_GIT_LOGS=false (or unset) → auto-commit/push logs & calibration outputs

Locally:

COMMIT_LOGS=true → will also commit/push from your machine (optional)

Optional:

GIT_LOG_MESSAGE="chore(logs): update forecasts & calibration"

GIT_REMOTE_NAME=origin

GIT_BRANCH_NAME=main (detected automatically; this is a safe fallback)

LLM/API keys (if used in your setup)

OPENROUTER_API_KEY, GOOGLE_API_KEY, XAI_API_KEY, etc.

Research keys: ASKNEWS_CLIENT_ID, ASKNEWS_CLIENT_SECRET, SERPER_API_KEY, etc.

Tip: Put non-secret defaults in .env.example and store secrets in GitHub → Settings → Secrets and variables → Actions.

Running Modes

Test set (no submit by default):
poetry run python run_forecaster.py --mode test_questions --limit 4

Single question by PID:
poetry run python run_forecaster.py --pid 22427

Tournament / Cup (CI workflows call these internally):

run_bot_on_tournament.yaml

run_bot_on_metaculus_cup.yaml
Include --submit in those workflows if you want automatic submissions.

Logging & Files Written

Machine CSV: configurable (FORECASTS_CSV_PATH, default forecast_logs/forecasts.csv)
Contains per-question details, per-model parsed outputs, final aggregation, timestamps, etc.

Human log: forecast_logs/runs/<timestamp>_<run_id>.md
Friendly narrative summary: which models ran, research snippets, final forecast.

These are committed automatically from GitHub Actions unless disabled.

Calibration (Autonomous Loop)

Input: the forecast CSV

Logic:

Keep resolved questions only.

Keep only forecasts made before resolution time.

Compute per-model loss by question type and class (e.g., binary vs numeric; topical classes if present).

Build class-conditional weights with shrinkage toward global weights.

Output:

calibration_weights.json

data/calibration_advice.txt (friendly notes)

Use: At the start of a run, Forecaster loads calibration_weights.json to weight ensemble members.

Important: Ensure the data/ folder exists in the repo so calibration_advice.txt can be written and committed.

Game-Theoretic Module (GTMC1)

Accepts an actor table with columns: name, position, capability, salience, risk_threshold (scales 0–100 for the first three).

Runs a deterministic Monte Carlo bargaining process (PCG64 RNG seeded from a fingerprint of the table + params).

Outputs:

exceedance_ge_50 (preferred probability-like signal when higher axis = “YES”),

median_of_final_medians,

coalition_rate,

dispersion,

and logs a per-run CSV under gtmc_logs/ (optional to commit).

You can enable/disable GTMC1 by question type or flags in cli.py.

Aggregation Math (Plain English)

Binary: Treat each model’s p(yes) (and GTMC1’s exceedance) as evidence with a confidence weight. Update a Beta prior; take the posterior mean as final probability (with quantile summaries P10/P50/P90).

MCQ: Treat each model’s probability vector as evidence and update a Dirichlet prior; take the posterior mean vector.

Numeric: Fit a Normal from each model’s P10/P50/P90, sample proportionally to confidence weights, and compute final P10/P50/P90 from the mixture.

Weights come from calibration_weights.json, updated by the calibration job.

GitHub Actions (What’s Included)

Fresh test runs vs cached test runs

Tournament and Metaculus Cup runners

Calibration refresh (scheduled)

Each workflow:

Sets the environment (paths, commit behavior),

Runs the bot (and/or update_calibration.py),

Auto-commits any modified files under:

forecast_logs/,

calibration_weights.json,

data/ (advice),

state/ (duplicate-protection, if enabled),

(optionally) other state you decide to persist.

Secrets You Must Add (GitHub → Actions)

METACULUS_TOKEN (for submission)

Any LLM / research API keys you plan to use:

OPENROUTER_API_KEY, GOOGLE_API_KEY, XAI_API_KEY, ASKNEWS_CLIENT_ID, ASKNEWS_CLIENT_SECRET, SERPER_API_KEY, etc.

Troubleshooting

“No calibration changes to commit.”
Normal if no new resolutions or no forecasts before the resolution time.

Numeric submission errors (CDF monotonicity etc.).
The numeric path enforces ordered quantiles (P10 ≤ P50 ≤ P90). If Metaculus still complains, reduce extremely steep distributions (too tiny sigma) or clip the extreme tails. (The current code already guards the common pitfalls.)

Logs not committing locally.
Set COMMIT_LOGS=true and ensure your git remote/branch are correct.

Advice file write error.
Ensure data/ folder exists and isn’t .gitignore’d.

Duplicate forecasts.
Make sure SEEN_GUARD_PATH is set to a path inside the repo (e.g., forecast_logs/state/seen_forecasts.jsonl) and that your workflow imports filter_post_ids/assert_not_seen/mark_post_seen successfully (see the compat layer in seen_guard.py above).

Metaculus API call hangs for minutes.

DuckDB connection inconsistencies or transaction errors.
- Always open databases through `resolver.db.duckdb_io.get_db(...)`. The helper canonicalises both filesystem paths and `duckdb:///` URLs (resolving relative paths against the current working directory and collapsing symlinks) so every caller reaches the same on-disk file and the shared connection cache can be reused.
- When both a `RESOLVER_DB_URL` environment variable and an explicit argument are provided, the explicit path wins. This prevents surprising cross-environment writes; a warning is logged the first time we override the environment value.
- Schema initialisation is wrapped in a Python-managed transaction. Do not embed `BEGIN`/`COMMIT` in schema snippets you pass to `duckdb_io.init_schema`; the helper strips those statements and handles rollbacks on errors for you.

