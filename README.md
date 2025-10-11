README (updated)
[![resolver-ci-fast](https://github.com/oughtinc/Pythia/actions/workflows/resolver-ci-fast.yml/badge.svg?branch=main)](https://github.com/oughtinc/Pythia/actions/workflows/resolver-ci-fast.yml)

What is Forecaster?

Forecaster is an AI forecasting assistant. Its job is to read forecasting questions (like those on Metaculus), gather relevant evidence, and combine the judgments of different models into one final forecast. It is designed to run automatically on GitHub, saving its forecasts and learning from past performance over time.

## CI guardrails and repo context checks

Our GitHub Actions workflows now include extra safety rails to ensure they always run inside the official [`oughtinc/Pythia`](https://github.com/oughtinc/Pythia) repository:

- Each job prints a **Debug repo context** block so you can see the repository slug, branch, commit SHA, and working directory tree size directly in the logs.
- An **anti-drift** script in `scripts/ci/` fails fast if any legacy references to the old repository slug sneak back into the tree.
- Database test jobs are protected with repo guards and file-existence checks so they only execute inside `oughtinc/Pythia` when the relevant tests are present.
- When you need to guard a step on the presence of a secret, assign it to an environment variable first (for example, `env.MY_TOKEN: ${{ secrets.MY_TOKEN }}`) and reference `env.MY_TOKEN` inside the `if:` expression. GitHub blocks direct `secrets.*` lookups in `if:` conditions at workflow-parse time.
- Resolver-specific diagnostics can be enabled in CI by exporting `RESOLVER_DEBUG=1`. When set, the resolver’s DuckDB selectors emit concise debug lines that show the backend, cutoff, `ym`, `iso3`, `hazard_code`, and row-count summaries during DB reads. The shared DuckDB connector also logs the canonicalised database path (for URLs like `duckdb:///tmp/db.duckdb`, `duckdb:////abs/path.duckdb`, relative file paths, or `:memory:`) so you can confirm reads and writes are pointed at the same file. The connector now self-heals cached handles: cache entries are evicted when `.close()` is called, stale connections are re-opened automatically, and cache hygiene is enforced at test boundaries as well as process exit. Set `RESOLVER_DISABLE_CONN_CACHE=1` to bypass caching entirely, or `RESOLVER_CONN_CACHE_MODE=thread` to scope caches to the current thread when running multithreaded diagnostics. `get_shared_duckdb_conn(db_url)` returns `(conn, resolved_path)` so tests can both use the live connection and assert which canonical DuckDB file is active. Snapshot writes also emit a one-line canonicalisation summary per table when `RESOLVER_DEBUG=1`, showing how messy semantics inputs were normalised before persistence.
- Database-backed resolver tests run twice in CI (with cache enabled and disabled) so regressions caused by cache state changes are caught immediately in both `.github/workflows/resolver-ci.yml` and `.github/workflows/resolver-ci-fast.yml`.
  - Cache-enabled runs expect repeated `duckdb_io.get_db(url)` calls to return the same connection object; cache-disabled runs (`RESOLVER_DISABLE_CONN_CACHE=1`) intentionally return fresh handles, and tests should branch accordingly.

If you fork the project under a different slug, update the guarded repository string inside the workflows before enabling CI.

### Resolver diagnostics

Set `RESOLVER_DIAG=1` to emit JSON-formatted diagnostics for DuckDB reads and writes. When enabled, the resolver logs the source file path, Python/DuckDB versions, cache resolution events, schema/index snapshots, input column samples, and post-write row counts before each upsert. CLI reads that fail to find data record the resolved database path and row-count summaries (total rows, keyed rows, and cutoff-filtered rows) to stderr so you can distinguish empty writes from URL mismatches. The flag leaves default logging untouched when unset, making it safe to toggle on only for triage runs or CI retries.

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

Resolver artifact hygiene

Nightly resolver runs can generate sizeable exports. The CI workflow calls
`resolver/tools/check_sizes.py` to warn when CSV/Parquet artifacts cross the
recommended 25 MB / 150 MB thresholds and to fail if the tracked repo contents
grow beyond ~2 GB. Temporary scratch files (`resolver/tmp/`,
`resolver/exports/*_working.*`, etc.) stay out of Git history via `.gitignore`.

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

### DuckDB setup for local development

Follow [resolver/docs/DUCKDB.md](resolver/docs/DUCKDB.md) to install DuckDB and
run the DuckDB-specific test suite locally. The same commands ensure
`python -c "import duckdb; print(duckdb.__version__)"` works before running
`pytest -q resolver/tests -k duckdb`.

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
- Inspect those CSVs locally or download them from the GitHub Actions job to understand which rows or columns diverged between the CSV export and DuckDB snapshot.

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

