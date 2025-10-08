# Resolver Overview (A1)

Resolver ingests humanitarian situation reports from multiple connectors, normalises them into staging CSVs, and exports monthly "new" deltas for downstream analytics. The ReliefWeb PDF branch adds attachment parsing with OCR fallback and household-to-people conversions.

## Documentation map

- [Pipeline overview](docs/pipeline_overview.md)
- [Connectors catalog](docs/connectors_catalog.md)
- [Data contracts](docs/data_contracts.md)
- [Precedence policy](docs/precedence.md)
- [ReliefWeb PDF pipeline](docs/reliefweb_pdf.md)
- [Operations run book](docs/operations.md)
- [Troubleshooting guide](docs/troubleshooting.md)
- [Governance & audit](docs/governance.md)

## Quick start

1. **Install dependencies** (once):
   ```bash
   pip install -r resolver/requirements.txt
   pip install -r resolver/requirements-dev.txt
   ```
2. **Generate staging CSVs (offline stubs):**
   ```bash
   python resolver/ingestion/run_all_stubs.py --retries 2
   ```
3. **Export and validate facts:**
   ```bash
   python resolver/tools/export_facts.py --in resolver/staging --out resolver/exports
   python resolver/tools/validate_facts.py --facts resolver/exports/facts.csv
   ```
4. **Resolve precedence and freeze a snapshot:**
   ```bash
   python resolver/tools/precedence_engine.py --facts resolver/exports/facts.csv --cutoff YYYY-MM-30
   python resolver/tools/freeze_snapshot.py --facts resolver/exports/facts.csv --resolved resolver/exports/resolved.csv --month YYYY-MM
   ```

5. **(Optional) Enable DuckDB dual-write + query layer:**
   ```bash
   export RESOLVER_DB_URL="duckdb:///$(pwd)/resolver/db/resolver.duckdb"
   python resolver/tools/export_facts.py --in resolver/staging --out resolver/exports
   python resolver/tools/freeze_snapshot.py \
     --facts resolver/exports/facts.csv \
     --resolved resolver/exports/resolved.csv \
     --deltas resolver/exports/deltas.csv \
     --month YYYY-MM
   ```
   The env var acts as the feature flag. When set, exports dual-write into
   `resolver/db/resolver.duckdb` while continuing to produce CSV/Parquet files.
   The first connection calls `duckdb_io.init_schema()` to apply the canonical
   tables and constraints defined in [`resolver/db/schema.sql`](db/schema.sql),
   so reruns always observe the same structure.

Refer to the [operations run book](docs/operations.md) for detailed command variants (including deltas and review tooling).

## Working with ReliefWeb PDFs

The ReliefWeb PDF branch is disabled by default in CI but can be enabled locally with feature flags:

```bash
RELIEFWEB_ENABLE_PDF=1 \
RELIEFWEB_PDF_ALLOW_NETWORK=0 \
RELIEFWEB_PDF_ENABLE_OCR=0 \
python resolver/ingestion/reliefweb_client.py
```

- Toggle OCR with `RELIEFWEB_PDF_ENABLE_OCR=1|0`.
- Provide custom people-per-household data via `RELIEFWEB_PPH_OVERRIDE_PATH=/path/to/file.csv`.
- Run mocked extractor tests with `pytest -q resolver/tests/ingestion/test_reliefweb_pdf.py`.

See [ReliefWeb PDF pipeline](docs/reliefweb_pdf.md) for heuristics, file contracts, and troubleshooting.

## Validation

Run the lightweight validator against any facts CSV/Parquet:

```bash
pip install pandas pyarrow pyyaml
python resolver/tools/validate_facts.py --facts resolver/samples/facts_sample.csv
```

Checks required columns, enums, and dates.

Verifies iso3 exists in resolver/data/countries.csv.

Verifies hazard_code/hazard_label/hazard_class match resolver/data/shocks.csv.

Ensures value >= 0, as_of_date <= publication_date <= today.

Blocks metric = in_need if source_type = media.

Requires unit = persons_cases when metric = cases.

PR checklist addition: ✅ Facts validate against registries (countries.csv, shocks.csv) with validate_facts.py.

## Snapshots

Create a monthly snapshot (validated, parquet + manifest):

```bash
pip install pandas pyarrow pyyaml
python resolver/tools/freeze_snapshot.py --facts resolver/samples/facts_sample.csv --resolved resolver/exports/resolved.csv --month 2025-09
```

PR checklist addition: ✅ If this PR changes facts or resolver logic, ensure a snapshot plan is documented and (when appropriate) a new snapshot was produced with the freezer.

## Export → Validate → Freeze (quickstart)

```bash
# 1) Export normalized facts from staging files/folder
python resolver/tools/export_facts.py --in resolver/staging --out resolver/exports

# 2) Validate against registries and schema
python resolver/tools/validate_facts.py --facts resolver/exports/facts.csv

# 3) Resolve canonical totals to resolved.csv
python resolver/tools/precedence_engine.py \
  --facts resolver/exports/facts.csv \
  --cutoff 2025-09-30

# 4) Build monthly new deltas (resolver/exports/deltas.csv)
python resolver/tools/make_deltas.py \
  --resolved resolver/exports/resolved.csv \
  --out resolver/exports/deltas.csv

# 5) Freeze a monthly snapshot for grading (facts + deltas)
python resolver/tools/freeze_snapshot.py \
  --facts resolver/exports/facts.csv \
  --resolved resolver/exports/resolved.csv \
  --deltas resolver/exports/deltas.csv \
  --month 2025-09


If you see validation errors, fix the staging inputs or tweak resolver/tools/export_config.yml.

## End-to-end (Stubs → Export → Validate → Freeze)

```bash
# 0) Ensure registries exist and include your latest countries/hazards
#    resolver/data/countries.csv
#    resolver/data/shocks.csv

# 1) Generate staging CSVs from stub connectors (no network)
python resolver/ingestion/run_all_stubs.py

# 2) Export canonical facts from staging
python resolver/tools/export_facts.py --in resolver/staging --out resolver/exports

# 3) Validate against registries & schema
python resolver/tools/validate_facts.py --facts resolver/exports/facts.csv

# 4) Resolve canonical totals to resolved.csv/jsonl
python resolver/tools/precedence_engine.py \
  --facts resolver/exports/facts.csv \
  --cutoff YYYY-MM-30

# 5) Build monthly new deltas (resolver/exports/deltas.csv)
python resolver/tools/make_deltas.py \
  --resolved resolver/exports/resolved.csv \
  --out resolver/exports/deltas.csv

# 6) Freeze a monthly snapshot (facts + deltas)
python resolver/tools/freeze_snapshot.py \
  --facts resolver/exports/facts.csv \
  --resolved resolver/exports/resolved.csv \
  --deltas resolver/exports/deltas.csv \
  --month YYYY-MM
```

This will create:

resolver/staging/*.csv (one per source)

resolver/exports/{facts.csv,resolved.csv,deltas.csv}

resolver/snapshots/YYYY-MM/{facts.parquet,manifest.json,deltas.csv}

resolver/db/resolver.duckdb (if `RESOLVER_DB_URL` is set)


## Remote-first state layout

When CI runs, it commits outputs into the repo so you can consume them directly:

- **PR state:** `resolver/state/pr/<PR_NUMBER>/...`
- **Nightly state:** `resolver/state/daily/<YYYY-MM-DD>/...`
- **Monthly state:** `resolver/state/monthly/<YYYY-MM>/{resolved.csv,deltas.csv}`
- **Monthly snapshots (authoritative for grading):** `resolver/snapshots/<YYYY-MM>/...`

This means the resolver can run from the remote alone (clone/pull → read files).


## Resolve at a cutoff (precedence engine)

Select one authoritative total per `(iso3, hazard_code)` applying A2 policy:

```bash
pip install pandas pyarrow pyyaml python-dateutil
python resolver/tools/precedence_engine.py \
  --facts resolver/exports/facts.csv \
  --cutoff 2025-09-30
```


Outputs:

resolver/exports/resolved.csv

resolver/exports/resolved.jsonl

resolver/exports/resolved_diagnostics.csv (conflict notes)

PR checklist addition: ✅ Precedence config reviewed (tools/precedence_config.yml) and results inspected (exports/resolved*.{csv,jsonl}).

## Monthly deltas (new PIN/PA)

Convert resolved totals into normalized monthly new values for downstream aggregation:

```bash
python resolver/tools/make_deltas.py \
  --resolved resolver/exports/resolved.csv \
  --out resolver/exports/deltas.csv \
  --lookback-months 24  # optional
```

All rows in `deltas.csv` are monthly "new" values with provenance. Stock series are differenced month over month; detected rebases set `rebase_flag=1` and clamp deltas to zero. Minor negative blips are clamped with `delta_negative_clamped=1`.

## DuckDB query layer

The resolver CLI and API read from the historical file-backed exports by
default. Opt into the DuckDB database by pointing at the database URL and
selecting the backend explicitly.

```bash
# CLI (default files; use --backend db or RESOLVER_CLI_BACKEND=db)
python resolver/cli/resolver_cli.py \
  --iso3 PHL --hazard_code TC --cutoff 2024-02-29 --series new --backend db --json_only

# API (uvicorn example; override default with RESOLVER_API_BACKEND=db)
RESOLVER_DB_URL=duckdb:///$(pwd)/resolver/db/resolver.duckdb \
  RESOLVER_API_BACKEND=db \
  uvicorn resolver.api.app:app --reload
# GET /resolve?iso3=PHL&hazard_code=TC&cutoff=2024-02-29&series=new&backend=db
```

Environment toggles:

- `RESOLVER_CLI_BACKEND`: choose `files`, `db`, or `auto` (prefer DB when
  available). Defaults to `files` for backwards compatibility.
- `RESOLVER_API_BACKEND`: same options for the API default backend; the query
  parameter `backend=` always wins when provided.

When `RESOLVER_DB_URL` is set, exports and freezer scripts remain backwards
compatible: files continue to be written for downstream consumers while the
database maintains parity through regression tests (`test_db_parity.py`,
`test_db_query_contract.py`, `test_monthly_deltas_primary.py`).

`resolved.csv` now includes a normalized `ym` column derived from the figure's `as_of_date` in the Europe/Istanbul timezone. The precedence engine enforces the configured publication lag when evaluating rows for a cutoff, ensuring deltas align to the month of the figure rather than the publication date.


### End-to-end (monthly)

```bash
# 1) Stage connectors → facts
python resolver/tools/export_facts.py --in resolver/staging --out resolver/exports
python resolver/tools/validate_facts.py --facts resolver/exports/facts.csv

# 2) Select authoritative values (stock totals)
python resolver/tools/precedence_engine.py --facts resolver/exports/facts.csv --cutoff YYYY-MM-DD

# 3) Build monthly NEW deltas (this enables summation across months)
python resolver/tools/make_deltas.py --resolved resolver/exports/resolved.csv --out resolver/exports/deltas.csv

# 4) Freeze snapshot (now includes deltas)
python resolver/tools/freeze_snapshot.py --facts resolver/exports/facts.csv --month YYYY-MM

# 5) Write repo state (per-month folder with resolved + deltas)
python resolver/tools/write_repo_state.py --month YYYY-MM

# Optional: coverage/gaps
python resolver/tools/gaps_report.py --deltas resolver/exports/deltas.csv --resolved resolver/exports/resolved.csv --months 3 --out resolver/exports/gaps_report.csv
```


## Ask the resolver (CLI)

```bash
# Country/Hazard by names
python resolver/cli/resolver_cli.py --country "Philippines" --hazard "Tropical Cyclone" --cutoff 2025-09-30

# Or by codes
python resolver/cli/resolver_cli.py --iso3 PHL --hazard_code TC --cutoff 2025-09-30

# Ask for stock totals instead of monthly new deltas
python resolver/cli/resolver_cli.py --iso3 PHL --hazard_code TC --cutoff 2025-09-30 --series stock
```

### Selection logic

- If the month of `--cutoff` is in the past, use `snapshots/YYYY-MM/facts.parquet`.
- If current month, use `exports/resolved_reviewed.csv` if present, else `exports/resolved.csv`.
- Default series is monthly `new` deltas when available; pass `--series stock` for totals. Missing deltas trigger a note and a stock fallback.

---
### Outputs

- A single JSON line (always) and a human-readable summary (unless `--json_only`).

---

**Definition of Done (DoD)**
- `resolver/ingestion/README.md` + `resolver/ingestion/checklist.yml` exist.
- Stubs exist and write staging CSVs: `reliefweb.csv`, `ifrc_go.csv`, `unhcr.csv`, `dtm.csv`, `who.csv`, `ipc.csv`.
- `run_all_stubs.py` runs all stubs and prints **✅ all stubs completed**.
- Root `resolver/README.md` shows the end-to-end commands.
- `resolver/tools/precedence_config.yml` exists with tiers and mapping that match A2.
- `resolver/tools/precedence_engine.py` runs on your exported facts and writes `resolved.csv/jsonl` + diagnostics.
- A local smoke test with `resolver/exports/facts_minimal.csv` succeeds and selects the expected rows.
- `resolver/README.md` updated with usage and a PR checklist line.

## HTTP API

Run a tiny FastAPI server:

```bash
pip install -r resolver/requirements.txt
uvicorn resolver.api.app:app --reload --port 8000
```

Query it:

```bash
curl "http://127.0.0.1:8000/resolve?iso3=PHL&hazard_code=TC&cutoff=2025-09-30"
```

### Developer setup (tests)

For local tests, install dev deps once:

```
python -m pip install -r resolver/requirements-dev.txt
```

Then run:

```
python -m pytest resolver/tests -q
```

OpenAPI docs at /docs.
