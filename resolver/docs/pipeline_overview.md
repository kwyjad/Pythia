# Resolver Pipeline Overview

This page provides a guided tour of the Resolver data pipeline from the first connector call to the publication of exports and (in a follow-on workflow) frozen monthly snapshots. It is intended for contributors who need to understand how staging inputs, validation, precedence, and ReliefWeb PDF parsing fit together.

```mermaid
flowchart LR
  A[Connectors (ACLED, DTM, IPC, IFRC GO, UNHCR, WFP mVAM, WHO PHE, HDX, GDACS, WorldPop, ReliefWeb)]
  A --> B[Staging CSVs: resolver/staging/*.csv]
  subgraph ReliefWeb PDF Branch
    A --> A1[ReliefWeb PDF Selector]
    A1 --> A2[PDF Text Extraction (native→OCR fallback)]
    A2 --> A3[Metric Parsing + HH→People Conversion]
    A3 --> A4[Tier-2 Monthly Deltas]
    A4 --> B_pdf[staging/reliefweb_pdf.csv]
  end
  B & B_pdf --> C[Schema Validation (schema.yml → tests)]
  C --> D[Delta Logic (PIN/PA new per month)]
  D --> E[Precedence Engine (tiers, tie-break, overrides)]
  E --> F[Exports (facts.csv, diagnostics)]
  F --> G[Snapshots (resolver/snapshots/YYYY-MM)]
  G --> H[API/UI (future), Analytics, Forecast Resolution]
```

## Layer Contracts & Data Flow

Resolver’s goal is to provide country–month–shock-level snapshots of humanitarian impact (PIN/PA), with a focus on **new PIN/PA per month** so month 1 + month 2 + month 3 can be aggregated safely.

The pipeline runs as:

1. **Connectors (staging)**
   - Fetch and normalise source-specific data (IOM DTM, IDMC, EM-DAT, ACLED, etc.).
   - Emit staging tables with standard columns (iso3, hazard codes, dates, raw counts).
2. **Export Facts (canonical facts)**
   - Map staging into canonical facts: `iso3`, `ym`, `hazard_code` / `hazard_class`, `metric`, `value`, `series_semantics`, `source`.
   - Write canonical outputs: `facts.csv` for the run and, when DB writes are enabled, `facts_resolved` / `facts_deltas` in DuckDB.
3. **Freeze Snapshot (per-month snapshots; separate workflow)**
   - Given canonical facts and a target month (`ym`), filter to that month, normalise required columns, and deduplicate resolved/deltas frames.
   - Optionally run EM-DAT validators when EM-DAT metrics are present.
   - Write a snapshot parquet for the month and update DuckDB snapshot metadata. The legacy in-backfill freeze stage is disabled; snapshots will be rebuilt via the DB-first snapshot workflow.
4. **Forecaster & APIs (consumers)**
   - Consult DuckDB and/or snapshot parquet files to obtain resolution-ready facts per country, month, and shock.
   - Use these as the scoring baseline for forecasting questions and downstream analysis.

This layered design is intentional: each layer can be tested, debugged, and evolved independently while keeping the monthly “new PIN/PA” objective intact.

### Connectors used in `resolver-initial-backfill`

The manual `resolver-initial-backfill` workflow currently runs only the four connectors that are stable end-to-end:

- **DTM** (IOM DTM)
- **IDMC** (internal displacement)
- **EM-DAT**
- **ACLED**

Other connectors are intentionally excluded from this workflow to avoid destabilising the ingestion run. The backfill writes canonical facts from these four sources into DuckDB for downstream snapshot, dashboard, and forecasting stages. Freeze/snapshot generation is intentionally detached from the backfill while the DB-first snapshot builder is rolled out; the GitHub Actions workflow now stops after export-and-DuckDB writes.

## Pipeline stages

- **Connector ingestion**
  Entry points live under `resolver/ingestion/*_client.py` and are orchestrated by [`resolver/ingestion/run_all_stubs.py`](../ingestion/run_all_stubs.py). Each connector writes a canonical CSV under `resolver/staging/`. The ReliefWeb client also drives the PDF selector when `RELIEFWEB_ENABLE_PDF=1`; see [ReliefWeb PDF](reliefweb_pdf.md).
- **ReliefWeb PDF branch**  
  [`resolver/ingestion/reliefweb_client.py`](../ingestion/reliefweb_client.py) hydrates report metadata, scores attachments, extracts text via [`resolver/ingestion/_pdf_text.py`](../ingestion/_pdf_text.py), and writes `resolver/staging/reliefweb_pdf.csv` plus manifest entries when the branch is enabled.
- **Schema validation**  
  Staging CSVs are checked against [`resolver/tools/schema.yml`](../tools/schema.yml) via [`resolver/tests/test_staging_schema_all.py`](../tests/test_staging_schema_all.py). A generated [SCHEMAS.md](../../SCHEMAS.md) provides column-level detail for exports and staging contracts.
- **Delta preparation**  
  Connectors output level values, but Resolver consumes monthly "new" deltas. Scripts such as [`resolver/tools/make_deltas.py`](../tools/make_deltas.py) and in-connector logic (for ReliefWeb PDFs) compute the month-over-month differences, including tier-2 ReliefWeb `series_semantics="new"` rows.
- **Precedence engine**  
  [`resolver/tools/precedence_engine.py`](../tools/precedence_engine.py) ranks candidates using tier policy, recency, completeness, and overrides. The logic is documented in [Precedence policy](precedence.md) and the governance appendix.
- **Exports**  
  [`resolver/tools/export_facts.py`](../tools/export_facts.py) consolidates staging inputs into `resolver/exports/facts.csv`, while [`resolver/tools/validate_facts.py`](../tools/validate_facts.py) enforces schema and registry rules before precedence runs.
- **Snapshots**
  The freezer [`resolver/tools/freeze_snapshot.py`](../tools/freeze_snapshot.py) writes immutable monthly bundles (`resolver/snapshots/YYYY-MM`) for downstream analytics, dashboards, and the forecast resolver.
  A DB-first successor is planned under [`resolver/snapshot`](../snapshot) with CLI orchestration at [`resolver/cli/snapshot_from_db.py`](../cli/snapshot_from_db.py) so monthly snapshots can be built directly from DuckDB.

### DB-backed snapshot builder (`facts_snapshot`)

In addition to the legacy freezer, the resolver includes a DB-backed snapshot builder under [`resolver.snapshot.builder`](../snapshot/builder.py):

- Reads canonical tables `facts_resolved`, `facts_deltas`, and connector-specific monthly tables (e.g., `acled_monthly_fatalities`).
- For each target `ym` (for example, `"2025-11"`), writes a unified `facts_snapshot` table plus a `snapshots` metadata table and can export `data/snapshots/<ym>/facts.parquet`.
- Operates purely from the DuckDB database (see [WRITING_TO_DUCKDB](WRITING_TO_DUCKDB.md)) and is idempotent: reruns replace prior rows for the month instead of appending duplicates.

## Additional references

- [Connectors catalog](connectors_catalog.md)
- [Data contracts](data_contracts.md)
- [ReliefWeb PDF branch](reliefweb_pdf.md)
- [Operations run book](operations.md)
- [Troubleshooting guide](troubleshooting.md)
- **Batch resolution** — Use `POST /resolve_batch` or `resolver/cli/resolver_cli.py batch-resolve --in queries.csv --out results.jsonl` to resolve multiple `(country, hazard, month, series)` questions in one call. The resolver automatically balances between DuckDB and CSV/Parquet backends based on availability. Batch input validation is handled by Pydantic v2 models so the CLI and API enforce identical rules.

## Series routing

Resolver exposes two user-facing series:

- **`series=stock`** queries the `facts_resolved` table and returns the stock value for the cutoff month (latest `as_of_date` on or before the cutoff).
- **`series=new`** queries the `facts_deltas` table and returns the month-over-month delta (`value_new`) for the cutoff month using the latest qualifying `as_of` timestamp. Provenance fields (`source_url`, `source_type`, `doc_title`, `definition_text`) are not stored on `facts_deltas`; the resolver returns these as empty strings unless a downstream join enriches them.

When running against DuckDB, Resolver materialises these answers directly from `facts_resolved.value` (stock totals) or `facts_deltas.value_new` (monthly deltas), keyed by the `(iso3, hazard_code, metric, ym)` tuple where `ym` is derived from the Europe/Istanbul cutoff month.

If the requested series has no data, Resolver does **not** fall back to the other series unless the operator explicitly opts in by setting `RESOLVER_ALLOW_SERIES_FALLBACK=1`. Both the CLI and API surface the `series_returned` field so clients can verify which series satisfied the request.
