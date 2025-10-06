# Resolver Pipeline Overview

This page provides a guided tour of the Resolver data pipeline from the first connector call to the publication of exports and frozen monthly snapshots. It is intended for contributors who need to understand how staging inputs, validation, precedence, and ReliefWeb PDF parsing fit together.

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

## Additional references

- [Connectors catalog](connectors_catalog.md)
- [Data contracts](data_contracts.md)
- [ReliefWeb PDF branch](reliefweb_pdf.md)
- [Operations run book](operations.md)
- [Troubleshooting guide](troubleshooting.md)
