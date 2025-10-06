# Data Contracts

Resolver normalises staging inputs and exported facts into a shared schema. This document highlights the canonical columns, ReliefWeb PDF additions, and pointers to the authoritative schema definitions.

## Canonical staging columns

All staging CSVs under `resolver/staging/` adhere to [`resolver/tools/schema.yml`](../tools/schema.yml). The generated [SCHEMAS.md](../../SCHEMAS.md) file provides column-level descriptions. Core columns include:

- Identifiers: `event_id`, `source`, `source_event_id`, `country_iso3`, `hazard_code`, `metric`
- Temporal fields: `as_of`, `month`, `publication_date`, `ingested_at`
- Figures: `value` (monthly new), `value_level` (optional stock), `unit`
- Provenance: `source_type`, `source_url`, `definition_text`, `method`, `method_value`
- Precedence hints: `tier`, `confidence`, `revision`

## ReliefWeb PDF staging contract

`resolver/staging/reliefweb_pdf.csv` extends the base schema with additional metadata needed for auditability:

- `pph_used` — numeric people-per-household multiplier applied when converting households → people
- `extraction_method` — `pdf_native` or `ocr` indicating which text layer produced the metric
- `pdf_score` — selector score used to pick the attachment
- `matched_phrase` — snippet showing the parsed metric context
- `series_semantics` — fixed to `new` to denote monthly deltas already computed in the connector
- `tier` — set to `2` so precedence can prioritise higher-tier sources when present

The ReliefWeb PDF manifest (`resolver/staging/reliefweb_pdf.csv.meta.json`) mirrors the standard manifest structure with additional keys for attachment `artifact_sha256`, selector `score`, and `extraction_method`.

## Facts exports

[`resolver/tools/export_facts.py`](../tools/export_facts.py) consolidates staging files into `resolver/exports/facts.csv`. Facts retain the staging fields and add:

- `ym` — `YYYY-MM` derived from `as_of` for monthly joins
- `series` — `new` or `stock` with `new` being the default for PIN/PA metrics
- `precedence_notes` — populated after review to document overrides

All People in Need (PIN) and People Affected (PA) figures in facts are stored as **monthly "new" values**. Stock totals are differenced during export or inside connectors (ReliefWeb PDF) before reaching precedence.

## People-per-household reference

Household conversions use `resolver/reference/avg_household_size.csv` as the base table. Overrides live in `resolver/reference/overrides/avg_household_size_overrides.yml`. A custom path can be supplied with `RELIEFWEB_PPH_OVERRIDE_PATH` when testing alternative datasets.
