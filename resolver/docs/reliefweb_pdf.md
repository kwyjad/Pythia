# ReliefWeb PDF Pipeline

The ReliefWeb PDF pipeline augments the classic ReliefWeb API connector by scoring attachments, extracting metrics from PDFs, and emitting tier-2 monthly deltas that fill gaps when higher-tier sources are silent. This page documents the full flow, configuration toggles, file contracts, and test hooks for the branch.

## What the pipeline does

1. **Select candidate PDFs:** `resolver/ingestion/reliefweb_client.py` queries the ReliefWeb API for reports with `application/pdf` attachments. Each attachment receives a heuristic score favouring structured situation reports (`sitreps`, tabular annexes) over infographics or narrative briefs.
2. **Extract text:** The connector first reads native text layers. When the native layer is sparse (`< min_text_chars_before_ocr`), it falls back to OCR through helpers in [`resolver/ingestion/_pdf_text.py`](../ingestion/_pdf_text.py). OCR can be disabled entirely when testing in minimal environments.
3. **Parse metrics:** Pattern matchers capture key humanitarian metrics (PIN, PA, IDPs, cases). Household-only figures are converted to people using the people-per-household (PPH) reference. Parsed metrics store the triggering phrase for audit.
4. **Compute tier-2 deltas:** The connector maintains per-report time series to derive month-over-month "new" values. Tier metadata is stamped as `tier=2` and `series_semantics="new"` before rows leave the connector.
5. **Publish staging outputs:** Results are written to `resolver/staging/reliefweb_pdf.csv` alongside `reliefweb_pdf.csv.meta.json` manifests containing attachment provenance, selector scores, and extraction modes.

The pipeline is designed to be hermetic in CI: attachments can be mocked and OCR disabled so the unit tests run without network access.

## Feature toggles and environment flags

| Variable | Purpose | Default |
|---|---|---|
| `RELIEFWEB_ENABLE_PDF=1|0` | Enable/disable the PDF branch | `0` in CI, `1` locally when testing |
| `RELIEFWEB_PDF_ENABLE_OCR=1|0` | Allow OCR fallback when native text is insufficient | `1` |
| `RELIEFWEB_PDF_ALLOW_NETWORK=1|0` | Permit live attachment downloads | `0` in CI |
| `RELIEFWEB_PPH_OVERRIDE_PATH=/path/to/file.csv` | Supply a custom PPH lookup table | unset |

Additional knobs live in [`ingestion/config/reliefweb.yml`](../ingestion/config/reliefweb.yml), including `preferred_pdf_titles`, language filters, and `min_text_chars_before_ocr`.

## File contracts

### Staging CSV (`resolver/staging/reliefweb_pdf.csv`)

Minimum columns:

```
as_of,source,event_id,country_iso3,month,hazard_type,metric,value,pph_used,
extraction_method,matched_phrase,pdf_score,method,method_value,series_semantics,
tier,confidence,publication_date,source_url,resource_url,definition_text
```

- `pph_used` records the multiplier applied during household → people conversion.
- `extraction_method` is `pdf_native` or `ocr` depending on the winning layer.
- `pdf_score` surfaces the selector confidence.
- `series_semantics` is always `new`; the connector already differences level values into monthly deltas.
- `tier` is fixed to `2` so precedence understands the branch hierarchy.

### Manifest (`resolver/staging/reliefweb_pdf.csv.meta.json`)

Adds attachment-level provenance:

- `artifact_sha256` and `artifact_path` for cached PDFs
- `pdf_score` mirroring the CSV
- `extraction_method` per record
- Selector rationale (`selector_notes`, `language`, `file_size_bytes`)

These fields power audit trails in precedence diagnostics.

## Heuristics summary

- **Attachment preference:** tables and structured situation reports outrank narrative briefs or infographics unless explicit keywords elevate them.
- **Language filters:** defaults favour English and UN working languages; others are skipped unless `allowed_languages` is configured.
- **File size caps:** attachments above the configured size threshold are skipped to avoid large OCR jobs.
- **Deduplication:** once an attachment is parsed it is cached under `resolver/staging/.cache/reliefweb/pdf/` to prevent repeated downloads.

## Delta logic

The connector tracks series by `(event_id, country_iso3, hazard_type, metric)` and compares the current level value to the previous level for the same lineage. When a report restates a level with no change the monthly delta resolves to zero and is emitted with `value=0`. Rebasings (large negative swings) trigger `method_details` notes and clamp to zero to avoid negative deltas.

## Testing hooks

- **Unit tests:** `pytest -q resolver/tests/ingestion/test_reliefweb_pdf.py` monkeypatches OCR and network calls so parsing logic runs against inline text fixtures.
- **Manual dry run:**
  ```bash
  RELIEFWEB_ENABLE_PDF=1 \
  RELIEFWEB_PDF_ALLOW_NETWORK=0 \
  RELIEFWEB_PDF_ENABLE_OCR=0 \
  python resolver/ingestion/reliefweb_client.py --limit 5
  ```
- **Schema checks:** `pytest -q resolver/tests/test_staging_schema_all.py` confirms the CSV header matches `schema.yml` (including `pph_used`, `extraction_method`, and `pdf_score`).

For additional operational guidance see the [run book](operations.md) and [troubleshooting guide](troubleshooting.md).
