# Operations Run Book

This run book covers the main resolver workflows, including the ReliefWeb PDF branch. It is designed for engineers and analysts running the pipeline locally or in CI.

## Quick start

1. **Ingest connectors (offline stubs):**
   ```bash
   python resolver/ingestion/run_all_stubs.py --retries 2
   ```
2. **Validate staging schemas:**
   ```bash
   pytest -q resolver/tests/test_staging_schema_all.py
   ```
3. **Export and validate facts:**
   ```bash
   python resolver/tools/export_facts.py --in resolver/staging --out resolver/exports
   python resolver/tools/validate_facts.py --facts resolver/exports/facts.csv
   ```
4. **Run precedence & deltas (optional for analytics):**
   ```bash
   python resolver/tools/precedence_engine.py --facts resolver/exports/facts.csv --cutoff YYYY-MM-30
   python resolver/tools/make_deltas.py --resolved resolver/exports/resolved.csv --out resolver/exports/deltas.csv
   ```
5. **Freeze a snapshot:**
   ```bash
   python resolver/tools/freeze_snapshot.py --facts resolver/exports/facts.csv --month YYYY-MM
   ```

## ReliefWeb PDF local runs

The PDF branch can be exercised without network access by enabling the feature flags and relying on mocked text extraction:

```bash
RELIEFWEB_ENABLE_PDF=1 \
RELIEFWEB_PDF_ALLOW_NETWORK=0 \
RELIEFWEB_PDF_ENABLE_OCR=0 \
python resolver/ingestion/reliefweb_client.py
```

To run the mocked extractor tests:

```bash
pytest -q resolver/tests/ingestion/test_reliefweb_pdf.py
```

### Feature toggles

- `RELIEFWEB_ENABLE_PDF=1|0` — enable or disable the PDF branch entirely
- `RELIEFWEB_PDF_ENABLE_OCR=1|0` — allow OCR fallback when native text is sparse
- `RELIEFWEB_PDF_ALLOW_NETWORK=1|0` — control attachment downloads (CI keeps this at `0`)
- `RELIEFWEB_PPH_OVERRIDE_PATH=/path/to/pph.csv` — optional CSV for household overrides

## Logs and observability

- Ingestion writes plain text and JSONL logs to `resolver/logs/ingestion/` by default.
- Override destinations with `RUNNER_LOG_DIR=/tmp/resolver-logs`.
- Set verbosity via `RUNNER_LOG_LEVEL=DEBUG` and format via `RUNNER_LOG_FORMAT=json`.

## Continuous integration

The `resolver-ci` workflow executes offline smoke tests plus the ReliefWeb PDF unit suite. When the optional Markdown link checker is enabled it runs after the tests and reports broken intra-repo links in the job logs without failing the build.
