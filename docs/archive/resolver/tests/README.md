# Resolver — Data Contract Tests

These tests enforce basic contracts across:
- Registries (`resolver/data/*.csv`)
- Exports (`resolver/exports/facts.csv`)
- Resolved outputs (`resolver/exports/resolved*.{csv,jsonl}`)
- Review queue (`resolver/review/review_queue.csv`)
- Snapshots (`resolver/snapshots/YYYY-MM/facts.parquet`), if present
- Remote-first state files under `resolver/state/**/exports/*.csv`

Mermaid diagrams in the docs (e.g., `resolver/docs/pipeline_overview.md`) render automatically on GitHub; view them there for the most up-to-date flow visuals.

## Run locally (cross-platform)

First install dev requirements:

```powershell
# Windows PowerShell
python -m pip install -r resolver/requirements-dev.txt
```

```bash
# macOS/Linux
python3 -m pip install -r resolver/requirements-dev.txt
```

Then run tests:

```bash
python -m pytest resolver/tests -q
```

Tests will skip gracefully if an expected file isn't present (e.g., snapshots),
but will fail if a file exists and violates the contract.

### Precedence multi-source overlaps

The precedence regression tests exercise a synthetic dataset with overlapping
sources to ensure the resolver consistently honours tier policy, recency, and
manual overrides.

```bash
pytest -q resolver/tests/test_precedence_multisource.py
```

Failures usually mean either:

- The policy config changed (e.g., new tier ordering) and the fixture needs to
  be updated, or
- A connector emitted unexpected fields (such as empty `as_of` values).

Inspect the failing assertion to see which country / hazard / metric combo broke
and adjust the precedence config or upstream mapping accordingly.

### Staging schema tests

After running the ingestor you can validate every staging CSV against the canonical
schema:

```bash
python resolver/ingestion/run_all_stubs.py
pytest -q resolver/tests/test_staging_schema_all.py
```

Failures list the offending CSV followed by the specific header or type issues
(missing columns, unexpected headers, invalid enums, etc.). Fix the connector
output so the CSV matches the schema, or update `resolver/tools/schema.yml` if
the schema needs to change.

### Hermetic connector tests
Header tests set `RESOLVER_SKIP_IFRCGO=1` and `RESOLVER_SKIP_RELIEFWEB=1` so no network is required.
Each connector must still produce a CSV with the canonical header (even if empty).

### ReliefWeb PDF tests

`resolver/tests/ingestion/test_reliefweb_pdf.py` monkeypatches the PDF text
extraction helper so we can exercise parsing, precedence, and delta logic using
plain strings.  No binary fixtures are required; see
`resolver/tests/fixtures/reliefweb_pdfs/README.md` for details.

### Running PDF unit tests (mocked)

```bash
pytest -q resolver/tests/ingestion/test_reliefweb_pdf.py
```
