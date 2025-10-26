# Contributing

## CI-parity local run (no DuckDB)

To mirror the CI fallback scenario without installing DuckDB:

1. Create and activate a clean virtual environment without the `duckdb` package.
2. Install the project in editable mode:
   ```bash
   pip install -e .
   ```
3. Run the fast-fixtures preflight to confirm the noop mode diagnostics succeed:
   ```bash
   python -m resolver.ingestion.preflight
   ```
4. Execute the offline smoke bootstrap for the DTM client:
   ```bash
   python -m resolver.ingestion.dtm_client --offline-smoke
   ```
5. Run the targeted pytest check that exercises the header writer without DuckDB:
   ```bash
   pytest -q resolver/tests/test_connectors_headers.py::test_dtm_header_written
   ```

Each command should exit successfully while operating entirely in the no-DuckDB fallback mode.
