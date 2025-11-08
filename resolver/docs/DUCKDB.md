# DuckDB Setup for Resolver Development

DuckDB is required for the resolver database parity and idempotency tests. Once
installed, any tests that rely on `pytest.importorskip("duckdb")` will execute
normally instead of being skipped.

## Installing via Poetry

```bash
poetry install --with dev
```

If you only want the resolver package extras, you can install them explicitly:

```bash
poetry install --with dev --with test
```

## Installing via pip

```bash
python -m pip install "duckdb>=1.0,<2"
```

## Verify the installation

```bash
python -c "import duckdb; print(duckdb.__version__)"
```

Once the command above prints a version, run the DuckDB-specific test subset:

```bash
pytest -q resolver/tests -k duckdb
```

These tests will now run in both local development and continuous integration
pipelines.

## IDMC flow semantics

The `resolver.cli.idmc_to_duckdb` wrapper normalizes staging paths and delegates
writes to the exporter with the `idmc-staging` strategy. Rows with
`metric=new_displacements` are forced to `series_semantics=new`, so they land in
`facts_deltas`, while accompanying stock rows continue to populate
`facts_resolved`. The wrapper verifies both tables after each run, so a flow-only
staging directory still reports success when the deltas table receives rows.
