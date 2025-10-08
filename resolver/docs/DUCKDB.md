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
