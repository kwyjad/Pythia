# Changelog

## Unreleased

- migrate batch resolve request/response models to Pydantic v2 while preserving validation behaviour and CLI/API compatibility.
- fix DuckDB-backed resolver lookups so `series="new"` reads `facts_deltas.value_new` (and stocks continue to read `facts_resolved.value`) while harmonising the pydantic/duckdb dependency pins.
- fix the DuckDB `series="new"` selector to only select columns that exist on `facts_deltas`, alias missing provenance fields to empty strings, and expose query errors instead of silently reporting "no data".
