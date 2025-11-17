# Changelog

## Unreleased

- fix ReliefWeb connector 400s by reading the `reliefweb_fields.yml` allowlist, pruning unsupported API fields on the fly, and retrying with the safe set when the API rejects a field.
- migrate batch resolve request/response models to Pydantic v2 while preserving validation behaviour and CLI/API compatibility.
- fix DuckDB-backed resolver lookups so `series="new"` reads `facts_deltas.value_new` (and stocks continue to read `facts_resolved.value`) while harmonising the pydantic/duckdb dependency pins.
- fix the DuckDB `series="new"` selector to only select columns that exist on `facts_deltas`, alias missing provenance fields to empty strings, and expose query errors instead of silently reporting "no data".
- fix the freeze snapshot CLI invocation order, gate EM-DAT normalization to true EM-DAT previews, and document the normalizer state in diagnostics summaries.
