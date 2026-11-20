# Changelog

## Unreleased

- check the GDACS history blocks the forecaster actually sent (`scripts/ci/check_gdacs_prompt_blocks.py`, run in the Sibyl job, in the debug bundle and as the `gdacs_prompt_block_invalid` issue in the cycle summary), parse them back with `forecaster.gdacs_history.parse_gdacs_prompt_block` (current and pre-October formats), list each recent Orange/Red alert month once in binary prompts (it listed one line per row, so a month appeared up to four times), and add the prompt's own GDACS history to the scored bundle's `inject_health.csv` beside the rebuild from today's database.
- fix ReliefWeb connector 400s by reading the `reliefweb_fields.yml` allowlist, pruning unsupported API fields on the fly, and retrying with the safe set when the API rejects a field.
- migrate batch resolve request/response models to Pydantic v2 while preserving validation behaviour and CLI/API compatibility.
- fix DuckDB-backed resolver lookups so `series="new"` reads `facts_deltas.value_new` (and stocks continue to read `facts_resolved.value`) while harmonising the pydantic/duckdb dependency pins.
- fix the DuckDB `series="new"` selector to only select columns that exist on `facts_deltas`, alias missing provenance fields to empty strings, and expose query errors instead of silently reporting "no data".
- ACLED CLI now derives `iso3` codes from `country` when missing, emits diagnostics for missing rates, and previews the grouped rows written to DuckDB.
