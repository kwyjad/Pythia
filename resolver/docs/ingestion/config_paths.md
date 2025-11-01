# Connector configuration paths

Connector YAML configuration files now live in a single canonical location: `resolver/ingestion/config/`.  The
`resolver/config/` directory is reserved for global configuration (for example `series_semantics.yml`) and remains as
an escape hatch while older jobs migrate.

The shared loader used by the ingestion connectors searches paths in the following order:

1. `resolver/ingestion/config/<connector>.yml`
2. `resolver/config/<connector>.yml` (fallback for legacy runs)

When both files exist and the contents are identical the loader keeps the copy from `resolver/ingestion/config` and
records that the duplicate is safe to remove.  If the files differ the ingestion version is preferred.  The loader will
emit a deprecation warning (or raise when `strict_mismatch=True`) that lists the differing top-level keys so that teams
can reconcile the two copies.

During the deprecation window the fallback path still works, but diagnostics include a warning asking contributors to
move the file to `resolver/ingestion/config`.  Once the duplicate is removed the warning disappears automatically.

## Checklist for contributors

* Add or update connector configs in `resolver/ingestion/config/`.
* Remove legacy duplicates from `resolver/config/` once downstream jobs have migrated.
* Treat `resolver/config/` as global scope only (shared constants, mappings, semantics files, etc.).
* If you need to verify that a run picked the expected file, check the connector diagnostics – the config source and any
  loader warnings are written alongside the usual metadata and appear in `diagnostics/ingestion/summary.md`.
