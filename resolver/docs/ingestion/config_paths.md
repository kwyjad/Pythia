# Connector configuration paths

Connector YAML configuration files are migrating toward a single canonical location: `resolver/ingestion/config/`.
Until every job flips over, the legacy directory `resolver/config/` remains in play for backwards compatibility.

The shared loader used by the ingestion connectors currently searches paths in the following order:

1. `resolver/config/<connector>.yml` (legacy path, preferred while tests and jobs depend on it)
2. `resolver/ingestion/config/<connector>.yml` (fallback while the migration completes)

When both files exist and the contents are identical the loader keeps the copy from `resolver/config` and records that
the duplicate is safe to delete.  If the files differ the legacy copy still wins for determinism, but the loader emits a
`duplicate-mismatch` warning (or raises when `strict_mismatch=True`) that lists the differing top-level keys so that
teams can reconcile the two copies.  Once the repository finishes migrating, we will flip the order back so the
ingestion path becomes authoritative.

During this transition the fallback path still works, but diagnostics include a warning asking contributors to move the
file under `resolver/config` if only the ingestion copy is present.  When both copies match, diagnostics flag the
duplicate as redundant so it can be removed once the switchover happens.

## Checklist for contributors

* Keep the authoritative legacy copy in `resolver/config/` up to date until the migration completes.
* Mirror any changes into `resolver/ingestion/config/` so the eventual switchover is a no-op.
* Remove legacy duplicates only after every job points at the ingestion path.
* If you need to verify that a run picked the expected file, check the connector diagnostics – the config source,
  loader warnings, and resolved path are written alongside the usual metadata and appear in
  `diagnostics/ingestion/summary.md`.
