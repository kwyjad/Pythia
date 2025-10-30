# Ingestion playbook

## DTM connector troubleshooting

### Missing `id_or_path`

The DTM client now runs a configuration preflight before any files are read. If
one or more entries in `resolver/ingestion/config/dtm.yml` omit the required
`id_or_path` field, the connector:

1. Writes a machine-readable report to `diagnostics/ingestion/dtm_config_issues.json`.
   Each invalid source is emitted with an `error` of `missing id_or_path` plus any
   other metadata the entry originally declared.
2. Emits a diagnostics row in `diagnostics/ingestion/connectors_report.jsonl`
   with `reason="missing id_or_path"`, `extras.invalid_sources`, and
   `extras.skipped_sources` summarising the skipped entries.
3. Continues with the remaining (valid) sources, writing header-only output when
   every entry was skipped. Pass `--fail-on-missing-config` (or set
   `DTM_STRICT=1`) to make the run exit immediately with code `2` instead.

Fix the report by adding a concrete `id_or_path` (Drive ID, S3 key, or absolute
path) to each invalid entry, then re-run the connector. The diagnostics line will
flip back to `status="ok"` and the config issues file will list `invalid: 0`.

### Debug triage

Set `LOG_LEVEL=DEBUG` to ask the connector for richer breadcrumbs. The resolved
sources snapshot (`diagnostics/ingestion/dtm_sources_resolved.json`) mirrors the
post-preflight source list so you can confirm inferred column names, resolved
paths, and skip reasons locally before a retry.

## IDMC skeleton smoke test

The IDMC connector is currently offline-only and exists to wire fixtures, normalization, and diagnostics. Run the smoke test locally with:

```bash
python -m resolver.ingestion.idmc.cli --skip-network
```

The command appends a diagnostics row to `diagnostics/ingestion/connectors.jsonl` and writes a normalized preview CSV to `diagnostics/ingestion/idmc/normalized_preview.csv`.
