# Static ingestion assets

- `iso3_master.csv` holds the canonical list of ISO3 codes used by ingestion
  connectors. The IDMC client resolves `countries: []` (or a missing `countries`
  block) to this roster so an empty list means “fetch every country” rather than
  “fetch none,” and honours environment overrides via `IDMC_COUNTRIES` or
  `IDMC_COUNTRIES_FILE` when you need to narrow the scope for a run.【F:resolver/ingestion/utils/country_utils.py†L118-L209】【F:resolver/ingestion/idmc/cli.py†L223-L323】 The CLI also accepts `--start`/`--end` (or `RESOLVER_START_ISO`/`RESOLVER_END_ISO`, falling back to `BACKFILL_*`) so CI workflows can inject the desired window while leaving local runs free to choose their own span.【F:resolver/ingestion/idmc/cli.py†L300-L384】
- `iso3_roster.csv` is a slimmer export that mirrors the master list and is used
  by some legacy scripts.
- `dtm_admin0_fake.csv` ships fixture rows that let the DTM exporter run in
  offline smoke tests.【F:resolver/tools/export_facts.py†L180-L212】
