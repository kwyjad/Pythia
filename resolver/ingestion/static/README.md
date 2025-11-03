# Static ingestion assets

- `iso3_master.csv` holds the canonical list of ISO3 codes used by ingestion
  connectors. The IDMC client resolves `countries: []` (or a missing `countries`
  block) to this roster so an empty list means “fetch every country” rather than
  “fetch none.”【F:resolver/ingestion/utils/country_utils.py†L7-L76】【F:resolver/ingestion/idmc/client.py†L433-L486】
- `iso3_roster.csv` is a slimmer export that mirrors the master list and is used
  by some legacy scripts.
- `dtm_admin0_fake.csv` ships fixture rows that let the DTM exporter run in
  offline smoke tests.【F:resolver/tools/export_facts.py†L180-L212】
