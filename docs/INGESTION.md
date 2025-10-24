# Ingestion Playbook

## DTM API (API-only)

The DTM connector now always calls the official IOM DTM API via the `dtmapi` Python package. Configuration lives in
`resolver/ingestion/config/dtm.yml` and contains a single `api:` block:

```yaml
enabled: true

api:
  admin_levels: [admin0, admin1, admin2]
  countries: []
  operations: []

output:
  measure: stock

field_aliases:
  idp_count:
    - "TotalIDPs"
    - "IDPTotal"
```

Adjust `admin_levels`, `countries`, or `operations` to scope the fetch. Leave `countries` and `operations` empty to pull all
available data.

### Secrets

* `DTM_API_KEY` (required) — primary subscription key from https://dtm-apim-portal.iom.int/.
* `DTM_API_SECONDARY_KEY` (optional) — secondary key used automatically when the primary key returns 401/403.

### Scoping fetches

* **Countries:** provide ISO names under `countries:`; an empty list fetches all.
* **Operations:** supply operation names when requesting admin2 data.
* **Date window:** the ingestion workflow window (`window_start`/`window_end`) is passed to the API automatically. Override at
  runtime via workflow inputs or export `DTM_NO_DATE_FILTER=true` to disable filtering.

Each run writes:

* `diagnostics/ingestion/dtm_run.json` — mode (`api`), window, filters, HTTP status counts, and per-level totals.
* `diagnostics/ingestion/dtm_api_request.json` — the request payload (admin levels, countries, operations, and window).
* `diagnostics/ingestion/dtm_api_response_sample.json` — the first 100 standardized output rows.
