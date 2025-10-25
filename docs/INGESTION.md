# Ingestion Playbook

## DTM API (API-only)

The DTM connector exclusively uses the official IOM DTM API via the `dtmapi` package. Configuration lives in
`resolver/ingestion/config/dtm.yml` and contains a single `api:` block.

### Minimal configuration

```yaml
enabled: true

api:
  admin_levels: [admin0, admin1, admin2]
  countries: []          # discovery always fetches the full catalog

output:
  measure: stock         # or "flow" for month-over-month deltas
```

Adjust `admin_levels` to scope the fetch. The connector ignores any configured `countries` list and automatically
discovers the full catalog each run. Admin2 pulls may require the caller to specify `operations` via CLI or config when the
API exposes multiple programmes for a given country.

### Required secrets

* `DTM_API_KEY` — IOM DTM subscription key from <https://dtm-apim-portal.iom.int/>. Store this in GitHub secrets for CI.

### Running locally

```bash
python -m resolver.ingestion.dtm_client \
  --strict-empty            # exit with code 3 when zero rows are written (optional) \
  --no-date-filter          # ignore RESOLVER_START_ISO/RESOLVER_END_ISO (optional)
  --offline-smoke           # short-circuit for fast tests; exits 0 without hitting the API
```

Environment variables:

* `RESOLVER_START_ISO` / `RESOLVER_END_ISO` define the inclusive window passed to the API.
* `DTM_NO_DATE_FILTER=1` mirrors `--no-date-filter`.
* `DTM_STRICT_EMPTY=1` mirrors `--strict-empty`.
* `DTM_OFFLINE_SMOKE=1` mirrors `--offline-smoke` for CI fast tests.

### Diagnostics

Every run emits the following diagnostics under `diagnostics/ingestion/`:

* `dtm_run.json` — run summary including window, resolved countries, HTTP counters, paging stats, row totals, a `totals`
  block, CLI args, and status.
* `dtm_api_request.json` — sanitized API payload (admin levels, countries, operations, window).
* `dtm_api_sample.json` — the first 100 standardized output rows; always written when zero rows are produced.
* `dtm_api_response_sample.json` — kept for backwards compatibility; mirrors `dtm_api_sample.json`.
* `dtm/discovery_countries.csv` — snapshot of the SDK discovery response.
* `dtm/discovery_fail.json` — populated when discovery fails or returns zero countries (reasons include `missing_key`,
  `invalid_key`, `empty_discovery`).
* `dtm/dtm_http.ndjson` — trace of every SDK request made by the connector.
* `dtm_client.log` — enriched log stream including `api_base_url`, `resolved_country_count`, `resolved_admin_levels`, `date_window`, `request_params_clean`, and (when applicable) `zero_rows_reason`.

The connectors report (`diagnostics/ingestion/connectors_report.jsonl`) mirrors the status (`ok`, `ok-empty`, `error`,
`skipped`) and includes the reason string surfaced by `dtm_client`.

### Zero-rows troubleshooting

When the API returns zero rows, inspect `dtm_api_request.json` and `dtm_api_sample.json` for the exact parameters used. Common
causes include:

* Country name mismatch — confirm the resolved names in `dtm_run.json` match those exposed by DTM discovery.
* Window out of range — the requested `window.start` / `window.end` may pre-date the API dataset.
* Missing operation filter — admin2 pulls require explicit `operations` when the API exposes multiple programmes.

Use `--strict-empty` or set `DTM_STRICT_EMPTY=1` in CI to fail builds when the connector writes zero rows. Real runs now exit with code `2` when the API returns no rows; the diagnostics and connectors report include a `zero_rows_reason` (`empty_country_list`, `invalid_indicator`, `api_empty_response`, or `filter_excluded_all`). Pass `--offline-smoke` (or set `DTM_OFFLINE_SMOKE=1`) to produce header-only diagnostics with exit code `0` without requiring credentials.
