# Ingestion Playbook

## DTM API (API-only)

The DTM connector exclusively uses the official IOM DTM API via the `dtmapi` package. Configuration lives in
`resolver/ingestion/config/dtm.yml` and contains a single `api:` block.

### Minimal configuration

```yaml
enabled: true

api:
  admin_levels: [admin1, admin0]
  countries: []          # empty => discover all available countries
  operations: []         # optional; required when fetching admin2 data

output:
  measure: stock         # or "flow" for month-over-month deltas

field_aliases:
  idp_count: ["TotalIDPs", "IDPTotal"]
```

Adjust `admin_levels`, `countries`, or `operations` to scope the fetch. Provide ISO3 codes or country names in
`countries`; the connector resolves them to the accepted API form at runtime. Leave `countries` empty to pull data for every
published country.

### Required secrets

* `DTM_API_PRIMARY_KEY` (preferred) or `DTM_API_KEY` — primary subscription key from <https://dtm-apim-portal.iom.int/>.
* `DTM_API_SECONDARY_KEY` (optional) — secondary key retried automatically when the primary key returns `401/403`.

### Running locally

```bash
python -m resolver.ingestion.dtm_client \
  --strict-empty            # exit with code 3 when zero rows are written (optional) \
  --no-date-filter          # ignore RESOLVER_START_ISO/RESOLVER_END_ISO (optional)
```

Environment variables:

* `RESOLVER_START_ISO` / `RESOLVER_END_ISO` define the inclusive window passed to the API.
* `DTM_NO_DATE_FILTER=1` mirrors `--no-date-filter`.
* `DTM_STRICT_EMPTY=1` mirrors `--strict-empty`.

### Diagnostics

Every run emits the following diagnostics under `diagnostics/ingestion/`:

* `dtm_run.json` — run summary including window, resolved countries, HTTP counters, paging stats, row totals, and status.
* `dtm_api_request.json` — sanitized API payload (admin levels, countries, operations, window).
* `dtm_api_sample.json` — the first 100 standardized output rows; always written when zero rows are produced.
* `dtm_api_response_sample.json` — kept for backwards compatibility; mirrors `dtm_api_sample.json`.

The connectors report (`diagnostics/ingestion/connectors_report.jsonl`) mirrors the status (`ok`, `ok-empty`, `error`,
`skipped`) and includes the reason string surfaced by `dtm_client`.

### Zero-rows troubleshooting

When the API returns zero rows, inspect `dtm_api_request.json` and `dtm_api_sample.json` for the exact parameters used. Common
causes include:

* Country name mismatch — confirm the resolved names in `dtm_run.json` match those exposed by DTM discovery.
* Window out of range — the requested `window.start` / `window.end` may pre-date the API dataset.
* Missing operation filter — admin2 pulls require explicit `operations` when the API exposes multiple programmes.

Use `--strict-empty` or set `DTM_STRICT_EMPTY=1` in CI to fail builds when the connector writes zero rows.
