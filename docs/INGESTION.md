# Ingestion Playbook

## DTM API mode

The DTM connector supports both legacy file ingestion and a modern API-backed mode. API mode is enabled when either of the
following configuration styles is used:

```yaml
# Top-level API block
api: {}

# or explicit source entry
sources:
  - type: api
```

A sample configuration is available at `resolver/ingestion/config/dtm.sample.yml`. Both styles may include additional fields
such as `admin_levels`, `countries`, and `operations` to scope the fetch.

### Secrets

Set the secret `DTM_API_PRIMARY_KEY` (exposed to the connector as the `DTM_API_KEY` environment variable) before running in API
mode. Without the key the connector will fall back to header-only output and log a warning.

### Scoping fetches

* **Countries:** provide ISO-3 names or labels under `countries:` to fetch a subset; leave empty (`[]`) to fetch all.
* **Operations:** supply operation names in `operations:` when requesting admin2 data.
* **Date window:** the ingestion workflow window (`window_start`/`window_end`) is passed to the API automatically. Override the
  window at runtime via workflow inputs or export `DTM_NO_DATE_FILTER=true` to disable date filtering.

When API mode is active the connector logs `Using DTM API mode (trigger=…)` and writes `diagnostics/ingestion/dtm_run.json`
containing the selected mode, trigger, filters, HTTP status counts, and row totals per admin level.
