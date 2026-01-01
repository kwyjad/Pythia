# Public APIs & Contracts

## FastAPI (pythia.api)

- **Authentication**
  - **Public (no auth)**: All read-only `GET` endpoints are public, including:
    - `/v1/diagnostics/kpi_scopes`
    - `/v1/diagnostics/summary`
    - `/v1/risk_index`, `/v1/rankings`
    - `/v1/questions`, `/v1/question_bundle`, `/v1/ui_runs/{ui_run_id}`
    - `/v1/forecasts/*`, `/v1/resolutions`
    - `/v1/downloads/forecasts.csv`, `/v1/downloads/forecasts.xlsx`
    - `/v1/llm/costs`, `/v1/llm/costs/summary`
    - `/v1/costs/total`, `/v1/costs/monthly`, `/v1/costs/runs`, `/v1/costs/latencies`, `/v1/costs/run_runtimes`
    - `/v1/downloads/total_costs.csv`, `/v1/downloads/monthly_costs.csv`, `/v1/downloads/run_costs.csv`
    - `/v1/calibration/weights`, `/v1/calibration/advice`
  - **Admin (token required)**: Action endpoints (e.g., `POST /v1/run`).
    - Server token is sourced from `PYTHIA_API_TOKEN` (preferred) or `PYTHIA_API_KEY` (legacy fallback).
    - Legacy header `X-Pythia-Token` is still accepted for backwards compatibility.
    - If no token environment variable is set, admin endpoints fail closed with
      `503 Admin token not configured`.
- `GET /v1/question_bundle`
  - Query params: `question_id` (required), `hs_run_id`, `forecaster_run_id`, `include_llm_calls` (bool, default false), `include_transcripts` (bool, default false), `limit_llm_calls` (int, default 200).
  - Returns a bundle containing: the question row, HS run/scenarios/country report, ensemble SPD rows plus per-model SPD rows, question context/resolutions (plus `context.scores` when available), and optional `llm_calls` (including transcripts only when requested).
- `GET /v1/questions`
  - When `latest_only=true`, each row includes nullable triage fields: `triage_score`, `triage_tier`, `triage_need_full_spd`, `triage_date` (YYYY-MM-DD).
- `GET /v1/diagnostics/kpi_scopes`
  - Query params: `metric_scope` (`PA` or `FATALITIES`, default `PA`), `year_month` (optional `YYYY-MM`).
  - Returns KPI counts for three scopes in a single payload:
    - `selected_run`: KPIs scoped to the selected run month.
    - `total_active`: `questions.status = 'active'`.
    - `total_all`: all questions.
  - Each scope returns `questions`, `forecasts`, `countries_with_forecasts`, `countries_total`, `countries` (legacy alias of `countries_with_forecasts`), `resolved_questions`, and `forecasts_by_hazard`.
  - Includes `available_months`, `selected_month`, `explanations`, `notes`, and `diagnostics` fields identifying the timestamp sources used for months and forecast counts.
- `GET /v1/risk_index`
  - Rows include `population` and per-capita fields `m1_pc..m6_pc` and `total_pc` for any metric when normalization is enabled and population data is available.
  - When `forecasts_ensemble` includes multiple ensemble aggregations (e.g., BayesMC + Mean), risk_index uses BayesMC per question when available, falling back to Mean.
- `GET /v1/downloads/forecasts.csv`
  - Streams a CSV export with one row per ISO3 × hazard × model × forecast_month.
  - Columns (in order): `ISO`, `country_name`, `year`, `month`, `forecast_month`, `metric`, `hazard`, `model`, `SPD_1..SPD_5`, `EIV`, `triage_score`, `triage_tier`, `hs_run_ID`.
- `GET /v1/downloads/forecasts.xlsx`
  - Streams an Excel export with one row per ISO3 × hazard × model × forecast_month.
  - If the Excel dependency is unavailable, responds with a redirect to `/v1/downloads/forecasts.csv`.
  - Columns (in order): `ISO`, `country_name`, `year`, `month`, `forecast_month`, `metric`, `hazard`, `model`, `SPD_1..SPD_5`, `EIV`, `triage_score`, `triage_tier`, `hs_run_ID`.
- `GET /v1/costs/total`, `/v1/costs/monthly`, `/v1/costs/runs`
  - Returns `tables.summary`, `tables.by_model`, and `tables.by_phase` using the same schema as the cost CSV downloads.
- `GET /v1/costs/latencies`
  - Returns `rows` with run/model/phase latency percentiles (p50/p90).
- `GET /v1/costs/run_runtimes`
  - Returns `rows` with per-run elapsed_ms totals by phase plus total.
  - Includes per-run p50/p90 for total elapsed_ms per question_id and per iso3.
- `GET /v1/downloads/total_costs.csv`, `/v1/downloads/monthly_costs.csv`, `/v1/downloads/run_costs.csv`
  - Streams tidy CSVs for costs with these columns:
    - `grain` (`total`, `monthly`, `run`)
    - `row_type` (`summary`, `by_model`, `by_phase`)
    - `year`, `month`, `run_id`
    - `model` (only populated for `by_model`)
    - `phase` (phase group for `by_phase`)
    - `total_cost_usd`, `n_questions`, `avg_cost_per_question`, `median_cost_per_question`
    - `n_countries`, `avg_cost_per_country`, `median_cost_per_country`

## resolver.query.db_reader

- `fetch_deltas_point(conn, *, ym, iso3, hazard_code, cutoff, preferred_metric)`
- `fetch_resolved_point(conn, *, ym, iso3, hazard_code, cutoff, preferred_metric)`

## resolver.db.conn_shared

- `normalize_duckdb_url(db_url)`
- `get_shared_duckdb_conn(db_url, *, force_reopen=False)`
- `clear_cached_connection(db_url)`

## resolver.ingestion.idmc

- `config.IdmcConfig` now exposes `api.base_url`, named `endpoints`, and `cache`
  controls (`dir`, `ttl_seconds`, `force_cache_only`).
- `client.fetch(cfg, *, network_mode="live", window_days=30, only_countries=None,
  base_url=None, cache_ttl=None)` returns the offline fixtures plus an
  `idus_view_flat` DataFrame and diagnostics with probe/http/cache fields.
- `http.http_get(url, *, headers=None, timeout=10.0, retries=2, backoff_s=0.5)`
  performs a guarded GET request (adds `Authorization` when `IDMC_API_TOKEN` is
  set) and surfaces retry diagnostics.
- `cache.cache_key`, `cache.cache_get`, and `cache.cache_put` provide a small
  file-backed cache for HTTP responses used by the connector and CLI.

### Upstream source reference

| Source | Base URL | Endpoints | Auth mode | Terms URL |
| --- | --- | --- | --- | --- |
| IDMC (IDU) | `https://www.internal-displacement.org` | `idus_json`, `idus_geo` (REST) | Optional bearer token via `IDMC_API_TOKEN` | _Pending legal confirmation_ |

## TypeScript / JavaScript

(none)
