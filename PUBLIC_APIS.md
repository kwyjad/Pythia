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
    - `/v1/downloads/triage.csv`
    - `/v1/hs_runs`, `/v1/hs_triage/all`
    - `/v1/llm/costs`, `/v1/llm/costs/summary`
    - `/v1/costs/total`, `/v1/costs/monthly`, `/v1/costs/runs`, `/v1/costs/latencies`, `/v1/costs/run_runtimes`
    - `/v1/downloads/total_costs.csv`, `/v1/downloads/monthly_costs.csv`, `/v1/downloads/run_costs.csv`
    - `/v1/resolver/connector_status`, `/v1/resolver/country_facts`
    - `/v1/calibration/weights`, `/v1/calibration/advice`
  - **Admin (token required)**: Action endpoints (e.g., `POST /v1/run`).
    - Server token is sourced from `PYTHIA_API_TOKEN` (preferred) or `PYTHIA_API_KEY` (legacy fallback).
    - Legacy header `X-Pythia-Token` is still accepted for backwards compatibility.
    - If no token environment variable is set, admin endpoints fail closed with
      `503 Admin token not configured`.
  - **Admin-only (debug)**: `/v1/debug/*` endpoints are gated by `FRED_DEBUG_TOKEN`
    and are excluded from the public API surface.
- `GET /v1/question_bundle`
  - Query params: `question_id` (required), `hs_run_id`, `forecaster_run_id`, `include_llm_calls` (bool, default false), `include_transcripts` (bool, default false), `limit_llm_calls` (int, default 200).
  - Returns a bundle containing: the question row, HS run/scenarios/country report, ensemble SPD rows plus per-model SPD rows, question context/resolutions (plus `context.scores` when available), and optional `llm_calls` (including transcripts only when requested).
  - `llm_calls` may include internal-only diagnostic fields (`status`, `error_type`, `error_message`, `hazard_scores_json`, `hazard_scores_parse_ok`, `response_format`) that are not part of the public contract.
- `GET /v1/questions`
  - When `latest_only=true`, each row includes nullable triage fields: `triage_score`, `triage_tier`, `triage_need_full_spd`, `triage_date` (YYYY-MM-DD).
  - When `latest_only=true`, each row may include nullable regime change fields: `regime_change_likelihood`, `regime_change_direction`, `regime_change_magnitude`, `regime_change_score`, `regime_change_level`.
- `GET /v1/countries`
  - Each row includes nullable `highest_rc_level` and `highest_rc_score` fields sourced from the latest HS triage run.
- `GET /v1/diagnostics/kpi_scopes`
  - Query params: `metric_scope` (`PA` or `FATALITIES`, default `PA`), `year_month` (optional `YYYY-MM`).
  - Returns KPI counts for three scopes in a single payload:
    - `selected_run`: KPIs scoped to the selected run month.
    - `total_active`: `questions.status = 'active'`.
    - `total_all`: all questions.
  - Each scope returns `questions`, `forecasts`, `countries_with_forecasts`, `countries` (legacy alias of `countries_with_forecasts`), `resolved_questions`, and `forecasts_by_hazard`.
  - `selected_run` additionally returns `countries_triaged`: distinct ISO3 triaged in the selected run-month (primary: `llm_calls` `phase='hs_triage'`; fallback: `hs_triage` table). `total_active`/`total_all` continue to return `countries_total` as distinct ISO3 in-scope from questions.
  - Includes `available_months`, `selected_month`, `explanations`, `notes`, and `diagnostics` fields identifying the timestamp sources used for months and forecast counts.
- `GET /v1/risk_index`
  - Rows include `population` and per-capita fields `m1_pc..m6_pc` and `total_pc` for any metric when normalization is enabled and population data is available.
  - When `forecasts_ensemble` includes multiple ensemble aggregations (e.g., BayesMC + Mean), risk_index uses BayesMC per question when available, falling back to Mean.
  - For `PA`, monthly values blend hazards with `alpha=0.1` (`max_h + alpha * (sum_h - max_h)`), and `total` is the peak month across months 1–6.
  - For `FATALITIES`, monthly values sum hazards, and `total` is the sum across months 1–6.
- `GET /v1/downloads/forecasts.csv`
  - Streams a CSV export with one row per ISO3 × hazard × model × forecast_month.
  - Columns (in order): `ISO`, `country_name`, `year`, `month`, `forecast_month`, `metric`, `hazard`, `model`, `SPD_1..SPD_5`, `EIV`, `triage_score`, `triage_tier`, `regime_change_likelihood`, `regime_change_direction`, `regime_change_magnitude`, `regime_change_score`, `hs_run_ID`.
- `GET /v1/downloads/forecasts.xlsx`
  - Streams an Excel export with one row per ISO3 × hazard × model × forecast_month.
  - If the Excel dependency is unavailable, responds with a redirect to `/v1/downloads/forecasts.csv`.
  - Columns (in order): `ISO`, `country_name`, `year`, `month`, `forecast_month`, `metric`, `hazard`, `model`, `SPD_1..SPD_5`, `EIV`, `triage_score`, `triage_tier`, `regime_change_likelihood`, `regime_change_direction`, `regime_change_magnitude`, `regime_change_score`, `hs_run_ID`.
- `GET /v1/downloads/triage.csv`
  - Streams a CSV export with one row per run × country.
  - Columns (in order): `Triage Year`, `Triage Month`, `Triage Date`, `Run ID`, `Triage model`, `ISO3`, `Country`, `Triage Score`, `Triage Tier`.
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
      - `n_countries` counts distinct country-triage instances (`run_id × iso3`) at the requested grain.
- `GET /v1/resolver/connector_status`
  - Returns `rows` with `source` (ACLED, IDMC, EM-DAT), `last_updated` (YYYY-MM-DD or null), and `rows_scanned`.
  - Optional `diagnostics` includes `facts_source_table`, `fallback_used`, `missing_tables_checked`, `date_column_used`, and `rows_total`.
  - `last_updated` prefers `created_at` when available (then `publication_date`, `as_of_date`, `as_of`, and `ym_proxy` fallback).
  - ACLED status prefers the selected facts table when ACLED facts are present; it falls back to `acled_monthly_fatalities` only when facts are absent. Diagnostics include `acled_status_source_table` and `acled_status_date_column_used`.
- `GET /v1/resolver/country_facts`
  - Query params: `iso3` (required 3-letter code), `limit` (optional, default 5000).
  - Returns `rows` for the requested ISO3 with `iso3`, `hazard`, `hazard_code`, `source_id`, `year`, `month`, `metric`, and `value`.
  - Optional `diagnostics` includes `facts_source_table`, `fallback_used`, `missing_tables_checked`, and `rows_returned`.
  - ACLED rows are sourced from `acled_monthly_fatalities` when present and add `acled_rows_added` + `acled_country_rows_total`.

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
