# Data Dictionary — Resolver (A3)

This dictionary defines the **facts** records the resolver stores and reads.  
All records **must** include both **country_name + iso3** and **hazard_label + hazard_code + hazard_class** (aligned to the registries in `/resolver/data/`).

## Canonical registries (authoritative)
- `/resolver/data/countries.csv` — columns: `country_name,iso3`
- `/resolver/data/shocks.csv` — columns: `hazard_code,hazard_label,hazard_class`  
  (`hazard_class` ∈ `natural | human-induced | epidemic`; **earthquakes excluded** per scope)

## Table: facts (append-only)
| Field | Type | Req | Description |
|---|---|:--:|---|
| event_id | string | R | Stable ID for a country–hazard–episode–revision |
| country_name | string | R | From countries registry |
| iso3 | string(3) | R | From countries registry |
| hazard_code | string | R | From shocks registry |
| hazard_label | string | R | From shocks registry |
| hazard_class | string | R | `natural|human-induced|epidemic` (from shocks registry) |
| metric | enum | R | `in_need` (PIN) \| `affected` \| `displaced` \| `cases` … |
| value | number | R | Non-negative integer (persons) |
| unit | enum | R | `persons` \| `persons_cases` (for outbreaks) |
| as_of_date | date(YYYY-MM-DD) | R | Date figure refers to |
| publication_date | date | R | Publication date of source |
| publisher | string | R | OCHA, IFRC, NDMA, UNHCR, IOM-DTM, WHO, etc. |
| source_type | enum | R | `appeal|sitrep|gov|cluster|agency|media` |
| source_url | string | R | Canonical URL of doc/portal |
| doc_title | string | R | Title of doc/page |
| definition_text | text | R | Verbatim local definition of who is counted |
| method | enum | R | `api|scrape|manual` |
| confidence | enum | R | `high|med|low` |
| revision | int | R | 1…n (newer supersedes older) |
| artifact_id | string | O | Path/key in object storage |
| artifact_sha256 | string | O | Hash of saved artifact |
| notes | text | O | Free-form notes |
| alt_value | number | O | Kept when conflict rule discards an eligible figure |
| alt_source_url | string | O | Source of alternative value |
| proxy_for | enum/null | O | `PIN` when using proxy (e.g., IPC P3+) |
| precedence_decision | text | O | Brief rationale (e.g., “HNO superset; NDMA older”) |
| ingested_at | datetime(UTC) | R | Insert timestamp |

**Constraints & validation**
- `value >= 0`; if `value > national_population(iso3)` ⇒ flag `confidence=low`.
- `as_of_date <= publication_date <= now`.
- Uniqueness hint (soft): `(iso3, hazard_code, metric, as_of_date, publisher, revision)`.
- `metric='in_need'` only when source explicitly publishes **PIN**.
- For outbreaks where only cases exist, use `metric=cases`, `unit=persons_cases` (do **not** coerce to PIN).

## Derived artifact: resolver_features

`data/resolver_features.parquet` materialises resolver deltas into features the Forecaster calibration loop consumes.

| Column | Type | Description |
| --- | --- | --- |
| country_iso3 | string | ISO-3166 alpha-3 country code. |
| hazard_code | string | Resolver hazard taxonomy slug. |
| metric | enum | `in_need`, `affected`, or `displaced` (configurable). |
| ym | string | Resolver month key (`YYYY-MM`). |
| delta_m1 | float | Monthly `value_new` from `facts_deltas` (baseline feature). |
| delta_m3_sum | float | Rolling 3-month sum of `delta_m1` per `(iso3, hazard, metric)`. |
| delta_m6_sum | float | Rolling 6-month sum of `delta_m1`. |
| delta_m12_sum | float | Rolling 12-month sum of `delta_m1`. |
| delta_zscore_6m | float | Z-score of `delta_m1` against the trailing 6-month mean/std. |
| sudden_spike_flag | boolean | True when `|delta_zscore_6m| >= 3`. |
| missing_month_flag | boolean | True when the prior observation is more than 1 month back. |
| as_of_date | string | ISO date used in precedence output (falls back to delta `as_of`). |
| as_of_recency_days | integer | Days between `as_of_date` and feature build timestamp. |
| source_tier | string | Precedence tier recorded in `facts_resolved`. |
| hazard_class | string | Hazard class from the shocks registry. |
| generated_at_utc | string | Feature build timestamp (`YYYY-MM-DDTHH:MM:SSZ`). |

## Table: emdat_pa

| Column | Type | Description |
| --- | --- | --- |
| iso3 | string | ISO 3166-1 alpha-3 country code. |
| ym | string | Month bucket in `YYYY-MM` aligned to EM-DAT disaster start dates. |
| shock_type | enum | `drought`, `tropical_cyclone`, or `flood` after EM-DAT subtype mapping. |
| pa | integer | People affected (EM-DAT Total Affected) summed per country-month-shock. |
| as_of_date | date | Data currency date from EM-DAT metadata (probe info timestamp). |
| publication_date | date | Latest EM-DAT `last_update` (fallback `entry_date`) contributing to the bucket. |
| source_id | string | Source identifier (`emdat`). |
| disno_first | string | Lowest EM-DAT `disno` contributing to the aggregate (traceability). |

## Table: acled_monthly_fatalities

Monthly fatalities sourced from ACLED and bucketed by country-month. The table schema and detailed notes live in [`data_dictionary/acled_monthly_fatalities.md`](data_dictionary/acled_monthly_fatalities.md).

- **Source:** ACLED
- **Primary key:** `iso3`, `month`
- **Metric:** Fatalities (people)
- **Granularity:** ISO3 country × month (`YYYY-MM-01`)
