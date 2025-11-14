# Data Dictionary — `acled_monthly_fatalities`

- **Source:** ACLED (Armed Conflict Location & Event Data Project)
- **Granularity:** Country-month (ISO3 + first of month)
- **Metric:** Fatalities (count)
- **Unit:** People
- **Primary key:** `iso3`, `month`

## Columns

| Column | Type | Notes |
| --- | --- | --- |
| `iso3` | string | ISO 3166-1 alpha-3 country code reported by ACLED. |
| `month` | date (`YYYY-MM-01`) | Month bucket derived from `event_date` (truncated to the first day of the month). |
| `fatalities` | integer | Sum of ACLED reported fatalities for the bucket. |
| `source` | string | Constant `"ACLED"` provenance label. |
| `updated_at` | timestamp (UTC) | Time the aggregation was produced. |

## Related resources

- Schema definition: [`SCHEMAS.md`](../../../SCHEMAS.md#dbacledmonthlyfatalities)
- Connector overview: [`ACLED` in the connectors catalog](../connectors_catalog.md)
