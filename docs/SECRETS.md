# Secrets Reference

The ingestion connectors only reach their upstream APIs when the expected secrets are present. Export these values locally (for
example via `direnv` or a `.env` file) and configure the same names under **Settings → Secrets and variables → Actions** when
running in GitHub Actions.

| Secret | Used by | Notes |
|---|---|---|
| `DTM_API_KEY` | `resolver.ingestion.dtm_client` | Single IOM DTM subscription key passed to the official `dtmapi` SDK. Real runs exit with code `2` if the key is missing or invalid; use `--offline-smoke` / `DTM_OFFLINE_SMOKE=1` for header-only diagnostics without credentials. |

Remove any legacy references to `DTM_API_PRIMARY_KEY`, `DTM_API_SECONDARY_KEY`, or `DTM_SUBSCRIPTION_KEY`; the connector now reads only `DTM_API_KEY`.
