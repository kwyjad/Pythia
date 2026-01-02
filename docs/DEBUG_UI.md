# Debug UI

The Debug UI provides internal diagnostics for Horizon Scan triage and downstream counts.
It is intentionally gated behind **both** a build-time flag (to hide the nav item) and a
server-side token (to secure the API endpoints).

## Enable the UI locally

1. **Expose the nav item** (client-safe):

```bash
export NEXT_PUBLIC_FRED_DEBUG_UI=1
```

2. **Enable the server endpoints** (server-only):

```bash
export FRED_DEBUG_TOKEN="your-secret-token"
```

3. **Start the services** as usual. The Debug nav link appears when the UI flag is set.

## Authenticate requests

All `/v1/debug/*` endpoints require the `X-Fred-Debug-Token` header.

Example:

```bash
curl -H "X-Fred-Debug-Token: your-secret-token" \
  "http://localhost:8000/v1/debug/hs_runs"
```

Behavior:

- If `FRED_DEBUG_TOKEN` is **unset**, debug endpoints return **404** (disabled).
- If the token is set but the header is **missing or wrong**, debug endpoints return **403**.

## Using the Debug UI

1. Open `/debug` once the UI flag is enabled.
2. Enter the debug token in the password field (stored in localStorage).
3. Select an HS run and optional filters (ISO3, hazard code).
4. Review triage rows, LLM call metadata, and country/run counts.

## Safety notes

- Do **not** enable Debug UI on public deployments without proper auth controls.
- The UI only shows a preview of LLM responses (truncated by default).
