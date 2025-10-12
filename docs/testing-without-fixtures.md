# Running Tests Without Large Fixtures

Codex runs `pytest -q resolver/tests -k "not slow and not nightly"` before a
pull request is opened. Several parity and staging-schema tests assume a local
checkout of large CSV/JSON fixtures that live outside the public repository. In
Codex (and on laptops that only clone this repo) those files are missing, which
used to make the fast test target fail before code review began.

This helper layer keeps the fast suite green without modifying CI behaviour:

- **Auto-skip heavy fixture tests locally.** When fixture directories such as
  `resolver/exports/` are absent _and_ the run is not on GitHub Actions, tests
  whose names include `parity`, `staging_schema`, `schema_parity`, or
  `export_parity` are marked as skipped with a clear explanation. CI is
  unaffected because it always runs with the full fixture checkout.
- **Provide tiny synthetic exports when possible.** For lighter-weight checks
  (for example, CLI/DB round-trips that only need schema-compliant rows) the
  session-scoped `synthetic_data_dir` fixture builds a small set of CSV files and
  points helper utilities at them. Synthetic fixtures are only generated when
  real exports are missing locally.

In CI the real datasets continue to run: we never auto-skip or synthesise when
`GITHUB_ACTIONS=true`.

## Environment variables

| Variable | Default | Effect |
| --- | --- | --- |
| `RESOLVER_ALLOW_SYNTHETIC_FIXTURES` | `1` (unless on GitHub Actions) | Set to `0` to disable synthetic fixture generation entirely. Tests that need external data will be skipped instead of using synthetic CSVs. |
| `RESOLVER_TEST_DATA_DIR` | _auto-set_ | When synthetic fixtures are active the conftest module points this to the generated directory. You can set it manually to force the helpers to use a custom fixture checkout. |

## Local quickstart

1. Ensure optional dependencies (`duckdb`, `pytest`, etc.) are installed.
2. Run `pytest -q resolver/tests -k "not slow and not nightly"`.
   - Without external fixtures you will see a short notice in the output:
     `[resolver-tests] Using synthetic fixtures from ...`.
   - Parity and staging-schema suites will be reported as skipped with the
     reason `Skipping fixture-dependent test: fixture files not present (local/Codex mode)`.
3. When you have the real data checkout, set `RESOLVER_ALLOW_SYNTHETIC_FIXTURES=0`
   (or simply keep the fixtures in place) and the full parity suite will run
   exactly as it does in CI.

These guardrails aim to provide fast feedback inside Codex while preserving the
full fidelity checks in GitHub Actions.
