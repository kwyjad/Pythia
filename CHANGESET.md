# Pending Changes

- Fix connectors runner ordering so env-derived safety flags (for example, `--soft-timeouts`) precede user-provided extras, keeping fast-test expectations intact, and update the Codex canary workflow to avoid disallowed `${{ runner.* }}` contexts so actionlint passes.
- feat(idmc): introduce offline IDMC (IDU) connector skeleton with fixtures, docs, and deterministic tests.
- fix(export-config): repair YAML mapping regression, add overlay stub, and guard parsing with a fast test.
- feat(idmc): enable IDU online fetch with reachability probe, cache utilities, CLI overrides, and offline-first tests.
- feat(idmc): convert IDU payloads into monthly new displacement rows with drop diagnostics, timing metrics, and zero-row rescue tooling.
- feat(idmc): feature flags & connector contract suite (offline deterministic).
- feat(idmc): wire IDU flow to facts (export adapter, gated, offline tests).
- feat(idmc): add optional hazard mapping with Resolver codes, CLI flag, and
  diagnostics preview for unmapped rows.
- feat(idmc): emit precedence-ready candidates, opt-in CLI wiring for
  selection, offline tests, and docs.
- feat(idmc): add rate limiting, month chunking, streaming cache, and
  performance diagnostics with offline tests.
- feat(idmc): add provenance manifests, diagnostics redaction, schema/docs, and
  compliance metadata for governance.
- feat(precedence): add union tool, offline tests, opt-in canary workflow, and
  documentation updates.
- ci(idmc): wire IDMC into monthly/backfill workflows with probe, harness defaults, and skip gating.
- ci(idmc): zero-row guardrails, staging bootstrap, and improved summary with config path.
- fix(dtm): prefer resolver/config for shared loader, surface config_path_used & countries_mode, restore fast tests.
- fix(ci): repair summarize_connectors indentation, add syntax guard, and ensure import sanity test covers the module.
- fix(hs-triage): stop truncating responses, map PASS_1/PASS_2 calls, use legacy error_text, and expose average source.
- fix(hs-triage): add provider cooldowns + retries, guaranteed fallback/repair, rerun lists, and coverage artifacts.
- fix(db): add llm_calls telemetry columns with migration/backfill helper and schema docs.
- fix(resolver-ui): include ACLED status and country facts from acled_monthly_fatalities.
