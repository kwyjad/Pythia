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
