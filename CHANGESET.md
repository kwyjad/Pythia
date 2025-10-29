# Pending Changes

- Fix connectors runner ordering so env-derived safety flags (for example, `--soft-timeouts`) precede user-provided extras, keeping fast-test expectations intact, and update the Codex canary workflow to avoid disallowed `${{ runner.* }}` contexts so actionlint passes.
