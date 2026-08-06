# Interpreter module

One short, plain-language report per cycle, explaining Fred's output to a
reader with no forecasting background. Part A covers the current forecast run
(what deserves attention and why); Part B covers the most recent scored run
(how well the system performed). Generation is fully automatic — no draft
state, no approval gate.

Status: **Phases 0–3 built** (deterministic metrics, input packs, the
generator itself). Phase 4 (deep validation), Phase 5 (API + dashboard +
map), Phase 6 (PDF + workflow wiring) are pending — the runner is NOT yet
invoked by any workflow.

## Design principles (short form)

1. All arithmetic happens in SQL/Python; the model explains, it never
   calculates. Every number arrives pre-computed in the input pack.
2. The model emits `{{fig:<key>}}` placeholders, never numerals in prose;
   the renderer substitutes values from the pack. Numeric hallucination is
   removed by construction.
3. Every claim carries its `question_ids`.
4. Binary (0–1) and SPD (0–2) Brier are never blended.
5. Failures are stored and visible (`status='failed_*'`), never silent.
6. Nothing here may fail the pipeline: `main()` returns 0 in every outcome.

## The pipeline

```
Phase 1 (pythia/tools/):        base_rate_spd -> compute_deviation (forecast_deviation)
                                score_baselines (__ext_climatology / __ext_uniform)
Phase 2 (scripts/ai_bundle/):   build_current_run_bundle (attention index, deltas,
                                blind spots, question records, 250k-token cap)
                                + skill columns in the scored bundle
Phase 3 (interpreter/):         pack -> Opus 5 (role: interpreter, effort=high)
                                -> JSON (schema-validated) -> markdown -> interpretations
```

## CLI

```
python -m interpreter.run \
  --db "$PYTHIA_DB_URL" \
  --kind combined \
  --pack ai_bundle/current_run_analysis__2026-08.zip \
  [--run-id fc_...] [--hs-run-id hs_...] [--scored-run-id 2026-08] \
  [--template-version v1] [--force] [--dry-run] [--out-dir ./out]
```

- `--kind scored` + the scored bundle: writes the performance-side
  interpretation (run in `compute_calibration_pythia.yml` once Phase 6 wires
  it).
- `--kind combined` + the current-run bundle: the full report; Part B is
  folded in from the most recent stored `scored` interpretation (its figure
  map is persisted in `figures_json`, so the scored pack is not needed at
  combined time). No scored row → an explicit "no scored run available yet"
  statement, never an omission.
- `--force` appends a new version row for the same run; default is skip when
  a `status='ok'` row exists.
- `--dry-run` assembles and (with `--out-dir`) writes the prompt, calls
  nothing, stores nothing.

## Package layout

```
interpreter/
  config.py     # env-driven knobs (see below)
  lexicon.py    # the fixed probability-word bands + appendix table
  schema.py     # the structured-output contract (jsonschema draft-07)
  packs.py      # loads the Phase 2 bundles; figure maps; token-capped input
  prompts.py    # template assembly from templates/{version}/
  templates/v1/ # system.md, current_run.md, scored_run.md, report_skeleton.md
  render.py     # JSON -> markdown; {{fig:...}} substitution; misses tracked
  store.py      # interpretations table writes; versioning; latest_scored
  run.py        # CLI orchestrator (model seam: interpreter.run._call_model)
  tests/
```

## Model configuration

- Role `interpreter` → the `claude` alias (claude-opus-5) in
  `pythia/config.yaml`; `PYTHIA_INTERPRETER_MODEL_ID` overrides.
- Thinking is requested explicitly via `ModelSpec.thinking` →
  `output_config: {effort: ...}` (wired in `forecaster/providers.py`, gated
  by `_ANTHROPIC_EFFORT_PREFIXES`; default level `high`,
  `PYTHIA_INTERPRETER_THINKING` overrides — thinking shares `max_tokens`,
  which is why the call runs at the 32k SPD-class ceiling).
- Temperature is never set (the opus-5 prefix guard would drop it anyway).
- Every call writes a rich `llm_calls` row with `phase='interpreter'`
  (`call_type='interpreter_<kind>'`), mapped to the `interpreter` bucket in
  `resolver/query/costs.py` and the Costs page. One call ≈ 250k in / ~20k
  out ≈ $2 at Opus 5 rates.
- No prompt caching (one call, no reuse), no batching (the report must be
  timely).

## Storage

`interpretations` (pythia/db/schema.py + interpreter/store.py, is_test
None-and-inherit): kind (`current`/`scored`/`combined`), run keys, integer
`version` per (kind, run key), template/model/thinking provenance,
`prompt_hash`/`pack_hash`, `content_json` (validated model output),
`content_md` (rendered report), `figures_json` (the resolved figure maps —
what makes the combined report reproducible without the scored pack),
`status` (`ok` / `failed_validation` / `failed_generation`),
`validation_json`, cost/tokens. Failed outputs are stored too, so they can
be inspected; the dashboard (Phase 5) renders them behind a warning banner.

## Env vars

| var | default | purpose |
|---|---|---|
| `PYTHIA_INTERPRETER_ENABLED` | `1` | kill switch |
| `PYTHIA_INTERPRETER_MODEL_ID` | unset | overrides the config role |
| `PYTHIA_INTERPRETER_TEMPLATE_VERSION` | `v1` | |
| `PYTHIA_INTERPRETER_MAX_PACK_TOKENS` | `250000` | model-input hard cap |
| `PYTHIA_INTERPRETER_TOP_N` | `8` | attention list length |
| `PYTHIA_INTERPRETER_PER_CAPITA_FLOOR` | `10000` | per-capita ranking floor |
| `PYTHIA_INTERPRETER_THINKING` | `high` | output_config.effort level |
| `PYTHIA_INTERPRETER_MAX_OUTPUT_TOKENS` | `32768` | shared with thinking |
| `PYTHIA_INTERPRETER_TIMEOUT_SEC` | `900` | call timeout |
| `PYTHIA_INTERPRETER_STRICT_VALIDATION` | `0` | Phase 4: failure suppresses publication |

## Validation (current state vs Phase 4)

Phase 3 validates SHAPE (jsonschema + kind-conditional requirements) and
tracks unresolved figure placeholders. Phase 4 adds: the bare-numeral prose
lint, lexicon band agreement, referential checks (every question_id exists
in the pack), and the independent-SQL numeric guard — plus
`PYTHIA_INTERPRETER_STRICT_VALIDATION=1` suppressing publication.

## Tests / CI

`interpreter/tests/` — fully deterministic, model call mocked, temp DuckDB,
bundle-directory fixtures. Run by `.github/workflows/interpreter-ci.yml`
(path-filtered on `interpreter/**` + `scripts/ai_bundle/**`, which also
re-runs the bundle suites — the interpreter consumes those bundles, so a
pack-shape change must re-run both sides).
