# Interpreter module

One short, plain-language report per cycle, explaining Fred's output to a
reader with no forecasting background. Part A covers the current forecast run
(what deserves attention and why); Part B covers the most recent scored run
(how well the system performed). Generation is fully automatic — no draft
state, no approval gate.

Status: **Phases 0–6 built — fully wired.** The combined report is generated
at the end of every forecast cycle (`run_sibyl.yml`), the scored report at
the end of every calibration cycle (`compute_calibration_pythia.yml`), the
PDF is rendered in CI and attached to the `pythia-data-latest` release, and
the dashboard's Report page links to it.

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
Phase 6 (interpreter/pdf.py):   interpretations row -> HTML -> WeasyPrint
                                -> report__{YYYY-MM}__v{n}.pdf + latest copy
                                -> pythia-interpreter-report artifact -> release
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
  interpretation (runs in `compute_calibration_pythia.yml` right after the
  scored-bundle step).
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
  pdf.py        # Phase 6: stored row -> HTML -> WeasyPrint -> PDF
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

## Validation (Phase 4 — `interpreter/validate.py`)

Four checks run after generation, before storage; each is reported
separately in `validation_json` and any failure sets
`status='failed_validation'` (the report is still stored and rendered, so
failures stay inspectable):

1. **schema** — jsonschema shape + kind-conditional requirements.
2. **referential** — every cited `question_id` exists in the pack; every
   figure reference (`figure_refs` entries AND inline `{{fig:...}}`
   placeholders) resolves against that entry's figure maps.
3. **numeric** — every referenced figure that can be re-derived with
   independent SQL (`forecast_deviation`, `hs_triage` via `questions`) is
   recomputed from the DB and compared (rtol 1e-6). A mismatch is a pack
   bug and fails the check. No DB / no source tables → the check is
   SKIPPED and says so — a missing DB never reads as a validated pack.
4. **prose** — the bare-numeral lint (digits outside a placeholder fail,
   with a whitelist for calendar month/year references), lexicon band
   agreement (a lexicon word attached to a probability placeholder must sit
   in that figure's band; lexicon phrases are matched longest-first and
   consumed, and ambiguous sentences — two distinct words — are skipped
   rather than guessed at), and a per-field length cap.

A crashed check is reported as a failed check, never an unhandled error.
`PYTHIA_INTERPRETER_STRICT_VALIDATION=1` additionally suppresses the
publication artifacts (`--out-dir` files; the Phase 5/6 consumers honour
the same flag) — the row is still stored, suppressed but never silent.

## API + dashboard (Phase 5)

`pythia/api/routes/interpreter.py` (registered in app.py; route modules never
import app.py):

- `GET /v1/interpreter/latest?kind=&include_test=` — highest version of the
  newest run; `has_interpretation: false` on a pre-interpreter DB (never a
  500).
- `GET /v1/interpreter/versions?kind=&run_id=&include_test=` — light rows
  for the version selector.
- `GET /v1/interpreter/{interpretation_id}` — one report (any status; the
  frontend banners non-ok rows). Declared last so literal routes win.
- `GET /v1/interpreter/attention_map?run_id=&include_test=` — per-ISO3
  attention (max `js_vs_baserate`/ln 2 over the run's questions, preferred
  aggregate per question). Coloured by attention, deliberately not raw risk.

Placeholders are resolved SERVER-side (`interpreter/render.py::
resolve_content` → the `content_resolved` field) so figure formatting has
exactly one implementation — the dashboard never re-formats numbers.

Frontend (`web/src/app/interpreter/`, Nav item "Report"): the attention map
(`RiskIndexMap` widened with optional injected-value props; click a
highlighted country to jump to its section), version selector, visible
warn/error banners for `failed_validation`/`failed_generation`, an explicit
no-report empty state, the report body rendered from `content_resolved`
with per-question links, the fixed lexicon + provenance appendix, the
shared glossary (`web/src/lib/score_glossary.ts` — the same constants the
Performance page tooltips import, so the two never drift), a **run
selector** defaulting to the most recent run (options are grouped per run by
`groupVersionsByRun`, labelled with the month the report is ABOUT via
`runMonthLabel`, which parses `hs_run_id` exactly as `pdf.py::month_label`
does; a per-run version selector appears only when that run has more than
one version), and a **Download PDF** button linking to the release asset (URL from
`/v1/version`'s `interpreter_report_url` manifest passthrough; the button
hides when no asset exists). When the release advertises a report PDF but the API serves no report, the
page says so explicitly ("still syncing") instead of rendering the
no-report empty state — the API's database lagging the release it just read
the manifest from is a soft-fail, and a soft-fail must never be
indistinguishable from "no data".

`/interpreter/print` is the print-ready view
(no nav, page breaks): the always-available `window.print()` fallback when
the release asset is missing.

## PDF + workflow wiring (Phase 6)

`interpreter/pdf.py` (`python -m interpreter.pdf --db ... --kind combined
--out-dir interpreter_out`) reads the newest `interpretations` row of the
kind (test-filtered by default; `--interpretation-id` pins an exact row),
converts its `content_md` to a self-contained HTML document with a small
deterministic subset converter (only what `render_markdown` emits —
headings, lists, pipe tables, bold/italic/code; inline code protects the
underscored question ids from emphasis parsing), and renders it with
**WeasyPrint** (chosen over a headless browser: fewer moving parts; the
import is lazy and the render seam `interpreter.pdf._render_pdf` is what
tests mock). Outputs `report__{YYYY-MM}__v{n}.pdf` plus the constant-name
copy `interpreter_report_latest.pdf`. A render failure keeps the HTML as
evidence and still exits 0; a non-`ok` row renders behind a validation
banner, unless `PYTHIA_INTERPRETER_STRICT_VALIDATION=1`, which suppresses
the PDF entirely (the row stays stored). `main()` returns 0 in every
outcome.

Where it runs — two insertion points, both inside workflows that already
own and upload the canonical DB (no new canonical-DB producer):

- **`run_sibyl.yml`** (the combined report): after `compute_deviation` and
  the current-run bundle, BEFORE the canonical DB upload and the publish
  dispatch — so the release order stays pythia → Sibyl → interpreter →
  publish and the published DB already contains the interpretations row.
  Steps: `interpreter.run --kind combined` (needs `ANTHROPIC_API_KEY`),
  then `interpreter.pdf` (apt pango top-up + `pip install weasyprint`),
  then upload of the **`pythia-interpreter-report`** artifact. All
  `continue-on-error` — a report bug must never cost the run its DB upload
  or its publish dispatch. The runner derives test mode from
  `hs_runs.is_test` for the interpreted run (the Sibyl pattern) so a
  test-mode cycle's report is stamped `is_test`.
- **`compute_calibration_pythia.yml`** (the scored report): right after the
  scored-bundle step, before the canonical DB upload — the row rides in the
  DB this workflow publishes. Has its own `Record stage start` +
  `stage_health --stage calibration_interpreter` so the call's cost is
  windowed; outputs upload as **`pythia-interpreter-scored`**.

### Backfill / recovery

`interpreter_backfill.yml` (manual dispatch) runs the interpreter over
already-completed forecast runs. Two uses: runs that finished before Phase 6
wired the interpreter in, and recovery for a cycle whose interpreter step
failed (every live step is `continue-on-error`, so a failure is silent by
design). Nothing is special-cased — it rebuilds exactly what the live path
would have had: `compute_deviation --run-id` (idempotent, delete-then-insert
per run) and `build_current_run_bundle --run-id`, then the same model call.

Inputs: `run_ids` (space/comma separated; blank = the latest production run),
`kind`, `force`, `include_test`, `db_run_id`, `upload_canonical` (default
true — without it the rows die with the runner), `publish` (default **false**,
so a backfill is inspectable before it goes public), and
`force_during_pipeline`. It is gated by `check_pipeline_active` and sits in
the `pythia-resolver-db` concurrency group for the same reason the nightly
backcast is: a run landing mid-pipeline would write rows the pipeline's final
canonical upload discards.

Because it re-uploads the canonical DB it **is a canonical-DB producer**, so
it is named in all three discovery lists and includes itself via
`extra-workflows` (otherwise a second backfill would start from a DB
predating the first). Per-run PDFs land in the diagnostics artifact; one
top-level render of the newest row goes to `pythia-interpreter-report` so the
publish step finds exactly one `interpreter_report_latest.pdf`.

Note on filenames: a backfilled report is named for the month it is ABOUT
(parsed from `hs_run_id`), not the month it was generated in — otherwise
July's report, generated in August, would be filed as August's.

**Release + manifest** (`publish_latest_data.yml`): a best-effort step
fetches `pythia-interpreter-report` from the source run (the Sibyl-chain
publish has one; the calibration-chain publish does not, which is expected
— the release keeps the previous PDF), uploads the PDFs to the
`pythia-data-latest` release with `--clobber`, and stamps
`interpreter_report_asset` / `interpreter_report_url` /
`interpreter_report_versioned_asset` into `manifest.json`. The URL key
reflects what is actually downloadable AFTER the publish (fresh upload OR
carried-over asset). `/v1/version` passes manifest keys through, which is
where the dashboard button reads it.

## Tests / CI

`interpreter/tests/` — fully deterministic, model call mocked, temp DuckDB,
bundle-directory fixtures. Run by `.github/workflows/interpreter-ci.yml`
(path-filtered on `interpreter/**` + `scripts/ai_bundle/**`, which also
re-runs the bundle suites — the interpreter consumes those bundles, so a
pack-shape change must re-run both sides).
