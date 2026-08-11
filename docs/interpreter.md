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
  names.py      # ISO3/hazard/metric codes -> the words a reader reads
  selection.py  # which forecasts the report covers, and under which heading
  gating.py     # v3: expected excess, the two-key gate, thin anchors
  gatecheck.py  # v3: run the gate over every run in a DB and print what it does
  panels.py     # v3: the selection panel, gate tags, appendix question table
  decisions.py  # v3: derived deadlines and the dated calendar page
  secondopinion.py # v3: Sibyl's direction, its unsettled trials, novel facts
  sector.py     # v3: rank gaps against ACAPS INFORM Severity / Risk Radar
  persistence.py # v3: (iso3,hazard,metric) matching, overlap movement, drops
  performance.py # v3: Part B — what resolves and when, small-sample discipline
  charts.py     # inline SVG of a forecast distribution (Fred palette)
  mapviz.py     # the printed attention map (vendored Natural Earth, no JS)
  schema.py     # the structured-output contract (jsonschema draft-07)
  packs.py      # loads the Phase 2 bundles; figure maps; token-capped input
  prompts.py    # template assembly from templates/{version}/
  templates/v2/ # system.md, current_run.md, scored_run.md, report_skeleton.md
  templates/v3/ # the shipped version: excess ordering, planning sentence
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
| `PYTHIA_INTERPRETER_TEMPLATE_VERSION` | `v4` | |
| `PYTHIA_INTERPRETER_MAX_PACK_TOKENS` | `250000` | model-input hard cap |
| `PYTHIA_INTERPRETER_TOP_N` | `8` | attention list length |
| `PYTHIA_INTERPRETER_PER_CAPITA_FLOOR` | `10000` | per-capita ranking floor |
| `PYTHIA_PUBLIC_BASE_URL` | `https://fredforecaster.org` | absolute question links in the PDF |
| `PYTHIA_INTERPRETER_THINKING` | `high` | output_config.effort level |
| `PYTHIA_INTERPRETER_MAX_OUTPUT_TOKENS` | `32768` | shared with thinking |
| `PYTHIA_INTERPRETER_TIMEOUT_SEC` | `900` | call timeout |
| `PYTHIA_INTERPRETER_STRICT_VALIDATION` | `0` | Phase 4: failure suppresses publication |
| `PYTHIA_INTERPRETER_VALIDATION_RETRIES` | `1` | correction passes after a failed validation |
| `PYTHIA_INTERPRETER_UNUSUAL_PERCENTILE` | `0.80` | v3: the "unusual" cut, within the run |
| `PYTHIA_INTERPRETER_MINPROB` | `0.25` | v3: the "material" probability, any horizon |
| `PYTHIA_INTERPRETER_THRESHOLD_PA` | `50000` | v3: people affected worth mobilising for |
| `PYTHIA_INTERPRETER_THRESHOLD_PA_POP_SHARE` | `0.01` | v3: the small-state alternative |
| `PYTHIA_INTERPRETER_THRESHOLD_FATALITIES` | `100` | v3 |
| `PYTHIA_INTERPRETER_THRESHOLD_PHASE3` | `1000000` | v3 |
| `PYTHIA_INTERPRETER_THRESHOLD_PHASE3_POP_SHARE` | `0.10` | v3 |
| `PYTHIA_INTERPRETER_BASERATE_MIN_OBS` | `12` | v3: below this an anchor is thin and is demoted |
| `PYTHIA_INTERPRETER_BASERATE_SMOOTHING` | `0.5` | v3: Jeffreys pseudo-count on the anchor |
| `PYTHIA_INTERPRETER_MAX_ENTRIES` | `5` | v3 |
| `PYTHIA_INTERPRETER_MIN_RESOLVED_FOR_SKILL` | `100` | v3: below this, skill is not claimed |
| `PYTHIA_INTERPRETER_SKILL_BOOTSTRAP` | `2000` | v3: resamples behind a printed interval |
| `PYTHIA_INTERPRETER_LEAD_TIME_{HAZARD}` | DR 3, ACE 2, FL/TC 1 | v3: months from decision to the month it covers |
| `PYTHIA_INTERPRETER_SIBYL_DISAGREEMENT_RATIO` | `1.5` | v3: when the second reader is called more alarmed |
| `PYTHIA_INTERPRETER_SIBYL_UNSETTLED_SHARE` | `0.25` | v3: inter-trial divergence that means "unsettled" |
| `PYTHIA_INTERPRETER_SIBYL_NOVEL_FACTS` | `2` | v3: novel findings printed per covered question |
| `PYTHIA_INTERPRETER_SECTOR_LIST_SIZE` | `5` | v3: entries per sector-gap list |
| `PYTHIA_INTERPRETER_SECTOR_MIN_RANK_GAP` | `5` | v3: places apart before a gap is worth printing |
| `PYTHIA_INTERPRETER_GATE_MODE` | `delta` | v4: `delta` (movement) or `level` (the pre-v4 rollback) |
| `PYTHIA_INTERPRETER_MAX_WATCHLIST` | `10` | v4: watchlist lines before the rest go to the appendix table |
| `PYTHIA_INTERPRETER_MAX_TENSIONS_PER_ENTRY` | `2` | v4: reconciled contradictions per entry |
| `PYTHIA_INTERPRETER_GENERIC_PHRASE_MAX` | `3` | v4: sections a stock phrase may appear in before the lint flags it |
| `PYTHIA_INTERPRETER_NOVEL_EVIDENCE_SCOPE` | `disagreement` | v4: `disagreement` or `all` — novel evidence from a reader that agreed is a research note, and it ran to two pages |

## What the report covers (`interpreter/selection.py`)

The pack, not the model, decides which forecasts the report talks about and
under which heading. Four boxes:

| box | who is in it |
|---|---|
| Potentially worsening situations: climate hazards | drought, flood, tropical cyclone sitting ABOVE what history would suggest |
| Potentially worsening situations: conflict | the same, for armed conflict |
| Major impact but roughly stable: climate hazards | heavy burden per head, no call that things are departing from the usual |
| Major impact but roughly stable: conflict | the same, for armed conflict |

**Direction is the point.** The divergence score `js_vs_baserate` is unsigned:
ranking on it would promote a forecast for sitting far BELOW its base rate,
which is not news. A ±5% deadband keeps forecasts that merely round off the
base rate out of the worsening sections.

### Ranking and the two-key gate (v4, `interpreter/gating.py`)

Ranking used to be on `log_ev_ratio`, a RATIO, and a ratio is dominated by
whichever base rate happens to sit closest to zero. A cyclone question whose
history puts almost all its weight on "nobody affected" divides by almost
nothing, so a modest tail produces a huge multiple. The August report led on
Indonesian cyclones and put famine in Sudan on page two.

Selection is a **two-key gate** stamped by `gating.gate_rows`, which returns
the counts the report prints. The first key is unchanged; the second was
redesigned in v4.

| key | test |
|---|---|
| unusual | `js_vs_baserate` at or above the run's `PYTHIA_INTERPRETER_UNUSUAL_PERCENTILE` (0.80) percentile |
| material worsening | at the peak horizon, `max(delta_p50, delta_p90) >= threshold` — the figure a planner acts on has RISEN by at least the size worth mobilising against |
| heavy burden (the level test) | at least `PYTHIA_INTERPRETER_MINPROB` (0.25) chance, in ANY month of the window, of passing that same size, whether or not anything changed |

**Materiality is two different tests, and v4 keeps them apart.** The LEVEL
test answers "is this a heavy burden" and is what puts Sudan in the report
whether or not anything moved. The MOVEMENT test answers "is this worsening".
Until v4 the level test did both jobs, and it cannot: a country with a
chronically large caseload clears an absolute level every month by
construction, so any positive excess then read as worsening. Ethiopia cleared
the hundred-deaths test at essentially certainty and led the September report
on five expected excess deaths; Indonesia's cyclone entry cleared it on a tail
whose planning figure never moved at all.

The movement is computed in `compute_deviation` against the SAME in-bucket
quantile interpolation the forecast uses, so the pair is one summary applied
to two distributions rather than two different summaries compared with each
other:

```
delta_p50 = forecast p50 at the peak horizon − base-rate p50
delta_p90 = forecast p90 at the peak horizon − base-rate p90
material  = max(delta_p50, delta_p90) >= movement_threshold
```

Three things follow. The gate is stated in the same units as the
recommendation, so a reader can check it. A shape change is caught honestly,
because a widened tail moves p90 even when p50 sits still — and the entry says
which figure moved (`movement_shape`), so the report never claims a rise the
planning figure does not support. And the explicit `rising` test disappears,
because a threshold is a positive number and clearing it IS the direction,
which also removes the old sensitivity to a near-zero denominator.

Binary questions have no size to plan against, so their movement is measured
in probability points against the same anchor and gated on the run's minimum
probability. Commensurable within its own kind, never mixed with people.

Thresholds are `min(absolute, share × population)` so a flat 50,000 floor does
not silently exclude every small island state from a cyclone report; the
movement threshold is the SAME number as the level one, because "worth
mobilising against" and "risen by enough to change a plan" deserve one answer.
The horizon that cleared is carried on the row (`peak_horizon`), because which
month cleared is part of the answer.

`PYTHIA_INTERPRETER_GATE_MODE=level` restores the pre-v4 behaviour without a
deploy. `python -m interpreter.gatecheck --db ... [--mode level]` runs the gate
over every run in a DB and prints what it admits, what it drops, WHICH
questions cleared and what the near misses were; `run_sibyl.yml` runs both
modes every cycle into the step summary and the `pythia-interpreter-gate-diagnostics`
artifact.

**Two lists, two orderings, each in the units that suit its purpose.** The
worsening list is ordered by `max(delta_p50, delta_p90)` in people or deaths
(`gating.movement_rank_key`); the heavy-burden list by the expected burden
itself, because "already in trouble" is a statement about size and not about
change. Each section heading states its own ordering
(`selection.CATEGORY_ORDERINGS`), so a reader who can see it can check it.

Gates, in the words the report prints: `larger than usual` (unusual AND
moved), `heavy burden` (the level test, without a claim that it is worsening),
`unusual, small scale` (the watchlist — tracked, not acted on, capped at
`PYTHIA_INTERPRETER_MAX_WATCHLIST` with each line carrying its movement, and
the overflow left to the appendix table which already holds every row).

**Thin anchors are demoted, not dropped.** Below
`PYTHIA_INTERPRETER_BASERATE_MIN_OBS` (12) historical observations a row is
ranked below every clear-anchored row and says so. Indonesia's near-empty
cyclone record may be a gap in GDACS and IFRC rather than a real absence of
impact; the honest response to not knowing is to rank it lower and say so.

### What the anchors rest on (v4 investigation)

The v3 report marked 138 of 185 anchors thin. That is a finding about the
anchor builder, not a threshold that needs lowering — a flag firing on
three-quarters of rows has stopped discriminating, and the demotion rule it
drives is doing nothing. `python -m interpreter.anchorcheck --db ...` reports
per (hazard, metric) what the anchors rest on and diagnoses which of the three
causes is in play. Two were real and both were fixed in
`pythia/tools/base_rate_spd.py`:

- **The conflict window was six months against a twelve-observation cutoff**,
  so no armed conflict anchor could ever clear the flag. Arithmetic, not
  evidence. `CONFLICT_WINDOW_MONTHS` is now 36, matching the Phase 3+ window
  and well inside what ACLED serves. The invariant that matters is that an
  anchor is drawn from the series that will resolve the question, not that it
  uses the same number of rows as a prose summary.
- **Quiet months were dropped rather than counted as observed zeros.** ACLED
  emits no row for a country-month with no events, and IDMC reports a country
  only when it records displacement, so an anchor built from present rows
  alone said displacement happens every month in a country IDMC reports twice
  a year. `COUNT_QUIET_MONTHS_AS_ZERO` fills them, behind the same two gates
  `source_coverage` applies on the resolution side: the month must be one the
  source was live for globally, and the country must appear in the source at
  all. Without both an ingestion gap becomes a run of quiet months.

"Counting events where months are wanted" was checked and is NOT happening:
`monthly_fatalities` groups by (iso3, month) already.

### The map shows direction (v4)

`interpreter/mapviz.py` and `/v1/interpreter/attention_map` both colour by the
SIGNED movement, scaled by the metric's own threshold so deaths and people in
crisis land on one ramp. One hue above the anchor, another below, pale neutral
near it, grey for no forecast, and a legend that names the direction. Where a
country carries several hazards the largest-magnitude movement wins, and the
legend says so. The old scale was unsigned, so Uganda appeared among the
countries furthest from usual on the same page as text saying its ensemble had
moved down. The frontend mirrors the ramps in
`web/src/app/interpreter/lib.ts`; the API keeps serving the old `attention`
field beside the new `movement` one so an older dashboard build still renders,
and guards the movement columns so a DB predating the migration serves a map
rather than a 500.

### The report's account of itself (`interpreter/panels.py`)

A reader must be able to see why something is in the report and why something
else is not. Three pieces, all GENERATED from the configuration and the gate's
own counts rather than written by the model — a model asked to describe the
selection rules will eventually describe them wrongly, and the reader has no
way to check:

- a boxed **"How these entries were chosen"** panel near the front, stating the
  ordering rule, both tests, the thresholds and the counts;
- a **gate tag** on every entry (`*Selected because: …*`);
- an appendix **question table** listing every question considered, which is
  what answers "why is my country not in the report".

All three read one counts dict, so they cannot disagree. Three numbers that
ought to match and are computed three times will eventually not match.

### What a planner is given (v3)

Bucket probability tables moved out of the entries and into the appendix, with
the bands named in words ("1 to 9,999 people", never `1-<10k`). A picture of
uncertainty is not something a response planner can act on, and it was crowding
out the two figures that are: `p50_peak` (plan against) and `p90_peak` (hold
contingency for), both at the peak horizon, with the month named
(`peak_month`) and `p_zero_peak` where the chance of nothing is worth saying.

**The planning sentence is GENERATED (v4), not written.** It is a frame with
two numbers in it and no judgement, and leaving it to the model produced
"+5 people more deaths" and, where both quantiles land in the open-ended top
band, a contingency figure identical to the planning figure ("Plan against
about 20,000,000 people. Hold contingency for 20,000,000 people"). Sudan read
exactly that. `panels.planning_sentence` prints one sentence using the band's
lower bound in that case and suppresses the contingency line, and rounds every
figure to its own scale: the number comes from interpolating inside a bucket
tens of thousands wide, so every digit past the third is a claim the scheme
cannot support. The `spd_shape` paragraph was cut in the same change — it
restated the two figures beside it, and the full distributions are already in
the appendix with their bands named in words.

**Membership is the gate's, not selection's.** `select_worsening` reads
`gate == "larger than usual"` and `select_stable_major` reads
`gate == "heavy burden"`; selection only orders (worsening by movement,
heavy burden by expected size, thin anchors demoted in both) and cuts to
`PYTHIA_INTERPRETER_MAX_ENTRIES` (5) across the whole report, split 60/40 in
favour of worsening with either side handing its unused slots to the other. The per-section knobs of v2 (a worsening MULTIPLE,
a minimum and a maximum per box) were removed: ranking by a multiple is the
defect the excess ordering exists to fix, and keeping a second policy beside
the gate would let the two disagree.

**No relabelling, structurally.** A row carries ONE gate and the two sections
read different gates, so a worsening entry crowded out by the cap is left out
of both rather than described as "roughly stable".

**Nothing is dropped silently.** Uncategorised entries the model still emits
render under "Other situations of note", where they are visible and obviously
odd. The August run produced 6/3/6/6 across the four boxes from 200 questions.

**Records follow the report, not the ranking.** Both the bundle builder and the
prompt assembly write question records in report order, so the rows the report
must cover are the last thing a token budget gives up; anything that still
loses its record is named, in the manifest and in the prompt.

### The decision calendar (v3, `interpreter/decisions.py`)

The most useful sentence in the August report appeared once, by accident: the
Vietnam entry noted the prepositioning window closes in September. A forecast a
reader cannot act on by a date is a fact, not a decision.

Every attention entry now carries a `decision_point` (required by the schema),
and the deadlines are collected onto one dated page near the front, ordered by
deadline. The split of work is deliberate:

- the **model writes the action** — one sentence naming what has to be decided.
  If it cannot name one, the entry does not belong in the report, and asking
  for it is how the model is made to justify its inclusion;
- the **deadline is derived** — the peak horizon the materiality gate found,
  less `PYTHIA_INTERPRETER_LEAD_TIME_{HAZARD}` (DR 3, ACE 2, FL/TC 1). The
  runner overwrites whatever month the model wrote, the same repair contract
  as the identity fields. A deadline already past is printed as it falls.

A row with no peak horizon has no derived deadline, and the calendar says "not
dated" rather than inventing one.

### The second reader (v3, `interpreter/secondopinion.py`)

Sibyl re-forecasts the most volatile questions by reading the open web; the
main pipeline reads structured connectors. It used to appear in one confidence
note on page fourteen. It now has a section:

- **direction of disagreement**, compared on expected value (a ratio, so
  questions about deaths and questions about millions share one rule) —
  `second opinion agrees` / `is more alarmed` / `is more cautious`. The tag is
  also stamped inline on every attention entry, so a reader does not have to
  reach the back to learn Sibyl disagreed;
- **unsettled trials**: `js_divergence_inter_trial` above a share of its ln 2
  maximum means careful research did not settle the question. That is a
  different signal from disagreeing with Fred, and it prints as its own flag;
- **what the second reader found that the main system did not** — sentences
  from Sibyl's trial belief traces whose content words are absent from the
  ASSEMBLED SPD PROMPT for that question (which carries every inject the
  ensemble had). Substring and token matching, deliberately: it is a prompt
  for attention, not a claim that a fact is new to the world. Below 500
  characters of comparison text nothing is claimed novel — with no prompt to
  compare against, every sentence looks new;
- **fixed caveat** the model may not drop: Sibyl has no scored track record,
  so it is a second reader, never a tiebreaker.

The dumbbell chart is drawn in `charts.second_opinion_chart` on a SHARE axis
(the centre line is agreement), because the covered questions mix deaths with
millions of people and one linear axis across both would say nothing. The
dashboard's own dumbbell is React, which a PDF cannot run, so the two share
the grammar rather than the code.

### Fred against the sector (v3, `interpreter/sector.py`)

Everyone knows Sudan and Somalia are catastrophic. The value is where Fred
departs from the prevailing view. Countries are ranked three ways — Fred's
summed positive excess, ACAPS INFORM Severity (newest snapshot, worst crisis
per country) and ACAPS Risk Radar (worst risk per country) — and the rank gaps
are printed in both directions, five each, above
`PYTHIA_INTERPRETER_SECTOR_MIN_RANK_GAP` (5) places.

Only countries in BOTH rankings are compared: treating absent ACAPS coverage
as a low sector rank would manufacture disagreement out of coverage. The fixed
`NOT_COMMENSURABLE` text says plainly that the three measure different things
and that a gap is a prompt to look, not a verdict.

### What changed, and persistence (v3, `interpreter/persistence.py`)

- **Matching is `(iso3, hazard_code, metric)`.** Without the metric a
  country's drought DEATHS question matched its drought PEOPLE AFFECTED
  question.
- **Movement is measured on the months the two runs share.** Consecutive runs
  overlap by five of six months at different horizon numbers, so the planning
  figure is keyed by CALENDAR month; comparing horizon 1 against horizon 1
  would read the window sliding forward as the forecast changing.
- **Persistence is counted from the reports themselves** (stored
  `interpretations` rows, newest first) and stops at the first month that did
  not flag it. Re-running today's thresholds over an old run would say what we
  WOULD have said, which is a different claim.
- **A dropped flag says why**: no question this month, below the gate, or
  still passing but ranked below the entries shown.

### Part B: the scored run (v3, `interpreter/performance.py`)

Part B used to resolve its run from the CLI argument and then fall back to the
most recent stored `scored` interpretation, of which there has never been one.
It could only ever report emptiness, and would have gone on doing so on the day
the first outcomes landed. It now probes `resolutions` and `scores` directly,
and logs loudly when outcomes exist with no scored interpretation made of them.

The **dormant state is informative**: how many question-months are due, on what
date, for which hazards, printed from `upcoming_resolutions` (a window month
resolves on the 28th of the month after it). A combined report with no scored
run behind it is not asked for a `performance` block at all — prose we discard
is prose the model learns to invent.

The **small-sample discipline** is built now because the failure happens once.
The first scored run will rest on well under a hundred question-months:

- below `PYTHIA_INTERPRETER_MIN_RESOLVED_FOR_SKILL` (100) the report DECLINES
  the claim. It states the count and offers no skill number, because a printed
  figure beside a hedge is the figure a reader keeps;
- every skill figure carries a seeded percentile bootstrap interval, and the
  interval is what gets printed. Pairs are resampled TOGETHER: the pairing is
  what makes the comparison fair;
- horizons are never pooled and score families never blended.

The **forecast diary** links back: a question flagged in an earlier report that
has since resolved, with what happened. Only resolved ones appear; padding it
with open questions would make it a second attention list.

### The proper-noun guard (v3, `interpreter/validate.py`)

The August report named "Typhoon Maysak" in its Vietnam entry. That string
appears nowhere in the pack, in a document whose footer tells the reader that
every figure in it is machine-derived. `check_proper_nouns` requires a proper
noun in prose to appear in the pack's evidence text (`packs.evidence_text`:
every file plus every question record, lowercased). Country and hazard names
from the system's own tables are always allowed, and a sentence-initial capital
is grammar rather than a name.

It **reports rather than fails** — the violations land in `validation_json` and
are quoted back to the model in the correction pass. A validator that failed a
whole report on an unusual spelling would be switched off within two months.
`_error_count` counts the violations so a correction that removes an invented
name is not thrown away as "no improvement".

### Checking the gate before shipping a threshold

```
python -m interpreter.gatecheck --db "$PYTHIA_DB_URL" [--include-test]
```

Runs the gate over every run with deviation rows and prints what it admits and
drops per run, plus any run that would produce no entries at all.
`gating.calibration_table` existed from the start with no caller, which in this
repo means it did not exist.

## The figure-key contract

A template may only name a `{{fig:...}}` key that something actually produces:
`packs._ATTENTION_FIG_KEYS` (per question), `_RUN_SUMMARY_KEYS` (run level) or
`_PERFORMANCE_KEYS` (scored pack). A key nothing produces renders as
`[figure unavailable]` and fails the referential check for the whole report.

`interpreter/tests/test_selection_names_charts.py::TestTemplateFigureContract`
scans every template for placeholder keys and fails if one is not produced. It
exists because `ev_multiple` shipped exactly that way: the system prompt told
the model to write it, the selection module computed it onto every attention
row, and the figure map never exposed it, so the first live report failed
validation with every worsening entry's headline number unresolvable.

`figure_refs` entries are normalised before lookup, so `eiv_nominal`,
`fig:eiv_nominal` and `{{fig:eiv_nominal}}` all resolve. The prefix carries no
meaning and rejecting it failed a report over punctuation.

## What the model actually writes (v4)

Everything numeric in this report is computed in SQL and rendered
deterministically. That is the right design and it does not change, but it
left the model summarising, which is not what a frontier model is for. v4 adds
four fields that require judgement and none of which can be computed:

| field | what it asks for |
|---|---|
| `tensions` (≤2 per entry) | two claims in the evidence that do not sit together, and an account of why both can be true or which is more credible. An acknowledged contradiction is worth more than a tidy narrative with one side buried. South Sudan's entry printed a 67% rise in civilian killings above a 4% fall in violent incidents as adjacent bullets with no comment. |
| `challenge` | what the adversarial check did to the reading: `held`, `weakened` or `changed_the_reading`, with the reasoning. Somalia is the live case — the challenge argued the coming short rains carry flood risk rather than drought, a serious objection to a drought entry, and the reader had no way to know whether it was taken seriously. |
| `second_opinion_explanation` | where the second reader differs, the SOURCE of the difference: different evidence, different weight on the same evidence, or a different reading of the time frame. "More cautious" is a label, not something a reader can act on. |
| `falsifier` (required) | what would have to be observed in the next thirty days for the call to look wrong. Cheap, genuinely analytical, and it makes the report testable before anything resolves. It also feeds the forecast diary: a falsifier that fired is the most legible way to report a miss. |

Two more at the top level: `cross_cutting` (five sentences at most, near the
front — are the entries connected by a common driver, or are they four
separate droughts that happen to fall in one month?) and
`scan_forecast_disagreements`, which was specified in the v3 plan and never
appeared in a published report. Where the scan called a high chance of change
and the ensemble came back at its anchor, one of the two is wrong, and nothing
outside this system produces that page.

All of them are linted like any other prose: the no-digits rule, the house
style and the code check apply to a reconciliation exactly as they apply to an
impact.

### Measuring the model rather than arguing about it (v4)

Two decisions are pending: whether Opus earns its cost, and whether the fields
above land or come back hollow. Two tools, and neither changes the production
model:

- **`check_generic_phrases`** (validate.py, the eighth check) counts stock
  phrases — access constraints, funding gaps, protection risks, plus the
  process vocabulary a reader does not want (base rate, the distribution,
  left tail) — per SECTION of the report, and flags any that appear in more
  than `PYTHIA_INTERPRETER_GENERIC_PHRASE_MAX` (3). It REPORTS, like the
  proper-noun check: a proxy for writing quality must never stop a report
  being published. It is the only check in the suite that can compare two
  models on one pack, because a weaker model reaches for the sentence that
  fits any crisis and that degradation is invisible to everything else here.
- **`python -m scripts.compare_interpreter_models --models A B`** runs one
  pack through two models (`interpreter.run --model-override`, `--force` so
  each gets its own stored version), writes both reports for reading, and
  prints the phrase counts beside a substance table: tensions per entry,
  challenge verdicts, falsifiers, and how many characters each model spent on
  the fields that matter. Opt-in in `interpreter_backfill.yml` via the
  `compare_models` input, because it costs a second full call per run and
  because the API key lives in CI. **The counts are a proxy. Deciding from
  the artifacts is a person's job.**

## Validation (Phase 4 — `interpreter/validate.py`)

Eight checks run after generation, before storage; each is reported
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

5. **style** — the house rules that can be checked mechanically: no em or
   en dashes, no banned words (`BANNED_PHRASES`), no "not X, but Y", no
   "on the other hand", and no bare ISO3 / metric enum / `HZ/METRIC` code
   where the reader needs a name. The code check matches the REAL ISO3 set
   from `resolver/data/countries.csv`, never any three-capital run: FAO, IOM
   and "Phase III" are ordinary humanitarian prose, and flagging them failed a
   correct report. Deliberately narrow: rhythm, variety and
   the habit of joining clauses with "and" are asked for in the prompt and
   judged by a reader, and a lint that scored them would fail honest prose.
6. **categories** — the pack decides which forecasts the report covers and
   under which heading; the model copies that decision across. Promoting,
   demoting, inventing or dropping an entry fails. Skipped when the pack
   carries no categorised rows (a scored interpretation has no attention
   list at all).
7. **proper_nouns** — a name in prose must appear in the pack's evidence.
   Reports rather than fails, but the violations are quoted back in the
   correction pass.
8. **generic_phrases** — how much of the report is written in the sentence
   that fits any crisis (see above). Reports rather than fails.

**One correction pass.** When validation fails, the runner quotes the exact
complaints back to the model and asks for the same report with only those
points fixed. The better of the two answers is kept, never the worse one, and
the retry's tokens are added to the interpretation's cost. Capped at one
(`PYTHIA_INTERPRETER_VALIDATION_RETRIES`, default 1): the failures worth
retrying are single-sentence slips, and a model that misses twice is telling
you something about the checks rather than about itself.

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

The page carries Fred's masthead (wordmark, strapline, the month the report
is about) and Fred's palette from `web/tailwind.config.ts`, so the printed
report and the dashboard read as one publication. Under the masthead sits the
attention map: `interpreter/mapviz.py` projects the vendored Natural Earth
boundaries the PA resolution machine already ships (1:50m, because 1:110m
drops the small island states) and fills each country from
`forecast_deviation` using the same table, preference order and ln 2 scaling
as `/v1/interpreter/attention_map`, so the printed map and the dashboard map
cannot disagree. It is drawn here rather than borrowed because the dashboard's
choropleth is JavaScript and a PDF cannot run JavaScript. Coordinates are
rounded to the pixel and consecutive duplicates dropped: the source carries
far more precision than a 960px map can show, and keeping it would put
megabytes into every PDF. A missing boundary file degrades the report to no
map, never a failed render.

Each attention entry also carries an inline SVG of its own forecast
distribution (`interpreter/charts.py`), drawn from the PACK, never from model
output. Question ids render as absolute links to their dashboard pages
(`PYTHIA_PUBLIC_BASE_URL`); a bare id in a printed report is a dead end.

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
pack-shape change must re-run both sides). Both are DIRECTORY-level pytest
invocations, so a new test file under either runs without being named — the
one place in this repo where that is true.

v3 suites: `test_v3_modules.py` (decisions, second opinion, sector,
persistence), `test_performance.py` (the dormant state, the small-sample
floor, the seeded bootstrap, and the guard that the report's score copy stays
verbatim-identical to `web/src/lib/score_glossary.ts`), `test_v3_report.py`
(section order, the calendar, the second reader, the sector lists, the PDF
contents page and map placement), the proper-noun class in `test_validate.py`
(a planted storm name is flagged and does NOT fail the report), and
`scripts/ai_bundle/tests/test_current_run_bundle_v3.py` (the whole chain over
a DB carrying the v3 deviation columns).
