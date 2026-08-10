# This month's forecast run

Set `kind` in your output to exactly `{{KIND}}`. That is a fact about this
request, not a judgement; getting it wrong fails the report.

Below is the pack for the run you are reporting on. Work only from it.

{{PACK}}

## What to produce

### `run_summary`

One short paragraph, three sentences at most, saying how wide the scan was
and what came out of it. Use the placeholders, never your own digits:

- `{{fig:countries_scanned}}` countries screened for regime change
- `{{fig:countries_with_questions}}` countries carried through to forecasts
- `{{fig:countries_track1}}` on the full ensemble, where the scan saw a
  possible change of regime
- `{{fig:countries_track2}}` on the single-model track, quiet but still
  worth a number

### `attention`

One entry per row in the attention index that carries a `category`. There
will be at most five. Do not add rows without one. Do not drop rows that have
one. Copy `iso3`, `hazard_code`, `metric`, `category` and `hazard_family`
exactly as the pack gives them, and set `rank` from `category_rank`.

The four boxes, and what each one means:

- **Potentially worsening situations, climate hazards.** Drought, flood and
  tropical cyclone where the forecast sits above what history would lead you
  to expect.
- **Potentially worsening situations, conflict.** The same, for armed
  conflict.
- **Major impact but roughly stable, climate hazards.** A heavy burden per
  head of population, with no call that things are departing from the usual.
  These are places already in trouble.
- **Major impact but roughly stable, conflict.** The same, for armed
  conflict.

For each entry write:

- `why_it_stands_out` — the point of the entry, in two or three sentences.
  Lead with how many MORE people are expected than history would suggest
  (`{{fig:excess_nominal}}`), because that is what the ordering is built on
  and what a reader needs first. The multiple may follow as a descriptor; it
  is never the point. Where the entry is marked with a thin anchor, say so
  plainly: the comparison rests on very few observations and the multiple
  should not be read literally.
- `planning_sentence` — the two figures a response planner can act on,
  written as a frame with placeholders and nothing else:

  > Plan against about `{{fig:p50_peak}}` in `{{fig:peak_month}}`. Hold
  > contingency for `{{fig:p90_peak}}`.

  Add a third sentence ONLY when the chance of nothing happening is worth
  saying: "There is a `{{fig:p_zero_peak}}` chance of nothing recorded at
  all." Leave it out when that chance is small.
- `spd_shape` — one or two sentences on where the probability sits and how
  much weight is in the tail. Is the system fairly sure of a moderate
  number, or is it holding a small chance of something much larger?
  **Write no digits here.** Do not name the size bands by their numbers.
  Say "the middle band", "the top band", "the band above it"; the full table
  is printed in the appendix for the reader who wants it.
- `what_the_model_was_reacting_to` — the evidence, briefly. Name the source
  where the pack names it. **Write no digits here either**, however
  tempting. A figure you read in a source document is still a number you
  typed, and this report's standing promise is that it contains none. Say
  "a government figure in the hundreds of thousands", "a sharp rise in
  recorded events", "roughly a fifth of the population".
- `decision_point` — the decision this entry calls for. Write the `action`:
  one sentence naming what has to be decided and by whom, in the language of
  a response plan. Prepositioning stock. Opening a pipeline. Moving money
  between appeals. Set `deadline_month` and `basis` from the decision
  calendar in the pack; both are recomputed from the pack after you answer,
  so a guess there is wasted work while a guess in `action` is not. **If you
  cannot name a decision, the entry does not belong in the report.** Say so
  in `why_it_stands_out` rather than inventing one.
- `impacts` — what this would mean for people, in one or two short items.
- `operational_challenges` — access, funding, season, in one or two short
  items. Leave it out if the pack says nothing useful.

Keep the whole entry under one hundred and twenty words. The report carries
five entries at most and it is read by people with a morning, not a day.

**One rule governs every field above: you write no digits.** Not in impacts,
not in operational challenges, not in the reason an entry stands out. Where
you need a figure the system computed, use its placeholder. Where you need a
figure the system did not compute, describe its size in words. A single
stray numeral fails the whole report, and the report says of itself that no
number in it was written by a language model.

### `changes_since_last_run`

Three or four short items from `deltas.json`: what entered, what left, what
moved most. Name countries, not codes. Do not restate the persistence table
or the movement table; the report prints those itself from the same file.

### `blind_spots`

What the report cannot see this month, from `blind_spots.json`. Be honest
and brief.

### `headline`

One sentence. The single thing a reader should take away.

## What you must NOT write

Several sections of the report are generated from the pack and printed
without passing through you. Do not write them, summarise them, or refer to
figures inside them:

- the decision calendar page,
- the second reader's section, including what Sibyl found and its caveat,
- the comparison against the sector's own severity rankings,
- the selection panel, the appendix question table and the score glossary,
- the schedule of what resolves and when.

Your part in those is the `action` sentence on each entry, and nothing else.

## Part B: how well the system has done

{{SCORED_SECTION}}

If the section above says no scored run is available, omit the `performance`
block entirely. The report prints its own account of what is due to resolve
and when, which is more use than a paragraph saying there is nothing yet.
Otherwise summarise it: the plain result, the skill against climatology with
its uncertainty, the best and worst calls, and the standing warning that
Track 1 and Track 2 cover different questions and cannot be compared on raw
scores.
