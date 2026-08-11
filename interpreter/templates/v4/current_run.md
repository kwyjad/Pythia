# This month's forecast run

Set `kind` in your output to exactly `{{KIND}}`. That is a fact about this
request, not a judgement; getting it wrong fails the report.

Below is the pack for the run you are reporting on. Work only from it.

{{PACK}}

## What to produce

Everything numeric in this report is already computed. Your work is the part
that cannot be: reconciling evidence that disagrees with itself, saying what
the challenge to a reading did to it, explaining why two processes reached
different answers, and naming what would show a call to be wrong. Those
fields are the reason a frontier model is doing this and not a template.

### `run_summary`

One short paragraph, three sentences at most, saying how wide the scan was
and what came out of it. Use the placeholders, never your own digits:

- `{{fig:countries_scanned}}` countries screened for regime change
- `{{fig:countries_with_questions}}` countries carried through to forecasts
- `{{fig:countries_track1}}` on the full ensemble, where the scan saw a
  possible change of regime
- `{{fig:countries_track2}}` on the single-model track, quiet but still
  worth a number

### `cross_cutting`

Five sentences at most, and fewer if you have less to say. Look across all
the entries at once. Are they connected? Four droughts in different regions
either share a driver, a climate signal, a funding withdrawal, a reporting
cycle, or they do not. Say which. If they are unconnected, say that plainly
and stop; a forced connection is worse than none.

### `attention`

One entry per row in the attention index that carries a `category`. There
will be at most five. Do not add rows without one. Do not drop rows that have
one. Copy `iso3`, `hazard_code`, `metric`, `category` and `hazard_family`
exactly as the pack gives them, and set `rank` from `category_rank`.

The four boxes, and what each one means:

- **Potentially worsening situations, climate hazards.** Drought, flood and
  tropical cyclone where the figure a planner would act on has RISEN above
  its historical equivalent by enough to change a plan.
- **Potentially worsening situations, conflict.** The same, for armed
  conflict.
- **Major impact but roughly stable, climate hazards.** A heavy burden, with
  no call that things are departing from the usual. These are places already
  in trouble.
- **Major impact but roughly stable, conflict.** The same, for armed
  conflict.

For each entry write:

- `why_it_stands_out` — the point of the entry, in two or three sentences.
  For a worsening entry, lead with what has MOVED. The pack tells you which
  of the two figures rose, in `movement_shape`. Where only the contingency
  figure rose, say so honestly: the number to plan against has not changed,
  the number to hold in reserve has. Where the entry is marked with a thin
  anchor, say plainly that the comparison rests on very few observations.
  **Do not write the planning figures here.** The report prints them itself
  in a generated sentence directly below your text, and repeating them
  produces the same number twice with different rounding.
- `what_the_model_was_reacting_to` — the evidence, briefly. Name the source
  where the pack names it. **Write no digits here**, however tempting. A
  figure you read in a source document is still a number you typed, and this
  report's standing promise is that it contains none. Say "a government
  figure in the hundreds of thousands", "a sharp rise in recorded events",
  "roughly a fifth of the population".
- `tensions` — at most two. Find the places where the evidence does not agree
  with itself, and account for them. A rise in one measure beside a fall in
  another over the same period. A government figure far from an agency's. A
  forecast signal pointing the opposite way to the reported situation. For
  each: state both claims, then reconcile them. Say why both can be true, or
  which is the more credible and why. **If you cannot reconcile them, say
  so.** An acknowledged contradiction is worth more to a reader than a tidy
  story with one side buried. Leave the field out only when the evidence
  genuinely does not conflict.
- `challenge` — the pack carries an adversarial check for the questions the
  scan flagged. Say what it did to the reading. `held` means the objection
  was considered and the reading survives. `weakened` means it stands but
  with less confidence, and the reasoning should say what would settle it.
  `changed_the_reading` means the entry is written differently because of it.
  One or two sentences. If the pack carries no challenge for this entry,
  leave the field out rather than inventing a verdict.
- `second_opinion_explanation` — only where the pack shows the second reader
  differing. Explain the SOURCE of the difference: did the two read different
  evidence, weigh the same evidence differently, or read the time frame
  differently? "More cautious" is a label, not an explanation, and a reader
  cannot act on a label.
- `falsifier` — one line. What would have to be observed in the next thirty
  days for this call to look wrong? Be specific enough that someone could
  check it. No digits: describe the threshold in words.
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
  items. Leave it out if the pack says nothing useful. Do not reach for the
  phrase that would fit any crisis: if the pack does not say access is
  constrained here, do not write that access is constrained.

Keep the whole entry under one hundred and eighty words. The report carries
five entries at most and it is read by people with a morning, not a day.

**One rule governs every field above: you write no digits.** Not in impacts,
not in a tension, not in a falsifier. Where you need a figure the system
computed, use its placeholder. Where you need a figure the system did not
compute, describe its size in words. A single stray numeral fails the whole
report, and the report says of itself that no number in it was written by a
language model.

### `scan_forecast_disagreements`

At most six. The scan reads the news for signs a situation is departing from
its pattern; the ensemble puts a number on the next six months. Where the
scan called a high chance of change and the ensemble came back sitting on its
historical anchor, or the reverse, one of the two is wrong. For each such
case, one entry: the question ids, and an explanation of where the
disagreement comes from. This is the most interesting page in the report and
nothing outside this system produces it. If the pack shows no such case, omit
the field.

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

- the planning and contingency figures for each entry, which are printed as a
  generated sentence under your text,
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
