# This month's forecast run

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

One entry per row in the attention index that carries a `category`. Do not
add rows without one. Do not drop rows that have one. Copy `iso3`,
`hazard_code`, `metric`, `category` and `hazard_family` exactly as the pack
gives them, and set `rank` from `category_rank`.

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
  For a worsening entry, say how far above the usual level the forecast sits
  and over what window. For a stable entry, say how heavy the burden is
  relative to the population.
- `spd_shape` — one or two sentences on where the probability sits and how
  much weight is in the tail. Is the system fairly sure of a moderate
  number, or is it holding a small chance of something much larger?
- `what_the_model_was_reacting_to` — the evidence, briefly. Name the source
  where the pack names it.
- `impacts` — what this would mean for people, in one or two short items.
- `operational_challenges` — access, funding, season, in one or two short
  items. Leave it out if the pack says nothing useful.

Keep the whole entry under one hundred and fifty words.

### `changes_since_last_run`

Three or four short items from `deltas.json`: what entered, what left, what
moved most. Name countries, not codes.

### `blind_spots`

What the report cannot see this month, from `blind_spots.json`. Be honest
and brief.

### `headline`

One sentence. The single thing a reader should take away.

## Part B: how well the system has done

{{SCORED_SECTION}}

If the section above says no scored run is available, write a `performance`
block whose `plain_summary` says plainly that no forecast window has closed
yet, so there is nothing to score. Do not imply performance is unknown for
some deeper reason. Otherwise summarise it: the plain result, the skill
against climatology, the best and worst calls, and the standing warning that
Track 1 and Track 2 cover different questions and cannot be compared on raw
scores.
