TASK: interpret the most recent SCORED run described in the pack below.
Produce the structured output for kind "scored" — the `performance` section
is the deliverable; `attention` is not required.

- `plain_summary`: how well the system did, in words a lay reader can check.
- `skill_statement`: against the climatology reference (`{{fig:skill_brier_spd}}`
  and, where present, `{{fig:skill_brier_binary}}`) — positive skill means
  the system beat the base rate; zero matched it; negative lost to it. Keep
  the two score families separate.
- `best_calls` / `worst_calls`: prefer high regime-change questions where the
  system's stance was tested; each carries its question_ids and, in prose,
  what was right or wrong about the REASONING, not just the score.
- `track_comparison`: Track 1 vs Track 2 — skill against each track's own
  climatology only (the fixed caveat in the system prompt applies).
- `sibyl_comparison`: from the pack's Sibyl material where present; the
  covered-set comparison only. If absent, say the deep-research track has no
  scored coverage this round.
- `vs_system_average`: this round against the system's running average, if
  the pack provides it; otherwise say the running average is not yet
  available.

INPUT PACK:

{{PACK}}
