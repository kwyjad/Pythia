TASK: interpret the CURRENT forecast run described in the pack below. Produce
the structured output for kind "{{KIND}}".

- Build the `attention` list ({{TOP_N}} entries at most) from the pack's
  attention_index: blend the four orderings (deviation from base rate,
  nominal impact, per-capita impact, RC-vs-deviation disagreement) and state
  each entry's `reason_code`. Use `deltas` for what changed since the
  previous run (`changes_since_last_run`) and `blind_spots` for what the
  system cannot see (`blind_spots`, `confidence_notes`).
- `what_the_model_was_reacting_to` comes from the question records: the RC
  rationale bullets, trigger signals, grounding evidence and scenarios — in
  your words, no quotes needed.
- Questions listed under `no_baserate_questions` cannot be assessed for
  base-rate deviation; if one earns attention on impact grounds, say plainly
  that its base-rate comparison is unavailable.

{{SCORED_SECTION}}

INPUT PACK:

{{PACK}}
