# Hazard resolution machine

**What this is.** For every country, month, and hazard (flood, drought,
tropical cyclone), this module produces exactly one answer to the question
*"how many people were affected?"* — either a credible number, a credible
zero, or an explicit "no data". Every answer carries full provenance: which
source said it, which records or documents it came from, when we retrieved
them, and which rule made the decision. The goal is that at least 80% of
country-month-hazard cells resolve to a number (including zero) rather than
"no data" — today's IFRC-GO-only resolution manages about 4%.

**How it decides — two layers.**

1. **Detection** — did a qualifying hazard happen at all? Physical and event
   sources answer this: IBTrACS storm tracks for cyclones, GDACS alerts for
   floods, IPC food-security data for droughts. If nothing qualifying
   occurred *and* a ReliefWeb keyword sweep confirms silence, the cell
   resolves to **zero**, and we record the evidence of absence.
2. **Impact** — if a hazard was detected, walk a fixed ladder of reporting
   sources, best first: **EM-DAT → figures extracted from ReliefWeb
   documents → IFRC GO field reports → IDMC displacement (a lower bound)**.
   The highest rung with a figure wins. The figure is sanity-checked against
   the GDACS exposed-population estimate (a ceiling — exposure is never used
   as the answer itself) and against national population. If the freeze date
   passes with no figure, the cell resolves to **"no data"** and is flagged
   for human review. Drought skips the ladder entirely and resolves from IPC
   data by a fixed rule.

**The rulebook.** Every threshold and policy switch lives in
[`rulebook.yaml`](rulebook.yaml) — cyclone wind/distance thresholds, the
GDACS alert colour that counts as a flood, the ladder order, the freeze
window, sanity-check multipliers, backcast start years. Change the YAML and
the machine behaves differently on its next run; nothing needs a programmer.
The file is validated on load and refuses anything that looks like a
credential (API keys come from environment variables only).

**Hard guarantees.**

- Reconciliation is deterministic: rules only, no AI judgement calls.
- The one AI step (reading ReliefWeb documents) only transcribes figures a
  document actually states — it never estimates or infers a number.
- Resolutions freeze 60 days after month-end and are never reopened. Later
  source revisions are logged in an audit table (`haz_revisions`) but do not
  change resolved values.
- GDACS is used for detection and plausibility ceilings only, never as a
  resolution value.

**Where the data lives.** All tables sit in the resolver DuckDB alongside
the existing `facts_resolved`: thin raw caches per source (`haz_raw_*`),
detection verdicts (`haz_triggers`), candidate figures
(`haz_impact_candidates`), final answers (`haz_resolutions`), the post-freeze
audit log (`haz_revisions`), and historical base rates
(`haz_base_rates_occurrence`, `haz_base_rates_severity`). See
[`schema.py`](schema.py).

**Running it (Phase 0).**

```bash
# Create/refresh the haz_* tables (idempotent, safe on a live DB)
python -m resolver.hazard_resolution.migrate

# Load national population denominators into haz_raw_population
python -m resolver.hazard_resolution.population
```

Later phases add the detection connectors, the ReliefWeb extraction step,
the reconciliation engine, and the backcast. Costs stay well under
USD 50/month: every data source is free; the only spend is the cheap-model
document extraction.
