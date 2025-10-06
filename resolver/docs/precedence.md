# Precedence Policy

Resolver's precedence engine selects a single authoritative figure per `(country_iso3, hazard_code, month, metric)` by applying tier policy, recency, and deterministic tie-breakers. This page summarises the policy and illustrates how ReliefWeb PDFs integrate into tier-2 decisions.

## Tier ordering

The default ordering is configured in [`resolver/tools/precedence_config.yml`](../tools/precedence_config.yml) and mirrors the governance policy:

1. `inter_agency_plan`
2. `ifrc_or_gov_sitrep`
3. `un_cluster_snapshot`
4. `reputable_ingo_un`
5. `media_discovery_only`
6. `reliefweb_pdf` (tier-2 branch)

ReliefWeb PDF rows are tagged with `tier=2` to ensure they are considered after tier-1 sources but before open media discovery. They are primarily used to fill gaps when higher tiers do not report monthly deltas for a given hazard month.

## Tie-breaking rules

When multiple rows land in the same tier:

1. Prefer the row with the most recent `as_of` date.
2. If tied, prefer the row with the most recent `publication_date`.
3. If still tied, prefer rows with complete values (non-null `value`).
4. Fall back to a deterministic order using the source code to avoid non-reproducible output.

Manual overrides created via [`resolver/tools/precedence_engine.py --overrides`](../tools/precedence_engine.py) or review workflows supersede automated outcomes. Overrides must log a note explaining the decision.

## Example scenarios

- **Tier-1 present, ReliefWeb PDF present:** If IFRC GO reports a PIN value for March 2025 and a ReliefWeb PDF emits a similar metric, the engine keeps the IFRC GO figure (tier `ifrc_or_gov_sitrep`). The ReliefWeb PDF row remains visible in diagnostics for transparency.
- **Tier-1 absent, ReliefWeb PDF fills gap:** Suppose IPC has no March 2025 update for a cyclone, but the ReliefWeb PDF pipeline produced a `people_in_need` delta. The precedence engine elevates the ReliefWeb PDF row (tier-2) so downstream exports carry a monthly new value instead of leaving a hole.
- **Conflicting ReliefWeb PDFs:** When multiple PDFs from the same month survive the selector, the engine uses `as_of` and `publication_date` tie-breakers. Duplicates are rare because the PDF branch computes month-over-month deltas per report series before they hit precedence.

## Review and auditability

All precedence choices are recorded in `resolver/exports/resolved_diagnostics.csv`. ReliefWeb PDF rows surface `extraction_method`, `pph_used`, and the attachment manifest so reviewers can trace back to the source document. See [Governance & audit](governance.md) for additional controls.
