# The resolver review's inputs in Drive

The monthly resolver-to-calibration review follows the template in Drive at
`Fred_archive/Performance_Reports/template/resolver_review_template.md`. That
copy is the one the scheduled session reads, and Kevin edits it by hand; this
note records the change to its inputs section (A2) that the archive now
supports, so the two can be kept in step.

## A2: read `resolver_review/resolver_review_inputs.json` first

Every chain with a Resolver Update stage now has a `resolver_review/` folder
beside its zip in
`Fred_archive/runs/<YYYY>/<YYYY-MM-DD>__scoring__gh<publish_id>/`. Read
`resolver_review/resolver_review_inputs.json` first. It holds one key per
section of the template, and the counts the review needs from files too large
to put in Drive. Open the lifted files in the same folder only to check a
figure or read a quote. Open the chain zip only when a file the review needs
is in neither place; `resolver_review/SKIPPED.md` names every file the folder
does not hold and why.

| Key | Template section | Built from |
|---|---|---|
| `chain` | 2 | each stage's `api_run.json` and `lineage.json`; the bundle's `run/workflow_run.json` and `run/step_timings.csv`; zero-second steps whose logs have content (matched by phase number) and every log with a `Traceback` (rule A4.15) |
| `issues` | 3 | `checks/issues.json`, and every other stage's `issues.json` |
| `ingestion` | 4 | `checks/reconciliation.md` (connectors not ok, not agreeing, or fetching nothing), `db/freshness.csv` (past threshold), `db/queries/crisiswatch_editions.csv`, the CrisisWatch line of `resolver_inspect.md`, the refresh runs in the chain |
| `contradictions` | 5 | `checks/contradictions.md`: every check with verdict, observed and expected |
| `pa_machine` | 6 | `hazard/acceptance_report.md` (its tables); `hazard/cell_ledger.csv` aggregated (status x hazard, reason x hazard for cells with no row, ceiling basis for resolved values, flags, months not yet ended); extraction budget this month and last; backcast progress; freeze dates (month end + `freeze_days`) for every provisional month (rule A4.17) |
| `conflict_displacement` | 7 | the four tables before and after (`db/signature_*.json`), the Phase 2.6 summaries or their absence, the ledger's latest month and assessed-at range, the IDMC conflict summary |
| `resolutions` | 8.1 to 8.3 | `reports/compute_resolutions.json` with the unresolved split, `resolutions_by_source_desc.csv` |
| `plausibility` | 8.4 | the checks run over the scored bundle's resolutions, the figures ledger and the cell ledger (see below) |
| `scoring` | 9 | the three score reports, and the hand-off check: resolved pairs with a forecast but no score |
| `calibration` | 10 | weights, groups weighted, family recalibration, centroid moves with a counterfactual, Sibyl steps |
| `known_issues` | 14 | `known_issues.yml`, each entry's `review_by` and whether it is overdue at the run date |

A section that could not be read is `{"stub": true, "reason": ...}`. Say so in
Appendix 1 and fall back to the lifted file or the zip.

## The plausibility checks

Each check gives `count` (cells a scored PA resolution reads, or the last
twelve months when the chain has no scored bundle), `count_last_12_months`,
`count_all_months`, up to ten `examples` (iso3, hazard, month, value, the
winning quote) and the scored resolutions and forecaster rows that rest on
the flagged cells. The winning figure for a cell is the accepted figure in
`hazard/figures_ledger.csv` whose persons value equals the cell's value, at
the lowest preference rank.

- `identical_fl_tc_pa_value`: one country-month, the same value under flood and cyclone PA.
- `winning_quote_names_only_earlier_years`: every year the quote names is before the cell's year ("2025a" counts as 2025). It does not read month names; read the quote for those.
- `household_conversion_where_quote_states_people`: a household figure whose quote also states a number of people.
- `one_figure_wins_two_months`: one document and quote winning the same hazard in more than one month.
- `value_above_10pct_of_population`: against `resolver/data/population.csv`.
- `round_extracted_figure`: a pointer, not a fault.
- `displacement_lower_bound_as_people_affected`: by design; counted.
- `zero_default_share_by_group`: event groups are marked exempt.
- `union_of_flagged_cells`: the first five together. The centroid counterfactual in `calibration.centroid_moves` leaves out resolutions on these cells: `counterfactual_new = old + ema_rate x (clean_mean - old)`, with the rate backed out of the run's own move.

All of these are derived figures (rule A4.1): name the key beside each table.
