# August 2026 audit — PA resolution machine PRs #843–#849

Full-integration audit of the six build phases (Phase 0 scaffold → Phase 6
prompt block), run 2026-08-05, immediately after all seven PRs merged. Read
this before re-reporting an issue against `resolver/hazard_resolution/`.

## Headline findings (all fixed in the audit PR)

1. **The machine was complete, tested, and entirely unrun.** No workflow
   invoked `resolve-hazards` / `haz-backcast` / `haz-base-rates` /
   `haz-acceptance`; the `haz_*` tables never existed in any production DB;
   the Phase 6 base-rate block rendered `""` on every production SPD prompt
   (behind a silent `except Exception: return ""` seam); extraction spend
   wrote no `llm_calls` row and had no Costs-page phase; six diagnostic
   scripts had zero awareness of any `haz_` table. → Wired: resolver_update
   Phase 2.5 (live months), nightly `haz_backcast.yml` (time-boxed,
   self-converging), `hazard_extraction` → `resolution` cost phase,
   inspect/summarize sections, `--summary-out` JSON artifacts.
2. **Extraction cache poisoning**: `status='error'` rows were permanent cache
   hits — one outage or missing API key permanently blinded documents AND
   burned the monthly call cap on calls that never left the process.
   → ok-only cache hits; budget/cap count billed calls only.
3. **Cache-key contamination**: keyed (doc, model, prompt_version) while the
   prompt varies by hazard/country/month — shared documents crossed answers
   between cells. → cell-scoped key (+ prompt_version bumped to v2).
4. **The prompt never named the target month** it told the model to filter
   by, and addressed countries as ISO3 codes. → month + country name
   interpolated.
5. **Quote verification never tied the value to the quote** — a fabricated
   number with any real sentence passed. → the value must appear in the
   quote (digits, grouped, or million/thousand phrasing);
   `value_not_found_in_quote` rejection reason.
6. **No per-country exception guard** in `resolve_triggered_cells`, and the
   CLI month loop caught return codes but not exceptions — one country could
   kill the month and all later months. → per-cell guards (ladder + drought),
   failures in the exit code, month-loop try/except.
7. **Backcast cost ledger always zero**: `record_month` wrote
   `extraction_calls`/`extraction_cost_usd`/`frozen_skipped` from keys
   `month_counts` never produced. → read back from `haz_doc_extractions` /
   `haz_revisions`.
8. **Provisional values leaked into severity base rates**
   (`compute_severity` filtered by year only). → frozen rows only.
9. Smaller: doc cap not enforced at read time (cache accumulation past 30);
   dead rulebook keys `extraction.max_output_tokens`/`request_timeout_sec`
   (now threaded through `call_chat_ms`); budget-cap `break` discarded
   already-paid cached figures (now `continue`); `cells_extracted` ignored
   cache-only cells; prompt_block's TC/FL-fatal drought-only rulebook lookup,
   header-window mismatch, all-`n/a` table rendering, dead `_LADDER_HAZARDS`;
   DB-URL precedence split-brain at the prompts.py seam (now env-first,
   matching `duckdb_io`); boundaries geojson missing from poetry `include`;
   `rulebook.py` missing from forecaster-ci paths; `haz-migrate` script
   registered.

## Answers to the owner's audit questions (as of the audit, pre-fix)

- **Q1 fire on the next resolver update, alongside IFRC GO?** No — the cron
  is the 28th (moved 2026-08-03), and nothing invoked the machine at all.
  IFRC GO itself was untouched. (Fixed: Phase 2.5.)
- **Q2 other data sources unaffected?** Yes — the seven PRs changed zero
  files under `resolver/tools|connectors|ingestion` or `pythia/tools`.
- **Q3 base rates wired?** Prompts: correctly injected into Track 1 + Track 2
  SPD for TC/FL/DR PA (cache-split-safe; deliberately NOT binary/Sibyl/
  scenario) but rendered nothing in production (empty tables).
  **Resolution/calibration: no bridge exists** — `compute_resolutions` never
  reads `haz_resolutions`; scores and calibration are strictly downstream of
  `resolutions`, so a future bridge propagates automatically.
- **Q4 costs wired?** No (fixed — see 1). Model-registry side was already
  clean (role, fallback, CONSUMED_ROLES, cost entry).
- **Q5/Q6 bugs/improvements?** See 2–9.
- **Q7 diagnostics?** None existed (fixed — see 1).

## Deliberately deferred (do NOT re-report as new findings)

- **The resolution bridge (shadow-mode flip).** `compute_resolutions` gaining
  `haz_resolutions` as the top PA source for TC/FL, the question-wording
  reconciliation ("as resolved by EM-DAT" vs the ladder the prompt block now
  describes vs the facts_resolved path that actually scores), and retiring
  the IFRC-GO-first PA path — all gated on the first monthly acceptance
  reports (owner decision, 2026-08-05). Until then a populated FL/TC PA
  prompt carries THREE inconsistent statements of the resolution rule; known
  and accepted for the shadow period.
- **ReliefWeb candidate-pool pagination**: `_payload` caps at
  `candidate_pool_size` (100) newest-first with no pagination — a >100-doc
  emergency can push early-month OCHA sitreps out of the pool before ranking
  runs. `totalCount` is recorded; revisit if `discarded_over_cap` shows it
  binding.
- **One asyncio loop per extraction call**: `_default_call` uses
  `asyncio.run` per document; `providers._LLM_SEMAPHORES` grows per loop id.
  Harmless at live-month volume; worth a shared loop if backcast extraction
  volume grows.
- **`acceptance.main` always exits 0** even when the report's own targets are
  missed — fine while the report is evidence rather than a gate; add a
  `--fail-below` when it starts gating the flip.
- **`_pace_reliefweb` module-global state** — correct single-threaded;
  revisit before any parallel ladder.
- **`duckdb_io.close_db` eviction churn** at the prompt seam (every TC/FL/DR
  PA question closes+evicts the shared cached connection) — pre-existing
  pattern shared with `_load_fewsnet_projection`; consolidate if it shows up
  in profiles.
- **Sibyl / binary prompts deliberately do NOT get the base-rate block**
  (binary is a different resolution definition; Sibyl's independence from
  structured injects is by design).
