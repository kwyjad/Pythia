# Precedence & Sources for Pythia “People Affected” (PA) and Conflict Fatalities

## Scope & principles
**Metrics in scope now**
- PA (People Affected) for natural hazards and displacement-related shocks
- Conflict fatalities (monthly flow)

**Not using**: PIN (People in Need)

**Semantics**: All monthly, admin0. Flows (e.g., “newly displaced”, “fatalities”) are `semantics=new`. EM-DAT “affected” is treated as PA per month.

**Auditability**: Every record carries: `source_system`, `collection_type`, `coverage`, `freshness_days`, `origin_iso3`, `destination_iso3`, `method_note`.

---

## 1) Hydro-met shocks (Flood, Drought, Cyclone)
**Source precedence**: EM-DAT → PA only (no PIN).

**Rationale**
- Global, consistent, long history.
- Already harmonized across hazards/countries.

**Implementation notes**
- Month attribution: prefer event end date; fallback start date; if spanning months, pick peak impact (fallback end date).
- Deduping: for multiple EM-DAT rows same country/month/hazard, keep max PA per event, then sum events.
- **Metadata**: `source_system=EMDAT`, `collection_type=event_db`, `hazard={flood|drought|cyclone}`.

**Metrics**
- `flood_affected_pa`, `drought_affected_pa`, `cyclone_affected_pa` (all `semantics=new` for the month they’re attributed to).

---

## 2) Conflict Escalation / Onset 1 (PA)
**Definition:**  
`conflict_onset1_pa = internal_displacement_new + cross_border_outflow_new`

### (A) Internal displacements (flows)
**Precedence**
1. **IDMC IDU** (curated, event-centric new displacements)
2. **DTM explicit flows** (“newly displaced / new arrivals / movements”)
3. **Δ(DTM stock)** (inferred, last resort; flag `collection_type=stock_inferred`)

**Rationale**
- IDU is curated & timely; DTM flow is observed but coverage varies; stock deltas can be noisy due to rebaselines.

### (B) Cross-border outflows (mirror from DI)
**Precedence**
1. **UNHCR ODP “new arrivals”** (by origin→destination)
2. **DTM Flow Monitoring** (corridor/site; flag partial coverage)

**Rationale**
- UNHCR arrivals align with protection stats; DTM adds timeliness where UNHCR lacks series.

**Tie-breakers** (A and B)
1. Coverage: national > corridor/site
2. Fresher `as_of`
3. Higher internal QA (if available)
4. If still tied: larger value (assume broader coverage ≈ larger)

**Metadata**
- `source_system`, `collection_type={curated_event|registration|flow_monitoring|stock_inferred}`, `coverage`, `freshness_days`, `origin_iso3`, `destination_iso3`, `method_note`.

---

## 3) Conflict Escalation / Onset 2 (fatalities)
**Source precedence**: **ACLED fatalities** only.  
**Notes**: sum monthly; `semantics=new`; `source_system=ACLED`, `collection_type=event_db`.

**Metric**
- `conflict_fatalities_new`

---

## 4) Displacement Influx (DI)
**Definition**: arrivals into a destination country (flows); mirror as **outflow** for origin (used in Onset 1).

**Precedence**
1. **UNHCR ODP “new arrivals”**
2. **DTM Flow Monitoring** (mark coverage)

**Tie-breakers**
- national > corridor/site; then fresher; then higher coverage %; then larger count.

**Metadata**
- `source_system={UNHCR|DTM}`, `collection_type={registration|flow_monitoring}`, `coverage`, `freshness_days`, `origin_iso3`, `destination_iso3`.

**Metric**
- `displacement_influx_new`

---

## Hygiene & edge cases
- Rebaselines: suppress negative or improbable Δ(stock) unless supported by an explicit flow; add `rebaseline_suspected=1`.
- Partial coverage: when corridor sums, set `coverage=corridor` (+ optional coverage score if known).
- Unknown origin in DI: do not impute to origin.
- Temporal roll-up: weekly → month sum.
- **Never** add IDMC and DTM flows together for the same country-month; pick per precedence.
- Resolver’s IDMC CLI can now emit precedence-ready candidates (`--write-candidates`)
  and dry-run the selector locally (`--run-precedence`).

---

## What this buys us
- **Clarity**: one unambiguous monthly PA figure per shock/country.
- **Timeliness with restraint**: curated/event-flows first, inference last.
- **Explainability**: metadata makes the choice defensible.
