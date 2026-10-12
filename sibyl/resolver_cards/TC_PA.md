How this question resolves (checked against pythia/tools/compute_resolutions.py, MACHINE_PA_HAZARDS):
- Source: Pythia's people-affected resolution machine (`haz_resolutions`). In a month with a qualifying cyclone (a best-track storm passing close enough at storm strength, per the rulebook) it takes the first figure on a fixed ladder: EM-DAT (not read today: no credential), then figures stated in ReliefWeb reports (each quote checked against the document), then IFRC GO (field reports, DREF and emergency appeals), then IDMC displacement records.
- An IDMC displacement count IS the answer when it is the best figure available. It is a LOWER BOUND on people affected: it counts only those displaced, which for a cyclone is often pre-emptive evacuations.
- A month with no qualifying cyclone and a silent ReliefWeb sweep resolves to ZERO. So p_zero for this question means "the machine answers zero".
- A month with a detected cyclone but no figure on any rung by the freeze deadline stays UNRESOLVED and is never scored.
- The answer is provisional until the freeze and may be revised before it; the forecast is scored on the answer held at scoring time.
- Satellite exposure estimates (GDACS "population exposed") are a detector and a ceiling on plausible figures, never the resolved value.
