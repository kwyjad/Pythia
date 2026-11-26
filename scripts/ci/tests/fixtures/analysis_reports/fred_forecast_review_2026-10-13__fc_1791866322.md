# Fred forecast pipeline review: 2026-10-13 (fc_1791866322)

| Field | Value |
|---|---|
| Forecast chain | 2026-10-13__forecast__fc_1791866322__gh38355281121 |
| HS run / forecaster run / Sibyl run | hs_20261013T041422 / fc_1791866322 / sibyl_1791868847099 |
| Pipeline stage runs, Sibyl run, Publish run | 38346198646, 38347109312, 38349905918; 38350797789; 38355281121 |
| Commit | 50b8b5f3356b9e2c3891451870d8dd896b0a5969 |
| Forecast window | 2026-11 to 2027-04 |
| Previous production forecast run | fc_1790831584, 2026-10-01 (hs_20261001T045127), from `run_comparison` |
| Inputs builder version | inputs_version 1, schema_version 1 |
| Previous review | none; this is the first forecast review |
| Template version | 1.0 (2026-11-24) |
| Prepared | 2026-11-25, scheduled run |

## 1. Summary

The run worked with faults. The ensemble held: every member landed every cell (2,148 SPD, 2,250 binary), the RC sync constraint held at 63 of 63, and no batch request fell back to synchronous. The worst fault is in the prompts. The GDACS "Recent event timeline" in SPD prompts and Track 2 binary prompts is one fixed 564-character block for every country and hazard. For Angola drought and Cambodia cyclone, the Track 1 binary prompt lists real recent events and the SPD prompt for the same pair shows the fixed block. Reach: Input, up to 61 questions, confirmed in every sampled prompt. The run asked 164 questions against 199 on 1 October and cost USD 89.86 against 51.35; all of the rise is Sibyl, while the forecast and HS phases got cheaper.

| Stage | Verdict | Worst issue | Reach |
|---|---|---|---|
| Chain and orchestration | worked with faults | no stage wrote chain diagnostics | contained |

## 2. Chain health and timeline

Excerpt ends here.
