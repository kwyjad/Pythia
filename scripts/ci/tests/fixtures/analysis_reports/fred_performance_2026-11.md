# Fred performance report: November 2026 scoring round

DRY RUN of template v1.0. Prepared by hand from the archived scoring chain, not by a scheduled run.

| Field | Value |
|---|---|
| Resolver Update date | 2026-11-11 (run 42999258374) |
| Scoring chain | pythia_2026-11-11__scoring__gh43009722469.zip |
| Scored bundle | scored_forecast_analysis__2026-11.zip, calibration run 43008755005, git a1186a0 |
| Bundle builder version | 1.0.0 |
| Signal taxonomy version | not stated in the scored bundle |
| Primary aggregate | ensemble_bayesmc_v2 (Track 1), track2_flash (Track 2) |
| Question-epoch window | last 12 months: 468 scored questions (258 SPD, 210 binary), observed months 2026-08 to 2026-10 |
| Forecast vintage | every scored question's latest forecast comes from a run between 15 July and 17 September 2026 |
| Template version | 1.0 |
| Prepared | 2026-11-20 |

## 1. Summary

Fred lost to climatology on conflict fatalities and on drought and cyclone event questions. It beat climatology on flood events, on Track 2 flood people affected and on Track 1 cyclone people affected. Conflict fatalities did worse against persistence: Fred lost 47 of 62 Track 1 questions to it. Two causes stand out. First, ensemble members chose starting distributions worse than the base rate they were shown, and their forecasts were too spread out. Second, the adversarial counter-evidence step, the most cited signal class, moved probability away from the outcome. Drought event misses come mostly from the prompt. Its GDACS history block repeated months in 40 of 84 prompts and showed zero past events in 11 where the DB holds some. Half of all drought events happened when Fred said under 10%. Sibyl and the ensemble are level on Sibyl's 28 paired questions. Sibyl is sharper on conflict, but costs about four times as much per question. These scores describe July to September forecasts only. None of the October changes (prior anchor, advice arm, recalibration, partial-month fix) is in this round. Before the 13 December run, check that the GDACS history block in current prompts is correct.

## 2. What resolved this round

*Corrected 20 November.* Excerpt ends here.
