# HS Regime-Change Scoring

Horizon Scanner (HS) captures a dedicated **regime-change / out-of-pattern risk** assessment for every hazard. The model returns a `regime_change` object, which is normalized, scored, and persisted into `hs_triage`.

## Key Concept: Regime Change vs Triage Score

- **`triage_score`** captures the *overall risk level*, including ongoing/chronic situations. A country with severe but steady conflict has a HIGH triage_score.
- **`regime_change`** captures *only* the probability and magnitude of a *departure from the established pattern*. That same country with steady conflict has LOW regime_change likelihood, because the pattern is continuing as expected.

A regime change means a statistically significant departure from the historical base rate — not just elevated risk. Most country-hazard pairs should have low RC values.

## Fields

Each hazard includes a normalized regime-change object with these fields:

- `likelihood` (0.0–1.0): likelihood of a base-rate break in the next 1–6 months.
- `magnitude` (0.0–1.0): how large the out-of-pattern change could be if it occurs.
- `direction`: one of `up`, `down`, `mixed`, `unclear`.
- `window`: one of `month_1`..`month_6`, `month_1-2`, `month_3-4`, `month_5-6`.
- `rationale_bullets`: concise bullets tying the regime-change risk to evidence pack signals.
- `trigger_signals`: optional list of specific signals that would indicate a break.

The normalized object is serialized into `regime_change_json` in `hs_triage`.

## Score and Level

HS computes a regime-change score and severity level:

- **Score**: `likelihood * magnitude` (defaults to `0.0` if either value is missing).
- **Level** (default thresholds):
  - Level 0: `likelihood < 0.45` **or** `score < 0.25`
  - Level 1: `likelihood >= 0.45` **and** `score >= 0.25`
  - Level 2: `likelihood >= 0.60` **and** `magnitude >= 0.50` (score >= 0.30)
  - Level 3: `likelihood >= 0.75` **and** `magnitude >= 0.60` (score >= 0.45)

## Expected Distribution

Across a full run of ~120 countries × 6 hazards (~720 assessments), the expected distribution is:

- ~80% at likelihood ≤ 0.10 (base-rate normal, no regime change signal)
- ~10% at likelihood 0.10–0.30 (watch)
- ~7% at likelihood 0.30–0.55 (emerging)
- ~3% at likelihood ≥ 0.55 (strong signal)

A run-level distribution check logs warnings when too many assessments exceed expected thresholds.

## `need_full_spd` Override

By default, HS forces `need_full_spd = TRUE` when regime-change risk is **Level 2+** (roughly `score >= 0.30`), even if the triage tier would otherwise be quiet.

## Environment Overrides

All thresholds are environment-overridable:

- Level thresholds:
  - `PYTHIA_HS_RC_LEVEL0_LIKELIHOOD` (default `0.45`)
  - `PYTHIA_HS_RC_LEVEL0_SCORE` (default `0.25`)
  - `PYTHIA_HS_RC_LEVEL1_LIKELIHOOD` (default `0.45`)
  - `PYTHIA_HS_RC_LEVEL1_SCORE` (default `0.25`)
  - `PYTHIA_HS_RC_LEVEL2_LIKELIHOOD` (default `0.60`)
  - `PYTHIA_HS_RC_LEVEL2_MAGNITUDE` (default `0.50`)
  - `PYTHIA_HS_RC_LEVEL3_LIKELIHOOD` (default `0.75`)
  - `PYTHIA_HS_RC_LEVEL3_MAGNITUDE` (default `0.60`)
- Force-full-SPD thresholds:
  - `PYTHIA_HS_RC_FORCE_LEVEL_MIN` (default `2`)
  - `PYTHIA_HS_RC_FORCE_SCORE_MIN` (default `0.30`)
- Distribution warning thresholds:
  - `PYTHIA_HS_RC_DIST_WARN_L1_FRAC` (default `0.25`) — warn if >25% of assessments are L1+
  - `PYTHIA_HS_RC_DIST_WARN_L2_FRAC` (default `0.15`) — warn if >15% of assessments are L2+
  - `PYTHIA_HS_RC_DIST_WARN_L3_FRAC` (default `0.08`) — warn if >8% of assessments are L3
