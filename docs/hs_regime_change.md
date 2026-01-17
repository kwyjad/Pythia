# HS Regime-Change Scoring

Horizon Scanner (HS) now captures a dedicated **regime-change / out-of-pattern risk** assessment for every hazard. The model returns a `regime_change` object, which is normalized, scored, and persisted into `hs_triage`.

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
  - Level 0: `likelihood < 0.35` **or** `score < 0.20`
  - Level 1: `likelihood >= 0.35` **and** `score >= 0.20`
  - Level 2: `likelihood >= 0.60` **and** `magnitude >= 0.50` (score >= 0.30)
  - Level 3: `likelihood >= 0.75` **and** `magnitude >= 0.60` (score >= 0.45)

## `need_full_spd` Override

By default, HS forces `need_full_spd = TRUE` when regime-change risk is **Level 2+** (roughly `score >= 0.30`), even if the triage tier would otherwise be quiet.

## Environment Overrides

All thresholds are environment-overridable:

- Level thresholds:
  - `PYTHIA_HS_RC_LEVEL0_LIKELIHOOD` (default `0.35`)
  - `PYTHIA_HS_RC_LEVEL0_SCORE` (default `0.20`)
  - `PYTHIA_HS_RC_LEVEL1_LIKELIHOOD` (default `0.35`)
  - `PYTHIA_HS_RC_LEVEL1_SCORE` (default `0.20`)
  - `PYTHIA_HS_RC_LEVEL2_LIKELIHOOD` (default `0.60`)
  - `PYTHIA_HS_RC_LEVEL2_MAGNITUDE` (default `0.50`)
  - `PYTHIA_HS_RC_LEVEL3_LIKELIHOOD` (default `0.75`)
  - `PYTHIA_HS_RC_LEVEL3_MAGNITUDE` (default `0.60`)
- Force-full-SPD thresholds:
  - `PYTHIA_HS_RC_FORCE_LEVEL_MIN` (default `2`)
  - `PYTHIA_HS_RC_FORCE_SCORE_MIN` (default `0.30`)
