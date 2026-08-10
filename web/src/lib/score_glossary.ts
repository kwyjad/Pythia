// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
// See the LICENSE file in the project root for details.

// The shared plain-language score/metric glossary. One copy, two consumers:
// the Performance page tooltips and the Interpreter page glossary — moved
// here (from PerformancePanel's module-private constants) so the two can
// never drift. Keep in sync with pythia/tools/compute_scores.py and
// web/src/app/performance/AboutScores.tsx's long-form copy.

export const TOOLTIP_BRIER_SPD =
  "Multiclass Brier score for SPD questions (PA, FATALITIES, PHASE3PLUS_IN_NEED). " +
  "Lower is better. Range: 0 (perfect) to 2. Not comparable to Binary Brier.";

export const TOOLTIP_BRIER_BINARY =
  "Brier score for binary EVENT_OCCURRENCE questions: (forecast_p − outcome)². " +
  "Lower is better. Range: 0 (perfect) to 1. Structurally near-zero for rare events " +
  "correctly forecast as unlikely. Not comparable to SPD Brier — the two families " +
  "are never averaged together.";

export const TOOLTIP_LOG =
  "Logarithmic scoring rule. Lower is better. Heavily penalises confident wrong predictions. Range: 0 (perfect) to +∞.";

export const TOOLTIP_CRPS =
  "Normalized Ranked Probability Score (the discrete form of CRPS): squared cumulative-distribution error across the ordered buckets, divided by K−1. Lower is better. Range: 0 (perfect) to 1 (all probability in the bucket farthest from the outcome).";

export const TOOLTIP_SAMPLES =
  "Number of individual scored data points (question × horizon combinations). Each question can produce multiple samples across different forecast horizons.";

export const TOOLTIP_EXTERNAL_BENCHMARK =
  "ViEWS (Violence & Impacts Early-Warning System) is an external conflict fatality " +
  "forecasting model from Uppsala University that produces monthly point forecasts " +
  "(expected fatality counts) for every country at lead times of 1–6 months — the " +
  "same horizons Fred uses.\n\n" +
  "Scoring method: ViEWS point forecasts are converted into synthetic 7-bucket " +
  "probability distributions using a log-normal model. Given a point forecast of X " +
  "fatalities, we construct a log-normal distribution with E[X] = point forecast and " +
  "integrate over Fred’s fatality bucket boundaries (0, 1–4, 5–24, 25–99, 100–499, 500–999, ≥1000) " +
  "to get bucket probabilities. These synthetic SPDs are then scored with the same " +
  "Brier, Log Loss, and CRPS functions used for Fred’s own models.\n\n" +
  "The log-normal spread parameter (σ) controls how uncertain the synthetic SPD is " +
  "around the point forecast. It is periodically optimized to minimize ViEWS’s Brier " +
  "score against resolved outcomes.\n\n" +
  "Purpose: This lets us directly compare ViEWS accuracy against Fred’s ensemble and " +
  "individual models on the same scale. The comparison also feeds back into Fred’s " +
  "calibration advice — if ViEWS outperforms the ensemble, forecasting models are " +
  "told to anchor more strongly on ViEWS conflict forecasts when they are provided as input.";

// --- Interpreter-page additions (deviation metrics + reference forecasters).
// Definitions mirror pythia/tools/compute_deviation.py and score_baselines.py.

export const TOOLTIP_JS_VS_BASERATE =
  "Jensen–Shannon divergence between the published forecast distribution " +
  "(averaged over the six window months) and the base-rate anchor the " +
  "forecaster was shown at prompt time. 0 = the forecast came back at the " +
  "base rate; the maximum (ln 2) = total disagreement. Shown as a share of " +
  "that maximum.";

export const TOOLTIP_LOG_EV_RATIO =
  "ln(expected impact of the forecast ÷ expected impact of the base rate), " +
  "signed: positive = the forecast expects MORE impact than history alone " +
  "would suggest, negative = less.";

export const TOOLTIP_EIV =
  "Expected impact value over the six-month window: the probability-weighted " +
  "bucket midpoints, blended with the same surge formula the risk index " +
  "uses. Binary event questions have no impact centroid, so no EIV.";

export const TOOLTIP_ATTENTION =
  "Blended attention rank across four orderings: deviation from base rate, " +
  "nominal expected impact, per-capita expected impact (with an absolute " +
  "floor so small islands don't dominate), and disagreement between the " +
  "Horizon Scanner's regime-change view and the ensemble's deviation.";

export const TOOLTIP_SKILL =
  "skill = 1 − (model score ÷ climatology score), computed per score family " +
  "and score type against the __ext_climatology reference (the base-rate " +
  "forecast). Positive = beat the base rate; zero = matched it; negative = " +
  "lost to it. Track 1 and Track 2 are disjoint populations — compare each " +
  "track against its own climatology, never raw scores across tracks.";

// --- The plain-English copy the monthly report prints -----------------------
// Mirrored verbatim in interpreter/performance.py (SCORE_EXPLANATIONS) and
// pinned by interpreter/tests/test_performance.py, so the PDF and the
// dashboard can never explain the same score two different ways. The tooltips
// above stay technical for the Performance page; these are for a reader who
// has never seen a Brier score.
export const REPORT_SCORE_EXPLANATIONS: Array<{ term: string; text: string }> = [
  {
    term: "Brier score",
    text:
      "How far the forecast sat from what happened, squared and averaged. " +
      "Lower is better, and zero is perfect. It rewards a forecast that put " +
      "its weight in the right place and punishes one that was confident " +
      "and wrong.",
  },
  {
    term: "Log loss",
    text:
      "The same idea, but far harsher on confidence. A forecast that ruled " +
      "something out and then watched it happen scores very badly indeed. " +
      "Lower is better.",
  },
  {
    term: "Ranked probability score",
    text:
      "Brier's cousin for questions about size. It gives partial credit for " +
      "being close: a forecast that expected the band next door scores " +
      "better than one that expected the other end of the scale. Lower is " +
      "better.",
  },
  {
    term: "Skill",
    text:
      "The score set against what you would have got by assuming a normal " +
      "year. Above zero means the system beat that; below zero means it " +
      "lost to it. This is the number that says whether the forecasting was " +
      "worth doing.",
  },
];

export const SCORE_GLOSSARY: Array<{ term: string; text: string }> = [
  { term: "SPD Brier", text: TOOLTIP_BRIER_SPD },
  { term: "Binary Brier", text: TOOLTIP_BRIER_BINARY },
  { term: "Log loss", text: TOOLTIP_LOG },
  { term: "CRPS (RPS)", text: TOOLTIP_CRPS },
  { term: "Skill vs climatology", text: TOOLTIP_SKILL },
  { term: "Deviation from base rate", text: TOOLTIP_JS_VS_BASERATE },
  { term: "Expected impact (EIV)", text: TOOLTIP_EIV },
  { term: "Direction (log EV ratio)", text: TOOLTIP_LOG_EV_RATIO },
  { term: "Attention rank", text: TOOLTIP_ATTENTION },
];
