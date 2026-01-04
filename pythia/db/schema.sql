-- Pythia
-- Copyright (c) 2025 Kevin Wyjad
-- Licensed under the Pythia Non-Commercial Public License v1.0.
-- See the LICENSE file in the project root for details.

-- HS runs (audit)
CREATE TABLE IF NOT EXISTS hs_runs (
  run_id TEXT PRIMARY KEY,
  started_at TIMESTAMP,
  finished_at TIMESTAMP,
  countries JSON,
  prompt_key TEXT,
  prompt_version TEXT,
  prompt_sha256 TEXT,
  commit_sha TEXT,
  config_sha TEXT,
  cost_usd DOUBLE DEFAULT 0,
  created_at TIMESTAMP DEFAULT now()
);

-- UI-triggered pipeline runs (API /v1/run)
CREATE TABLE IF NOT EXISTS ui_runs (
  ui_run_id   TEXT PRIMARY KEY,
  started_at  TIMESTAMP,
  finished_at TIMESTAMP,
  countries   JSON,
  status      TEXT,         -- queued|running|ok|failed
  error       TEXT,
  created_at  TIMESTAMP DEFAULT now()
);

-- HS scenarios (one row per scenario narrative)
CREATE TABLE IF NOT EXISTS hs_scenarios (
  scenario_id TEXT PRIMARY KEY,
  run_id TEXT,
  iso3 TEXT,
  country_name TEXT,
  hazard_code TEXT,
  hazard_label TEXT,
  likely_window_month TEXT,              -- YYYY-MM
  markdown TEXT,                         -- full narrative (MD)
  scenario_title TEXT,                   -- display title
  probability_text TEXT,                 -- raw probability string
  probability_pct DOUBLE,                -- numeric probability (e.g., 70.0)
  pin_best_guess BIGINT,                 -- PIN best guess
  pa_best_guess BIGINT,                  -- PA best guess
  json JSON,                             -- exact HS JSON block
  created_at TIMESTAMP DEFAULT now()
);

-- Questions registry (two rows per scenario: PIN & PA)
CREATE TABLE IF NOT EXISTS questions (
  question_id TEXT PRIMARY KEY,
  scenario_id TEXT,
  run_id TEXT,
  iso3 TEXT,
  country_name TEXT,
  hazard_code TEXT,
  hazard_label TEXT,
  metric TEXT CHECK (metric IN ('PIN','PA','FATALITIES')),
  target_month TEXT,                     -- YYYY-MM
  wording TEXT,
  best_guess_value DOUBLE,               -- best guess for metric
  hs_json JSON,                          -- per-question subset
  status TEXT DEFAULT 'active',          -- draft|active|frozen|resolved|archived
  supersedes_question_id TEXT,
  created_at TIMESTAMP DEFAULT now()
);

-- Per-model forecasts in 5 bins + optional binary against HS best-guess
-- Used for both ensemble inputs and per-model SPD storage (one row per bucket).
CREATE TABLE IF NOT EXISTS forecasts_raw (
  forecast_id TEXT PRIMARY KEY,
  question_id TEXT,
  model_name TEXT,
  model_version TEXT,
  prompt_key TEXT,
  prompt_version TEXT,
  prompt_sha256 TEXT,
  horizon_m INTEGER,                     -- 1..6
  class_bin TEXT,                        -- one of five bins
  p DOUBLE,                              -- probability for class_bin
  threshold_value DOUBLE,                -- HS best-guess snapshot
  p_over_threshold DOUBLE,               -- binary prob over threshold
  run_id TEXT,
  tokens_in INTEGER,
  tokens_out INTEGER,
  cost_usd DOUBLE,
  created_at TIMESTAMP DEFAULT now()
);

-- Aggregated (ensemble) forecasts
CREATE TABLE IF NOT EXISTS forecasts_ensemble (
  question_id TEXT,
  horizon_m INTEGER,
  class_bin TEXT,
  p DOUBLE,
  p_over_threshold DOUBLE,
  aggregator TEXT,                       -- Bayes_MC / GTMC1etc
  ensemble_version TEXT,
  created_at TIMESTAMP DEFAULT now(),
  PRIMARY KEY (question_id, horizon_m, class_bin)
);

-- Resolutions (link to Resolver monthly deltas)
CREATE TABLE IF NOT EXISTS resolutions (
  question_id TEXT,
  observed_month TEXT,                   -- YYYY-MM equals target_month
  value DOUBLE,
  source_snapshot_ym TEXT,               -- resolver snapshot month used
  created_at TIMESTAMP DEFAULT now(),
  PRIMARY KEY (question_id, observed_month)
);

-- Scores (model and ensemble)
CREATE TABLE IF NOT EXISTS scores (
  question_id TEXT,
  horizon_m INTEGER,
  metric TEXT,
  score_type TEXT,                       -- brier|log|crps
  model_name TEXT,                       -- NULL => ensemble
  value DOUBLE,
  created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS calibration_advice (
  as_of_month TEXT,                      -- YYYY-MM
  hazard_code TEXT,
  metric TEXT,
  advice TEXT,
  created_at TIMESTAMP DEFAULT now(),
  PRIMARY KEY (as_of_month, hazard_code, metric)
);

-- Model calibration weights for SPD forecasts (per hazard, metric, and as_of_month)
CREATE TABLE IF NOT EXISTS calibration_weights (
  as_of_month TEXT,          -- 'YYYY-MM'
  hazard_code TEXT,          -- e.g. 'FLOOD', 'CONFLICT'
  metric TEXT,               -- 'PA', 'FATALITIES', etc.
  model_name TEXT,           -- null for ensemble, or LLM model slug
  weight DOUBLE,
  n_questions INTEGER,       -- distinct resolved question concepts used
  n_samples INTEGER,         -- total (question_id, horizon_m) samples
  avg_brier DOUBLE,
  avg_log DOUBLE,
  avg_crps DOUBLE,
  created_at TIMESTAMP DEFAULT now(),
  PRIMARY KEY (as_of_month, hazard_code, metric, model_name)
);

-- LLM call logs (costs & usage)
CREATE TABLE IF NOT EXISTS llm_calls (
  call_id TEXT PRIMARY KEY,
  component TEXT,                        -- HS|Researcher|Forecaster|GTMC1|Aggregator
  model_name TEXT,
  prompt_key TEXT,
  prompt_version TEXT,
  tokens_in INTEGER,
  tokens_out INTEGER,
  cost_usd DOUBLE,
  latency_ms INTEGER,
  success BOOLEAN,
  llm_profile TEXT,                      -- "test" | "prod" (or others if added)
  hs_run_id TEXT,                        -- fk -> hs_runs.run_id (nullable)
  ui_run_id TEXT,                        -- fk -> ui_runs.ui_run_id (nullable)
  forecaster_run_id TEXT,                -- Forecaster CLI run_id (nullable)
  iso3 TEXT,
  hazard_code TEXT,
  metric TEXT,
  phase TEXT,
  status TEXT,                         -- ok|error
  error_type TEXT,                     -- timeout|rate_limit|provider_error|parse_error
  error_message TEXT,                  -- truncated message
  hazard_scores_json TEXT,             -- JSON map hazard_code -> score
  hazard_scores_parse_ok BOOLEAN,
  response_format TEXT,                -- json|fenced_json|text
  created_at TIMESTAMP DEFAULT now()
);

-- Component/model performance (marginal contribution tests etc.)
CREATE TABLE IF NOT EXISTS component_performance (
  question_id TEXT,
  model_name TEXT,
  component TEXT,
  horizon_m INTEGER,
  metric TEXT,
  score_type TEXT,
  score_value DOUBLE,
  run_id TEXT,
  created_at TIMESTAMP DEFAULT now()
);

-- Optional: populations (can be filled from WorldPop)
CREATE TABLE IF NOT EXISTS populations (
  iso3 TEXT,
  year INTEGER,
  population BIGINT,
  PRIMARY KEY (iso3, year)
);

-- Bucket centroids for SPD expected values
CREATE TABLE IF NOT EXISTS bucket_centroids (
  hazard_code TEXT,     -- e.g. 'FLOOD', 'CONFLICT'; NULL/'' = all-hazards if you ever want it
  metric      TEXT,     -- e.g. 'PA' or 'FATALITIES'
  class_bin   TEXT,     -- must match the relevant SPD class bins (PA: '<10k'..'>=500k'; conflict fatalities: '<5'..'>=500')
  ev          DOUBLE,   -- E[metric | bucket, hazard_code]
  n_obs       BIGINT,
  updated_at  TIMESTAMP DEFAULT now(),
  PRIMARY KEY (hazard_code, metric, class_bin)
);
