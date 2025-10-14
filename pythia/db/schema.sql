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
  metric TEXT CHECK (metric IN ('PIN','PA')),
  target_month TEXT,                     -- YYYY-MM
  wording TEXT,
  best_guess_value DOUBLE,               -- best guess for metric
  hs_json JSON,                          -- per-question subset
  status TEXT DEFAULT 'active',          -- draft|active|frozen|resolved|archived
  supersedes_question_id TEXT,
  created_at TIMESTAMP DEFAULT now()
);

-- Per-model forecasts in 5 bins + optional binary against HS best-guess
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

-- Calibration advice
CREATE TABLE IF NOT EXISTS calibration_advice (
  as_of_month TEXT,                      -- YYYY-MM
  shock_code TEXT,
  model_name TEXT,
  weight DOUBLE,
  notes TEXT,
  created_at TIMESTAMP DEFAULT now(),
  PRIMARY KEY (as_of_month, shock_code, model_name)
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
