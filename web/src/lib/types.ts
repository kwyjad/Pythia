export type VersionResponse = {
  latest_hs_run_id?: string | null;
  latest_hs_created_at?: string | null;
  latest_data_at?: string | null;
  latest_calibration_at?: string | null;
  latest_scores_at?: string | null;
  manifest?: Record<string, unknown> | null;
};

export type DiagnosticsSummaryResponse = {
  questions_by_status?: Array<{ status?: string | null; n?: number | null }>;
  questions_with_forecasts?: number | null;
  questions_with_resolutions?: number | null;
  questions_with_scores?: number | null;
  latest_hs_run?: Record<string, unknown> | null;
  latest_calibration?: Record<string, unknown> | null;
};

export type DiagnosticsKpiScope = {
  label: string;
  questions: number;
  forecasts: number;
  countries?: number;
  countries_triaged?: number;
  countries_total?: number;
  countries_with_forecasts?: number;
  resolved_questions: number;
  forecasts_by_hazard: Record<string, number>;
};

export type ForecastRun = {
  run_id: string;
  started_at: string | null;
  n_questions: number;
  is_latest: boolean;
  is_test?: boolean;
};

export type DiagnosticsKpiScopesResponse = {
  available_months: Array<{
    year_month: string;
    label: string;
    is_latest: boolean;
  }>;
  available_runs?: ForecastRun[];
  selected_month: string | null;
  selected_run_id?: string | null;
  scopes: Record<string, DiagnosticsKpiScope>;
  explanations?: string[] | null;
  diagnostics?: Record<string, unknown> | null;
  notes?: string[] | null;
};

export type RiskIndexResponse = {
  metric: "PA" | "FATALITIES" | "EVENT_OCCURRENCE" | "PHASE3PLUS_IN_NEED" | string;
  target_month: string | null;
  horizon_m?: number | null;
  normalize?: boolean | null;
  rows?: RiskIndexRow[];
  metric_type?: "binary" | "spd" | null;
};

export type RiskView =
  | "ALL_METRICS_SUMMARY"
  | "PA_EIV"
  | "PA_PC"
  | "FATALITIES_EIV"
  | "FATALITIES_PC"
  | "EVENT_OCCURRENCE"
  | "PHASE3PLUS_EIV"
  | "PHASE3PLUS_PC";

export type CountriesRow = {
  iso3: string;
  n_questions: number;
  n_forecasted: number;
  country_name?: string | null;
  last_triaged?: string | null;
  last_forecasted?: string | null;
  highest_rc_level?: number | null;
  highest_rc_score?: number | null;
  in_country_list?: boolean;
};

export type CountriesResponse = {
  rows: CountriesRow[];
};

export type RiskIndexRow = {
  iso3: string;
  country_name?: string | null;
  n_hazards_forecasted?: number | null;
  m1?: number | null;
  m2?: number | null;
  m3?: number | null;
  m4?: number | null;
  m5?: number | null;
  m6?: number | null;
  total?: number | null;
  expected_value?: number | null;
  population?: number | null;
  m1_pc?: number | null;
  m2_pc?: number | null;
  m3_pc?: number | null;
  m4_pc?: number | null;
  m5_pc?: number | null;
  m6_pc?: number | null;
  total_pc?: number | null;
  per_capita?: number | null;
  metric_type?: "binary" | "spd" | null;
};

export type RunSummaryHazardBreakdown = {
  hazard_code: string;
  count: number;
};

export type RunSummaryMetric = {
  metric: string;
  label: string;
  questions: number;
  countries: number;
  hazards: RunSummaryHazardBreakdown[];
};

export type RunSummaryRcByHazard = {
  hazard_code: string;
  L0: number;
  L1: number;
  L2: number;
  L3: number;
};

export type RunSummaryResponse = {
  run_id: string | null;
  hs_run_id: string | null;
  updated_at: string | null;
  coverage: {
    countries_scanned: number;
    hazard_pairs_assessed: number;
    seasonal_screenouts: number;
    acled_low_activity: number;
    pairs_with_questions: number;
    total_questions: number;
    countries_with_forecasts: number;
    countries_no_questions: number;
    triaged_quiet: number;
  };
  metrics: RunSummaryMetric[];
  rc_assessment: {
    total_assessed: number;
    levels: { L0: number; L1: number; L2: number; L3: number };
    l1_plus_rate: number;
    by_hazard: RunSummaryRcByHazard[];
    countries_by_level: { L1: number; L2: number; L3: number };
  };
  tracks: {
    track1: { questions: number; countries: number; models: number };
    track2: { questions: number; countries: number };
  };
  ensemble: { expected: number; ok: number };
  cost: {
    total_usd: number;
    total_tokens: number;
    by_phase: Array<{ phase: string; label: string; cost_usd: number }>;
  };
  llm_health: {
    total_calls: number;
    errors: number;
    error_rate: number;
  };
  performance: {
    resolved_questions: number;
    total_questions: number;
    brier: { avg: number | null; median: number | null };
    log: { avg: number | null; median: number | null };
    crps: { avg: number | null; median: number | null };
  };
  /** Parallel deep-research track coverage; null when no Sibyl run exists. */
  sibyl?: {
    sibyl_run_id: string;
    budget_capped: boolean;
    run_cost_usd: number;
    opus_cost_usd: number;
    brave_cost_usd: number;
    n_selected: number;
    n_forecast: number;
    n_skipped: number;
    n_skipped_budget_cap: number;
    k: number;
    aggregation: string | null;
  } | null;
};

export type QuestionsResponse = {
  rows: Array<{
    question_id: string;
    hs_run_id?: string | null;
    iso3: string;
    hazard_code: string;
    metric: string;
    target_month: string;
    window_start_date?: string | null;
    forecast_date?: string | null;
    forecast_horizon_max?: number | null;
    eiv_total?: number | null;
    eiv_peak?: number | null;
    triage_score?: number | null;
    triage_tier?: string | null;
    triage_need_full_spd?: boolean | null;
    triage_date?: string | null;
    regime_change_likelihood?: number | null;
    regime_change_direction?: string | null;
    regime_change_magnitude?: number | null;
    regime_change_score?: number | null;
    regime_change_level?: number | null;
    status?: string | null;
    wording?: string | null;
  }>;
};

/**
 * Brier score family. `binary` = EVENT_OCCURRENCE (range 0-1); `spd` =
 * multiclass PA/FATALITIES/PHASE3PLUS_IN_NEED (range 0-2). The two are on
 * different scales and must never be averaged together. Optional because a
 * frontend build may be deployed ahead of the API that adds the field.
 */
export type ScoreFamily = "binary" | "spd";

export type PerformanceSummaryRow = {
  hazard_code: string;
  metric: string;
  score_family?: ScoreFamily;
  score_type: string;
  model_name: string | null;
  n_samples: number;
  n_questions: number;
  avg_value: number | null;
  median_value: number | null;
  resolution_rate?: number | null;
  n_scored?: number | null;
  n_total?: number | null;
};

export type PerformanceRunRow = {
  forecaster_run_id: string | null;
  hs_run_id: string;
  run_date: string | null;
  hazard_code: string;
  metric: string;
  score_family?: ScoreFamily;
  score_type: string;
  model_name: string | null;
  n_samples: number;
  n_questions: number;
  avg_value: number | null;
  median_value: number | null;
};

export type ResolutionRateRow = {
  hazard_code: string;
  metric: string;
  total_questions: number;
  resolved_questions: number;
  skipped_questions: number;
  /**
   * Questions whose earliest horizon (window_start_date) is after the
   * resolution pipeline's calendar cutoff (previous complete month). These
   * are structurally unresolvable until the calendar advances — typically
   * brand-new questions from the latest epoch. The dashboard uses this to
   * distinguish a "waiting for first horizon" state from a real 0% scored.
   * Optional for backward compatibility with older API versions.
   */
  pending_too_new?: number;
  resolution_rate: number;
};

export type ResolutionRatesResponse = {
  rows: ResolutionRateRow[];
};

export type PerformanceScoresResponse = {
  summary_rows: PerformanceSummaryRow[];
  run_rows: PerformanceRunRow[];
  track_counts?: {
    track1: number;
    track2: number;
    /**
     * Total distinct scored questions, including legacy questions that
     * pre-date the Track 1/2 split (where ``q.track IS NULL``). Optional
     * for backward compatibility with older API versions.
     */
    total?: number;
  };
};

export type QuestionBundleResponse = {
  question: Record<string, unknown>;
  hs?:
    | {
        hs_run?: unknown;
        triage?: unknown;
        scenarios?: unknown[];
        scenario_ids?: unknown[];
        country_report?: unknown;
      }
    | null;
  forecast?:
    | {
        forecaster_run_id?: string | null;
        research?: unknown;
        ensemble_spd?: unknown[];
        raw_spd?: unknown[];
        scenario_writer?: unknown[];
        bucket_labels?: string[];
        bucket_centroids?: number[];
      }
    | null;
  context?:
    | {
        question_context?: unknown;
        resolutions?: unknown[];
        scores?: unknown[];
      }
    | null;
  llm_calls?:
    | {
        included?: boolean;
        transcripts_included?: boolean;
        rows?: unknown[];
        by_phase?: Record<string, unknown[]>;
      }
    | null;
};

// --- Sibyl (parallel deep-research track) ----------------------------------

export type SibylRun = {
  sibyl_run_id: string;
  hs_run_id?: string | null;
  as_of?: string | null;
  model?: string | null;
  k?: number | null;
  max_steps?: number | null;
  aggregation?: string | null;
  run_hard_cap_usd?: number | null;
  budget_capped?: boolean | null;
  run_cost_usd?: number | null;
  opus_cost_usd?: number | null;
  brave_cost_usd?: number | null;
  n_selected?: number | null;
  n_forecast?: number | null;
  n_skipped?: number | null;
  created_at?: string | null;
  config?: Record<string, unknown> | null;
};

export type SibylQuestionRow = {
  sibyl_run_id?: string;
  run_id?: string | null;
  question_id: string;
  iso3?: string | null;
  hazard_code?: string | null;
  metric?: string | null;
  status?: string | null;
  skip_reason?: string | null;
  as_of?: string | null;
  k?: number | null;
  aggregation?: string | null;
  volatility_score?: number | null;
  triage_score?: number | null;
  js_divergence_vs_standard?: number | null;
  js_divergence_inter_trial?: number | null;
  cost_usd?: number | null;
  opus_cost_usd?: number | null;
  brave_cost_usd?: number | null;
  pooled_quantiles?: Record<string, number> | null;
};

export type SibylSummaryResponse = {
  run: SibylRun | null;
  questions: SibylQuestionRow[];
};

export type SibylQuestionsResponse = {
  sibyl_run_id: string | null;
  rows: SibylQuestionRow[];
};

export type SibylTrialStep = {
  step: number;
  action: string;
  action_input: string;
  tool_ok?: boolean | null;
  belief?: {
    quantiles?: Record<string, number>;
    confidence?: string;
    evidence_higher?: string[];
    evidence_lower?: string[];
    open_questions?: string[];
    baserate_reconciliation?: string;
    step_rationale?: string;
  };
  repaired?: boolean;
};

export type SibylTrial = {
  trial_index: number;
  perspective?: string;
  quantiles?: Record<string, number> | null;
  confidence?: string;
  belief_trace?: SibylTrialStep[];
  evidence_higher?: string[];
  evidence_lower?: string[];
  source_urls?: string[];
  steps_used?: number;
  submitted?: boolean;
  cost?: { opus_usd?: number; brave_usd?: number; total_usd?: number };
  error?: string | null;
};

export type SibylQuestionDetailResponse = {
  record: SibylQuestionRow & {
    trials?: SibylTrial[] | null;
    bucket_probs?: number[] | null;
    leakage?: Record<string, unknown> | null;
  };
  question?: Record<string, unknown> | null;
  bucket_labels: string[];
  standard_spd: {
    model_name: string | null;
    by_month: Record<string, number[]>;
  };
};

export type SibylRunsResponse = {
  rows: SibylRun[];
};
