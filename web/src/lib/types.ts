export type VersionResponse = {
  latest_hs_run_id?: string | null;
  latest_hs_created_at?: string | null;
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

export type DiagnosticsKpiScopesResponse = {
  available_months: Array<{
    year_month: string;
    label: string;
    is_latest: boolean;
  }>;
  selected_month: string | null;
  scopes: Record<string, DiagnosticsKpiScope>;
  explanations?: string[] | null;
  diagnostics?: Record<string, unknown> | null;
  notes?: string[] | null;
};

export type RiskIndexResponse = {
  metric: "PA" | "FATALITIES" | string;
  target_month: string | null;
  horizon_m?: number | null;
  normalize?: boolean | null;
  rows?: RiskIndexRow[];
};

export type RiskView = "PA_EIV" | "PA_PC" | "FATALITIES_EIV" | "FATALITIES_PC";

export type CountriesRow = {
  iso3: string;
  n_questions: number;
  n_forecasted: number;
  country_name?: string | null;
  last_triaged?: string | null;
  last_forecasted?: string | null;
  highest_rc_level?: number | null;
  highest_rc_score?: number | null;
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
};

export type QuestionsResponse = {
  rows: Array<{
    question_id: string;
    hs_run_id?: string | null;
    iso3: string;
    hazard_code: string;
    metric: string;
    target_month: string;
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

export type PerformanceSummaryRow = {
  hazard_code: string;
  metric: string;
  score_type: string;
  model_name: string | null;
  n_samples: number;
  n_questions: number;
  avg_value: number | null;
  median_value: number | null;
};

export type PerformanceRunRow = {
  hs_run_id: string;
  run_date: string | null;
  hazard_code: string;
  metric: string;
  score_type: string;
  model_name: string | null;
  n_samples: number;
  n_questions: number;
  avg_value: number | null;
  median_value: number | null;
};

export type PerformanceScoresResponse = {
  summary_rows: PerformanceSummaryRow[];
  run_rows: PerformanceRunRow[];
  track_counts?: {
    track1: number;
    track2: number;
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
