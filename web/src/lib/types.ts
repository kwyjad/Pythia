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

export type RiskIndexResponse = {
  metric: string;
  target_month: string | null;
  rows?: Array<{
    iso3: string;
    horizon_m?: number | null;
    expected_value?: number | null;
    per_capita?: number | null;
  }>;
};

export type QuestionsResponse = {
  rows: Array<{
    question_id: string;
    hs_run_id?: string | null;
    iso3: string;
    hazard_code: string;
    metric: string;
    target_month: string;
    status?: string | null;
    wording?: string | null;
  }>;
};

export type QuestionBundleResponse = {
  question: Record<string, unknown>;
  hs?: Record<string, unknown> | null;
  forecast?: Record<string, unknown> | null;
  context?: Record<string, unknown> | null;
  llm_calls?: Record<string, unknown> | null;
};
