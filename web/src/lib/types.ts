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
  metric: "PA" | "FATALITIES" | string;
  target_month: string | null;
  horizon_m?: number | null;
  normalize?: boolean | null;
  rows?: RiskIndexRow[];
};

export type RiskView = "PA_EIV" | "PA_PC" | "FATALITIES_EIV";

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
