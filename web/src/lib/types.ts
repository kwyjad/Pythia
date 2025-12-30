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

export type RiskView = "PA_EIV" | "PA_PC" | "FATALITIES_EIV" | "FATALITIES_PC";

export type CountriesRow = {
  iso3: string;
  n_questions: number;
  n_forecasted: number;
  country_name?: string | null;
  last_triaged?: string | null;
  last_forecasted?: string | null;
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
    status?: string | null;
    wording?: string | null;
  }>;
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
