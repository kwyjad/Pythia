"use client";

import { useEffect, useMemo, useState } from "react";

import InfoTooltip from "../../components/InfoTooltip";
import KpiCard from "../../components/KpiCard";
import SortableTable, { SortableColumn } from "../../components/SortableTable";
import { apiGet } from "../../lib/api";
import { formatModelName } from "../../lib/model_names";
import type {
  PerformanceRunRow,
  PerformanceScoresResponse,
  PerformanceSummaryRow,
  ResolutionRateRow,
  ResolutionRatesResponse,
  ScoreFamily,
  SibylComparisonResponse,
} from "../../lib/types";
import AboutScores from "./AboutScores";
import SibylComparison from "./SibylComparison";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

// Top-level page tabs: the live scoring dashboard vs. the scoring-methodology
// reference. Kept distinct from `ViewMode` (the Detailed Scores sub-tabs).
type PageTab = "dashboard" | "about";

type ViewMode = "total" | "by_hazard" | "by_run" | "by_model";

type PerformancePanelProps = {
  initialData: PerformanceScoresResponse;
  initialSibyl?: SibylComparisonResponse;
  includeTest?: boolean;
};

// Aggregated row used by Total, By Hazard, and By Model views after pivoting.
// `avg_brier`/`n_questions`/`n_samples` are the SPD family (multiclass, 0-2);
// the `*_binary` fields are the EVENT_OCCURRENCE family (0-1). The two families
// are never blended into a single average.
type PivotedRow = {
  key: string;
  label: string;
  avg_brier: number | null;
  median_brier: number | null;
  avg_brier_binary: number | null;
  median_brier_binary: number | null;
  avg_log: number | null;
  median_log: number | null;
  avg_crps: number | null;
  median_crps: number | null;
  n_questions: number;
  n_samples: number;
  n_questions_binary: number;
  n_samples_binary: number;
};

// Aggregated row used by the By Run view after pivoting.
type RunPivotedRow = {
  key: string;
  forecaster_run_id: string | null;
  hs_run_id: string;
  run_date: string | null;
  avg_brier: number | null;
  median_brier: number | null;
  avg_brier_binary: number | null;
  median_brier_binary: number | null;
  avg_log: number | null;
  median_log: number | null;
  avg_crps: number | null;
  median_crps: number | null;
  n_questions: number;
  n_samples: number;
  n_questions_binary: number;
  n_samples_binary: number;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const VIEW_LABELS: { key: ViewMode; label: string }[] = [
  { key: "total", label: "Total" },
  { key: "by_hazard", label: "By Hazard" },
  { key: "by_run", label: "By Run" },
  { key: "by_model", label: "By Model" },
];

// Shared tooltip texts (consistent across all views)
const TOOLTIP_BRIER_SPD =
  "Multiclass Brier score for SPD questions (PA, FATALITIES, PHASE3PLUS_IN_NEED). " +
  "Lower is better. Range: 0 (perfect) to 2. Not comparable to Binary Brier.";
const TOOLTIP_BRIER_BINARY =
  "Brier score for binary EVENT_OCCURRENCE questions: (forecast_p − outcome)². " +
  "Lower is better. Range: 0 (perfect) to 1. Structurally near-zero for rare events " +
  "correctly forecast as unlikely. Not comparable to SPD Brier — the two families " +
  "are never averaged together.";
const TOOLTIP_LOG =
  "Logarithmic scoring rule. Lower is better. Heavily penalises confident wrong predictions. Range: 0 (perfect) to +\u221E.";
const TOOLTIP_CRPS =
  "Normalized Ranked Probability Score (the discrete form of CRPS): squared cumulative-distribution error across the ordered buckets, divided by K\u22121. Lower is better. Range: 0 (perfect) to 1 (all probability in the bucket farthest from the outcome).";
const TOOLTIP_SAMPLES =
  "Number of individual scored data points (question \u00D7 horizon combinations). Each question can produce multiple samples across different forecast horizons.";
const TOOLTIP_EXTERNAL_BENCHMARK =
  "ViEWS (Violence & Impacts Early-Warning System) is an external conflict fatality " +
  "forecasting model from Uppsala University that produces monthly point forecasts " +
  "(expected fatality counts) for every country at lead times of 1\u20136 months \u2014 the " +
  "same horizons Fred uses.\n\n" +
  "Scoring method: ViEWS point forecasts are converted into synthetic 7-bucket " +
  "probability distributions using a log-normal model. Given a point forecast of X " +
  "fatalities, we construct a log-normal distribution with E[X] = point forecast and " +
  "integrate over Fred\u2019s fatality bucket boundaries (0, 1\u20134, 5\u201324, 25\u201399, 100\u2013499, 500\u2013999, \u22651000) " +
  "to get bucket probabilities. These synthetic SPDs are then scored with the same " +
  "Brier, Log Loss, and CRPS functions used for Fred\u2019s own models.\n\n" +
  "The log-normal spread parameter (\u03C3) controls how uncertain the synthetic SPD is " +
  "around the point forecast. It is periodically optimized to minimize ViEWS\u2019s Brier " +
  "score against resolved outcomes.\n\n" +
  "Purpose: This lets us directly compare ViEWS accuracy against Fred\u2019s ensemble and " +
  "individual models on the same scale. The comparison also feeds back into Fred\u2019s " +
  "calibration advice \u2014 if ViEWS outperforms the ensemble, forecasting models are " +
  "told to anchor more strongly on ViEWS conflict forecasts when they are provided as input.";

/** Check if a model_name represents an ensemble method. */
const isEnsembleModel = (name: string | null): boolean => {
  if (name == null) return true;
  const lower = name.toLowerCase();
  return lower.startsWith("ensemble");
};

/** Check if a model_name represents an external benchmark (not a Pythia model). */
const isExternalBenchmark = (name: string | null): boolean => {
  if (name == null) return false;
  return name.startsWith("__ext_");
};

/** Pick the preferred ensemble model name from available model names. */
const pickDefaultEnsemble = (models: string[]): string | null => {
  const bayesmc = models.find((m) =>
    m.toLowerCase().includes("ensemble_bayesmc"),
  );
  if (bayesmc) return bayesmc;
  const anyEnsemble = models.find((m) =>
    m.toLowerCase().startsWith("ensemble"),
  );
  if (anyEnsemble) return anyEnsemble;
  return models[0] ?? null;
};

/** Friendly display name for a model, including external benchmarks. */
const displayModelName = (name: string | null): string => {
  if (name == null) return "Ensemble";
  if (name === "__ext_views") return "ViEWS (external benchmark)";
  if (name.startsWith("__ext_")) {
    // Future external benchmarks: strip prefix and capitalize
    return name.replace("__ext_", "").replace(/_/g, " ") + " (external)";
  }
  // Delegates to the shared prettifier, which maps sibyl + the ensemble/track2
  // aggregate slugs to friendly labels (and base model ids to product names).
  return formatModelName(name);
};

const formatScore = (value: number | null | undefined) => {
  if (value == null) return "\u2014";
  return value.toLocaleString(undefined, {
    minimumFractionDigits: 4,
    maximumFractionDigits: 4,
  });
};

const formatInt = (value: number | null | undefined) => {
  if (value == null) return "\u2014";
  return value.toLocaleString();
};

/**
 * Resolve a row's Brier family. Prefers the API-provided `score_family`, but
 * falls back to deriving it from the metric so the UI stays correct even if the
 * frontend is deployed ahead of the API that adds the field.
 */
const familyOf = (row: {
  score_family?: ScoreFamily;
  metric?: string;
}): ScoreFamily =>
  row.score_family ??
  ((row.metric ?? "").toUpperCase() === "EVENT_OCCURRENCE" ? "binary" : "spd");

/**
 * Compute weighted average + weighted median of `scoreType` across rows.
 * When `family` is provided, only rows of that Brier family are included, so a
 * single call never blends binary (0-1) and multiclass SPD (0-2) Brier scales.
 */
function aggregateScore<
  T extends {
    score_type: string;
    score_family?: ScoreFamily;
    metric?: string;
    avg_value: number | null;
    median_value: number | null;
    n_samples: number;
    n_questions: number;
  },
>(
  rows: T[],
  scoreType: string,
  family?: ScoreFamily,
): {
  avg: number | null;
  median: number | null;
  nQuestions: number;
  nSamples: number;
} {
  const matching = rows.filter(
    (r) =>
      r.score_type === scoreType &&
      (family === undefined || familyOf(r) === family),
  );
  let totalWeight = 0;
  let weightedSumAvg = 0;
  let weightedSumMedian = 0;
  let medianWeight = 0;
  let nQuestions = 0;
  let nSamples = 0;
  for (const r of matching) {
    if (r.avg_value != null && r.n_samples > 0) {
      weightedSumAvg += r.avg_value * r.n_samples;
      totalWeight += r.n_samples;
    }
    if (r.median_value != null && r.n_samples > 0) {
      weightedSumMedian += r.median_value * r.n_samples;
      medianWeight += r.n_samples;
    }
    nQuestions += r.n_questions;
    nSamples += r.n_samples;
  }
  return {
    avg: totalWeight > 0 ? weightedSumAvg / totalWeight : null,
    median: medianWeight > 0 ? weightedSumMedian / medianWeight : null,
    nQuestions,
    nSamples,
  };
}

// ---------------------------------------------------------------------------
// Column definitions
// ---------------------------------------------------------------------------

const PIVOTED_COLUMNS: Array<SortableColumn<PivotedRow>> = [
  {
    key: "label",
    label: "Name",
    sortValue: (row) => row.label,
    defaultSortDirection: "asc",
    render: (row) => {
      const isExternal = row.key.startsWith("__ext_");
      const isSibyl = row.key === "sibyl";
      return (
        <span className={`font-medium ${isExternal ? "text-amber-400" : "text-fred-text"}`}>
          {row.label}
          {isExternal ? (
            <InfoTooltip text={TOOLTIP_EXTERNAL_BENCHMARK} />
          ) : null}
          {isSibyl ? (
            <span
              className="ml-2 rounded-sm px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-white"
              style={{ backgroundColor: "#4a3aa7" }}
            >
              deep research
            </span>
          ) : null}
        </span>
      );
    },
  },
  {
    key: "avg_brier",
    label: (
      <span className="inline-flex items-center gap-1">
        SPD Brier <InfoTooltip text={TOOLTIP_BRIER_SPD} />
      </span>
    ),
    sortValue: (row) => row.avg_brier,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.avg_brier),
  },
  {
    key: "median_brier",
    label: "Mdn SPD Brier",
    sortValue: (row) => row.median_brier,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.median_brier),
  },
  {
    key: "avg_brier_binary",
    label: (
      <span className="inline-flex items-center gap-1">
        Binary Brier <InfoTooltip text={TOOLTIP_BRIER_BINARY} />
      </span>
    ),
    sortValue: (row) => row.avg_brier_binary,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.avg_brier_binary),
  },
  {
    key: "avg_log",
    label: (
      <span className="inline-flex items-center gap-1">
        Avg Log Loss <InfoTooltip text={TOOLTIP_LOG} />
      </span>
    ),
    sortValue: (row) => row.avg_log,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.avg_log),
  },
  {
    key: "median_log",
    label: "Mdn Log Loss",
    sortValue: (row) => row.median_log,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.median_log),
  },
  {
    key: "avg_crps",
    label: (
      <span className="inline-flex items-center gap-1">
        Avg CRPS <InfoTooltip text={TOOLTIP_CRPS} />
      </span>
    ),
    sortValue: (row) => row.avg_crps,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.avg_crps),
  },
  {
    key: "median_crps",
    label: "Mdn CRPS",
    sortValue: (row) => row.median_crps,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.median_crps),
  },
  {
    key: "n_questions",
    label: "SPD Q",
    sortValue: (row) => row.n_questions,
    defaultSortDirection: "desc",
    render: (row) => formatInt(row.n_questions),
  },
  {
    key: "n_questions_binary",
    label: "Binary Q",
    sortValue: (row) => row.n_questions_binary,
    defaultSortDirection: "desc",
    render: (row) => formatInt(row.n_questions_binary),
  },
  {
    key: "n_samples",
    label: (
      <span className="inline-flex items-center gap-1">
        Samples <InfoTooltip text={TOOLTIP_SAMPLES} />
      </span>
    ),
    sortValue: (row) => row.n_samples,
    defaultSortDirection: "desc",
    render: (row) => formatInt(row.n_samples),
  },
];

const RUN_COLUMNS: Array<SortableColumn<RunPivotedRow>> = [
  {
    key: "run_date",
    label: "Run Date",
    sortValue: (row) => row.run_date ?? "",
    defaultSortDirection: "desc",
    render: (row) => (
      <span className="font-medium text-fred-text">
        {row.run_date ?? row.hs_run_id}
      </span>
    ),
  },
  {
    key: "avg_brier",
    label: (
      <span className="inline-flex items-center gap-1">
        SPD Brier <InfoTooltip text={TOOLTIP_BRIER_SPD} />
      </span>
    ),
    sortValue: (row) => row.avg_brier,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.avg_brier),
  },
  {
    key: "median_brier",
    label: "Mdn SPD Brier",
    sortValue: (row) => row.median_brier,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.median_brier),
  },
  {
    key: "avg_brier_binary",
    label: (
      <span className="inline-flex items-center gap-1">
        Binary Brier <InfoTooltip text={TOOLTIP_BRIER_BINARY} />
      </span>
    ),
    sortValue: (row) => row.avg_brier_binary,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.avg_brier_binary),
  },
  {
    key: "avg_log",
    label: (
      <span className="inline-flex items-center gap-1">
        Avg Log Loss <InfoTooltip text={TOOLTIP_LOG} />
      </span>
    ),
    sortValue: (row) => row.avg_log,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.avg_log),
  },
  {
    key: "median_log",
    label: "Mdn Log Loss",
    sortValue: (row) => row.median_log,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.median_log),
  },
  {
    key: "avg_crps",
    label: (
      <span className="inline-flex items-center gap-1">
        Avg CRPS <InfoTooltip text={TOOLTIP_CRPS} />
      </span>
    ),
    sortValue: (row) => row.avg_crps,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.avg_crps),
  },
  {
    key: "median_crps",
    label: "Mdn CRPS",
    sortValue: (row) => row.median_crps,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.median_crps),
  },
  {
    key: "n_questions",
    label: "SPD Q",
    sortValue: (row) => row.n_questions,
    defaultSortDirection: "desc",
    render: (row) => formatInt(row.n_questions),
  },
  {
    key: "n_questions_binary",
    label: "Binary Q",
    sortValue: (row) => row.n_questions_binary,
    defaultSortDirection: "desc",
    render: (row) => formatInt(row.n_questions_binary),
  },
  {
    key: "n_samples",
    label: (
      <span className="inline-flex items-center gap-1">
        Samples <InfoTooltip text={TOOLTIP_SAMPLES} />
      </span>
    ),
    sortValue: (row) => row.n_samples,
    defaultSortDirection: "desc",
    render: (row) => formatInt(row.n_samples),
  },
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function PerformancePanel({
  initialData,
  initialSibyl,
  includeTest,
}: PerformancePanelProps) {
  const [pageTab, setPageTab] = useState<PageTab>("dashboard");
  const [view, setView] = useState<ViewMode>("total");
  const [metric, setMetric] = useState<string | null>(null);
  const [track, setTrack] = useState<number | null>(null);
  const [data, setData] = useState<PerformanceScoresResponse>(initialData);
  const [loading, setLoading] = useState(false);
  const [resolutionRates, setResolutionRates] = useState<ResolutionRateRow[]>(
    [],
  );

  // Fetch resolution rates on mount
  useEffect(() => {
    apiGet<ResolutionRatesResponse>("/diagnostics/resolution_rates", {
      include_test: includeTest || undefined,
    })
      .then((res) => setResolutionRates(res.rows ?? []))
      .catch(() => {});
  }, [includeTest]);

  // ---- Detect available ensemble models ------------------------------------

  const ensembleModels = useMemo(() => {
    const names = new Set<string>();
    for (const row of data.summary_rows) {
      if (row.model_name && isEnsembleModel(row.model_name)) {
        names.add(row.model_name);
      }
    }
    return Array.from(names).sort();
  }, [data]);

  const [selectedEnsemble, setSelectedEnsemble] = useState<string | null>(
    null,
  );

  const activeEnsemble = useMemo(() => {
    if (selectedEnsemble && ensembleModels.includes(selectedEnsemble)) {
      return selectedEnsemble;
    }
    return pickDefaultEnsemble(ensembleModels);
  }, [selectedEnsemble, ensembleModels]);

  // ---- Fetch when metric changes -------------------------------------------

  const refetchScores = async (
    newMetric: string | null,
    newTrack: number | null,
  ) => {
    setLoading(true);
    try {
      const params: Record<string, string | boolean | undefined> = {};
      if (newMetric) params.metric = newMetric;
      if (newTrack != null) params.track = String(newTrack);
      params.include_test = includeTest || undefined;
      const response = await apiGet<PerformanceScoresResponse>(
        "/performance/scores",
        params,
      );
      setData(response);
    } catch (error) {
      console.warn("Failed to fetch performance scores:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleMetricChange = async (newMetric: string | null) => {
    setMetric(newMetric);
    await refetchScores(newMetric, track);
  };

  const handleTrackChange = async (newTrack: number | null) => {
    setTrack(newTrack);
    await refetchScores(metric, newTrack);
  };

  // ---- KPI cards -----------------------------------------------------------

  const kpiStats = useMemo(() => {
    const ensembleRows = activeEnsemble
      ? data.summary_rows.filter(
          (r) => r.model_name === activeEnsemble && !isExternalBenchmark(r.model_name),
        )
      : data.summary_rows.filter(
          (r) => isEnsembleModel(r.model_name) && !isExternalBenchmark(r.model_name),
        );
    const allHazards = new Set(data.summary_rows.map((r) => r.hazard_code));
    const allModels = new Set(
      data.summary_rows.map((r) => r.model_name).filter(Boolean),
    );
    const allRuns = new Set(data.run_rows.map((r) => r.forecaster_run_id ?? r.hs_run_id));
    const brierSpd = aggregateScore(ensembleRows, "brier", "spd");
    const brierBinary = aggregateScore(ensembleRows, "brier", "binary");
    return {
      hazards: allHazards.size,
      models: allModels.size,
      runs: allRuns.size,
      totalQuestions: brierSpd.nQuestions,
      totalQuestionsBinary: brierBinary.nQuestions,
      overallBrierSpd: brierSpd.avg,
      overallBrierBinary: brierBinary.avg,
    };
  }, [data, activeEnsemble]);

  // ---- Helper to build pivoted rows from summary data ----------------------

  function buildPivoted(
    rows: PerformanceSummaryRow[],
    groupBy: (r: PerformanceSummaryRow) => string,
    labelFor: (key: string) => string,
  ): PivotedRow[] {
    const groups = new Map<string, PerformanceSummaryRow[]>();
    for (const row of rows) {
      const gk = groupBy(row);
      const existing = groups.get(gk) ?? [];
      existing.push(row);
      groups.set(gk, existing);
    }
    const result: PivotedRow[] = [];
    for (const [gk, gRows] of groups) {
      const brierSpd = aggregateScore(gRows, "brier", "spd");
      const brierBinary = aggregateScore(gRows, "brier", "binary");
      const log = aggregateScore(gRows, "log", "spd");
      const crps = aggregateScore(gRows, "crps", "spd");
      result.push({
        key: gk,
        label: labelFor(gk),
        avg_brier: brierSpd.avg,
        median_brier: brierSpd.median,
        avg_brier_binary: brierBinary.avg,
        median_brier_binary: brierBinary.median,
        avg_log: log.avg,
        median_log: log.median,
        avg_crps: crps.avg,
        median_crps: crps.median,
        n_questions: brierSpd.nQuestions,
        n_samples: brierSpd.nSamples,
        n_questions_binary: brierBinary.nQuestions,
        n_samples_binary: brierBinary.nSamples,
      });
    }
    return result;
  }

  // ---- Total view ----------------------------------------------------------

  const totalRows = useMemo(
    (): PivotedRow[] =>
      buildPivoted(
        data.summary_rows,
        (r) => r.model_name ?? "__null__",
        (k) => displayModelName(k === "__null__" ? null : k),
      ),
    [data],
  );

  // ---- By Hazard view ------------------------------------------------------

  const hazardRows = useMemo((): PivotedRow[] => {
    const filtered = activeEnsemble
      ? data.summary_rows.filter((r) => r.model_name === activeEnsemble)
      : data.summary_rows;
    return buildPivoted(
      filtered,
      (r) => r.hazard_code,
      (k) => k,
    );
  }, [data, activeEnsemble]);

  // ---- By Run view ---------------------------------------------------------

  const runRows = useMemo((): RunPivotedRow[] => {
    const filtered = activeEnsemble
      ? data.run_rows.filter((r) => r.model_name === activeEnsemble)
      : data.run_rows;
    const groups = new Map<string, PerformanceRunRow[]>();
    for (const row of filtered) {
      // Group by forecaster_run_id (the run that produced forecasts), falling
      // back to hs_run_id for backward compat / external benchmark scores.
      const groupKey = row.forecaster_run_id ?? row.hs_run_id;
      const existing = groups.get(groupKey) ?? [];
      existing.push(row);
      groups.set(groupKey, existing);
    }
    const result: RunPivotedRow[] = [];
    for (const [runId, gRows] of groups) {
      const brierSpd = aggregateScore(gRows, "brier", "spd");
      const brierBinary = aggregateScore(gRows, "brier", "binary");
      const log = aggregateScore(gRows, "log", "spd");
      const crps = aggregateScore(gRows, "crps", "spd");
      const runDate = gRows[0]?.run_date ?? null;
      const forecasterRunId = gRows[0]?.forecaster_run_id ?? null;
      const hsRunId = gRows[0]?.hs_run_id ?? runId;
      result.push({
        key: runId,
        forecaster_run_id: forecasterRunId,
        hs_run_id: hsRunId,
        run_date: runDate,
        avg_brier: brierSpd.avg,
        median_brier: brierSpd.median,
        avg_brier_binary: brierBinary.avg,
        median_brier_binary: brierBinary.median,
        avg_log: log.avg,
        median_log: log.median,
        avg_crps: crps.avg,
        median_crps: crps.median,
        n_questions: brierSpd.nQuestions,
        n_samples: brierSpd.nSamples,
        n_questions_binary: brierBinary.nQuestions,
        n_samples_binary: brierBinary.nSamples,
      });
    }
    return result;
  }, [data, activeEnsemble]);

  // ---- By Model view -------------------------------------------------------

  const modelRows = useMemo((): PivotedRow[] => {
    const pivoted = buildPivoted(
      data.summary_rows,
      (r) => r.model_name ?? "__null__",
      (k) => displayModelName(k === "__null__" ? null : k),
    );
    // Sort: ensembles first, then regular models, then external benchmarks last
    return pivoted.sort((a, b) => {
      const aExt = a.key.startsWith("__ext_") ? 2 : isEnsembleModel(a.key === "__null__" ? null : a.key) ? 0 : 1;
      const bExt = b.key.startsWith("__ext_") ? 2 : isEnsembleModel(b.key === "__null__" ? null : b.key) ? 0 : 1;
      if (aExt !== bExt) return aExt - bExt;
      return a.label.localeCompare(b.label);
    });
  }, [data]);

  // ---- Render --------------------------------------------------------------

  const isEmpty =
    data.summary_rows.length === 0 && data.run_rows.length === 0;

  // Defensive filter: drop blocked hazards (DI, HW, CU, ACO) from the
  // coverage panel even if the API hasn't been updated yet. We don't forecast
  // these anymore, so their tiles would always sit at 0% and create the
  // impression of broken calibration.
  const BLOCKED_HAZARDS = new Set(["DI", "HW", "CU", "ACO"]);
  const visibleResolutionRates = resolutionRates.filter(
    (row) => !BLOCKED_HAZARDS.has(row.hazard_code.toUpperCase()),
  );

  // Summary across all coverage rows: how many questions are eligible
  // for resolution (cutoff has passed), how many of those are scored, how
  // many are pending (too new). Pre-Track questions are included here via
  // the questions table — this is the count the user is looking for.
  const coverageTotals = useMemo(() => {
    let total = 0;
    let resolved = 0;
    let pending = 0;
    for (const row of visibleResolutionRates) {
      total += row.total_questions;
      resolved += row.resolved_questions;
      pending += row.pending_too_new ?? 0;
    }
    const eligible = Math.max(0, total - pending);
    return { total, resolved, pending, eligible };
  }, [visibleResolutionRates]);

  const totalScored =
    data.track_counts?.total ??
    (data.track_counts
      ? (data.track_counts.track1 ?? 0) + (data.track_counts.track2 ?? 0)
      : 0);

  return (
    <div className="space-y-6">
      {/* Top-level page tabs: live dashboard vs. scoring methodology */}
      <div className="flex gap-1 rounded-lg border border-fred-secondary bg-fred-bg/40 p-1">
        {(
          [
            { key: "dashboard", label: "Dashboard" },
            { key: "about", label: "About performance scores" },
          ] as { key: PageTab; label: string }[]
        ).map(({ key, label }) => (
          <button
            key={key}
            type="button"
            className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
              pageTab === key
                ? "bg-fred-primary text-fred-bg"
                : "text-fred-text hover:bg-fred-bg/60"
            }`}
            onClick={() => setPageTab(key)}
          >
            {label}
          </button>
        ))}
      </div>

      {pageTab === "about" ? <AboutScores /> : null}

      {pageTab === "dashboard" ? (
        <>
      {/* Summary KPI cards */}
      <section className="rounded-lg border border-fred-secondary bg-fred-surface/50 p-4">
        <div className="mb-3 flex items-baseline justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-fred-text">
            Summary
          </h2>
          <span className="text-xs text-fred-muted">
            Counts cover every question in the database, including legacy
            questions before the Track 1/2 split.
          </span>
        </div>
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
          <KpiCard
            label={
              <span className="inline-flex items-center gap-1">
                Resolved Questions{" "}
                <InfoTooltip
                  text={
                    "Questions with at least one resolved horizon (ground " +
                    "truth from the Resolver). Excludes blocked hazards " +
                    "(DI/HW/CU/ACO). Pending questions whose earliest " +
                    "horizon is in the future are not counted here."
                  }
                />
              </span>
            }
            value={`${formatInt(coverageTotals.resolved)} / ${formatInt(coverageTotals.eligible)}`}
          />
          <KpiCard
            label={
              <span className="inline-flex items-center gap-1">
                Scored Questions{" "}
                <InfoTooltip
                  text={
                    "Distinct questions with at least one row in the scores " +
                    "table. Includes legacy pre-Track questions that don't " +
                    "carry a track tag."
                  }
                />
              </span>
            }
            value={formatInt(totalScored)}
          />
          <KpiCard
            label={
              <span className="inline-flex items-center gap-1">
                Track Split{" "}
                <InfoTooltip
                  text={
                    "Breakdown of scored questions by track. Track 1 = full " +
                    "ensemble; Track 2 = a single fast model (currently " +
                    "gemini-3.5-flash). Older questions predate this split " +
                    "and are counted only in Scored Questions above."
                  }
                />
              </span>
            }
            value={`T1 ${formatInt(data.track_counts?.track1 ?? 0)} / T2 ${formatInt(data.track_counts?.track2 ?? 0)}`}
          />
          <KpiCard label="Hazards" value={formatInt(kpiStats.hazards)} />
          <KpiCard label="Models" value={formatInt(kpiStats.models)} />
          <KpiCard
            label={
              <span className="inline-flex items-center gap-1">
                Ensemble Brier{" "}
                <InfoTooltip
                  text={
                    "Average ensemble Brier score, split by family because the " +
                    "two scales are not comparable. SPD = multiclass " +
                    "(PA/FATALITIES/PHASE3PLUS_IN_NEED, range 0-2). Binary = " +
                    "EVENT_OCCURRENCE (range 0-1, near-zero for rare events " +
                    "correctly forecast as unlikely). Lower is better."
                  }
                />
              </span>
            }
            value={
              <span className="inline-flex flex-col leading-tight">
                <span>SPD {formatScore(kpiStats.overallBrierSpd)}</span>
                <span className="text-fred-muted">
                  Binary {formatScore(kpiStats.overallBrierBinary)}
                </span>
              </span>
            }
          />
        </div>
      </section>

      {/* Resolution Coverage Summary */}
      {visibleResolutionRates.length > 0 && (
        <section className="rounded-lg border border-fred-secondary bg-fred-surface p-4">
          <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-fred-text">
              Resolution Coverage
            </h2>
            <span className="text-xs text-fred-muted">
              Eligible {formatInt(coverageTotals.eligible)} · Resolved{" "}
              {formatInt(coverageTotals.resolved)} · Pending (too new){" "}
              {formatInt(coverageTotals.pending)}
            </span>
          </div>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {visibleResolutionRates.map((row) => {
              const pending = row.pending_too_new ?? 0;
              const eligible = Math.max(
                0,
                row.total_questions - pending,
              );
              // Eligible rate = resolved / (total - pending). This is the
              // honest accuracy number — denominator excludes questions that
              // can't possibly be resolved yet.
              const eligibleRate =
                eligible > 0 ? row.resolved_questions / eligible : null;
              const isAllPending =
                row.total_questions > 0 && eligible === 0 && pending > 0;
              const colorClass = isAllPending
                ? "text-fred-muted"
                : eligibleRate == null
                  ? "text-fred-muted"
                  : eligibleRate >= 0.9
                    ? "text-green-500"
                    : eligibleRate >= 0.5
                      ? "text-yellow-500"
                      : "text-red-500";
              const metricLabel =
                row.metric === "EVENT_OCCURRENCE"
                  ? "Binary (event)"
                  : row.metric === "PHASE3PLUS_IN_NEED"
                    ? "Phase 3+"
                    : row.metric;
              const pctLabel =
                eligibleRate == null
                  ? "—"
                  : `${(eligibleRate * 100).toFixed(1)}% scored`;
              return (
                <div
                  key={`${row.hazard_code}-${row.metric}`}
                  className="rounded border border-fred-secondary bg-fred-surface px-3 py-2"
                  title={
                    isAllPending
                      ? `${pending} questions in this group are from the latest forecast epoch — their earliest horizon hasn't reached the resolution calendar cutoff yet.`
                      : pending > 0
                        ? `${pending} additional question(s) too new to resolve yet.`
                        : undefined
                  }
                >
                  <div className="text-[11px] font-semibold uppercase tracking-wide text-fred-muted">
                    {row.hazard_code} / {metricLabel}
                  </div>
                  <div className={`mt-1 text-sm font-medium ${colorClass}`}>
                    {isAllPending ? "Awaiting first horizon" : pctLabel}
                  </div>
                  <div className="text-xs text-fred-muted">
                    {isAllPending ? (
                      <>
                        0 of 0 eligible — {pending} too new, will resolve next
                        month
                      </>
                    ) : (
                      <>
                        {row.resolved_questions} of {eligible} eligible
                        {pending > 0 ? ` · ${pending} too new` : null}
                      </>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* Sibyl deep-research track: understand it + head-to-head vs the ensemble */}
      {initialSibyl ? (
        <SibylComparison data={initialSibyl} includeTest={includeTest} />
      ) : null}

      {/* Detailed Scores: filters + table grouped together */}
      <section className="rounded-lg border border-fred-secondary bg-fred-surface p-4">
        <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-fred-text">
            Detailed Scores
          </h2>
          {loading ? (
            <span className="text-xs text-fred-muted">Loading…</span>
          ) : null}
        </div>

        {/* Controls — directly above the table they control */}
        <div className="mb-3 flex flex-wrap items-center gap-3">
          {/* View selector tabs */}
          <div className="flex gap-1 rounded-lg border border-fred-secondary bg-fred-bg/40 p-1">
            {VIEW_LABELS.map(({ key, label }) => (
              <button
                key={key}
                type="button"
                className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                  view === key
                    ? "bg-fred-primary text-fred-bg"
                    : "text-fred-text hover:bg-fred-bg/60"
                }`}
                onClick={() => setView(key)}
              >
                {label}
              </button>
            ))}
          </div>

          {/* Metric filter */}
          <select
            className="rounded-md border border-fred-secondary bg-fred-bg/40 px-3 py-1.5 text-sm text-fred-text"
            value={metric ?? ""}
            onChange={(e) => handleMetricChange(e.target.value || null)}
          >
            <option value="">All Metrics</option>
            <option value="PA">People Affected (PA)</option>
            <option value="FATALITIES">Fatalities</option>
            <option value="EVENT_OCCURRENCE">Event Occurrence (binary)</option>
            <option value="PHASE3PLUS_IN_NEED">Phase 3+ (IPC)</option>
          </select>

          {/* Track filter */}
          <select
            className="rounded-md border border-fred-secondary bg-fred-bg/40 px-3 py-1.5 text-sm text-fred-text"
            value={track != null ? String(track) : ""}
            onChange={(e) =>
              handleTrackChange(e.target.value ? Number(e.target.value) : null)
            }
          >
            <option value="">All Tracks</option>
            <option value="1">Track 1</option>
            <option value="2">Track 2</option>
          </select>

          {/* Ensemble selector (shown for By Hazard and By Run views) */}
          {(view === "by_hazard" || view === "by_run") &&
          ensembleModels.length > 0 ? (
            <select
              className="rounded-md border border-fred-secondary bg-fred-bg/40 px-3 py-1.5 text-sm text-fred-text"
              value={activeEnsemble ?? ""}
              onChange={(e) => setSelectedEnsemble(e.target.value || null)}
            >
              {ensembleModels.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          ) : null}
        </div>

        {/* Table */}
        {isEmpty ? (
          <div className="rounded-lg border border-slate-800 bg-slate-900/40 px-4 py-6 text-center text-slate-300">
            No scoring data available yet. Scores are generated once forecasts
            have been resolved against ground truth.
          </div>
        ) : (
          <div className="overflow-x-auto rounded-lg border border-fred-secondary">
            {view === "total" && (
              <SortableTable
                columns={PIVOTED_COLUMNS}
                rows={totalRows}
                rowKey={(row) => row.key}
                initialSortKey="avg_brier"
                initialSortDirection="asc"
                tableLayout="auto"
                emptyMessage="No scores available."
                rowClassName={(row) =>
                  row.key.startsWith("__ext_")
                    ? "bg-amber-500/5 border-l-2 border-l-amber-400/40"
                    : undefined
                }
              />
            )}

            {view === "by_hazard" && (
              <SortableTable
                columns={PIVOTED_COLUMNS}
                rows={hazardRows}
                rowKey={(row) => row.key}
                initialSortKey="avg_brier"
                initialSortDirection="asc"
                tableLayout="auto"
                emptyMessage="No hazard-level scores available."
              />
            )}

            {view === "by_run" && (
              <SortableTable
                columns={RUN_COLUMNS}
                rows={runRows}
                rowKey={(row) => row.key}
                initialSortKey="run_date"
                initialSortDirection="desc"
                tableLayout="auto"
                emptyMessage="No per-run scores available."
              />
            )}

            {view === "by_model" && (
              <SortableTable
                columns={PIVOTED_COLUMNS}
                rows={modelRows}
                rowKey={(row) => row.key}
                initialSortKey="avg_brier"
                initialSortDirection="asc"
                tableLayout="auto"
                emptyMessage="No per-model scores available."
                rowClassName={(row) =>
                  row.key.startsWith("__ext_")
                    ? "bg-amber-500/5 border-l-2 border-l-amber-400/40"
                    : undefined
                }
              />
            )}
          </div>
        )}
      </section>
        </>
      ) : null}
    </div>
  );
}
