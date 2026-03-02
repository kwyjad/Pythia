"use client";

import { useMemo, useState } from "react";

import InfoTooltip from "../../components/InfoTooltip";
import KpiCard from "../../components/KpiCard";
import SortableTable, { SortableColumn } from "../../components/SortableTable";
import { apiGet } from "../../lib/api";
import type {
  PerformanceRunRow,
  PerformanceScoresResponse,
  PerformanceSummaryRow,
} from "../../lib/types";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type ViewMode = "total" | "by_hazard" | "by_run" | "by_model";

type PerformancePanelProps = {
  initialData: PerformanceScoresResponse;
};

// Aggregated row used by Total, By Hazard, and By Model views after pivoting.
type PivotedRow = {
  key: string;
  label: string;
  avg_brier: number | null;
  median_brier: number | null;
  avg_log: number | null;
  median_log: number | null;
  avg_crps: number | null;
  median_crps: number | null;
  n_questions: number;
  n_samples: number;
};

// Aggregated row used by the By Run view after pivoting.
type RunPivotedRow = {
  key: string;
  hs_run_id: string;
  run_date: string | null;
  avg_brier: number | null;
  median_brier: number | null;
  avg_log: number | null;
  median_log: number | null;
  avg_crps: number | null;
  median_crps: number | null;
  n_questions: number;
  n_samples: number;
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
const TOOLTIP_BRIER =
  "Brier score measures the accuracy of probabilistic predictions. Lower is better. Range: 0 (perfect) to 1.";
const TOOLTIP_LOG =
  "Logarithmic scoring rule. Lower is better. Heavily penalises confident wrong predictions. Range: 0 (perfect) to +\u221E.";
const TOOLTIP_CRPS =
  "Continuous Ranked Probability Score. Lower is better. Measures full distribution accuracy. Range: 0 (perfect) to +\u221E.";
const TOOLTIP_SAMPLES =
  "Number of individual scored data points (question \u00D7 horizon combinations). Each question can produce multiple samples across different forecast horizons.";

/** Check if a model_name represents an ensemble method. */
const isEnsembleModel = (name: string | null): boolean => {
  if (name == null) return true;
  const lower = name.toLowerCase();
  return lower.startsWith("ensemble");
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

/** Friendly display name for a model. */
const displayModelName = (name: string | null): string => {
  if (name == null) return "Ensemble";
  return name;
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

/** Compute weighted average + weighted median of `scoreType` across rows. */
function aggregateScore<
  T extends {
    score_type: string;
    avg_value: number | null;
    median_value: number | null;
    n_samples: number;
    n_questions: number;
  },
>(
  rows: T[],
  scoreType: string,
): {
  avg: number | null;
  median: number | null;
  nQuestions: number;
  nSamples: number;
} {
  const matching = rows.filter((r) => r.score_type === scoreType);
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
    render: (row) => (
      <span className="font-medium text-fred-text">{row.label}</span>
    ),
  },
  {
    key: "avg_brier",
    label: (
      <span className="inline-flex items-center gap-1">
        Avg Brier <InfoTooltip text={TOOLTIP_BRIER} />
      </span>
    ),
    sortValue: (row) => row.avg_brier,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.avg_brier),
  },
  {
    key: "median_brier",
    label: "Mdn Brier",
    sortValue: (row) => row.median_brier,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.median_brier),
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
    label: "Questions",
    sortValue: (row) => row.n_questions,
    defaultSortDirection: "desc",
    render: (row) => formatInt(row.n_questions),
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
        Avg Brier <InfoTooltip text={TOOLTIP_BRIER} />
      </span>
    ),
    sortValue: (row) => row.avg_brier,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.avg_brier),
  },
  {
    key: "median_brier",
    label: "Mdn Brier",
    sortValue: (row) => row.median_brier,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.median_brier),
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
    label: "Questions",
    sortValue: (row) => row.n_questions,
    defaultSortDirection: "desc",
    render: (row) => formatInt(row.n_questions),
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
}: PerformancePanelProps) {
  const [view, setView] = useState<ViewMode>("total");
  const [metric, setMetric] = useState<string | null>(null);
  const [track, setTrack] = useState<number | null>(null);
  const [data, setData] = useState<PerformanceScoresResponse>(initialData);
  const [loading, setLoading] = useState(false);

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
      const params: Record<string, string> = {};
      if (newMetric) params.metric = newMetric;
      if (newTrack != null) params.track = String(newTrack);
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
      ? data.summary_rows.filter((r) => r.model_name === activeEnsemble)
      : data.summary_rows.filter((r) => isEnsembleModel(r.model_name));
    const allHazards = new Set(data.summary_rows.map((r) => r.hazard_code));
    const allModels = new Set(
      data.summary_rows.map((r) => r.model_name).filter(Boolean),
    );
    const allRuns = new Set(data.run_rows.map((r) => r.hs_run_id));
    const brier = aggregateScore(ensembleRows, "brier");
    return {
      hazards: allHazards.size,
      models: allModels.size,
      runs: allRuns.size,
      totalQuestions: brier.nQuestions,
      overallBrier: brier.avg,
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
      const brier = aggregateScore(gRows, "brier");
      const log = aggregateScore(gRows, "log");
      const crps = aggregateScore(gRows, "crps");
      result.push({
        key: gk,
        label: labelFor(gk),
        avg_brier: brier.avg,
        median_brier: brier.median,
        avg_log: log.avg,
        median_log: log.median,
        avg_crps: crps.avg,
        median_crps: crps.median,
        n_questions: brier.nQuestions,
        n_samples: brier.nSamples,
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
      const existing = groups.get(row.hs_run_id) ?? [];
      existing.push(row);
      groups.set(row.hs_run_id, existing);
    }
    const result: RunPivotedRow[] = [];
    for (const [runId, gRows] of groups) {
      const brier = aggregateScore(gRows, "brier");
      const log = aggregateScore(gRows, "log");
      const crps = aggregateScore(gRows, "crps");
      const runDate = gRows[0]?.run_date ?? null;
      result.push({
        key: runId,
        hs_run_id: runId,
        run_date: runDate,
        avg_brier: brier.avg,
        median_brier: brier.median,
        avg_log: log.avg,
        median_log: log.median,
        avg_crps: crps.avg,
        median_crps: crps.median,
        n_questions: brier.nQuestions,
        n_samples: brier.nSamples,
      });
    }
    return result;
  }, [data, activeEnsemble]);

  // ---- By Model view -------------------------------------------------------

  const modelRows = useMemo(
    (): PivotedRow[] =>
      buildPivoted(
        data.summary_rows,
        (r) => r.model_name ?? "__null__",
        (k) => displayModelName(k === "__null__" ? null : k),
      ),
    [data],
  );

  // ---- Render --------------------------------------------------------------

  const isEmpty =
    data.summary_rows.length === 0 && data.run_rows.length === 0;

  return (
    <div className="space-y-6">
      {/* KPI cards */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
        <KpiCard
          label="Scored Track 1"
          value={formatInt(data.track_counts?.track1 ?? 0)}
        />
        <KpiCard
          label="Scored Track 2"
          value={formatInt(data.track_counts?.track2 ?? 0)}
        />
        <KpiCard label="Hazards" value={formatInt(kpiStats.hazards)} />
        <KpiCard label="Models" value={formatInt(kpiStats.models)} />
        <KpiCard label="HS Runs" value={formatInt(kpiStats.runs)} />
        <KpiCard
          label={
            <span className="inline-flex items-center gap-1">
              Ensemble Brier{" "}
              <InfoTooltip text="Average Brier score across all ensemble forecasts. Lower is better." />
            </span>
          }
          value={formatScore(kpiStats.overallBrier)}
        />
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4">
        {/* View selector tabs */}
        <div className="flex gap-1 rounded-lg border border-fred-secondary bg-fred-surface p-1">
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
          className="rounded-md border border-fred-secondary bg-fred-surface px-3 py-1.5 text-sm text-fred-text"
          value={metric ?? ""}
          onChange={(e) => handleMetricChange(e.target.value || null)}
        >
          <option value="">All Metrics</option>
          <option value="PA">People Affected (PA)</option>
          <option value="FATALITIES">Fatalities</option>
        </select>

        {/* Track filter */}
        <select
          className="rounded-md border border-fred-secondary bg-fred-surface px-3 py-1.5 text-sm text-fred-text"
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
            className="rounded-md border border-fred-secondary bg-fred-surface px-3 py-1.5 text-sm text-fred-text"
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

        {loading ? (
          <span className="text-xs text-fred-muted">Loading...</span>
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
            />
          )}
        </div>
      )}
    </div>
  );
}
