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
  avg_log: number | null;
  avg_crps: number | null;
  n_questions: number;
  n_samples: number;
};

// Aggregated row used by the By Run view after pivoting.
type RunPivotedRow = {
  key: string;
  hs_run_id: string;
  run_date: string | null;
  avg_brier: number | null;
  avg_log: number | null;
  avg_crps: number | null;
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

/** Compute a weighted average of `field` across `rows`, weighted by n_samples. */
function weightedAvg(
  rows: PerformanceSummaryRow[],
  scoreType: string,
): { avg: number | null; nQuestions: number; nSamples: number } {
  const matching = rows.filter((r) => r.score_type === scoreType);
  let totalWeight = 0;
  let weightedSum = 0;
  let nQuestions = 0;
  let nSamples = 0;
  for (const r of matching) {
    if (r.avg_value != null && r.n_samples > 0) {
      weightedSum += r.avg_value * r.n_samples;
      totalWeight += r.n_samples;
    }
    nQuestions += r.n_questions;
    nSamples += r.n_samples;
  }
  return {
    avg: totalWeight > 0 ? weightedSum / totalWeight : null,
    nQuestions,
    nSamples,
  };
}

function weightedAvgRun(
  rows: PerformanceRunRow[],
  scoreType: string,
): { avg: number | null; nQuestions: number; nSamples: number } {
  const matching = rows.filter((r) => r.score_type === scoreType);
  let totalWeight = 0;
  let weightedSum = 0;
  let nQuestions = 0;
  let nSamples = 0;
  for (const r of matching) {
    if (r.avg_value != null && r.n_samples > 0) {
      weightedSum += r.avg_value * r.n_samples;
      totalWeight += r.n_samples;
    }
    nQuestions += r.n_questions;
    nSamples += r.n_samples;
  }
  return {
    avg: totalWeight > 0 ? weightedSum / totalWeight : null,
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
    render: (row) => <span className="font-medium text-fred-text">{row.label}</span>,
  },
  {
    key: "avg_brier",
    label: (
      <span className="inline-flex items-center gap-1">
        Avg Brier{" "}
        <InfoTooltip text="Brier score measures the accuracy of probabilistic predictions. Lower is better. Range: 0 (perfect) to 1." />
      </span>
    ),
    sortValue: (row) => row.avg_brier,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.avg_brier),
  },
  {
    key: "avg_log",
    label: (
      <span className="inline-flex items-center gap-1">
        Avg Log Loss{" "}
        <InfoTooltip text="Logarithmic scoring rule. Lower is better. Heavily penalises confident wrong predictions." />
      </span>
    ),
    sortValue: (row) => row.avg_log,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.avg_log),
  },
  {
    key: "avg_crps",
    label: (
      <span className="inline-flex items-center gap-1">
        Avg CRPS{" "}
        <InfoTooltip text="Continuous Ranked Probability Score. Lower is better. Measures full distribution accuracy." />
      </span>
    ),
    sortValue: (row) => row.avg_crps,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.avg_crps),
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
    label: "Samples",
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
        Avg Brier{" "}
        <InfoTooltip text="Brier score measures the accuracy of probabilistic predictions. Lower is better." />
      </span>
    ),
    sortValue: (row) => row.avg_brier,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.avg_brier),
  },
  {
    key: "avg_log",
    label: (
      <span className="inline-flex items-center gap-1">
        Avg Log Loss{" "}
        <InfoTooltip text="Logarithmic scoring rule. Lower is better." />
      </span>
    ),
    sortValue: (row) => row.avg_log,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.avg_log),
  },
  {
    key: "avg_crps",
    label: (
      <span className="inline-flex items-center gap-1">
        Avg CRPS{" "}
        <InfoTooltip text="Continuous Ranked Probability Score. Lower is better." />
      </span>
    ),
    sortValue: (row) => row.avg_crps,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.avg_crps),
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
    label: "Samples",
    sortValue: (row) => row.n_samples,
    defaultSortDirection: "desc",
    render: (row) => formatInt(row.n_samples),
  },
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function PerformancePanel({ initialData }: PerformancePanelProps) {
  const [view, setView] = useState<ViewMode>("total");
  const [metric, setMetric] = useState<string | null>(null);
  const [data, setData] = useState<PerformanceScoresResponse>(initialData);
  const [loading, setLoading] = useState(false);

  // ---- Fetch when metric changes -------------------------------------------

  const handleMetricChange = async (newMetric: string | null) => {
    setMetric(newMetric);
    setLoading(true);
    try {
      const params: Record<string, string> = {};
      if (newMetric) params.metric = newMetric;
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

  // ---- KPI cards -----------------------------------------------------------

  const kpiStats = useMemo(() => {
    const ensembleRows = data.summary_rows.filter((r) => r.model_name == null);
    const allHazards = new Set(ensembleRows.map((r) => r.hazard_code));
    const allModels = new Set(
      data.summary_rows.map((r) => r.model_name).filter(Boolean),
    );
    const allRuns = new Set(data.run_rows.map((r) => r.hs_run_id));
    const brier = weightedAvg(ensembleRows, "brier");
    return {
      hazards: allHazards.size,
      models: allModels.size,
      runs: allRuns.size,
      totalQuestions: brier.nQuestions,
      overallBrier: brier.avg,
    };
  }, [data]);

  // ---- Total view ----------------------------------------------------------

  const totalRows = useMemo((): PivotedRow[] => {
    const ensembleRows = data.summary_rows.filter((r) => r.model_name == null);
    if (ensembleRows.length === 0) return [];
    const brier = weightedAvg(ensembleRows, "brier");
    const log = weightedAvg(ensembleRows, "log");
    const crps = weightedAvg(ensembleRows, "crps");
    return [
      {
        key: "ensemble-total",
        label: "Ensemble (all hazards)",
        avg_brier: brier.avg,
        avg_log: log.avg,
        avg_crps: crps.avg,
        n_questions: brier.nQuestions,
        n_samples: brier.nSamples,
      },
    ];
  }, [data]);

  // ---- By Hazard view ------------------------------------------------------

  const hazardRows = useMemo((): PivotedRow[] => {
    const ensembleRows = data.summary_rows.filter((r) => r.model_name == null);
    const groups = new Map<string, PerformanceSummaryRow[]>();
    for (const row of ensembleRows) {
      const existing = groups.get(row.hazard_code) ?? [];
      existing.push(row);
      groups.set(row.hazard_code, existing);
    }
    const result: PivotedRow[] = [];
    for (const [hazard, rows] of groups) {
      const brier = weightedAvg(rows, "brier");
      const log = weightedAvg(rows, "log");
      const crps = weightedAvg(rows, "crps");
      result.push({
        key: hazard,
        label: hazard,
        avg_brier: brier.avg,
        avg_log: log.avg,
        avg_crps: crps.avg,
        n_questions: brier.nQuestions,
        n_samples: brier.nSamples,
      });
    }
    return result;
  }, [data]);

  // ---- By Run view ---------------------------------------------------------

  const runRows = useMemo((): RunPivotedRow[] => {
    const groups = new Map<string, PerformanceRunRow[]>();
    for (const row of data.run_rows) {
      const existing = groups.get(row.hs_run_id) ?? [];
      existing.push(row);
      groups.set(row.hs_run_id, existing);
    }
    const result: RunPivotedRow[] = [];
    for (const [runId, rows] of groups) {
      const brier = weightedAvgRun(rows, "brier");
      const log = weightedAvgRun(rows, "log");
      const crps = weightedAvgRun(rows, "crps");
      const runDate = rows[0]?.run_date ?? null;
      result.push({
        key: runId,
        hs_run_id: runId,
        run_date: runDate,
        avg_brier: brier.avg,
        avg_log: log.avg,
        avg_crps: crps.avg,
        n_questions: brier.nQuestions,
        n_samples: brier.nSamples,
      });
    }
    return result;
  }, [data]);

  // ---- By Model view -------------------------------------------------------

  const modelRows = useMemo((): PivotedRow[] => {
    const groups = new Map<string, PerformanceSummaryRow[]>();
    for (const row of data.summary_rows) {
      const modelKey = row.model_name ?? "__ensemble__";
      const existing = groups.get(modelKey) ?? [];
      existing.push(row);
      groups.set(modelKey, existing);
    }
    const result: PivotedRow[] = [];
    for (const [modelKey, rows] of groups) {
      const brier = weightedAvg(rows, "brier");
      const log = weightedAvg(rows, "log");
      const crps = weightedAvg(rows, "crps");
      result.push({
        key: modelKey,
        label: modelKey === "__ensemble__" ? "Ensemble" : modelKey,
        avg_brier: brier.avg,
        avg_log: log.avg,
        avg_crps: crps.avg,
        n_questions: brier.nQuestions,
        n_samples: brier.nSamples,
      });
    }
    return result;
  }, [data]);

  // ---- Render --------------------------------------------------------------

  const isEmpty =
    data.summary_rows.length === 0 && data.run_rows.length === 0;

  return (
    <div className="space-y-6">
      {/* KPI cards */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
        <KpiCard label="Scored Questions" value={formatInt(kpiStats.totalQuestions)} />
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
              emptyMessage="No ensemble scores available."
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
