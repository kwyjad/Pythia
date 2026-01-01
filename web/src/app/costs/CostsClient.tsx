"use client";

import { useMemo } from "react";
import { useSearchParams } from "next/navigation";

import SortableTable, { SortableColumn } from "../../components/SortableTable";

export type CostsRow = {
  grain: string;
  row_type: string;
  year: number | null;
  month: number | null;
  run_id: string | null;
  model: string | null;
  phase: string | null;
  total_cost_usd: number | null;
  n_questions: number | null;
  avg_cost_per_question: number | null;
  median_cost_per_question: number | null;
  n_countries: number | null;
  avg_cost_per_country: number | null;
  median_cost_per_country: number | null;
};

export type CostsResponse = {
  tables: {
    summary: CostsRow[];
    by_model: CostsRow[];
    by_phase: CostsRow[];
  };
};

export type LatencyRow = {
  run_id: string | null;
  year: number | null;
  month: number | null;
  model: string | null;
  phase: string | null;
  n_calls: number | null;
  p50_elapsed_ms: number | null;
  p90_elapsed_ms: number | null;
};

export type LatenciesResponse = {
  rows: LatencyRow[];
};

export type RunRuntimeRow = {
  run_date: string | null;
  run_id: string | null;
  year: number | null;
  month: number | null;
  n_questions: number | null;
  n_countries: number | null;
  question_p50_ms: number | null;
  question_p90_ms: number | null;
  country_p50_ms: number | null;
  country_p90_ms: number | null;
  web_search_ms: number | null;
  hs_ms: number | null;
  research_ms: number | null;
  forecast_ms: number | null;
  scenario_ms: number | null;
  other_ms: number | null;
  total_ms: number | null;
};

export type RunRuntimesResponse = {
  rows: RunRuntimeRow[];
};

type CostsClientProps = {
  total: CostsResponse["tables"];
  monthly: CostsResponse["tables"];
  runs: CostsResponse["tables"];
  latencies: LatencyRow[];
  runRuntimes: RunRuntimeRow[];
};

type CostsTableVisibility = {
  showYearMonth: boolean;
  showRun: boolean;
  showModel: boolean;
  showPhase: boolean;
};

const numberFormatter = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 2,
});

const currencyFormatter = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 4,
});

const formatNumber = (value: number | null | undefined) => {
  if (value === null || value === undefined) return "—";
  return numberFormatter.format(value);
};

const formatCurrency = (value: number | null | undefined) => {
  if (value === null || value === undefined) return "—";
  return `$${currencyFormatter.format(value)}`;
};

const formatDurationMs = (value: number | null | undefined) => {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  const totalSeconds = value / 1000;
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = (totalSeconds % 60).toFixed(1);
  const secondsPadded = seconds.padStart(4, "0");
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(
    2,
    "0"
  )}:${secondsPadded}`;
};

const formatMonth = (year: number | null, month: number | null) => {
  if (!year || !month) return "—";
  return `${year}-${String(month).padStart(2, "0")}`;
};

const sortYearMonth = (row: { year: number | null; month: number | null }) => {
  if (!row.year || !row.month) return null;
  return row.year * 100 + row.month;
};

const buildCostsColumns = (
  visibility: CostsTableVisibility
): Array<SortableColumn<CostsRow>> => [
  {
    key: "year_month",
    label: "Year-Month",
    headerClassName: "text-left",
    cellClassName: "text-left tabular-nums",
    sortValue: (row) => sortYearMonth(row),
    render: (row) => formatMonth(row.year, row.month),
    isVisible: visibility.showYearMonth,
    defaultSortDirection: "desc",
  },
  {
    key: "run_id",
    label: "Run",
    headerClassName: "text-left",
    cellClassName: "text-left",
    sortValue: (row) => row.run_id ?? "",
    render: (row) => row.run_id ?? "—",
    isVisible: visibility.showRun,
    defaultSortDirection: "asc",
  },
  {
    key: "model",
    label: "Model",
    headerClassName: "text-left",
    cellClassName: "text-left",
    sortValue: (row) => row.model ?? "",
    render: (row) => row.model ?? "—",
    isVisible: visibility.showModel,
    defaultSortDirection: "asc",
  },
  {
    key: "phase",
    label: "Phase",
    headerClassName: "text-left",
    cellClassName: "text-left",
    sortValue: (row) => row.phase ?? "",
    render: (row) => row.phase ?? "—",
    isVisible: visibility.showPhase,
    defaultSortDirection: "asc",
  },
  {
    key: "total_cost_usd",
    label: "Total cost (USD)",
    headerClassName: "text-right",
    cellClassName: "text-right tabular-nums",
    sortValue: (row) => row.total_cost_usd,
    render: (row) => formatCurrency(row.total_cost_usd),
    defaultSortDirection: "desc",
  },
  {
    key: "n_questions",
    label: "Questions",
    headerClassName: "text-right",
    cellClassName: "text-right tabular-nums",
    sortValue: (row) => row.n_questions,
    render: (row) => formatNumber(row.n_questions),
    defaultSortDirection: "desc",
  },
  {
    key: "avg_cost_per_question",
    label: "Avg/Q",
    headerClassName: "text-right",
    cellClassName: "text-right tabular-nums",
    sortValue: (row) => row.avg_cost_per_question,
    render: (row) => formatCurrency(row.avg_cost_per_question),
    defaultSortDirection: "desc",
  },
  {
    key: "median_cost_per_question",
    label: "Median/Q",
    headerClassName: "text-right",
    cellClassName: "text-right tabular-nums",
    sortValue: (row) => row.median_cost_per_question,
    render: (row) => formatCurrency(row.median_cost_per_question),
    defaultSortDirection: "desc",
  },
  {
    key: "n_countries",
    label: "Countries",
    headerClassName: "text-right",
    cellClassName: "text-right tabular-nums",
    sortValue: (row) => row.n_countries,
    render: (row) => formatNumber(row.n_countries),
    defaultSortDirection: "desc",
  },
  {
    key: "avg_cost_per_country",
    label: "Avg/Country",
    headerClassName: "text-right",
    cellClassName: "text-right tabular-nums",
    sortValue: (row) => row.avg_cost_per_country,
    render: (row) => formatCurrency(row.avg_cost_per_country),
    defaultSortDirection: "desc",
  },
  {
    key: "median_cost_per_country",
    label: "Median/Country",
    headerClassName: "text-right",
    cellClassName: "text-right tabular-nums",
    sortValue: (row) => row.median_cost_per_country,
    render: (row) => formatCurrency(row.median_cost_per_country),
    defaultSortDirection: "desc",
  },
];

const CostsTable = ({
  rows,
  emptyMessage,
  visibility,
}: {
  rows: CostsRow[];
  emptyMessage: string;
  visibility: CostsTableVisibility;
}) => {
  const columns = useMemo(
    () => buildCostsColumns(visibility),
    [
      visibility.showYearMonth,
      visibility.showRun,
      visibility.showModel,
      visibility.showPhase,
    ]
  );

  return (
    <div className="overflow-x-auto rounded-lg border border-fred-secondary">
      <SortableTable
        columns={columns}
        emptyMessage={emptyMessage}
        initialSortDirection="desc"
        initialSortKey="total_cost_usd"
        rowKey={(row) =>
          [
            row.grain,
            row.row_type,
            row.year ?? "none",
            row.month ?? "none",
            row.run_id ?? "none",
            row.model ?? "none",
            row.phase ?? "none",
          ].join("-")
        }
        rows={rows}
        tableLayout="auto"
      />
    </div>
  );
};

const LatencyTable = ({
  rows,
  emptyMessage,
}: {
  rows: LatencyRow[];
  emptyMessage: string;
}) => {
  const columns = useMemo<Array<SortableColumn<LatencyRow>>>(
    () => [
      {
        key: "year_month",
        label: "Year-Month",
        headerClassName: "text-left",
        cellClassName: "text-left tabular-nums",
        sortValue: (row) => sortYearMonth(row),
        render: (row) => formatMonth(row.year, row.month),
        defaultSortDirection: "desc",
      },
      {
        key: "run_id",
        label: "Run",
        headerClassName: "text-left",
        cellClassName: "text-left",
        sortValue: (row) => row.run_id ?? "",
        render: (row) => row.run_id ?? "—",
        defaultSortDirection: "asc",
      },
      {
        key: "model",
        label: "Model",
        headerClassName: "text-left",
        cellClassName: "text-left",
        sortValue: (row) => row.model ?? "",
        render: (row) => row.model ?? "—",
        defaultSortDirection: "asc",
      },
      {
        key: "phase",
        label: "Phase",
        headerClassName: "text-left",
        cellClassName: "text-left",
        sortValue: (row) => row.phase ?? "",
        render: (row) => row.phase ?? "—",
        defaultSortDirection: "asc",
      },
      {
        key: "n_calls",
        label: "Calls",
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) => row.n_calls,
        render: (row) => formatNumber(row.n_calls),
        defaultSortDirection: "desc",
      },
      {
        key: "p50_elapsed_ms",
        label: "P50",
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) => row.p50_elapsed_ms,
        render: (row) => formatDurationMs(row.p50_elapsed_ms),
        defaultSortDirection: "desc",
      },
      {
        key: "p90_elapsed_ms",
        label: "P90",
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) => row.p90_elapsed_ms,
        render: (row) => formatDurationMs(row.p90_elapsed_ms),
        defaultSortDirection: "desc",
      },
    ],
    []
  );

  return (
    <div className="overflow-x-auto rounded-lg border border-fred-secondary">
      <SortableTable
        columns={columns}
        emptyMessage={emptyMessage}
        initialSortDirection="desc"
        initialSortKey="p90_elapsed_ms"
        rowKey={(row) =>
          [
            row.year ?? "none",
            row.month ?? "none",
            row.run_id ?? "none",
            row.model ?? "none",
            row.phase ?? "none",
          ].join("-")
        }
        rows={rows}
        tableLayout="auto"
      />
    </div>
  );
};

const RunRuntimesTable = ({
  rows,
  emptyMessage,
}: {
  rows: RunRuntimeRow[];
  emptyMessage: string;
}) => {
  const columns = useMemo<Array<SortableColumn<RunRuntimeRow>>>(
    () => [
      {
        key: "run_date",
        label: "Run date",
        headerClassName: "text-left",
        cellClassName: "text-left tabular-nums",
        sortValue: (row) => (row.run_date ? Date.parse(row.run_date) : null),
        render: (row) => row.run_date ?? "—",
        defaultSortDirection: "desc",
      },
      {
        key: "run_id",
        label: "Run",
        headerClassName: "text-left",
        cellClassName: "text-left",
        sortValue: (row) => row.run_id ?? "",
        render: (row) => row.run_id ?? "—",
        defaultSortDirection: "asc",
      },
      {
        key: "n_questions",
        label: "Questions",
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) => row.n_questions,
        render: (row) => formatNumber(row.n_questions),
        defaultSortDirection: "desc",
      },
      {
        key: "n_countries",
        label: "Countries",
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) => row.n_countries,
        render: (row) => formatNumber(row.n_countries),
        defaultSortDirection: "desc",
      },
      {
        key: "question_p50_ms",
        label: "P50 / question",
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) => row.question_p50_ms,
        render: (row) => formatDurationMs(row.question_p50_ms),
        defaultSortDirection: "desc",
      },
      {
        key: "question_p90_ms",
        label: "P90 / question",
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) => row.question_p90_ms,
        render: (row) => formatDurationMs(row.question_p90_ms),
        defaultSortDirection: "desc",
      },
      {
        key: "country_p50_ms",
        label: "P50 / country",
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) => row.country_p50_ms,
        render: (row) => formatDurationMs(row.country_p50_ms),
        defaultSortDirection: "desc",
      },
      {
        key: "country_p90_ms",
        label: "P90 / country",
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) => row.country_p90_ms,
        render: (row) => formatDurationMs(row.country_p90_ms),
        defaultSortDirection: "desc",
      },
      {
        key: "web_search_ms",
        label: "Web search",
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) => row.web_search_ms,
        render: (row) => formatDurationMs(row.web_search_ms),
        defaultSortDirection: "desc",
      },
      {
        key: "hs_ms",
        label: "HS",
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) => row.hs_ms,
        render: (row) => formatDurationMs(row.hs_ms),
        defaultSortDirection: "desc",
      },
      {
        key: "research_ms",
        label: "Research",
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) => row.research_ms,
        render: (row) => formatDurationMs(row.research_ms),
        defaultSortDirection: "desc",
      },
      {
        key: "forecast_ms",
        label: "Forecast",
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) => row.forecast_ms,
        render: (row) => formatDurationMs(row.forecast_ms),
        defaultSortDirection: "desc",
      },
      {
        key: "scenario_ms",
        label: "Scenario",
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) => row.scenario_ms,
        render: (row) => formatDurationMs(row.scenario_ms),
        defaultSortDirection: "desc",
      },
      {
        key: "other_ms",
        label: "Other",
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) => row.other_ms,
        render: (row) => formatDurationMs(row.other_ms),
        defaultSortDirection: "desc",
      },
      {
        key: "total_ms",
        label: "Total run time",
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) => row.total_ms,
        render: (row) => formatDurationMs(row.total_ms),
        defaultSortDirection: "desc",
      },
    ],
    []
  );

  return (
    <div className="overflow-x-auto rounded-lg border border-fred-secondary">
      <SortableTable
        columns={columns}
        emptyMessage={emptyMessage}
        initialSortDirection="desc"
        initialSortKey="run_date"
        rowKey={(row) =>
          [row.run_date ?? "none", row.run_id ?? "none"].join("-")
        }
        rows={rows}
        tableLayout="auto"
      />
    </div>
  );
};

export default function CostsClient({
  total,
  monthly,
  runs,
  latencies,
  runRuntimes,
}: CostsClientProps) {
  const searchParams = useSearchParams();
  const showDebug = searchParams?.get("debug_costs") === "1";
  const debugStats = useMemo(() => {
    const totals = runRuntimes
      .map((row) => row.total_ms)
      .filter((value): value is number => value !== null && value !== undefined);
    const nullTotals =
      runRuntimes.length - totals.length;
    const minTotal = totals.length ? Math.min(...totals) : null;
    const maxTotal = totals.length ? Math.max(...totals) : null;
    return {
      count: runRuntimes.length,
      nullTotals,
      minTotal,
      maxTotal,
    };
  }, [runRuntimes]);

  return (
    <div className="space-y-8">
      <section className="space-y-2">
        <h1 className="text-3xl font-semibold">Costs</h1>
        <p className="text-sm text-fred-text">
          Review LLM spend and latency across total, monthly, and run-level
          aggregates.
        </p>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Total costs</h2>
        <CostsTable
          rows={total.summary}
          emptyMessage="No total cost summary available."
          visibility={{
            showYearMonth: false,
            showRun: false,
            showModel: false,
            showPhase: false,
          }}
        />
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="space-y-2">
            <h3 className="text-sm font-semibold">By model</h3>
            <CostsTable
              rows={total.by_model}
              emptyMessage="No model breakdown available."
              visibility={{
                showYearMonth: false,
                showRun: false,
                showModel: true,
                showPhase: false,
              }}
            />
          </div>
          <div className="space-y-2">
            <h3 className="text-sm font-semibold">By phase</h3>
            <CostsTable
              rows={total.by_phase}
              emptyMessage="No phase breakdown available."
              visibility={{
                showYearMonth: false,
                showRun: false,
                showModel: false,
                showPhase: true,
              }}
            />
          </div>
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Monthly costs</h2>
        <CostsTable
          rows={monthly.summary}
          emptyMessage="No monthly summary available."
          visibility={{
            showYearMonth: true,
            showRun: false,
            showModel: false,
            showPhase: false,
          }}
        />
        <details className="space-y-3 rounded-lg border border-fred-secondary bg-fred-surface p-4">
          <summary className="cursor-pointer text-sm font-semibold text-fred-secondary">
            View monthly breakdowns by model and phase
          </summary>
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="space-y-2">
              <h3 className="text-sm font-semibold">By model</h3>
              <CostsTable
                rows={monthly.by_model}
                emptyMessage="No monthly model data."
                visibility={{
                  showYearMonth: true,
                  showRun: false,
                  showModel: true,
                  showPhase: false,
                }}
              />
            </div>
            <div className="space-y-2">
              <h3 className="text-sm font-semibold">By phase</h3>
              <CostsTable
                rows={monthly.by_phase}
                emptyMessage="No monthly phase data."
                visibility={{
                  showYearMonth: true,
                  showRun: false,
                  showModel: false,
                  showPhase: true,
                }}
              />
            </div>
          </div>
        </details>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Run costs</h2>
        <CostsTable
          rows={runs.summary}
          emptyMessage="No run summary available."
          visibility={{
            showYearMonth: true,
            showRun: true,
            showModel: false,
            showPhase: false,
          }}
        />
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="space-y-2">
            <h3 className="text-sm font-semibold">By model</h3>
            <CostsTable
              rows={runs.by_model}
              emptyMessage="No run model breakdown available."
              visibility={{
                showYearMonth: true,
                showRun: true,
                showModel: true,
                showPhase: false,
              }}
            />
          </div>
          <div className="space-y-2">
            <h3 className="text-sm font-semibold">By phase</h3>
            <CostsTable
              rows={runs.by_phase}
              emptyMessage="No run phase breakdown available."
              visibility={{
                showYearMonth: true,
                showRun: true,
                showModel: false,
                showPhase: true,
              }}
            />
          </div>
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Run runtimes</h2>
        <RunRuntimesTable
          rows={runRuntimes}
          emptyMessage="No runtime data available."
        />
        {showDebug ? (
          <details className="space-y-2 rounded-lg border border-fred-secondary bg-fred-surface p-4 text-sm">
            <summary className="cursor-pointer font-semibold text-fred-secondary">
              Debug runtimes
            </summary>
            <div className="space-y-1">
              <div>Runtime rows: {formatNumber(debugStats.count)}</div>
              <div>Rows missing total_ms: {formatNumber(debugStats.nullTotals)}</div>
              <div>
                Min total: {formatDurationMs(debugStats.minTotal)}
              </div>
              <div>
                Max total: {formatDurationMs(debugStats.maxTotal)}
              </div>
            </div>
          </details>
        ) : null}
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Run latencies</h2>
        <LatencyTable rows={latencies} emptyMessage="No latency data available." />
      </section>
    </div>
  );
}
