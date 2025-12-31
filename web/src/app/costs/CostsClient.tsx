"use client";

import { useMemo } from "react";

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

type CostsClientProps = {
  total: CostsResponse["tables"];
  monthly: CostsResponse["tables"];
  runs: CostsResponse["tables"];
  latencies: LatencyRow[];
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
        label: "P50 (ms)",
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) => row.p50_elapsed_ms,
        render: (row) => formatNumber(row.p50_elapsed_ms),
        defaultSortDirection: "desc",
      },
      {
        key: "p90_elapsed_ms",
        label: "P90 (ms)",
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) => row.p90_elapsed_ms,
        render: (row) => formatNumber(row.p90_elapsed_ms),
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

export default function CostsClient({
  total,
  monthly,
  runs,
  latencies,
}: CostsClientProps) {
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
        <h2 className="text-xl font-semibold">Run latencies</h2>
        <LatencyTable rows={latencies} emptyMessage="No latency data available." />
      </section>
    </div>
  );
}
