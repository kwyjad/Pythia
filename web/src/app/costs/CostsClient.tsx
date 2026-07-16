"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";

import { apiGet } from "../../lib/api";
import { formatModelName } from "../../lib/model_names";

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
  prediction_markets_ms: number | null;
  sibyl_ms: number | null;
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

// Canonical phase order + friendly labels (mirrors CANONICAL_PHASES in
// resolver/query/costs.py). Keep in sync — the phase_group backend never emits a
// value outside this set.
export const PHASE_ORDER = [
  "web_search",
  "hs",
  "research",
  "forecast",
  "scenario",
  "prediction_markets",
  "sibyl",
  "other",
] as const;

export const PHASE_LABELS: Record<string, string> = {
  web_search: "Web search",
  hs: "Horizon scan",
  research: "Research",
  forecast: "Forecast",
  scenario: "Scenario",
  prediction_markets: "Prediction markets",
  sibyl: "Sibyl",
  other: "Other",
};

// Validated categorical palette (dataviz skill, light surface #fcfcfb). One fixed
// hue per phase, assigned by entity (never by rank), so a track filter that drops
// phases never repaints the survivors.
export const PHASE_COLORS: Record<string, string> = {
  web_search: "#2a78d6",
  hs: "#008300",
  research: "#e87ba4",
  forecast: "#eda100",
  scenario: "#1baf7a",
  prediction_markets: "#eb6834",
  sibyl: "#4a3aa7",
  other: "#e34948",
};

const phaseRank = (phase: string | null) =>
  phase ? PHASE_ORDER.indexOf(phase as (typeof PHASE_ORDER)[number]) : -1;

const formatPhase = (phase: string | null | undefined) => {
  if (!phase) return "—";
  return PHASE_LABELS[phase] ?? phase;
};

const phaseColor = (phase: string | null) =>
  (phase && PHASE_COLORS[phase]) || PHASE_COLORS.other;

const formatModel = (model: string | null | undefined) =>
  model ? formatModelName(model) : "—";

const PhaseLegend = () => (
  <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-fred-muted" aria-hidden>
    {PHASE_ORDER.map((phase) => (
      <li key={phase} className="flex items-center gap-1.5">
        <span
          className="inline-block h-2.5 w-2.5 rounded-sm"
          style={{ backgroundColor: PHASE_COLORS[phase] }}
        />
        {PHASE_LABELS[phase]}
      </li>
    ))}
  </ul>
);

// Horizontal spend-by-phase bar. Direct labels ($ + %) satisfy the relief rule for
// the low-contrast light-mode hues; the adjacent "By phase" table is the a11y view.
const PhaseSpendBar = ({ rows }: { rows: CostsRow[] }) => {
  const byPhase = new Map<string, number>();
  for (const row of rows) {
    const phase = row.phase ?? "other";
    byPhase.set(phase, (byPhase.get(phase) ?? 0) + (row.total_cost_usd ?? 0));
  }
  const entries = PHASE_ORDER.filter((p) => byPhase.has(p)).map((p) => ({
    phase: p,
    value: byPhase.get(p) ?? 0,
  }));
  const total = entries.reduce((acc, e) => acc + e.value, 0);
  if (!entries.length || total <= 0) {
    return <p className="text-sm text-fred-muted">No phase spend to chart.</p>;
  }
  return (
    <div className="space-y-2">
      {entries.map(({ phase, value }) => {
        const pct = (value / total) * 100;
        return (
          <div key={phase} className="flex items-center gap-3 text-xs">
            <span className="w-32 shrink-0 text-fred-text">{formatPhase(phase)}</span>
            <div
              className="h-4 flex-1 overflow-hidden rounded-sm bg-fred-bg"
              role="img"
              aria-label={`${formatPhase(phase)}: ${formatCurrency(value)} (${pct.toFixed(1)}%)`}
            >
              <div
                className="h-full rounded-sm"
                style={{ width: `${Math.max(pct, 0.5)}%`, backgroundColor: phaseColor(phase) }}
                title={`${formatPhase(phase)}: ${formatCurrency(value)} (${pct.toFixed(1)}%)`}
              />
            </div>
            <span className="w-28 shrink-0 text-right tabular-nums text-fred-text">
              {formatCurrency(value)}
            </span>
            <span className="w-12 shrink-0 text-right tabular-nums text-fred-muted">
              {pct.toFixed(1)}%
            </span>
          </div>
        );
      })}
    </div>
  );
};

// Monthly spend-over-time, stacked by phase. Inline SVG (no chart dependency);
// segments carry native <title> hover tooltips. One axis (USD), one column per month.
const MonthlySpendTrend = ({ rows }: { rows: CostsRow[] }) => {
  const monthMap = new Map<string, { year: number; month: number; byPhase: Map<string, number> }>();
  for (const row of rows) {
    if (!row.year || !row.month) continue;
    const key = `${row.year}-${String(row.month).padStart(2, "0")}`;
    let entry = monthMap.get(key);
    if (!entry) {
      entry = { year: row.year, month: row.month, byPhase: new Map() };
      monthMap.set(key, entry);
    }
    const phase = row.phase ?? "other";
    entry.byPhase.set(phase, (entry.byPhase.get(phase) ?? 0) + (row.total_cost_usd ?? 0));
  }
  const months = Array.from(monthMap.entries())
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([key, entry]) => ({ key, ...entry }));
  if (!months.length) {
    return <p className="text-sm text-fred-muted">No monthly spend to chart.</p>;
  }
  const monthTotals = months.map((m) =>
    Array.from(m.byPhase.values()).reduce((acc, v) => acc + v, 0)
  );
  const maxTotal = Math.max(...monthTotals, 0);
  if (maxTotal <= 0) {
    return <p className="text-sm text-fred-muted">No monthly spend to chart.</p>;
  }

  const width = Math.max(months.length * 56 + 60, 320);
  const height = 220;
  const padLeft = 52;
  const padBottom = 28;
  const padTop = 8;
  const plotH = height - padBottom - padTop;
  const bandW = (width - padLeft) / months.length;
  const barW = Math.min(bandW * 0.62, 40);
  const gap = 2; // surface gap between stacked segments

  return (
    <div className="overflow-x-auto">
      <svg
        width="100%"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMinYMin meet"
        role="img"
        aria-label="Monthly LLM spend, stacked by phase"
        style={{ maxWidth: "100%", minWidth: Math.min(width, 320) }}
      >
        {/* y-axis reference lines + labels */}
        {[0, 0.5, 1].map((frac) => {
          const y = padTop + plotH * (1 - frac);
          return (
            <g key={frac}>
              <line x1={padLeft} y1={y} x2={width} y2={y} stroke="#E5E5E5" strokeWidth={1} />
              <text x={padLeft - 6} y={y + 3} textAnchor="end" fontSize={10} fill="#6B7280">
                ${currencyFormatter.format(maxTotal * frac)}
              </text>
            </g>
          );
        })}
        {months.map((m, i) => {
          const x = padLeft + i * bandW + (bandW - barW) / 2;
          let cursorY = padTop + plotH;
          const segments = PHASE_ORDER.filter((p) => (m.byPhase.get(p) ?? 0) > 0).map((phase) => {
            const value = m.byPhase.get(phase) ?? 0;
            const segH = (value / maxTotal) * plotH;
            const top = cursorY - segH;
            const rect = (
              <rect
                key={phase}
                x={x}
                y={top}
                width={barW}
                height={Math.max(segH - gap, 0)}
                rx={2}
                fill={phaseColor(phase)}
              >
                <title>
                  {`${m.key} · ${formatPhase(phase)}: ${formatCurrency(value)}`}
                </title>
              </rect>
            );
            cursorY = top;
            return rect;
          });
          return (
            <g key={m.key}>
              {segments}
              <text
                x={x + barW / 2}
                y={height - padBottom + 14}
                textAnchor="middle"
                fontSize={9}
                fill="#6B7280"
              >
                {`${String(m.month).padStart(2, "0")}/${String(m.year).slice(2)}`}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
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
    sortValue: (row) => formatModel(row.model),
    render: (row) => formatModel(row.model),
    isVisible: visibility.showModel,
    defaultSortDirection: "asc",
  },
  {
    key: "phase",
    label: "Phase",
    headerClassName: "text-left",
    cellClassName: "text-left",
    sortValue: (row) => phaseRank(row.phase),
    render: (row) => formatPhase(row.phase),
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
        sortValue: (row) => formatModel(row.model),
        render: (row) => formatModel(row.model),
        defaultSortDirection: "asc",
      },
      {
        key: "phase",
        label: "Phase",
        headerClassName: "text-left",
        cellClassName: "text-left",
        sortValue: (row) => phaseRank(row.phase),
        render: (row) => formatPhase(row.phase),
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
        key: "prediction_markets_ms",
        label: "Pred. markets",
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) => row.prediction_markets_ms,
        render: (row) => formatDurationMs(row.prediction_markets_ms),
        defaultSortDirection: "desc",
      },
      {
        key: "sibyl_ms",
        label: "Sibyl",
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) => row.sibyl_ms,
        render: (row) => formatDurationMs(row.sibyl_ms),
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

type TrackFilter = "all" | "1" | "2";

const TRACK_OPTIONS: { value: TrackFilter; label: string }[] = [
  { value: "all", label: "Tracks 1 and 2" },
  { value: "1", label: "Track 1" },
  { value: "2", label: "Track 2" },
];

const emptyTables = { summary: [] as CostsRow[], by_model: [] as CostsRow[], by_phase: [] as CostsRow[] };

export default function CostsClient({
  total: initialTotal,
  monthly: initialMonthly,
  runs: initialRuns,
  latencies: initialLatencies,
  runRuntimes: initialRunRuntimes,
}: CostsClientProps) {
  const searchParams = useSearchParams();
  const showDebug = searchParams?.get("debug_costs") === "1";

  const [trackFilter, setTrackFilter] = useState<TrackFilter>("all");
  const [total, setTotal] = useState(initialTotal);
  const [monthly, setMonthly] = useState(initialMonthly);
  const [runs, setRuns] = useState(initialRuns);
  const [latencies, setLatencies] = useState(initialLatencies);
  const [runRuntimes, setRunRuntimes] = useState(initialRunRuntimes);
  const [loading, setLoading] = useState(false);

  const fetchCosts = useCallback(async (track: TrackFilter) => {
    setLoading(true);
    const params = track !== "all" ? { track: Number(track) } : undefined;
    try {
      const [t, m, r, l, rt] = await Promise.all([
        apiGet<CostsResponse>("/costs/total", params).catch(() => ({ tables: emptyTables })),
        apiGet<CostsResponse>("/costs/monthly", params).catch(() => ({ tables: emptyTables })),
        apiGet<CostsResponse>("/costs/runs", params).catch(() => ({ tables: emptyTables })),
        apiGet<LatenciesResponse>("/costs/latencies", params).catch(() => ({ rows: [] as LatencyRow[] })),
        apiGet<RunRuntimesResponse>("/costs/run_runtimes", params).catch(() => ({ rows: [] as RunRuntimeRow[] })),
      ]);
      setTotal(t.tables);
      setMonthly(m.tables);
      setRuns(r.tables);
      setLatencies(l.rows);
      setRunRuntimes(rt.rows);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (trackFilter === "all") {
      setTotal(initialTotal);
      setMonthly(initialMonthly);
      setRuns(initialRuns);
      setLatencies(initialLatencies);
      setRunRuntimes(initialRunRuntimes);
      return;
    }
    fetchCosts(trackFilter);
  }, [trackFilter, fetchCosts, initialTotal, initialMonthly, initialRuns, initialLatencies, initialRunRuntimes]);

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
        <div className="flex items-center gap-3">
          <label className="text-xs font-semibold text-fred-text">Track</label>
          <div className="flex gap-1">
            {TRACK_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                type="button"
                className={`rounded-md px-3 py-1 text-xs font-semibold transition-colors ${
                  trackFilter === opt.value
                    ? "bg-fred-secondary text-white"
                    : "bg-fred-surface text-fred-text border border-fred-secondary hover:bg-fred-secondary/10"
                }`}
                onClick={() => setTrackFilter(opt.value)}
                disabled={loading}
              >
                {opt.label}
              </button>
            ))}
          </div>
          {loading ? (
            <span className="text-xs text-fred-muted">Loading...</span>
          ) : null}
        </div>
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
        <div className="space-y-3 rounded-lg border border-fred-secondary bg-fred-surface p-4">
          <h3 className="text-sm font-semibold">Spend by phase</h3>
          <PhaseSpendBar rows={total.by_phase} />
          <p className="text-xs text-fred-muted">
            Sibyl = Opus reasoning + Brave web search. Other = spend that could not be
            attributed to a known phase.
          </p>
        </div>
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
        <div className="space-y-3 rounded-lg border border-fred-secondary bg-fred-surface p-4">
          <h3 className="text-sm font-semibold">Spend over time, by phase</h3>
          <MonthlySpendTrend rows={monthly.by_phase} />
          <PhaseLegend />
        </div>
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
