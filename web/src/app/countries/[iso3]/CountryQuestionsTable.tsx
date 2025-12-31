"use client";

import Link from "next/link";
import { useEffect, useMemo } from "react";

import SortableTable, { SortableColumn } from "../../../components/SortableTable";

type CountryQuestionRow = {
  question_id: string;
  hs_run_id?: string | null;
  hazard_code: string;
  metric: string;
  target_month: string;
  status?: string | null;
  wording?: string | null;
  forecast_date?: string | null;
  first_forecast_month?: string | null;
  last_forecast_month?: string | null;
  eiv_total?: number | null;
  triage_score?: number | null;
  triage_date?: string | null;
};

type CountryQuestionsTableProps = {
  rows: CountryQuestionRow[];
};

const parseDateValue = (value?: string | null) => {
  if (!value) return null;
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? null : parsed;
};

const parseYearMonthValue = (value?: string | null) => {
  if (!value) return null;
  const match = value.match(/^(\d{4})-(\d{2})$/);
  if (!match) return null;
  const year = Number(match[1]);
  const month = Number(match[2]);
  if (!Number.isFinite(year) || !Number.isFinite(month)) return null;
  if (month < 1 || month > 12) return null;
  return year * 12 + month;
};

const formatNumber = (value?: number | null) => {
  if (value == null) return "—";
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
};

const formatDate = (value?: string | null) => value ?? "—";

const formatEiv = (value?: number | null) => {
  if (value == null) return "—";
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
};

const buildColumns = (): Array<SortableColumn<CountryQuestionRow>> => [
  {
    key: "question",
    label: "Question",
    sortValue: (row) => row.wording ?? row.question_id ?? "",
    defaultSortDirection: "asc",
    render: (row) => (
      <div className="space-y-1">
        <div className="font-medium text-white">{row.wording ?? "Untitled"}</div>
        <Link
          className="text-sky-300 underline underline-offset-2 hover:text-sky-200"
          href={`/questions/${row.question_id}?hs_run_id=${encodeURIComponent(
            row.hs_run_id ?? ""
          )}`}
        >
          {row.question_id}
        </Link>
      </div>
    ),
  },
  {
    key: "hazard_code",
    label: "Hazard",
    sortValue: (row) => row.hazard_code ?? "",
    defaultSortDirection: "asc",
    render: (row) => row.hazard_code ?? "—",
  },
  {
    key: "metric",
    label: "Metric",
    sortValue: (row) => row.metric ?? "",
    defaultSortDirection: "asc",
    render: (row) => row.metric ?? "—",
  },
  {
    key: "triage_score",
    label: "Triage Score",
    sortValue: (row) => row.triage_score ?? null,
    defaultSortDirection: "desc",
    render: (row) => formatNumber(row.triage_score),
  },
  {
    key: "triage_date",
    label: "Triage Date",
    sortValue: (row) => parseDateValue(row.triage_date),
    defaultSortDirection: "desc",
    render: (row) => formatDate(row.triage_date),
  },
  {
    key: "forecast_date",
    label: "Forecast Date",
    sortValue: (row) => parseDateValue(row.forecast_date),
    defaultSortDirection: "desc",
    render: (row) => formatDate(row.forecast_date),
  },
  {
    key: "first_forecast_month",
    label: "First Forecast Month",
    sortValue: (row) => parseYearMonthValue(row.first_forecast_month),
    defaultSortDirection: "desc",
    render: (row) => formatDate(row.first_forecast_month),
  },
  {
    key: "last_forecast_month",
    label: "Last Forecast Month",
    sortValue: (row) => parseYearMonthValue(row.last_forecast_month),
    defaultSortDirection: "desc",
    render: (row) => formatDate(row.last_forecast_month),
  },
  {
    key: "eiv_total",
    label: "6 Month EIV",
    sortValue: (row) => row.eiv_total ?? null,
    defaultSortDirection: "desc",
    render: (row) => formatEiv(row.eiv_total),
  },
  {
    key: "status",
    label: "Status",
    sortValue: (row) => row.status ?? "",
    defaultSortDirection: "asc",
    render: (row) => row.status ?? "—",
  },
];

const useDebugCountry = () => {
  return useMemo(() => {
    if (typeof window === "undefined") return false;
    const params = new URLSearchParams(window.location.search);
    if (params.get("debug_country") === "1") return true;
    try {
      return window.localStorage.getItem("pythia_debug_country") === "1";
    } catch {
      return false;
    }
  }, []);
};

export default function CountryQuestionsTable({ rows }: CountryQuestionsTableProps) {
  const debugEnabled = useDebugCountry();
  const columns = useMemo(() => buildColumns(), []);

  const debugSummary = useMemo(() => {
    let eivMin: number | null = null;
    let eivMax: number | null = null;
    let missingTriageScore = 0;
    let missingTriageDate = 0;
    let missingForecastDate = 0;
    let missingEivTotal = 0;
    let missingLastForecastMonth = 0;

    rows.forEach((row) => {
      if (row.triage_score == null) missingTriageScore += 1;
      if (!row.triage_date) missingTriageDate += 1;
      if (!row.forecast_date) missingForecastDate += 1;
      if (row.eiv_total == null) missingEivTotal += 1;
      if (!row.last_forecast_month) missingLastForecastMonth += 1;
      if (row.eiv_total != null) {
        eivMin = eivMin == null ? row.eiv_total : Math.min(eivMin, row.eiv_total);
        eivMax = eivMax == null ? row.eiv_total : Math.max(eivMax, row.eiv_total);
      }
    });

    return {
      rowCount: rows.length,
      missingTriageScore,
      missingTriageDate,
      missingForecastDate,
      missingEivTotal,
      missingLastForecastMonth,
      eivMin,
      eivMax,
    };
  }, [rows]);

  useEffect(() => {
    if (!debugEnabled) return;
    console.groupCollapsed("[CountryQuestionsTable] debug");
    console.log("rowCount:", debugSummary.rowCount);
    console.log("missing triage_score:", debugSummary.missingTriageScore);
    console.log("missing triage_date:", debugSummary.missingTriageDate);
    console.log("missing forecast_date:", debugSummary.missingForecastDate);
    console.log("missing eiv_total:", debugSummary.missingEivTotal);
    console.log("missing last_forecast_month:", debugSummary.missingLastForecastMonth);
    console.log("eiv_min:", debugSummary.eivMin);
    console.log("eiv_max:", debugSummary.eivMax);
    console.groupEnd();
  }, [debugEnabled, debugSummary]);

  return (
    <div className="space-y-3">
      {debugEnabled ? (
        <details className="rounded-lg border border-slate-800 bg-slate-900/50 px-3 py-2 text-xs text-slate-300">
          <summary className="cursor-pointer text-slate-200">
            Country table debug
          </summary>
          <div className="mt-2 space-y-1">
            <div>rowCount: {debugSummary.rowCount}</div>
            <div>missing triage_score: {debugSummary.missingTriageScore}</div>
            <div>missing triage_date: {debugSummary.missingTriageDate}</div>
            <div>missing forecast_date: {debugSummary.missingForecastDate}</div>
            <div>missing eiv_total: {debugSummary.missingEivTotal}</div>
            <div>
              missing last_forecast_month: {debugSummary.missingLastForecastMonth}
            </div>
            <div>
              eiv_min: {debugSummary.eivMin == null ? "n/a" : formatEiv(debugSummary.eivMin)}
            </div>
            <div>
              eiv_max: {debugSummary.eivMax == null ? "n/a" : formatEiv(debugSummary.eivMax)}
            </div>
          </div>
        </details>
      ) : null}
      <SortableTable
        columns={columns}
        rows={rows}
        rowKey={(row) => row.question_id}
        initialSortKey="eiv_total"
        initialSortDirection="desc"
        tableLayout="auto"
      />
    </div>
  );
}
