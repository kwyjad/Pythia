"use client";

import Link from "next/link";
import { useMemo, useState } from "react";

import SortableTable, { SortableColumn } from "../../components/SortableTable";

type QuestionsRow = {
  question_id: string;
  hs_run_id?: string | null;
  iso3: string;
  country_name?: string | null;
  hazard_code: string;
  metric: string;
  forecast_date?: string | null;
  first_forecast_month?: string | null;
  last_forecast_month?: string | null;
  status?: string;
  wording?: string;
  eiv_total?: number | null;
};

type QuestionsTableProps = {
  rows: QuestionsRow[];
};

const parseYearMonth = (value?: string | null) => {
  if (!value) return null;
  const [yearStr, monthStr] = value.split("-");
  const year = Number(yearStr);
  const month = Number(monthStr);
  if (!Number.isFinite(year) || !Number.isFinite(month)) return null;
  return year * 12 + month;
};

const contains = (haystack: string, needle: string) =>
  haystack.toLowerCase().includes(needle.toLowerCase());

const parseDateValue = (value: string) => {
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? null : parsed;
};

const parseNumberValue = (value: string) => {
  const num = Number(value);
  return Number.isNaN(num) ? null : num;
};

export default function QuestionsTable({ rows }: QuestionsTableProps) {
  const [selectedIso3, setSelectedIso3] = useState<string[]>([]);
  const [countryNameQuery, setCountryNameQuery] = useState("");
  const [questionQuery, setQuestionQuery] = useState("");
  const [selectedHazards, setSelectedHazards] = useState<string[]>([]);
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>([]);
  const [selectedStatuses, setSelectedStatuses] = useState<string[]>([]);
  const [forecastDateFrom, setForecastDateFrom] = useState("");
  const [forecastDateTo, setForecastDateTo] = useState("");
  const [firstMonthFrom, setFirstMonthFrom] = useState("");
  const [firstMonthTo, setFirstMonthTo] = useState("");
  const [lastMonthFrom, setLastMonthFrom] = useState("");
  const [lastMonthTo, setLastMonthTo] = useState("");
  const [eivMin, setEivMin] = useState("");
  const [eivMax, setEivMax] = useState("");

  const iso3Options = useMemo(
    () => Array.from(new Set(rows.map((row) => row.iso3).filter(Boolean))).sort(),
    [rows]
  );
  const hazardOptions = useMemo(
    () =>
      Array.from(new Set(rows.map((row) => row.hazard_code).filter(Boolean))).sort(),
    [rows]
  );
  const metricOptions = useMemo(
    () => Array.from(new Set(rows.map((row) => row.metric).filter(Boolean))).sort(),
    [rows]
  );
  const statusOptions = useMemo(
    () =>
      Array.from(new Set(rows.map((row) => row.status ?? "").filter(Boolean))).sort(),
    [rows]
  );

  const filteredRows = useMemo(() => {
    const iso3Set = new Set(selectedIso3);
    const hazardSet = new Set(selectedHazards);
    const metricSet = new Set(selectedMetrics);
    const statusSet = new Set(selectedStatuses);
    const forecastFromValue = forecastDateFrom ? parseDateValue(forecastDateFrom) : null;
    const forecastToValue = forecastDateTo ? parseDateValue(forecastDateTo) : null;
    const eivMinValue = eivMin ? parseNumberValue(eivMin) : null;
    const eivMaxValue = eivMax ? parseNumberValue(eivMax) : null;

    return rows.filter((row) => {
      if (iso3Set.size > 0 && !iso3Set.has(row.iso3)) return false;
      if (countryNameQuery) {
        const name = row.country_name ?? "";
        if (!contains(name, countryNameQuery)) return false;
      }
      if (questionQuery) {
        const questionText = row.wording ?? "";
        const idText = row.question_id ?? "";
        if (!contains(questionText, questionQuery) && !contains(idText, questionQuery)) {
          return false;
        }
      }
      if (hazardSet.size > 0 && !hazardSet.has(row.hazard_code)) return false;
      if (metricSet.size > 0 && !metricSet.has(row.metric)) return false;
      if (statusSet.size > 0 && !statusSet.has(row.status ?? "")) return false;

      if (forecastFromValue != null || forecastToValue != null) {
        if (!row.forecast_date) return false;
        const rowForecast = parseDateValue(row.forecast_date);
        if (rowForecast == null) return false;
        if (forecastFromValue != null && rowForecast < forecastFromValue) return false;
        if (forecastToValue != null && rowForecast > forecastToValue) return false;
      }

      if (firstMonthFrom || firstMonthTo) {
        const month = row.first_forecast_month ?? "";
        if (!month) return false;
        if (firstMonthFrom && month < firstMonthFrom) return false;
        if (firstMonthTo && month > firstMonthTo) return false;
      }

      if (lastMonthFrom || lastMonthTo) {
        const month = row.last_forecast_month ?? "";
        if (!month) return false;
        if (lastMonthFrom && month < lastMonthFrom) return false;
        if (lastMonthTo && month > lastMonthTo) return false;
      }

      if (eivMinValue != null || eivMaxValue != null) {
        if (row.eiv_total == null) return false;
        const eiv = row.eiv_total;
        if (eivMinValue != null && eiv < eivMinValue) return false;
        if (eivMaxValue != null && eiv > eivMaxValue) return false;
      }

      return true;
    });
  }, [
    countryNameQuery,
    eivMax,
    eivMin,
    firstMonthFrom,
    firstMonthTo,
    forecastDateFrom,
    forecastDateTo,
    lastMonthFrom,
    lastMonthTo,
    questionQuery,
    rows,
    selectedHazards,
    selectedIso3,
    selectedMetrics,
    selectedStatuses,
  ]);

  const clearFilters = () => {
    setSelectedIso3([]);
    setCountryNameQuery("");
    setQuestionQuery("");
    setSelectedHazards([]);
    setSelectedMetrics([]);
    setSelectedStatuses([]);
    setForecastDateFrom("");
    setForecastDateTo("");
    setFirstMonthFrom("");
    setFirstMonthTo("");
    setLastMonthFrom("");
    setLastMonthTo("");
    setEivMin("");
    setEivMax("");
  };

  const columns = useMemo<Array<SortableColumn<QuestionsRow>>>(
    () => [
      {
        key: "iso3",
        label: "ISO3",
        headerClassName: "text-left",
        cellClassName: "text-left",
        sortValue: (row) => row.iso3,
        render: (row) => (
          <Link
            href={`/countries/${row.iso3}`}
            className="block w-full px-0 py-0 text-sky-300 underline underline-offset-2 hover:text-sky-200"
          >
            {row.iso3}
          </Link>
        ),
        defaultSortDirection: "asc",
      },
      {
        key: "country_name",
        label: "Name",
        headerClassName: "text-left",
        cellClassName: "text-left",
        sortValue: (row) => row.country_name ?? "",
        render: (row) => row.country_name ?? "",
        defaultSortDirection: "asc",
      },
      {
        key: "question",
        label: "Question",
        headerClassName: "text-left",
        cellClassName: "text-left",
        sortValue: (row) => row.wording ?? row.question_id,
        render: (row) => (
          <div>
            <div className="font-medium text-white">{row.wording}</div>
            <Link
              href={`/questions/${row.question_id}?hs_run_id=${encodeURIComponent(
                row.hs_run_id ?? ""
              )}`}
              className="text-sky-300 underline underline-offset-2 hover:text-sky-200"
            >
              {row.question_id}
            </Link>
          </div>
        ),
        defaultSortDirection: "asc",
      },
      {
        key: "hazard_code",
        label: "Hazard",
        headerClassName: "text-left",
        cellClassName: "text-left",
        sortValue: (row) => row.hazard_code,
        render: (row) => row.hazard_code,
        defaultSortDirection: "asc",
      },
      {
        key: "metric",
        label: "Metric",
        headerClassName: "text-left",
        cellClassName: "text-left",
        sortValue: (row) => row.metric,
        render: (row) => row.metric,
        defaultSortDirection: "asc",
      },
      {
        key: "forecast_date",
        label: "Forecast Date",
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) =>
          row.forecast_date ? Date.parse(row.forecast_date) : null,
        render: (row) => row.forecast_date ?? "",
        defaultSortDirection: "desc",
      },
      {
        key: "first_forecast_month",
        label: "First Forecast Month",
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) => parseYearMonth(row.first_forecast_month),
        render: (row) => row.first_forecast_month ?? "",
        defaultSortDirection: "desc",
      },
      {
        key: "last_forecast_month",
        label: "Last Forecast Month",
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) => parseYearMonth(row.last_forecast_month),
        render: (row) => row.last_forecast_month ?? "",
        defaultSortDirection: "desc",
      },
      {
        key: "status",
        label: "Status",
        headerClassName: "text-left",
        cellClassName: "text-left",
        sortValue: (row) => row.status ?? "",
        render: (row) => row.status ?? "",
        defaultSortDirection: "asc",
      },
      {
        key: "eiv_total",
        label: "EIV",
        headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (row) => row.eiv_total ?? null,
        render: (row) =>
          row.eiv_total != null ? Math.round(row.eiv_total).toLocaleString() : "",
        defaultSortDirection: "desc",
      },
    ],
    []
  );

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="grid w-full grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <label className="flex flex-col gap-1 text-xs text-slate-400">
            ISO3
            <select
              className="rounded border border-slate-800 bg-slate-950 px-2 py-1 text-xs text-slate-200"
              multiple
              value={selectedIso3}
              onChange={(event) =>
                setSelectedIso3(
                  Array.from(event.target.selectedOptions).map((option) => option.value)
                )
              }
            >
              {iso3Options.map((iso3) => (
                <option key={iso3} value={iso3}>
                  {iso3}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-xs text-slate-400">
            Name
            <input
              className="rounded border border-slate-800 bg-slate-950 px-2 py-1 text-xs text-slate-200"
              placeholder="Search name"
              type="text"
              value={countryNameQuery}
              onChange={(event) => setCountryNameQuery(event.target.value)}
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-slate-400">
            Question
            <input
              className="rounded border border-slate-800 bg-slate-950 px-2 py-1 text-xs text-slate-200"
              placeholder="Search text or ID"
              type="text"
              value={questionQuery}
              onChange={(event) => setQuestionQuery(event.target.value)}
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-slate-400">
            Hazard
            <select
              className="rounded border border-slate-800 bg-slate-950 px-2 py-1 text-xs text-slate-200"
              multiple
              value={selectedHazards}
              onChange={(event) =>
                setSelectedHazards(
                  Array.from(event.target.selectedOptions).map((option) => option.value)
                )
              }
            >
              {hazardOptions.map((hazard) => (
                <option key={hazard} value={hazard}>
                  {hazard}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-xs text-slate-400">
            Metric
            <select
              className="rounded border border-slate-800 bg-slate-950 px-2 py-1 text-xs text-slate-200"
              multiple
              value={selectedMetrics}
              onChange={(event) =>
                setSelectedMetrics(
                  Array.from(event.target.selectedOptions).map((option) => option.value)
                )
              }
            >
              {metricOptions.map((metric) => (
                <option key={metric} value={metric}>
                  {metric}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-xs text-slate-400">
            Status
            <select
              className="rounded border border-slate-800 bg-slate-950 px-2 py-1 text-xs text-slate-200"
              multiple
              value={selectedStatuses}
              onChange={(event) =>
                setSelectedStatuses(
                  Array.from(event.target.selectedOptions).map((option) => option.value)
                )
              }
            >
              {statusOptions.map((status) => (
                <option key={status} value={status}>
                  {status}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-xs text-slate-400">
            Forecast Date From
            <input
              className="rounded border border-slate-800 bg-slate-950 px-2 py-1 text-xs text-slate-200"
              type="date"
              value={forecastDateFrom}
              onChange={(event) => setForecastDateFrom(event.target.value)}
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-slate-400">
            Forecast Date To
            <input
              className="rounded border border-slate-800 bg-slate-950 px-2 py-1 text-xs text-slate-200"
              type="date"
              value={forecastDateTo}
              onChange={(event) => setForecastDateTo(event.target.value)}
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-slate-400">
            First Forecast Month From
            <input
              className="rounded border border-slate-800 bg-slate-950 px-2 py-1 text-xs text-slate-200"
              type="month"
              value={firstMonthFrom}
              onChange={(event) => setFirstMonthFrom(event.target.value)}
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-slate-400">
            First Forecast Month To
            <input
              className="rounded border border-slate-800 bg-slate-950 px-2 py-1 text-xs text-slate-200"
              type="month"
              value={firstMonthTo}
              onChange={(event) => setFirstMonthTo(event.target.value)}
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-slate-400">
            Last Forecast Month From
            <input
              className="rounded border border-slate-800 bg-slate-950 px-2 py-1 text-xs text-slate-200"
              type="month"
              value={lastMonthFrom}
              onChange={(event) => setLastMonthFrom(event.target.value)}
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-slate-400">
            Last Forecast Month To
            <input
              className="rounded border border-slate-800 bg-slate-950 px-2 py-1 text-xs text-slate-200"
              type="month"
              value={lastMonthTo}
              onChange={(event) => setLastMonthTo(event.target.value)}
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-slate-400">
            EIV Min
            <input
              className="rounded border border-slate-800 bg-slate-950 px-2 py-1 text-xs text-slate-200"
              min="0"
              step="1"
              type="number"
              value={eivMin}
              onChange={(event) => setEivMin(event.target.value)}
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-slate-400">
            EIV Max
            <input
              className="rounded border border-slate-800 bg-slate-950 px-2 py-1 text-xs text-slate-200"
              min="0"
              step="1"
              type="number"
              value={eivMax}
              onChange={(event) => setEivMax(event.target.value)}
            />
          </label>
        </div>
        <div className="flex flex-col items-end gap-2">
          <button
            className="rounded border border-slate-700 px-3 py-2 text-xs text-slate-200 hover:text-white"
            type="button"
            onClick={clearFilters}
          >
            Clear all Filters
          </button>
          <div className="text-xs text-slate-400">
            Showing {filteredRows.length.toLocaleString()} of{" "}
            {rows.length.toLocaleString()} questions
          </div>
        </div>
      </div>
      <SortableTable
        columns={columns}
        emptyMessage="No questions available. No data in DB snapshot."
        rowKey={(row) => row.question_id}
        rows={filteredRows}
        initialSortKey="eiv_total"
        initialSortDirection="desc"
      />
    </div>
  );
}
