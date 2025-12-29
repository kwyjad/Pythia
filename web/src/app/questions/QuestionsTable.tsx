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

const contains = (haystack: string, needle: string) =>
  haystack.toLowerCase().includes(needle.toLowerCase());

const parseMmYYYY = (value: string) => {
  const match = value.match(/^(\d{2})-(\d{4})$/);
  if (!match) return null;
  const month = Number(match[1]);
  const year = Number(match[2]);
  if (!Number.isFinite(year) || !Number.isFinite(month)) return null;
  if (month < 1 || month > 12) return null;
  return year * 12 + month;
};

const rowForecastMonthKey = (value?: string | null) => {
  if (!value) return null;
  const match = value.match(/^(\d{4})-(\d{2})-\d{2}$/);
  if (!match) return null;
  const year = Number(match[1]);
  const month = Number(match[2]);
  if (!Number.isFinite(year) || !Number.isFinite(month)) return null;
  if (month < 1 || month > 12) return null;
  return year * 12 + month;
};

const rowYearMonthKey = (value?: string | null) => {
  if (!value) return null;
  const match = value.match(/^(\d{4})-(\d{2})$/);
  if (!match) return null;
  const year = Number(match[1]);
  const month = Number(match[2]);
  if (!Number.isFinite(year) || !Number.isFinite(month)) return null;
  if (month < 1 || month > 12) return null;
  return year * 12 + month;
};

const parseNumberValue = (value: string) => {
  const num = Number(value);
  return Number.isNaN(num) ? null : num;
};

const EIV_TOOLTIP =
  "EIV = sum over months 1–6 of (sum over buckets of p(bucket, month) × centroid(bucket)). Uses PA centroids for PA questions and fatalities centroids for FATALITIES questions.";

const renderEivHeader = (label: string) => (
  <span className="inline-flex items-center gap-1">
    {label}
    <span
      className="rounded-full border border-slate-500 px-1 text-xs text-slate-300"
      title={EIV_TOOLTIP}
    >
      ?
    </span>
  </span>
);

export default function QuestionsTable({ rows }: QuestionsTableProps) {
  const [selectedCountries, setSelectedCountries] = useState<string[]>([]);
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

  const countryOptions = useMemo(
    () =>
      Array.from(
        new Set(
          rows
            .map((row) => row.country_name ?? row.iso3)
            .map((value) => value ?? "")
            .filter(Boolean)
        )
      ).sort(),
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
    const countrySet = new Set(selectedCountries);
    const hazardSet = new Set(selectedHazards);
    const metricSet = new Set(selectedMetrics);
    const statusSet = new Set(selectedStatuses);
    const forecastFromValue = forecastDateFrom ? parseMmYYYY(forecastDateFrom) : null;
    const forecastToValue = forecastDateTo ? parseMmYYYY(forecastDateTo) : null;
    const firstMonthFromValue = firstMonthFrom ? parseMmYYYY(firstMonthFrom) : null;
    const firstMonthToValue = firstMonthTo ? parseMmYYYY(firstMonthTo) : null;
    const lastMonthFromValue = lastMonthFrom ? parseMmYYYY(lastMonthFrom) : null;
    const lastMonthToValue = lastMonthTo ? parseMmYYYY(lastMonthTo) : null;
    const eivMinValue = eivMin ? parseNumberValue(eivMin) : null;
    const eivMaxValue = eivMax ? parseNumberValue(eivMax) : null;

    return rows.filter((row) => {
      if (countrySet.size > 0) {
        const countryValue = row.country_name ?? row.iso3;
        if (!countryValue || !countrySet.has(countryValue)) return false;
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
        const rowForecast = rowForecastMonthKey(row.forecast_date);
        if (rowForecast == null) return false;
        if (forecastFromValue != null && rowForecast < forecastFromValue) return false;
        if (forecastToValue != null && rowForecast > forecastToValue) return false;
      }

      if (firstMonthFromValue != null || firstMonthToValue != null) {
        const month = rowYearMonthKey(row.first_forecast_month);
        if (month == null) return false;
        if (firstMonthFromValue != null && month < firstMonthFromValue) return false;
        if (firstMonthToValue != null && month > firstMonthToValue) return false;
      }

      if (lastMonthFromValue != null || lastMonthToValue != null) {
        const month = rowYearMonthKey(row.last_forecast_month);
        if (month == null) return false;
        if (lastMonthFromValue != null && month < lastMonthFromValue) return false;
        if (lastMonthToValue != null && month > lastMonthToValue) return false;
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
    selectedCountries,
    selectedMetrics,
    selectedStatuses,
  ]);

  const clearFilters = () => {
    setSelectedCountries([]);
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
        headerClassName: "text-left whitespace-nowrap",
        cellClassName: "text-left whitespace-nowrap",
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
        label: "Country",
        headerClassName: "text-left whitespace-nowrap",
        cellClassName: "text-left whitespace-nowrap",
        sortValue: (row) => row.country_name ?? "",
        render: (row) => row.country_name ?? "",
        defaultSortDirection: "asc",
      },
      {
        key: "question",
        label: "Question",
        headerClassName: "text-left",
        cellClassName: "text-left min-w-[40ch] align-top",
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
        headerClassName: "text-left whitespace-nowrap",
        cellClassName: "text-left whitespace-nowrap",
        sortValue: (row) => row.hazard_code,
        render: (row) => row.hazard_code,
        defaultSortDirection: "asc",
      },
      {
        key: "metric",
        label: "Metric",
        headerClassName: "text-left whitespace-nowrap",
        cellClassName: "text-left whitespace-nowrap",
        sortValue: (row) => row.metric,
        render: (row) => row.metric,
        defaultSortDirection: "asc",
      },
      {
        key: "forecast_date",
        label: "Forecast Date",
        headerClassName: "text-right whitespace-nowrap",
        cellClassName: "text-right tabular-nums whitespace-nowrap",
        sortValue: (row) => rowForecastMonthKey(row.forecast_date),
        render: (row) => row.forecast_date ?? "",
        defaultSortDirection: "desc",
      },
      {
        key: "first_forecast_month",
        label: "First Forecast Month",
        headerClassName: "text-right whitespace-nowrap",
        cellClassName: "text-right tabular-nums whitespace-nowrap",
        sortValue: (row) => rowYearMonthKey(row.first_forecast_month),
        render: (row) => row.first_forecast_month ?? "",
        defaultSortDirection: "desc",
      },
      {
        key: "last_forecast_month",
        label: "Last Forecast Month",
        headerClassName: "text-right whitespace-nowrap",
        cellClassName: "text-right tabular-nums whitespace-nowrap",
        sortValue: (row) => rowYearMonthKey(row.last_forecast_month),
        render: (row) => row.last_forecast_month ?? "",
        defaultSortDirection: "desc",
      },
      {
        key: "status",
        label: "Status",
        headerClassName: "text-left whitespace-nowrap",
        cellClassName: "text-left whitespace-nowrap",
        sortValue: (row) => row.status ?? "",
        render: (row) => row.status ?? "",
        defaultSortDirection: "asc",
      },
      {
        key: "eiv_total",
        label: renderEivHeader("EIV"),
        headerClassName: "text-right whitespace-nowrap",
        cellClassName: "text-right tabular-nums whitespace-nowrap",
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
        <div className="flex w-full flex-col gap-3">
          <div className="flex flex-wrap items-end gap-3 lg:flex-nowrap">
            <label className="flex w-[15ch] flex-col gap-1 text-xs text-slate-400">
              Country
              <select
                className="rounded border border-slate-800 bg-slate-950 px-2 py-1 text-xs text-slate-200"
                multiple
                value={selectedCountries}
                onChange={(event) =>
                  setSelectedCountries(
                    Array.from(event.target.selectedOptions).map(
                      (option) => option.value
                    )
                  )
                }
              >
                {countryOptions.map((country) => (
                  <option key={country} value={country}>
                    {country}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1 text-xs text-slate-400">
              Hazard
              <select
                className="rounded border border-slate-800 bg-slate-950 px-2 py-1 text-xs text-slate-200"
                multiple
                value={selectedHazards}
                onChange={(event) =>
                  setSelectedHazards(
                    Array.from(event.target.selectedOptions).map(
                      (option) => option.value
                    )
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
                    Array.from(event.target.selectedOptions).map(
                      (option) => option.value
                    )
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
                    Array.from(event.target.selectedOptions).map(
                      (option) => option.value
                    )
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
            <label className="flex w-[25ch] flex-col gap-1 text-xs text-slate-400">
              Question
              <input
                className="rounded border border-slate-800 bg-slate-950 px-2 py-1 text-xs text-slate-200"
                placeholder="Search text or ID"
                type="text"
                value={questionQuery}
                onChange={(event) => setQuestionQuery(event.target.value)}
              />
            </label>
            <label className="flex w-[10ch] flex-col gap-1 text-xs text-slate-400">
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
            <label className="flex w-[10ch] flex-col gap-1 text-xs text-slate-400">
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
          <div className="flex flex-wrap gap-6">
            <div className="flex flex-col gap-3">
              <label className="flex flex-col gap-1 text-xs text-slate-400">
                Forecast Date From
                <input
                  className="rounded border border-slate-800 bg-slate-950 px-2 py-1 text-xs text-slate-200"
                  inputMode="numeric"
                  pattern="\\d{2}-\\d{4}"
                  placeholder="mm-yyyy"
                  type="text"
                  value={forecastDateFrom}
                  onChange={(event) => setForecastDateFrom(event.target.value)}
                />
              </label>
              <label className="flex flex-col gap-1 text-xs text-slate-400">
                Forecast Date To
                <input
                  className="rounded border border-slate-800 bg-slate-950 px-2 py-1 text-xs text-slate-200"
                  inputMode="numeric"
                  pattern="\\d{2}-\\d{4}"
                  placeholder="mm-yyyy"
                  type="text"
                  value={forecastDateTo}
                  onChange={(event) => setForecastDateTo(event.target.value)}
                />
              </label>
            </div>
            <div className="flex flex-col gap-3">
              <label className="flex flex-col gap-1 text-xs text-slate-400">
                First Forecast Month From
                <input
                  className="rounded border border-slate-800 bg-slate-950 px-2 py-1 text-xs text-slate-200"
                  inputMode="numeric"
                  pattern="\\d{2}-\\d{4}"
                  placeholder="mm-yyyy"
                  type="text"
                  value={firstMonthFrom}
                  onChange={(event) => setFirstMonthFrom(event.target.value)}
                />
              </label>
              <label className="flex flex-col gap-1 text-xs text-slate-400">
                First Forecast Month To
                <input
                  className="rounded border border-slate-800 bg-slate-950 px-2 py-1 text-xs text-slate-200"
                  inputMode="numeric"
                  pattern="\\d{2}-\\d{4}"
                  placeholder="mm-yyyy"
                  type="text"
                  value={firstMonthTo}
                  onChange={(event) => setFirstMonthTo(event.target.value)}
                />
              </label>
            </div>
            <div className="flex flex-col gap-3">
              <label className="flex flex-col gap-1 text-xs text-slate-400">
                Last Forecast Month From
                <input
                  className="rounded border border-slate-800 bg-slate-950 px-2 py-1 text-xs text-slate-200"
                  inputMode="numeric"
                  pattern="\\d{2}-\\d{4}"
                  placeholder="mm-yyyy"
                  type="text"
                  value={lastMonthFrom}
                  onChange={(event) => setLastMonthFrom(event.target.value)}
                />
              </label>
              <label className="flex flex-col gap-1 text-xs text-slate-400">
                Last Forecast Month To
                <input
                  className="rounded border border-slate-800 bg-slate-950 px-2 py-1 text-xs text-slate-200"
                  inputMode="numeric"
                  pattern="\\d{2}-\\d{4}"
                  placeholder="mm-yyyy"
                  type="text"
                  value={lastMonthTo}
                  onChange={(event) => setLastMonthTo(event.target.value)}
                />
              </label>
            </div>
          </div>
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
            {rows.length.toLocaleString()} forecasts
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
        tableLayout="auto"
        dense
      />
    </div>
  );
}
