"use client";

import Link from "next/link";
import type { ReactNode } from "react";
import { useEffect, useMemo, useState } from "react";

import InfoTooltip from "../../components/InfoTooltip";
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
  triage_score?: number | null;
  triage_tier?: string | null;
  triage_need_full_spd?: boolean | null;
  regime_change_score?: number | null;
  track?: number | null;
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
  "PA uses 6-month EIV as the peak month (not a sum). FATALITIES uses 6-month cumulative expected deaths. Values come from sum over buckets of p(bucket, month) × centroid(bucket).";

const TRIAGE_TOOLTIP =
  "HS triage_score (0–1) estimates risk of unusually high recorded impact in the next 1–6 months using evidence + base-rate signals. Scores map to tiers (priority ≥ 0.50, quiet < 0.50). Priority hazards enter the forecasting pipeline; quiet hazards may appear but have no EIV.";
const RC_SCORE_TOOLTIP = "Regime change score (probability × magnitude).";

const renderHeaderClamp = (content: ReactNode) => (
  <div className="max-h-[3.6em] overflow-hidden whitespace-normal leading-tight">
    {content}
  </div>
);

const renderEivHeader = (label: ReactNode) => (
  <span className="inline-flex items-start gap-1">
    {label}
    <span
      className="rounded-full border border-fred-secondary px-1 text-xs text-fred-secondary"
      title={EIV_TOOLTIP}
    >
      ?
    </span>
  </span>
);

const renderTriageHeader = (label: string) => (
  <span className="inline-flex items-center gap-1">
    {label}
    <InfoTooltip text={TRIAGE_TOOLTIP} />
  </span>
);

const renderRcHeader = (label: string) => (
  <span className="inline-flex items-center gap-1">
    {label}
    <InfoTooltip text={RC_SCORE_TOOLTIP} />
  </span>
);

export default function QuestionsTable({ rows }: QuestionsTableProps) {
  const [selectedCountries, setSelectedCountries] = useState<string[]>([]);
  const [questionQuery, setQuestionQuery] = useState("");
  const [selectedHazards, setSelectedHazards] = useState<string[]>([]);
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>([]);
  const [selectedStatuses, setSelectedStatuses] = useState<string[]>([]);
  const [selectedTracks, setSelectedTracks] = useState<string[]>([]);
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
  const trackOptions = useMemo(
    () =>
      Array.from(
        new Set(
          rows
            .map((row) => (row.track != null ? `Track ${row.track}` : ""))
            .filter(Boolean),
        ),
      ).sort(),
    [rows],
  );

  const filteredRows = useMemo(() => {
    const countrySet = new Set(selectedCountries);
    const hazardSet = new Set(selectedHazards);
    const metricSet = new Set(selectedMetrics);
    const statusSet = new Set(selectedStatuses);
    const trackSet = new Set(selectedTracks);
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
      if (trackSet.size > 0) {
        const trackLabel = row.track != null ? `Track ${row.track}` : "";
        if (!trackSet.has(trackLabel)) return false;
      }

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
    selectedTracks,
  ]);

  const clearFilters = () => {
    setSelectedCountries([]);
    setQuestionQuery("");
    setSelectedHazards([]);
    setSelectedMetrics([]);
    setSelectedStatuses([]);
    setSelectedTracks([]);
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
        label: renderHeaderClamp("ISO3"),
        headerClassName: "text-left whitespace-normal leading-tight",
        cellClassName: "text-left whitespace-nowrap",
        sortValue: (row) => row.iso3,
        render: (row) => (
          <Link
            href={`/countries/${row.iso3}`}
            className="block w-full px-0 py-0 text-fred-primary underline underline-offset-2 hover:text-fred-secondary"
          >
            {row.iso3}
          </Link>
        ),
        defaultSortDirection: "asc",
      },
      {
        key: "country_name",
        label: renderHeaderClamp("Country"),
        headerClassName: "text-left whitespace-normal leading-tight",
        cellClassName: "text-left",
        sortValue: (row) => row.country_name ?? "",
        render: (row) => row.country_name ?? "",
        defaultSortDirection: "asc",
      },
      {
        key: "question",
        label: renderHeaderClamp("Question"),
        headerClassName: "text-left whitespace-normal leading-tight",
        cellClassName: "text-left align-top",
        sortValue: (row) => row.wording ?? row.question_id,
        render: (row) => (
          <div>
            <div className="max-h-[4.2em] max-w-full overflow-hidden whitespace-normal break-words font-medium leading-tight text-fred-text">
              {row.wording}
            </div>
            <Link
              href={`/questions/${row.question_id}?hs_run_id=${encodeURIComponent(
                row.hs_run_id ?? ""
              )}`}
              className="text-xs text-fred-primary underline underline-offset-2 hover:text-fred-secondary"
            >
              {row.question_id}
            </Link>
          </div>
        ),
        defaultSortDirection: "asc",
      },
      {
        key: "hazard_code",
        label: renderHeaderClamp("Hazard"),
        headerClassName: "text-left whitespace-normal leading-tight",
        cellClassName: "text-left whitespace-nowrap",
        sortValue: (row) => row.hazard_code,
        render: (row) => row.hazard_code,
        defaultSortDirection: "asc",
      },
      {
        key: "metric",
        label: renderHeaderClamp("Metric"),
        headerClassName: "text-left whitespace-normal leading-tight",
        cellClassName: "text-left whitespace-nowrap",
        sortValue: (row) => row.metric,
        render: (row) => row.metric,
        defaultSortDirection: "asc",
      },
      {
        key: "forecast_date",
        label: renderHeaderClamp("Forecast Date"),
        headerClassName: "text-left whitespace-normal leading-tight",
        cellClassName: "text-left tabular-nums whitespace-nowrap",
        sortValue: (row) => rowForecastMonthKey(row.forecast_date),
        render: (row) => row.forecast_date ?? "",
        defaultSortDirection: "desc",
      },
      {
        key: "first_forecast_month",
        label: renderHeaderClamp("First Forecast Month"),
        headerClassName: "text-left whitespace-normal leading-tight",
        cellClassName: "text-left tabular-nums whitespace-nowrap",
        sortValue: (row) => rowYearMonthKey(row.first_forecast_month),
        render: (row) => row.first_forecast_month ?? "",
        defaultSortDirection: "desc",
      },
      {
        key: "last_forecast_month",
        label: renderHeaderClamp("Last Forecast Month"),
        headerClassName: "text-left whitespace-normal leading-tight",
        cellClassName: "text-left tabular-nums whitespace-nowrap",
        sortValue: (row) => rowYearMonthKey(row.last_forecast_month),
        render: (row) => row.last_forecast_month ?? "",
        defaultSortDirection: "desc",
      },
      {
        key: "status",
        label: renderHeaderClamp("Status"),
        headerClassName: "text-left whitespace-normal leading-tight",
        cellClassName: "text-left whitespace-nowrap",
        sortValue: (row) => row.status ?? "",
        render: (row) => row.status ?? "",
        defaultSortDirection: "asc",
      },
      {
        key: "triage_score",
        label: renderHeaderClamp(renderTriageHeader("Triage Score")),
        headerClassName: "text-right whitespace-normal leading-tight",
        cellClassName: "text-right tabular-nums whitespace-nowrap",
        sortValue: (row) => row.triage_score ?? null,
        render: (row) =>
          row.triage_score != null ? row.triage_score.toFixed(2) : "",
        defaultSortDirection: "desc",
      },
      {
        key: "regime_change_score",
        label: renderHeaderClamp(renderRcHeader("RC Score")),
        headerClassName: "text-right whitespace-normal leading-tight",
        cellClassName: "text-right tabular-nums whitespace-nowrap",
        sortValue: (row) => row.regime_change_score ?? null,
        render: (row) =>
          row.regime_change_score != null
            ? row.regime_change_score.toFixed(2)
            : "",
        defaultSortDirection: "desc",
      },
      {
        key: "track",
        label: "Track",
        headerClassName: "text-center whitespace-nowrap",
        cellClassName: "text-center whitespace-nowrap",
        sortValue: (row) => row.track ?? 0,
        render: (row) =>
          row.track != null ? `Track ${row.track}` : "\u2014",
        defaultSortDirection: "asc",
      },
      {
        key: "eiv_total",
        label: renderHeaderClamp(
          renderEivHeader(
            <span className="flex flex-col">
              <span>6-Month EIV</span>
              <span className="text-[11px] text-fred-muted">
                / cumulative expected deaths
              </span>
            </span>
          )
        ),
        headerClassName: "text-right whitespace-normal leading-tight",
        cellClassName: "text-right tabular-nums whitespace-nowrap",
        sortValue: (row) => row.eiv_total ?? null,
        render: (row) =>
          row.eiv_total != null ? Math.round(row.eiv_total).toLocaleString() : "",
        defaultSortDirection: "desc",
      },
    ],
    []
  );

  const columnWidths = useMemo(
    () => [
      "5ch",
      "12ch",
      "40ch",
      "6ch",
      "7ch",
      "10ch",
      "10ch",
      "10ch",
      "7ch",
      "7ch",
      "7ch",
      "7ch",
      "12ch",
    ],
    []
  );

  useEffect(() => {
    if (typeof window === "undefined") return;
    const debugEnabled =
      new URLSearchParams(window.location.search).get("debug_table") === "1";
    if (!debugEnabled) return;
    const columnConfig = columns.map((column, index) => ({
      key: column.key,
      width: columnWidths[index] ?? "auto",
    }));
    console.log("[QuestionsTable] column config", columnConfig);
  }, [columns, columnWidths]);

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="flex w-full flex-col gap-3">
          <div className="flex flex-wrap items-start gap-3 lg:flex-nowrap">
            <label className="flex w-[15ch] flex-col gap-1 text-xs text-fred-text">
              Country
              <select
                className="rounded border border-fred-secondary bg-fred-surface px-2 py-1 text-xs text-fred-text"
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
            <label className="flex flex-col gap-1 text-xs text-fred-text">
              Hazard
              <select
                className="rounded border border-fred-secondary bg-fred-surface px-2 py-1 text-xs text-fred-text"
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
            <label className="flex flex-col gap-1 text-xs text-fred-text">
              Metric
              <select
                className="rounded border border-fred-secondary bg-fred-surface px-2 py-1 text-xs text-fred-text"
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
            <label className="flex flex-col gap-1 text-xs text-fred-text">
              Status
              <select
                className="rounded border border-fred-secondary bg-fred-surface px-2 py-1 text-xs text-fred-text"
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
            <label className="flex flex-col gap-1 text-xs text-fred-text">
              Track
              <select
                className="rounded border border-fred-secondary bg-fred-surface px-2 py-1 text-xs text-fred-text"
                multiple
                value={selectedTracks}
                onChange={(event) =>
                  setSelectedTracks(
                    Array.from(event.target.selectedOptions).map(
                      (option) => option.value
                    )
                  )
                }
              >
                {trackOptions.map((track) => (
                  <option key={track} value={track}>
                    {track}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex w-[25ch] flex-col gap-1 text-xs text-fred-text">
              Question
              <input
                className="rounded border border-fred-secondary bg-fred-surface px-2 py-1 text-xs text-fred-text"
                placeholder="Search text or ID"
                type="text"
                value={questionQuery}
                onChange={(event) => setQuestionQuery(event.target.value)}
              />
            </label>
            <label className="flex w-[10ch] flex-col gap-1 text-xs text-fred-text">
              EIV Min
              <input
                className="rounded border border-fred-secondary bg-fred-surface px-2 py-1 text-xs text-fred-text"
                min="0"
                step="1"
                type="number"
                value={eivMin}
                onChange={(event) => setEivMin(event.target.value)}
              />
            </label>
            <label className="flex w-[10ch] flex-col gap-1 text-xs text-fred-text">
              EIV Max
              <input
                className="rounded border border-fred-secondary bg-fred-surface px-2 py-1 text-xs text-fred-text"
                min="0"
                step="1"
                type="number"
                value={eivMax}
                onChange={(event) => setEivMax(event.target.value)}
              />
            </label>
            <div className="grid grid-cols-3 gap-3">
              <div className="flex flex-col gap-3">
                <label className="flex flex-col gap-1 text-xs text-fred-text">
                  Forecast Date From
                  <input
                    className="w-[10ch] rounded border border-fred-secondary bg-fred-surface px-2 py-1 text-xs text-fred-text"
                    inputMode="numeric"
                    pattern="\\d{2}-\\d{4}"
                    placeholder="mm-yyyy"
                    type="text"
                    value={forecastDateFrom}
                    onChange={(event) => setForecastDateFrom(event.target.value)}
                  />
                </label>
                <label className="flex flex-col gap-1 text-xs text-fred-text">
                  Forecast Date To
                  <input
                    className="w-[10ch] rounded border border-fred-secondary bg-fred-surface px-2 py-1 text-xs text-fred-text"
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
                <label className="flex flex-col gap-1 text-xs text-fred-text">
                  First Forecast Month From
                  <input
                    className="w-[10ch] rounded border border-fred-secondary bg-fred-surface px-2 py-1 text-xs text-fred-text"
                    inputMode="numeric"
                    pattern="\\d{2}-\\d{4}"
                    placeholder="mm-yyyy"
                    type="text"
                    value={firstMonthFrom}
                    onChange={(event) => setFirstMonthFrom(event.target.value)}
                  />
                </label>
                <label className="flex flex-col gap-1 text-xs text-fred-text">
                  First Forecast Month To
                  <input
                    className="w-[10ch] rounded border border-fred-secondary bg-fred-surface px-2 py-1 text-xs text-fred-text"
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
                <label className="flex flex-col gap-1 text-xs text-fred-text">
                  Last Forecast Month From
                  <input
                    className="w-[10ch] rounded border border-fred-secondary bg-fred-surface px-2 py-1 text-xs text-fred-text"
                    inputMode="numeric"
                    pattern="\\d{2}-\\d{4}"
                    placeholder="mm-yyyy"
                    type="text"
                    value={lastMonthFrom}
                    onChange={(event) => setLastMonthFrom(event.target.value)}
                  />
                </label>
                <label className="flex flex-col gap-1 text-xs text-fred-text">
                  Last Forecast Month To
                  <input
                    className="w-[10ch] rounded border border-fred-secondary bg-fred-surface px-2 py-1 text-xs text-fred-text"
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
        </div>
        <div className="flex flex-col items-end gap-2">
          <button
            className="rounded border border-fred-secondary bg-fred-secondary px-3 py-2 text-xs text-fred-surface hover:opacity-90"
            type="button"
            onClick={clearFilters}
          >
            Clear all Filters
          </button>
          <div className="text-xs text-fred-text">
            Showing {filteredRows.length.toLocaleString()} of{" "}
            {rows.length.toLocaleString()} forecasts
          </div>
        </div>
      </div>
      <div className="md:hidden">
        <div className="space-y-3">
          {filteredRows.map((row) => (
            <div
              key={row.question_id}
              className="rounded-lg border border-fred-secondary bg-fred-surface p-4 shadow-fredCard"
            >
              <div className="flex items-center justify-between gap-3 text-xs text-fred-muted">
                <Link
                  href={`/countries/${row.iso3}`}
                  className="text-fred-primary underline underline-offset-2 hover:text-fred-secondary"
                >
                  {row.country_name ?? row.iso3}
                </Link>
                <span>{row.status ?? "Unknown status"}</span>
              </div>
              <div className="mt-2 max-h-[3.2em] overflow-hidden text-sm font-semibold leading-tight text-fred-text">
                {row.wording}
              </div>
              <Link
                href={`/questions/${row.question_id}?hs_run_id=${encodeURIComponent(
                  row.hs_run_id ?? ""
                )}`}
                className="mt-1 block text-xs text-fred-primary underline underline-offset-2 hover:text-fred-secondary"
              >
                {row.question_id}
              </Link>
              <div className="mt-3 grid gap-2 text-xs text-fred-text">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="rounded-full border border-fred-secondary px-2 py-0.5 text-[11px]">
                    {row.hazard_code}
                  </span>
                  <span className="rounded-full border border-fred-secondary px-2 py-0.5 text-[11px]">
                    {row.metric}
                  </span>
                </div>
                <div>
                  <span className="text-fred-muted">Forecast date:</span>{" "}
                  {row.forecast_date ?? "n/a"}
                </div>
                <div>
                  <span className="text-fred-muted">Window:</span>{" "}
                  {row.first_forecast_month ?? "n/a"} →{" "}
                  {row.last_forecast_month ?? "n/a"}
                </div>
                <div>
                  <span className="text-fred-muted">Triage score:</span>{" "}
                  {row.triage_score != null ? row.triage_score.toFixed(2) : "n/a"}
                </div>
                <div>
                  <span className="text-fred-muted">RC score:</span>{" "}
                  {row.regime_change_score != null
                    ? row.regime_change_score.toFixed(2)
                    : "n/a"}
                </div>
                <div>
                  <span className="text-fred-muted">6-Month EIV:</span>{" "}
                  {row.eiv_total != null
                    ? Math.round(row.eiv_total).toLocaleString()
                    : "n/a"}
                </div>
              </div>
            </div>
          ))}
          {filteredRows.length === 0 ? (
            <div className="rounded-lg border border-fred-secondary bg-fred-surface px-4 py-3 text-sm text-fred-muted">
              No questions available. No data in DB snapshot.
            </div>
          ) : null}
        </div>
      </div>
      <div className="hidden md:block overflow-x-hidden">
        <SortableTable
          colGroup={
            <>
              {columnWidths.map((width, index) => (
                <col key={`${width}-${index}`} style={{ width }} />
              ))}
            </>
          }
          columns={columns}
          emptyMessage="No questions available. No data in DB snapshot."
          rowKey={(row) => row.question_id}
          rows={filteredRows}
          initialSortKey="eiv_total"
          initialSortDirection="desc"
          tableLayout="fixed"
          dense
        />
      </div>
    </div>
  );
}
