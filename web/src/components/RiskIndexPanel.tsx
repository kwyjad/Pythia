"use client";

import { useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";

import { apiGet } from "../lib/api";
import type {
  CountriesRow,
  DiagnosticsKpiScopesResponse,
  RiskIndexResponse,
  RiskView,
} from "../lib/types";
import KpiCard from "./KpiCard";
import RiskIndexMap from "./RiskIndexMap";
import RiskIndexTable from "./RiskIndexTable";
import RunMonthSelector from "./RunMonthSelector";

const VIEW_OPTIONS: Array<{ value: RiskView; label: string }> = [
  { value: "PA_EIV", label: "People Affected (PA) EIV" },
  { value: "PA_PC", label: "People Affected (PA) per capita EIV" },
  { value: "FATALITIES_EIV", label: "Armed Conflict (ACE) fatalities EIV" },
  { value: "FATALITIES_PC", label: "Armed Conflict (ACE) fatalities per capita EIV" },
];

type RiskIndexPanelProps = {
  initialResponse: RiskIndexResponse;
  countriesRows: CountriesRow[];
  kpiScopes: DiagnosticsKpiScopesResponse;
  mapHeightClassName?: string;
};

const buildParams = (view: RiskView) => {
  switch (view) {
    case "PA_EIV":
      return { metric: "PA", horizon_m: 6, normalize: false };
    case "PA_PC":
      return { metric: "PA", horizon_m: 6, normalize: true };
    case "FATALITIES_EIV":
      return {
        metric: "FATALITIES",
        hazard_code: "ACE",
        horizon_m: 6,
        normalize: false,
      };
    case "FATALITIES_PC":
      return {
        metric: "FATALITIES",
        hazard_code: "ACE",
        horizon_m: 6,
        normalize: true,
      };
    default:
      return { metric: "PA", horizon_m: 6, normalize: false };
  }
};

const metricScopeForView = (view: RiskView) =>
  view === "FATALITIES_EIV" || view === "FATALITIES_PC" ? "FATALITIES" : "PA";

const addMonthsYYYYMM = (value: string, months: number): string | null => {
  const parts = value.split("-");
  if (parts.length < 2) return null;
  const year = Number(parts[0]);
  const month = Number(parts[1]);
  if (!Number.isFinite(year) || !Number.isFinite(month)) return null;
  if (month < 1 || month > 12) return null;
  const total = year * 12 + (month - 1) + months;
  if (!Number.isFinite(total)) return null;
  const outYear = Math.floor(total / 12);
  const outMonth = (total % 12) + 1;
  return `${String(outYear).padStart(4, "0")}-${String(outMonth).padStart(2, "0")}`;
};

const HAZARD_LABELS: Record<string, string> = {
  ACE: "Armed Conflict",
  FL: "Flood",
  DR: "Drought",
  TC: "Tropical Cyclone",
  DI: "Displacement Inflow",
  EP: "Earthquake",
  HW: "Heatwave",
  LS: "Landslide",
  VO: "Volcano",
  WH: "Wildfire",
};

export default function RiskIndexPanel({
  initialResponse,
  countriesRows,
  kpiScopes,
  mapHeightClassName,
}: RiskIndexPanelProps) {
  const [view, setView] = useState<RiskView>("PA_EIV");
  const [rows, setRows] = useState(initialResponse.rows ?? []);
  const [targetMonth, setTargetMonth] = useState(initialResponse.target_month);
  const [metric, setMetric] = useState(initialResponse.metric);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [kpiData, setKpiData] = useState(kpiScopes);
  const [selectedRunMonth, setSelectedRunMonth] = useState(
    kpiScopes.selected_month
  );
  const [kpiError, setKpiError] = useState<string | null>(null);
  const [runMonthFallback, setRunMonthFallback] = useState(false);
  const searchParams = useSearchParams();
  const showKpiDebug = searchParams?.get("debug_kpi") === "1";

  const isPerCapita = view === "PA_PC" || view === "FATALITIES_PC";

  const fetchKpiScopes = async (
    metricScope: string,
    runMonth: string | null
  ) => {
    setKpiError(null);
    try {
      const response = await apiGet<DiagnosticsKpiScopesResponse>(
        "/diagnostics/kpi_scopes",
        {
          metric_scope: metricScope,
          ...(runMonth ? { year_month: runMonth } : {}),
        }
      );
      setKpiData(response);
      setSelectedRunMonth(response.selected_month);
    } catch (fetchError) {
      console.warn("KPI scopes unavailable:", fetchError);
      setKpiError("KPI scopes unavailable (API error).");
    }
  };

  const handleViewChange = async (nextView: RiskView) => {
    setView(nextView);
    setIsLoading(true);
    setError(null);
    try {
      const forecastTargetMonthForRun = selectedRunMonth
        ? addMonthsYYYYMM(selectedRunMonth, 1)
        : null;
      const response = await apiGet<RiskIndexResponse>(
        "/risk_index",
        {
          ...buildParams(nextView),
          ...(forecastTargetMonthForRun
            ? { target_month: forecastTargetMonthForRun }
            : {}),
        }
      );
      setRows(response.rows ?? []);
      setTargetMonth(response.target_month ?? null);
      setMetric(response.metric);
      setRunMonthFallback(
        Boolean(
          forecastTargetMonthForRun &&
            response.target_month &&
            response.target_month !== forecastTargetMonthForRun
        )
      );
      void fetchKpiScopes(metricScopeForView(nextView), selectedRunMonth);
    } catch (fetchError) {
      console.warn("Risk index unavailable:", fetchError);
      setError("Risk index unavailable (API error).");
      setRunMonthFallback(false);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRunMonthChange = async (yearMonth: string) => {
    setSelectedRunMonth(yearMonth);
    setIsLoading(true);
    setError(null);
    try {
      await fetchKpiScopes(metricScopeForView(view), yearMonth);
      const forecastTargetMonthForRun = addMonthsYYYYMM(yearMonth, 1);
      const response = await apiGet<RiskIndexResponse>("/risk_index", {
        ...buildParams(view),
        ...(forecastTargetMonthForRun
          ? { target_month: forecastTargetMonthForRun }
          : {}),
      });
      setRows(response.rows ?? []);
      setTargetMonth(response.target_month ?? null);
      setMetric(response.metric);
      setRunMonthFallback(
        Boolean(
          forecastTargetMonthForRun &&
            response.target_month &&
            response.target_month !== forecastTargetMonthForRun
        )
      );
    } catch (fetchError) {
      console.warn("Risk index unavailable:", fetchError);
      setError("Risk index unavailable (API error).");
      setRunMonthFallback(false);
    } finally {
      setIsLoading(false);
    }
  };

  const selectedScope = kpiData.scopes?.selected_run;
  const hazardEntries = useMemo(() => {
    const entries = Object.entries(
      selectedScope?.forecasts_by_hazard ?? {}
    ).filter(([, value]) => typeof value === "number");
    return entries.sort(([a], [b]) => a.localeCompare(b));
  }, [selectedScope]);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-3">
            <label className="flex items-center gap-2 text-sm text-fred-text">
              <span>View</span>
              <select
                className="w-[30ch] max-w-full rounded-md border border-fred-secondary bg-fred-surface px-3 py-2 text-sm text-fred-text focus:outline-none focus:ring-2 focus:ring-fred-primary/30 sm:w-[40ch] md:w-[48ch]"
                disabled={isLoading}
                onChange={(event) =>
                  handleViewChange(event.target.value as RiskView)
                }
                value={view}
              >
                {VIEW_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <RunMonthSelector
              availableMonths={kpiData.available_months ?? []}
              onChange={handleRunMonthChange}
              selectedMonth={selectedRunMonth}
            />
          </div>
          <div className="space-y-1 text-xs text-fred-muted">
            <div>
              World overview • Jenks breaks calculated from the selected risk values.
            </div>
            <div>
              Metric {metric} • {targetMonth ?? "latest"}
            </div>
            {runMonthFallback ? (
              <div className="text-amber-300">
                No forecasts found for selected run month; showing latest
                available month instead.
              </div>
            ) : null}
          </div>
        </div>
      </div>

      {error ? (
        <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
          {error}
        </div>
      ) : null}

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_360px]">
        <RiskIndexMap
          countriesRows={countriesRows}
          heightClassName={mapHeightClassName}
          riskRows={rows}
          view={view}
        />
        <div className="space-y-4" data-testid="risk-index-kpi-panel">
          <div className="rounded-lg border border-fred-secondary bg-fred-surface p-4 shadow-fredCard">
            <div className="flex items-center justify-between gap-3">
              <div className="text-xs uppercase tracking-wide text-fred-muted">
                {selectedRunMonth
                  ? `Selected run ${selectedRunMonth}`
                  : "Selected run"}
              </div>
            </div>
            {kpiError ? (
              <div className="mt-3 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
                {kpiError}
              </div>
            ) : null}
            <div className="mt-3 grid grid-cols-3 gap-3">
              <KpiCard
                label="Questions"
                value={selectedScope?.questions ?? 0}
              />
              <KpiCard
                label="Forecasts"
                value={selectedScope?.forecasts ?? 0}
              />
              <KpiCard
                label="Countries with Forecasts"
                value={
                  selectedScope?.countries_with_forecasts ??
                  selectedScope?.countries ??
                  0
                }
              />
            </div>
            <div className="mt-3">
              <KpiCard
                label="Countries Triaged"
                value={selectedScope?.countries_triaged ?? 0}
              />
              {showKpiDebug ? (
                <div className="mt-1 text-[11px] text-fred-muted">
                  countries_triaged_source:{" "}
                  {String(
                    kpiData.diagnostics?.countries_triaged_source ?? "unknown"
                  )}
                </div>
              ) : null}
            </div>
            <p className="mt-3 text-xs text-fred-muted">
              {kpiData.explanations?.[0] ??
                "Questions can exceed forecasts when some runs stop at triage or research."}
            </p>
            <div className="mt-4">
              <KpiCard
                label="Resolved questions"
                value={selectedScope?.resolved_questions ?? 0}
              />
            </div>
            <div className="mt-4">
              <div className="text-xs uppercase tracking-wide text-fred-muted">
                Forecasts by hazard type
              </div>
              <div className="mt-3 grid grid-cols-2 gap-3">
                {hazardEntries.length ? (
                  hazardEntries.map(([code, count]) => (
                    <div
                      key={code}
                      className="rounded-lg border border-fred-secondary/70 bg-fred-surface px-3 py-2 text-sm"
                    >
                      <div className="text-xs text-fred-muted">
                        {(HAZARD_LABELS[code] ?? "Hazard") + ` (${code})`}
                      </div>
                      <div className="mt-1 text-lg font-semibold text-fred-primary">
                        {count}
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-xs text-fred-muted">
                    No hazard forecasts in this scope.
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="w-full overflow-x-auto rounded-lg border border-fred-secondary bg-fred-surface">
        <RiskIndexTable
          mode={isPerCapita ? "percap" : "raw"}
          rows={rows}
          targetMonth={targetMonth}
          metric={metric}
        />
      </div>
    </div>
  );
}
