"use client";

import { useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";

import { apiGet } from "../lib/api";
import type {
  CountriesResponse,
  CountriesRow,
  DiagnosticsKpiScopesResponse,
  RiskIndexResponse,
  RiskView,
} from "../lib/types";
import InfoTooltip from "./InfoTooltip";
import KpiCard from "./KpiCard";
import RiskIndexMap from "./RiskIndexMap";
import RiskIndexTable from "./RiskIndexTable";
import RunMonthSelector from "./RunMonthSelector";
import RunSelector from "./RunSelector";

const VIEW_OPTIONS: Array<{ value: RiskView; label: string }> = [
  { value: "PA_EIV", label: "People affected (absolute)" },
  { value: "PA_PC", label: "People affected (per capita)" },
  { value: "FATALITIES_EIV", label: "Fatalities (absolute)" },
  { value: "FATALITIES_PC", label: "Fatalities (per capita)" },
  { value: "EVENT_OCCURRENCE", label: "Event probability" },
  { value: "PHASE3PLUS_EIV", label: "Phase 3+ population (absolute)" },
  { value: "PHASE3PLUS_PC", label: "Phase 3+ (% of population)" },
];

type RiskIndexPanelProps = {
  initialResponse: RiskIndexResponse;
  countriesRows: CountriesRow[];
  kpiScopes: DiagnosticsKpiScopesResponse;
  mapHeightClassName?: string;
  includeTest?: boolean;
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
    case "EVENT_OCCURRENCE":
      return { metric: "EVENT_OCCURRENCE", horizon_m: 6, normalize: false };
    case "PHASE3PLUS_EIV":
      return {
        metric: "PHASE3PLUS_IN_NEED",
        hazard_code: "DR",
        horizon_m: 6,
        normalize: false,
      };
    case "PHASE3PLUS_PC":
      return {
        metric: "PHASE3PLUS_IN_NEED",
        hazard_code: "DR",
        horizon_m: 6,
        normalize: true,
      };
    default:
      return { metric: "PA", horizon_m: 6, normalize: false };
  }
};

const metricScopeForView = (view: RiskView) => {
  if (view === "FATALITIES_EIV" || view === "FATALITIES_PC") return "FATALITIES";
  if (view === "EVENT_OCCURRENCE") return "EVENT_OCCURRENCE";
  if (view === "PHASE3PLUS_EIV" || view === "PHASE3PLUS_PC") return "PHASE3PLUS_IN_NEED";
  return "PA";
};

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

const RC_LEVEL_TOOLTIP =
  "RC means Regime Change, and refers to a score 0-1 that reflects Fred's expectation of a significant change from the base rate";

export default function RiskIndexPanel({
  initialResponse,
  countriesRows,
  kpiScopes,
  mapHeightClassName,
  includeTest,
}: RiskIndexPanelProps) {
  const [view, setView] = useState<RiskView>("PA_EIV");
  const [rows, setRows] = useState(initialResponse.rows ?? []);
  const [targetMonth, setTargetMonth] = useState(initialResponse.target_month);
  const [metric, setMetric] = useState(initialResponse.metric);
  const [countries, setCountries] = useState<CountriesRow[]>(countriesRows);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [kpiData, setKpiData] = useState(kpiScopes);
  const [selectedRunMonth, setSelectedRunMonth] = useState(
    kpiScopes.selected_month
  );
  const [kpiError, setKpiError] = useState<string | null>(null);
  const [runMonthFallback, setRunMonthFallback] = useState(false);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(
    kpiScopes.selected_run_id ?? null
  );
  const searchParams = useSearchParams();
  const showKpiDebug = searchParams?.get("debug_kpi") === "1";

  const isPerCapita =
    view === "PA_PC" ||
    view === "FATALITIES_PC" ||
    view === "PHASE3PLUS_PC" ||
    view === "EVENT_OCCURRENCE";

  const fetchCountries = async (
    metricScope: string,
    runMonth: string | null,
    runId?: string | null
  ) => {
    try {
      const response = await apiGet<CountriesResponse>("/countries", {
        metric_scope: metricScope,
        ...(runMonth ? { year_month: runMonth } : {}),
        ...(runId ? { forecaster_run_id: runId } : {}),
        include_test: includeTest || undefined,
      });
      setCountries(response.rows ?? []);
    } catch (fetchError) {
      console.warn("Countries unavailable:", fetchError);
    }
  };

  const fetchKpiScopes = async (
    metricScope: string,
    runMonth: string | null,
    runId?: string | null
  ): Promise<DiagnosticsKpiScopesResponse | null> => {
    setKpiError(null);
    try {
      const response = await apiGet<DiagnosticsKpiScopesResponse>(
        "/diagnostics/kpi_scopes",
        {
          metric_scope: metricScope,
          ...(runMonth ? { year_month: runMonth } : {}),
          ...(runId ? { forecaster_run_id: runId } : {}),
          include_test: includeTest || undefined,
        }
      );
      setKpiData(response);
      setSelectedRunMonth(response.selected_month);
      return response;
    } catch (fetchError) {
      console.warn("KPI scopes unavailable:", fetchError);
      setKpiError("KPI scopes unavailable (API error).");
      return null;
    }
  };

  const fetchRiskIndex = async (
    nextView: RiskView,
    runMonth: string | null,
    runId: string | null
  ) => {
    const forecastTargetMonthForRun = runMonth
      ? addMonthsYYYYMM(runMonth, 1)
      : null;
    const response = await apiGet<RiskIndexResponse>("/risk_index", {
      ...buildParams(nextView),
      ...(forecastTargetMonthForRun
        ? { target_month: forecastTargetMonthForRun }
        : {}),
      ...(runId ? { forecaster_run_id: runId } : {}),
      include_test: includeTest || undefined,
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
  };

  const handleViewChange = async (nextView: RiskView) => {
    setView(nextView);
    setIsLoading(true);
    setError(null);
    try {
      await fetchRiskIndex(nextView, selectedRunMonth, selectedRunId);
      const nextMetricScope = metricScopeForView(nextView);
      void fetchKpiScopes(nextMetricScope, selectedRunMonth, selectedRunId);
      void fetchCountries(nextMetricScope, selectedRunMonth, selectedRunId);
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
    setSelectedRunId(null);
    setIsLoading(true);
    setError(null);
    try {
      const currentMetricScope = metricScopeForView(view);
      const kpiResponse = await fetchKpiScopes(currentMetricScope, yearMonth);
      const newRunId = kpiResponse?.selected_run_id ?? null;
      void fetchCountries(currentMetricScope, yearMonth, newRunId);
      setSelectedRunId(newRunId);
      await fetchRiskIndex(view, yearMonth, newRunId);
    } catch (fetchError) {
      console.warn("Risk index unavailable:", fetchError);
      setError("Risk index unavailable (API error).");
      setRunMonthFallback(false);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRunChange = async (runId: string) => {
    setSelectedRunId(runId);
    setIsLoading(true);
    setError(null);
    try {
      const currentMetricScope = metricScopeForView(view);
      await fetchRiskIndex(view, selectedRunMonth, runId);
      void fetchKpiScopes(currentMetricScope, selectedRunMonth, runId);
      void fetchCountries(currentMetricScope, selectedRunMonth, runId);
    } catch (fetchError) {
      console.warn("Risk index unavailable:", fetchError);
      setError("Risk index unavailable (API error).");
      setRunMonthFallback(false);
    } finally {
      setIsLoading(false);
    }
  };

  const selectedScope = kpiData.scopes?.selected_run;
  const rcLevelCounts = useMemo(() => {
    const counts = { level1: 0, level2: 0, level3: 0 };
    countries.forEach((row) => {
      const level = row.highest_rc_level;
      if (level === 1) counts.level1 += 1;
      if (level === 2) counts.level2 += 1;
      if (level === 3) counts.level3 += 1;
    });
    return counts;
  }, [countries]);
  const hazardEntries = useMemo(() => {
    const entries = Object.entries(
      selectedScope?.forecasts_by_hazard ?? {}
    ).filter(([, value]) => typeof value === "number");
    return entries.sort(([a], [b]) => a.localeCompare(b));
  }, [selectedScope]);

  const defaultMapHeightClassName =
    mapHeightClassName ??
    "h-[360px] sm:h-[420px] md:h-[520px] lg:h-[720px]";
  const paEivMapHeightClassName =
    "h-[360px] sm:h-[420px] md:h-[520px] lg:h-[520px]";
  const resolvedMapHeightClassName =
    view === "PA_EIV" ? paEivMapHeightClassName : defaultMapHeightClassName;

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
            <RunSelector
              availableRuns={kpiData.available_runs ?? []}
              selectedRunId={selectedRunId}
              onChange={handleRunChange}
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
        <div className="space-y-4">
          <RiskIndexMap
            countriesRows={countries}
            heightClassName={resolvedMapHeightClassName}
            riskRows={rows}
            view={view}
          />
        </div>
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
            <div className="mt-3 grid grid-cols-2 gap-3">
              <div>
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
              <KpiCard
                label="Resolved questions"
                value={selectedScope?.resolved_questions ?? 0}
              />
            </div>
            <div className="mt-3 grid grid-cols-3 gap-3">
              <KpiCard
                label={
                  <span className="inline-flex items-center gap-1">
                    RC L1 countries <InfoTooltip text={RC_LEVEL_TOOLTIP} />
                  </span>
                }
                value={rcLevelCounts.level1}
              />
              <KpiCard
                label={
                  <span className="inline-flex items-center gap-1">
                    RC L2 countries <InfoTooltip text={RC_LEVEL_TOOLTIP} />
                  </span>
                }
                value={rcLevelCounts.level2}
              />
              <KpiCard
                label={
                  <span className="inline-flex items-center gap-1">
                    RC L3 countries <InfoTooltip text={RC_LEVEL_TOOLTIP} />
                  </span>
                }
                value={rcLevelCounts.level3}
              />
            </div>
            <p className="mt-3 text-xs text-fred-muted">
              {kpiData.explanations?.[0] ??
                "RC L1, L2, and L3 questions are forecasted with the full Fred LLM ensemble. All others, which are expected to stay close to base rates, are forecasted with a single LLM."}
            </p>
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

      <div className="w-full max-h-[520px] overflow-x-auto overflow-y-auto rounded-lg border border-fred-secondary bg-fred-surface">
        <RiskIndexTable
          countriesRows={countries}
          mode={isPerCapita ? "percap" : "raw"}
          rows={rows}
          targetMonth={targetMonth}
          metric={metric}
          stickyHeader
        />
      </div>
    </div>
  );
}
