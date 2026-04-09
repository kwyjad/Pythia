"use client";

import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";

import { apiGet, fetchRunSummary } from "../lib/api";
import type {
  CountriesResponse,
  CountriesRow,
  DiagnosticsKpiScopesResponse,
  PerformanceScoresResponse,
  RiskIndexResponse,
  RiskView,
  RunSummaryResponse,
} from "../lib/types";
import RiskIndexMap from "./RiskIndexMap";
import RiskIndexTable from "./RiskIndexTable";
import RunMonthSelector from "./RunMonthSelector";
import RunSelector from "./RunSelector";
import RunSummaryView from "./RunSummaryView";

const VIEW_OPTIONS: Array<{ value: RiskView; label: string }> = [
  { value: "ALL_METRICS_SUMMARY", label: "All metrics (summary)" },
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
  if (view === "ALL_METRICS_SUMMARY") return "PA";
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

export default function RiskIndexPanel({
  initialResponse,
  countriesRows,
  kpiScopes,
  mapHeightClassName,
  includeTest,
}: RiskIndexPanelProps) {
  const [view, setView] = useState<RiskView>("ALL_METRICS_SUMMARY");
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
  const [summaryData, setSummaryData] = useState<RunSummaryResponse | null>(null);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [perfScores, setPerfScores] = useState<PerformanceScoresResponse | null>(null);
  const searchParams = useSearchParams();
  const showKpiDebug = searchParams?.get("debug_kpi") === "1";

  const loadSummary = async (
    runMonth: string | null,
    runId?: string | null
  ) => {
    setSummaryLoading(true);
    try {
      const response = await fetchRunSummary({
        ...(runMonth ? { year_month: runMonth } : {}),
        ...(runId ? { forecaster_run_id: runId } : {}),
        include_test: includeTest || undefined,
      });
      setSummaryData(response);
    } catch (fetchError) {
      console.warn("Run summary unavailable:", fetchError);
    } finally {
      setSummaryLoading(false);
    }
  };

  // Load summary data on mount since ALL_METRICS_SUMMARY is the default view
  useEffect(() => {
    void loadSummary(selectedRunMonth, selectedRunId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const isSummaryView = view === "ALL_METRICS_SUMMARY";
  const isPerCapita =
    view === "PA_PC" ||
    view === "FATALITIES_PC" ||
    view === "PHASE3PLUS_PC" ||
    view === "EVENT_OCCURRENCE";

  const fetchPerfScores = async (metricScope: string) => {
    try {
      const resp = await apiGet<PerformanceScoresResponse>(
        "/performance/scores",
        { metric: metricScope, include_test: includeTest || undefined }
      );
      setPerfScores(resp);
    } catch {
      console.warn("Performance scores unavailable");
      setPerfScores(null);
    }
  };

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
    // When a specific run is selected, let the API auto-select the correct
    // target_month for that run's questions (target_month conventions differ
    // between old and new question formats).  Only pass an explicit
    // target_month when no run is selected and we have a runMonth to derive
    // it from.
    const forecastTargetMonthForRun =
      !runId && runMonth ? addMonthsYYYYMM(runMonth, 1) : null;
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
      if (nextView === "ALL_METRICS_SUMMARY") {
        await loadSummary(selectedRunMonth, selectedRunId);
        // Still load kpi data for available months/runs
        const nextMetricScope = metricScopeForView(nextView);
        void fetchKpiScopes(nextMetricScope, selectedRunMonth, selectedRunId);
      } else {
        await fetchRiskIndex(nextView, selectedRunMonth, selectedRunId);
        const nextMetricScope = metricScopeForView(nextView);
        void fetchKpiScopes(nextMetricScope, selectedRunMonth, selectedRunId);
        void fetchCountries(nextMetricScope, selectedRunMonth, selectedRunId);
        void fetchPerfScores(nextMetricScope);
      }
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
      setSelectedRunId(newRunId);
      if (view === "ALL_METRICS_SUMMARY") {
        await loadSummary(yearMonth, newRunId);
      } else {
        void fetchCountries(currentMetricScope, yearMonth, newRunId);
        await fetchRiskIndex(view, yearMonth, newRunId);
      }
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
      if (view === "ALL_METRICS_SUMMARY") {
        await loadSummary(selectedRunMonth, runId);
        const currentMetricScope = metricScopeForView(view);
        void fetchKpiScopes(currentMetricScope, selectedRunMonth, runId);
      } else {
        const currentMetricScope = metricScopeForView(view);
        await fetchRiskIndex(view, selectedRunMonth, runId);
        void fetchKpiScopes(currentMetricScope, selectedRunMonth, runId);
        void fetchCountries(currentMetricScope, selectedRunMonth, runId);
      }
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
          {isSummaryView ? (
            <div className="text-xs text-fred-muted">
              Select a non-summary metric view to see the forecast index map and
              country-level results.
            </div>
          ) : (
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
          )}
        </div>
      </div>

      {error ? (
        <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
          {error}
        </div>
      ) : null}

      {isSummaryView ? (
        summaryLoading ? (
          <div className="py-12 text-center text-sm text-fred-muted">
            Loading run summary...
          </div>
        ) : summaryData ? (
          <RunSummaryView data={summaryData} />
        ) : (
          <div className="py-12 text-center text-sm text-fred-muted">
            No summary data available. Select a run month to view the summary.
          </div>
        )
      ) : (
        <>
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_360px]">
            <div className="space-y-4">
              <RiskIndexMap
                countriesRows={countries}
                heightClassName={resolvedMapHeightClassName}
                riskRows={rows}
                view={view}
              />
            </div>
            <div className="space-y-3" data-testid="risk-index-kpi-panel">
              {/* Coverage */}
              <div className="rounded-lg border border-fred-secondary bg-fred-surface p-3 shadow-fredCard">
                <div className="text-[11px] uppercase tracking-wide text-fred-muted">
                  Coverage
                </div>
                {kpiError ? (
                  <div className="mt-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-2 py-1 text-[11px] text-amber-200">
                    {kpiError}
                  </div>
                ) : null}
                <div className="mt-2 grid grid-cols-2 gap-2 text-sm">
                  <div>
                    <div className="text-[11px] text-fred-muted">Forecasts</div>
                    <div className="text-lg font-semibold text-fred-primary">{selectedScope?.forecasts ?? 0}</div>
                  </div>
                  <div>
                    <div className="text-[11px] text-fred-muted">Countries</div>
                    <div className="text-lg font-semibold text-fred-primary">
                      {selectedScope?.countries_with_forecasts ?? selectedScope?.countries ?? 0}
                      <span className="ml-1 text-xs font-normal text-fred-muted">
                        / {selectedScope?.countries_triaged ?? 0} triaged
                      </span>
                    </div>
                  </div>
                  <div>
                    <div className="text-[11px] text-fred-muted">Resolved</div>
                    <div className="text-lg font-semibold text-fred-primary">{selectedScope?.resolved_questions ?? 0}</div>
                  </div>
                  <div>
                    <div className="text-[11px] text-fred-muted">By hazard</div>
                    <div className="mt-0.5 flex flex-wrap gap-1">
                      {hazardEntries.length ? (
                        hazardEntries.map(([code, count]) => (
                          <span key={code} className="inline-block rounded-full bg-fred-primary/10 px-1.5 py-0.5 text-[11px] font-medium text-fred-primary">
                            {code} {count}
                          </span>
                        ))
                      ) : (
                        <span className="text-[11px] text-fred-muted">—</span>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* RC Assessment */}
              <div className="rounded-lg border border-fred-secondary bg-fred-surface p-3 shadow-fredCard">
                <div className="text-[11px] uppercase tracking-wide text-fred-muted">
                  Regime Change
                </div>
                <div className="mt-2 flex h-5 w-full overflow-hidden rounded">
                  {([
                    { key: "level0" as const, color: "bg-teal-600", count: countries.length - rcLevelCounts.level1 - rcLevelCounts.level2 - rcLevelCounts.level3 },
                    { key: "level1" as const, color: "bg-amber-500", count: rcLevelCounts.level1 },
                    { key: "level2" as const, color: "bg-orange-500", count: rcLevelCounts.level2 },
                    { key: "level3" as const, color: "bg-red-600", count: rcLevelCounts.level3 },
                  ] as const).map(({ key, color, count }) => {
                    const total = countries.length || 1;
                    const pct = (count / total) * 100;
                    if (count === 0) return null;
                    return (
                      <div
                        key={key}
                        className={`${color} flex items-center justify-center text-[10px] font-semibold text-white`}
                        style={{ width: `${pct}%`, minWidth: count > 0 ? "18px" : 0 }}
                      >
                        {pct > 8 ? count : ""}
                      </div>
                    );
                  })}
                </div>
                <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-fred-muted">
                  <span><span className="inline-block h-2 w-2 rounded-sm bg-teal-600" /> L0</span>
                  <span><span className="inline-block h-2 w-2 rounded-sm bg-amber-500" /> L1 ({rcLevelCounts.level1})</span>
                  <span><span className="inline-block h-2 w-2 rounded-sm bg-orange-500" /> L2 ({rcLevelCounts.level2})</span>
                  <span><span className="inline-block h-2 w-2 rounded-sm bg-red-600" /> L3 ({rcLevelCounts.level3})</span>
                </div>
                <p className="mt-1.5 text-[11px] text-fred-muted">
                  L1+ countries forecast with full ensemble; L0 with single model.
                </p>
              </div>

              {/* Performance */}
              <div className="rounded-lg border border-fred-secondary bg-fred-surface p-3 shadow-fredCard">
                <div className="text-[11px] uppercase tracking-wide text-fred-muted">
                  Performance
                </div>
                {(() => {
                  const rows = perfScores?.summary_rows ?? [];
                  const ensembleRows = rows.filter(
                    (r) => r.model_name != null && r.model_name.startsWith("ensemble_")
                  );
                  const brierRow = ensembleRows.find((r) => r.score_type === "brier");
                  const logRow = ensembleRows.find((r) => r.score_type === "log");
                  const crpsRow = ensembleRows.find((r) => r.score_type === "crps");
                  const fmt = (v: number | null | undefined) =>
                    v != null ? v.toFixed(3) : "—";
                  return (
                    <div className="mt-2 grid grid-cols-2 gap-2 text-sm">
                      <div>
                        <div className="text-[11px] text-fred-muted">Resolved</div>
                        <div className="text-lg font-semibold text-fred-primary">
                          {selectedScope?.resolved_questions ?? 0}
                        </div>
                      </div>
                      <div>
                        <div className="text-[11px] text-fred-muted">Brier</div>
                        <div className="text-lg font-semibold text-fred-primary">
                          {fmt(brierRow?.avg_value)}
                        </div>
                        <div className="text-[10px] text-fred-muted">
                          med {fmt(brierRow?.median_value)}
                        </div>
                      </div>
                      <div>
                        <div className="text-[11px] text-fred-muted">Log Loss</div>
                        <div className="text-lg font-semibold text-fred-primary">
                          {fmt(logRow?.avg_value)}
                        </div>
                        <div className="text-[10px] text-fred-muted">
                          med {fmt(logRow?.median_value)}
                        </div>
                      </div>
                      <div>
                        <div className="text-[11px] text-fred-muted">CRPS</div>
                        <div className="text-lg font-semibold text-fred-primary">
                          {fmt(crpsRow?.avg_value)}
                        </div>
                        <div className="text-[10px] text-fred-muted">
                          med {fmt(crpsRow?.median_value)}
                        </div>
                      </div>
                    </div>
                  );
                })()}
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
        </>
      )}
    </div>
  );
}
