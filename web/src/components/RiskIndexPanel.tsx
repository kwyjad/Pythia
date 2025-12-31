"use client";

import { useMemo, useState, type ReactNode } from "react";

import { apiGet } from "../lib/api";
import type { CountriesRow, RiskIndexResponse, RiskView } from "../lib/types";
import RiskIndexMap from "./RiskIndexMap";
import RiskIndexTable from "./RiskIndexTable";

const POPULATION_HELPER =
  "Per-capita requires population data (populations table or resolver/data/population.csv).";

const VIEW_OPTIONS: Array<{ value: RiskView; label: string }> = [
  { value: "PA_EIV", label: "People Affected (PA) EIV" },
  { value: "PA_PC", label: "People Affected (PA) per capita EIV" },
  { value: "FATALITIES_EIV", label: "Armed Conflict (ACE) fatalities EIV" },
  { value: "FATALITIES_PC", label: "Armed Conflict (ACE) fatalities per capita EIV" },
];

type RiskIndexPanelProps = {
  initialResponse: RiskIndexResponse;
  countriesRows: CountriesRow[];
  aside?: ReactNode;
  mapHeightClassName?: string;
};

const buildParams = (view: RiskView) => {
  switch (view) {
    case "PA_PC":
    case "PA_EIV":
      return { metric: "PA", horizon_m: 6, normalize: true };
    case "FATALITIES_PC":
    case "FATALITIES_EIV":
      return {
        metric: "FATALITIES",
        hazard_code: "ACE",
        horizon_m: 6,
        normalize: true,
      };
    default:
      return { metric: "PA", horizon_m: 6, normalize: true };
  }
};

export default function RiskIndexPanel({
  initialResponse,
  countriesRows,
  aside,
  mapHeightClassName,
}: RiskIndexPanelProps) {
  const [view, setView] = useState<RiskView>("PA_EIV");
  const [rows, setRows] = useState(initialResponse.rows ?? []);
  const [targetMonth, setTargetMonth] = useState(initialResponse.target_month);
  const [metric, setMetric] = useState(initialResponse.metric);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isPerCapita = view === "PA_PC" || view === "FATALITIES_PC";

  const populationAvailable = useMemo(
    () => rows.some((row) => typeof row.population === "number"),
    [rows]
  );

  const handleViewChange = async (nextView: RiskView) => {
    setView(nextView);
    setIsLoading(true);
    setError(null);
    try {
      const response = await apiGet<RiskIndexResponse>(
        "/risk_index",
        buildParams(nextView)
      );
      setRows(response.rows ?? []);
      setTargetMonth(response.target_month ?? null);
      setMetric(response.metric);
    } catch (fetchError) {
      console.warn("Risk index unavailable:", fetchError);
      setError("Risk index unavailable (API error).");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-2 text-sm text-fred-text">
            <span>View</span>
            <select
              className="w-[30ch] max-w-full rounded-md border border-fred-border bg-fred-surface px-3 py-2 text-sm text-fred-text focus:outline-none focus:ring-2 focus:ring-fred-primary/30 sm:w-[40ch] md:w-[48ch]"
              disabled={isLoading}
              onChange={(event) =>
                handleViewChange(event.target.value as RiskView)
              }
              value={view}
            >
              {VIEW_OPTIONS.map((option) => (
                <option
                  key={option.value}
                  disabled={
                    (option.value === "PA_PC" ||
                      option.value === "FATALITIES_PC") &&
                    !populationAvailable
                  }
                  value={option.value}
                >
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <span className="text-xs text-fred-muted">
            World overview • Jenks breaks calculated from the selected risk values.
          </span>
        </div>

        <span className="text-sm text-fred-muted">
          Metric {metric} • {targetMonth ?? "latest"}
        </span>
      </div>

      {!populationAvailable ? (
        <p className="text-xs text-fred-muted">{POPULATION_HELPER}</p>
      ) : null}

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
        {aside ?? null}
      </div>

      <div className="w-full overflow-x-auto rounded-lg border border-fred-border bg-fred-surface">
        <RiskIndexTable
          mode={isPerCapita ? "percap" : "raw"}
          rows={rows}
          targetMonth={targetMonth}
        />
      </div>
    </div>
  );
}
