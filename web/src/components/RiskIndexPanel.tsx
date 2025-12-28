"use client";

import { useMemo, useState } from "react";

import { apiGet } from "../lib/api";
import type { RiskIndexResponse, RiskView } from "../lib/types";
import RiskIndexTable from "./RiskIndexTable";

const POPULATION_HELPER =
  "Per-capita requires population data (populations table). Not available in this snapshot.";

const VIEW_OPTIONS: Array<{ value: RiskView; label: string }> = [
  { value: "PA_EIV", label: "PA EIV" },
  { value: "PA_PC", label: "PA per-capita EIV" },
  { value: "FATALITIES_EIV", label: "ACE fatalities EIV" },
];

type RiskIndexPanelProps = {
  initialResponse: RiskIndexResponse;
};

const buildParams = (view: RiskView) => {
  switch (view) {
    case "PA_PC":
    case "PA_EIV":
      return { metric: "PA", horizon_m: 6, normalize: true };
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

export default function RiskIndexPanel({ initialResponse }: RiskIndexPanelProps) {
  const [view, setView] = useState<RiskView>("PA_EIV");
  const [rows, setRows] = useState(initialResponse.rows ?? []);
  const [targetMonth, setTargetMonth] = useState(initialResponse.target_month);
  const [metric, setMetric] = useState(initialResponse.metric);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
        <label className="flex items-center gap-2 text-sm text-slate-300">
          <span>View</span>
          <select
            className="rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100"
            disabled={isLoading}
            onChange={(event) => handleViewChange(event.target.value as RiskView)}
            value={view}
          >
            {VIEW_OPTIONS.map((option) => (
              <option
                key={option.value}
                disabled={option.value === "PA_PC" && !populationAvailable}
                value={option.value}
              >
                {option.label}
              </option>
            ))}
          </select>
        </label>

        <span className="text-sm text-slate-400">
          Metric {metric} • {targetMonth ?? "latest"}
        </span>
      </div>

      {!populationAvailable ? (
        <p className="text-xs text-slate-500">{POPULATION_HELPER}</p>
      ) : null}

      {error ? (
        <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
          {error}
        </div>
      ) : null}

      <div className="w-full overflow-x-auto rounded-lg border border-slate-800">
        <RiskIndexTable
          mode={view === "PA_PC" ? "percap" : "raw"}
          rows={rows}
          targetMonth={targetMonth}
        />
      </div>
    </div>
  );
}
