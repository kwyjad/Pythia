"use client";

import { useMemo, useState } from "react";

import FactsTab from "./tabs/FactsTab";
import ConflictForecastsTab from "./tabs/ConflictForecastsTab";
import SituationTab from "./tabs/SituationTab";
import ClimateTab from "./tabs/ClimateTab";
import ContextSignalsTab from "./tabs/ContextSignalsTab";
import GlobalTab from "./tabs/GlobalTab";

export type ConnectorStatusRow = {
  source: string;
  last_updated: string | null;
  rows_scanned: number;
};

export type ResolverCountryOption = {
  iso3: string;
  country_name?: string | null;
};

export type DbSummaryTable = {
  name: string;
  row_count: number;
  last_updated: string | null;
  has_iso3: boolean;
};

type TabKey = "facts" | "conflict" | "situation" | "climate" | "context" | "global";

type ResolverClientProps = {
  countries: ResolverCountryOption[];
  connectorStatus: ConnectorStatusRow[];
  dbSummary: DbSummaryTable[];
};

const TABS: { key: TabKey; label: string; requiresCountry: boolean }[] = [
  { key: "facts", label: "Facts", requiresCountry: true },
  { key: "conflict", label: "Conflict Forecasts", requiresCountry: true },
  { key: "situation", label: "Situation Reports", requiresCountry: true },
  { key: "climate", label: "Climate", requiresCountry: true },
  { key: "context", label: "Context Signals", requiresCountry: true },
  { key: "global", label: "Global", requiresCountry: false },
];

const resolveIso3 = (
  input: string,
  countryByIso3: Map<string, ResolverCountryOption>
) => {
  const trimmed = input.trim();
  if (!trimmed) return null;
  const upper = trimmed.toUpperCase();
  if (countryByIso3.has(upper)) return upper;
  const match = trimmed.match(/\(([A-Za-z]{3})\)/);
  if (match) {
    const iso3 = match[1].toUpperCase();
    if (countryByIso3.has(iso3)) return iso3;
  }
  const lower = trimmed.toLowerCase();
  for (const option of countryByIso3.values()) {
    if ((option.country_name ?? "").toLowerCase() === lower) {
      return option.iso3;
    }
  }
  return null;
};

function freshnessColor(dateStr: string | null): string {
  if (!dateStr) return "text-fred-muted";
  const d = new Date(dateStr);
  const days = (Date.now() - d.getTime()) / (1000 * 60 * 60 * 24);
  if (days <= 7) return "text-green-400";
  if (days <= 30) return "text-yellow-400";
  return "text-red-400";
}

const TABLE_DISPLAY_NAMES: Record<string, string> = {
  facts_resolved: "Facts Resolved",
  facts_deltas: "Facts Deltas",
  acled_monthly_fatalities: "ACLED Fatalities",
  conflict_forecasts: "Conflict Forecasts",
  reliefweb_reports: "ReliefWeb",
  acled_political_events: "ACLED Political",
  acaps_inform_severity: "ACAPS INFORM",
  acaps_risk_radar: "ACAPS Risk Radar",
  acaps_daily_monitoring: "ACAPS Daily",
  acaps_humanitarian_access: "ACAPS Access",
  seasonal_forecasts: "NMME Seasonal",
  ipc_phases: "IPC Phases",
  enso_state: "ENSO",
  seasonal_tc_outlooks: "Seasonal TC",
  seasonal_tc_context_cache: "TC Context",
  hdx_signals: "HDX Signals",
  crisiswatch_entries: "CrisisWatch",
};

export default function ResolverClient({
  countries,
  dbSummary,
}: ResolverClientProps) {
  const [activeTab, setActiveTab] = useState<TabKey>("facts");
  const [countryInput, setCountryInput] = useState("");
  const [selectedIso3, setSelectedIso3] = useState<string | null>(null);

  const countryByIso3 = useMemo(() => {
    const map = new Map<string, ResolverCountryOption>();
    countries.forEach((c) => {
      if (c.iso3) map.set(c.iso3.toUpperCase(), { ...c, iso3: c.iso3.toUpperCase() });
    });
    return map;
  }, [countries]);

  const countryName = selectedIso3
    ? countryByIso3.get(selectedIso3)?.country_name ?? null
    : null;

  const currentTabDef = TABS.find((t) => t.key === activeTab);
  const showCountrySelector = currentTabDef?.requiresCountry !== false;

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-3xl font-semibold">Resolver Data Explorer</h1>
        <p className="text-sm text-fred-text">
          Browse all data collected by the Resolver pipeline across Phases 1-4.
        </p>
      </section>

      {/* DB Summary Cards */}
      {dbSummary.length > 0 && (
        <section className="space-y-2">
          <h2 className="text-sm font-semibold text-fred-muted">Data inventory</h2>
          <div className="grid gap-2 grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
            {dbSummary.map((tbl) => (
              <div
                key={tbl.name}
                className="rounded-lg border border-fred-secondary bg-fred-surface px-3 py-2"
              >
                <div className="text-xs font-semibold text-fred-primary truncate">
                  {TABLE_DISPLAY_NAMES[tbl.name] ?? tbl.name}
                </div>
                <div className="text-lg font-semibold text-fred-text tabular-nums">
                  {tbl.row_count.toLocaleString()}
                </div>
                <div className={`text-xs ${freshnessColor(tbl.last_updated)}`}>
                  {tbl.last_updated ?? "No data"}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Country selector + Tab bar */}
      <section className="space-y-4">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-end">
          {showCountrySelector && (
            <div className="flex flex-col gap-1">
              <label className="text-sm font-semibold text-fred-primary">
                Country
              </label>
              <input
                list="country-options"
                className="w-64 rounded-md border border-fred-secondary bg-fred-surface px-3 py-2 text-sm text-fred-text"
                placeholder="Type ISO3 or country name"
                value={countryInput}
                onChange={(e) => {
                  setCountryInput(e.target.value);
                  setSelectedIso3(resolveIso3(e.target.value, countryByIso3));
                }}
              />
              <datalist id="country-options">
                {countries.map((c) => (
                  <option key={c.iso3} value={c.iso3}>
                    {c.country_name ? `${c.country_name} (${c.iso3})` : c.iso3}
                  </option>
                ))}
              </datalist>
              {selectedIso3 && (
                <span className="text-xs text-fred-muted">
                  Selected: {countryName ?? selectedIso3}
                </span>
              )}
            </div>
          )}
        </div>

        {/* Top-level tabs */}
        <div className="flex gap-1 border-b border-fred-secondary overflow-x-auto">
          {TABS.map((tab) => (
            <button
              key={tab.key}
              className={`whitespace-nowrap px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                activeTab === tab.key
                  ? "border-fred-primary text-fred-primary"
                  : "border-transparent text-fred-muted hover:text-fred-text"
              }`}
              onClick={() => setActiveTab(tab.key)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        <div>
          {activeTab === "facts" && (
            <FactsTab selectedIso3={selectedIso3} countryName={countryName} />
          )}
          {activeTab === "conflict" && (
            <ConflictForecastsTab selectedIso3={selectedIso3} countryName={countryName} />
          )}
          {activeTab === "situation" && (
            <SituationTab selectedIso3={selectedIso3} countryName={countryName} />
          )}
          {activeTab === "climate" && (
            <ClimateTab selectedIso3={selectedIso3} countryName={countryName} />
          )}
          {activeTab === "context" && (
            <ContextSignalsTab selectedIso3={selectedIso3} countryName={countryName} />
          )}
          {activeTab === "global" && <GlobalTab />}
        </div>
      </section>
    </div>
  );
}
