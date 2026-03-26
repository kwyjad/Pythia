"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import SortableTable, { SortableColumn } from "../../components/SortableTable";
import { apiGet } from "../../lib/api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ResolverCountryOption = {
  iso3: string;
  country_name?: string | null;
};

export type SourceInventoryItem = {
  key: string;
  label: string;
  category: string;
  last_updated: string | null;
  global_rows: number;
  country_rows: number | null;
  has_iso3: boolean;
};

type SourceDataResponse = {
  rows: Record<string, unknown>[];
  columns: string[];
};

// Keep these for backward compat with page.tsx (legacy props still accepted)
export type ConnectorStatusRow = {
  source: string;
  last_updated: string | null;
  rows_scanned: number;
};

export type DbSummaryTable = {
  name: string;
  row_count: number;
  last_updated: string | null;
  has_iso3: boolean;
};

// ---------------------------------------------------------------------------
// Category definitions (static)
// ---------------------------------------------------------------------------

const CATEGORIES = [
  {
    key: "resolution_data",
    title: "Resolution Data",
    sources: ["ifrc", "idmc", "acled", "gdacs", "fewsnet", "acled_fatalities"],
  },
  {
    key: "conflict_forecasts",
    title: "Conflict Forecasts",
    sources: ["views", "conflictforecast", "acled_cast", "crisiswatch"],
  },
  {
    key: "weather_climate",
    title: "Weather and Climate",
    sources: ["nmme", "enso", "seasonal_tc", "tc_context"],
  },
  {
    key: "situation_reports",
    title: "Situation Reports",
    sources: ["reliefweb", "acaps_daily", "acled_political"],
  },
  {
    key: "other_alerts",
    title: "Other Alerts",
    sources: ["hdx_signals", "acaps_risk_radar"],
  },
  {
    key: "other",
    title: "Other",
    sources: ["acaps_inform", "acaps_access", "ipc_phases"],
  },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

const formatValue = (value: unknown) => {
  if (value === null || value === undefined) return "";
  const num = Number(value);
  if (Number.isNaN(num)) return String(value);
  if (typeof value === "string" && value.includes("-")) return value; // dates
  return num.toLocaleString(undefined, { maximumFractionDigits: 2 });
};

// ---------------------------------------------------------------------------
// SourceAccordion
// ---------------------------------------------------------------------------

type SourceAccordionProps = {
  sourceKey: string;
  inventory: SourceInventoryItem | null;
  selectedIso3: string | null;
};

function SourceAccordion({
  sourceKey,
  inventory,
  selectedIso3,
}: SourceAccordionProps) {
  const [expanded, setExpanded] = useState(false);
  const [allColumns, setAllColumns] = useState(false);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<SourceDataResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Track which iso3 + allColumns combo was used for the cached data
  const [fetchedFor, setFetchedFor] = useState<string>("");

  const fetchKey = `${selectedIso3 ?? ""}_${allColumns}`;
  const isEmpty = inventory && inventory.global_rows === 0;

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params: Record<string, string | number | boolean> = {
        source: sourceKey,
        limit: 500,
        all_columns: allColumns,
      };
      if (selectedIso3 && inventory?.has_iso3) {
        params.iso3 = selectedIso3;
      }
      const res = await apiGet<SourceDataResponse>(
        "/resolver/source_data",
        params
      );
      setData(res);
      setFetchedFor(fetchKey);
    } catch {
      setError("Failed to load data.");
    } finally {
      setLoading(false);
    }
  }, [sourceKey, selectedIso3, allColumns, fetchKey, inventory?.has_iso3]);

  // Re-fetch when expanded and parameters change
  useEffect(() => {
    if (expanded && !isEmpty && fetchedFor !== fetchKey) {
      loadData();
    }
  }, [expanded, isEmpty, fetchKey, fetchedFor, loadData]);

  const handleToggle = () => {
    if (isEmpty) return;
    const willExpand = !expanded;
    setExpanded(willExpand);
    if (willExpand && fetchedFor !== fetchKey) {
      loadData();
    }
  };

  const handleAllColumnsToggle = () => {
    setAllColumns((prev) => !prev);
    // fetchedFor won't match, so useEffect will re-fetch
  };

  // Build generic columns from response
  const columns = useMemo<Array<SortableColumn<Record<string, unknown>>>>(() => {
    const colNames = data?.columns ?? [];
    if (colNames.length === 0 && data?.rows?.length) {
      return Object.keys(data.rows[0]).map(
        (k): SortableColumn<Record<string, unknown>> => ({
          key: k,
          label: k.replace(/_/g, " "),
          headerClassName: "text-left",
          cellClassName: "text-left",
          sortValue: (r) => {
            const v = r[k];
            if (v === null || v === undefined) return null;
            return typeof v === "number" ? v : String(v);
          },
          render: (r) => formatValue(r[k]),
        })
      );
    }
    return colNames.map(
      (k): SortableColumn<Record<string, unknown>> => ({
        key: k,
        label: k.replace(/_/g, " "),
        headerClassName: "text-left",
        cellClassName: "text-left",
        sortValue: (r) => {
          const v = r[k];
          if (v === null || v === undefined) return null;
          return typeof v === "number" ? v : String(v);
        },
        render: (r) => formatValue(r[k]),
      })
    );
  }, [data]);

  // --- Summary line ---
  const label = inventory?.label ?? sourceKey;
  const lastUpdated = inventory?.last_updated;
  const globalRows = inventory?.global_rows ?? 0;
  const countryRows = inventory?.country_rows;
  const hasIso3 = inventory?.has_iso3 ?? true;

  return (
    <div
      className={`rounded-lg border ${
        isEmpty
          ? "border-fred-secondary/50 opacity-50"
          : "border-fred-secondary"
      } bg-fred-surface`}
    >
      {/* Summary row */}
      <button
        type="button"
        className={`flex w-full items-center gap-3 px-4 py-3 text-left text-sm ${
          isEmpty ? "cursor-default" : "cursor-pointer hover:bg-fred-bg/40"
        }`}
        onClick={handleToggle}
        disabled={!!isEmpty}
      >
        <span className="text-fred-muted text-xs w-4">
          {isEmpty ? "" : expanded ? "▼" : "▶"}
        </span>
        <span className="font-semibold text-fred-primary min-w-[160px]">
          {label}
        </span>
        <span className={`text-xs ${freshnessColor(lastUpdated ?? null)}`}>
          {lastUpdated ?? "No data"}
        </span>
        <span className="text-xs text-fred-muted ml-auto tabular-nums">
          {isEmpty ? (
            "No data"
          ) : hasIso3 ? (
            <>
              Global: {globalRows.toLocaleString()} rows
              {countryRows != null && (
                <> | Country: {countryRows.toLocaleString()} rows</>
              )}
            </>
          ) : (
            <>{globalRows.toLocaleString()} rows (global)</>
          )}
        </span>
      </button>

      {/* Expanded table */}
      {expanded && !isEmpty && (
        <div className="border-t border-fred-secondary px-4 pb-4 pt-2 space-y-2">
          <div className="flex items-center gap-3">
            <label className="flex items-center gap-1.5 text-xs text-fred-muted cursor-pointer">
              <input
                type="checkbox"
                checked={allColumns}
                onChange={handleAllColumnsToggle}
                className="rounded border-fred-secondary"
              />
              Show all columns
            </label>
            {data && (
              <span className="text-xs text-fred-muted">
                {data.rows.length.toLocaleString()} rows loaded
              </span>
            )}
          </div>

          {loading && (
            <p className="text-sm text-fred-muted">Loading...</p>
          )}
          {error && <p className="text-sm text-red-400">{error}</p>}

          {data && !loading && (
            <div className="overflow-x-auto rounded-lg border border-fred-secondary max-h-[500px] overflow-y-auto">
              <SortableTable
                columns={columns}
                rows={data.rows}
                rowKey={(r) => JSON.stringify(r)}
                emptyMessage="No data found."
                tableLayout="auto"
                dense
                stickyHeader
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

type ResolverClientProps = {
  countries: ResolverCountryOption[];
  // Legacy props (still accepted but not used for primary rendering)
  connectorStatus?: ConnectorStatusRow[];
  dbSummary?: DbSummaryTable[];
};

export default function ResolverClient({
  countries,
}: ResolverClientProps) {
  const [countryInput, setCountryInput] = useState("");
  const [selectedIso3, setSelectedIso3] = useState<string | null>(null);
  const [inventory, setInventory] = useState<SourceInventoryItem[]>([]);
  const [inventoryLoading, setInventoryLoading] = useState(true);

  const countryByIso3 = useMemo(() => {
    const map = new Map<string, ResolverCountryOption>();
    countries.forEach((c) => {
      if (c.iso3)
        map.set(c.iso3.toUpperCase(), { ...c, iso3: c.iso3.toUpperCase() });
    });
    return map;
  }, [countries]);

  const countryName = selectedIso3
    ? countryByIso3.get(selectedIso3)?.country_name ?? null
    : null;

  // Fetch inventory on mount and when iso3 changes
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setInventoryLoading(true);
      try {
        const params: Record<string, string> = {};
        if (selectedIso3) params.iso3 = selectedIso3;
        const res = await apiGet<{ sources: SourceInventoryItem[] }>(
          "/resolver/source_inventory",
          params
        );
        if (!cancelled) setInventory(res.sources);
      } catch {
        // keep stale inventory on error
      } finally {
        if (!cancelled) setInventoryLoading(false);
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [selectedIso3]);

  const inventoryMap = useMemo(() => {
    const map = new Map<string, SourceInventoryItem>();
    inventory.forEach((s) => map.set(s.key, s));
    return map;
  }, [inventory]);

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-3xl font-semibold">Resolver</h1>
        <p className="text-sm text-fred-muted">
          Browse all data collected by the Resolver pipeline
        </p>
      </section>

      {/* Country selector */}
      <section>
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
                {c.country_name
                  ? `${c.country_name} (${c.iso3})`
                  : c.iso3}
              </option>
            ))}
          </datalist>
          {selectedIso3 && (
            <span className="text-xs text-fred-muted">
              Selected: {countryName ?? selectedIso3}
            </span>
          )}
        </div>
      </section>

      {inventoryLoading && inventory.length === 0 && (
        <p className="text-sm text-fred-muted">Loading data inventory...</p>
      )}

      {/* Category sections with accordion sources */}
      {CATEGORIES.map((cat) => (
        <section key={cat.key} className="space-y-2">
          <h2 className="text-lg font-semibold text-fred-text border-b border-fred-secondary pb-1">
            {cat.title}
          </h2>
          <div className="space-y-1">
            {cat.sources.map((srcKey) => (
              <SourceAccordion
                key={srcKey}
                sourceKey={srcKey}
                inventory={inventoryMap.get(srcKey) ?? null}
                selectedIso3={selectedIso3}
              />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}
