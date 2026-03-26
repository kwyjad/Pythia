"use client";

import { useEffect, useMemo, useState } from "react";

import SortableTable, { SortableColumn } from "../../../components/SortableTable";
import { apiGet } from "../../../lib/api";

type FactsSubTab = "facts_resolved" | "facts_deltas" | "acled_fatalities";

type ResolverFactRow = {
  iso3: string;
  hazard: string | null;
  hazard_code: string | null;
  source_id: string | null;
  year: number | null;
  month: number | null;
  metric: string | null;
  value: number | null;
};

type DeltaRow = Record<string, unknown>;
type FatalityRow = Record<string, unknown>;

const formatValue = (value: unknown) => {
  if (value === null || value === undefined) return "";
  const num = Number(value);
  if (Number.isNaN(num)) return String(value);
  return num.toLocaleString(undefined, { maximumFractionDigits: 2 });
};

const buildHazardDisplay = (row: ResolverFactRow) => {
  const code = (row.hazard_code ?? "").trim();
  const label = (row.hazard ?? "").trim();
  if (label && code && label.toUpperCase() !== code.toUpperCase()) {
    return `${label} (${code})`;
  }
  return label || code || "Unknown";
};

type FactsTabProps = {
  selectedIso3: string | null;
  countryName: string | null;
};

export default function FactsTab({ selectedIso3, countryName }: FactsTabProps) {
  const [subTab, setSubTab] = useState<FactsSubTab>("facts_resolved");
  const [factsRows, setFactsRows] = useState<ResolverFactRow[]>([]);
  const [deltaRows, setDeltaRows] = useState<DeltaRow[]>([]);
  const [fatalityRows, setFatalityRows] = useState<FatalityRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [sourceFilter, setSourceFilter] = useState("");
  const [hazardFilter, setHazardFilter] = useState("");
  const [metricFilter, setMetricFilter] = useState("");

  useEffect(() => {
    if (!selectedIso3) {
      setFactsRows([]);
      setDeltaRows([]);
      setFatalityRows([]);
      return;
    }
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        if (subTab === "facts_resolved") {
          const res = await apiGet<{ rows: ResolverFactRow[] }>(
            "/resolver/country_facts",
            { iso3: selectedIso3 }
          );
          if (!cancelled) setFactsRows(res.rows ?? []);
        } else if (subTab === "facts_deltas") {
          const res = await apiGet<{ rows: DeltaRow[] }>(
            "/resolver/facts_deltas",
            { iso3: selectedIso3, limit: 2000 }
          );
          if (!cancelled) setDeltaRows(res.rows ?? []);
        } else {
          const res = await apiGet<{ rows: FatalityRow[] }>(
            "/resolver/acled_monthly_fatalities",
            { iso3: selectedIso3, limit: 2000 }
          );
          if (!cancelled) setFatalityRows(res.rows ?? []);
        }
      } catch {
        if (!cancelled) setError("Failed to load data.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => { cancelled = true; };
  }, [selectedIso3, subTab]);

  // --- facts_resolved view ---
  const factsView = useMemo(() => {
    return factsRows.map((row) => ({
      ...row,
      hazard_display: buildHazardDisplay(row),
      hazard_key: row.hazard_code ?? row.hazard ?? "",
      source_key: row.source_id ?? "",
      metric_key: row.metric ?? "",
    }));
  }, [factsRows]);

  const filteredFacts = useMemo(() => {
    return factsView.filter((row) => {
      if (sourceFilter && row.source_key !== sourceFilter) return false;
      if (hazardFilter && row.hazard_key !== hazardFilter) return false;
      if (metricFilter && row.metric_key !== metricFilter) return false;
      return true;
    });
  }, [factsView, sourceFilter, hazardFilter, metricFilter]);

  const sourceOptions = useMemo(() => {
    const s = new Set<string>();
    factsView.forEach((r) => { if (r.source_key) s.add(r.source_key); });
    return Array.from(s).sort();
  }, [factsView]);

  const hazardOptions = useMemo(() => {
    const m = new Map<string, string>();
    factsView.forEach((r) => {
      if (r.hazard_key) m.set(r.hazard_key, r.hazard_display);
    });
    return Array.from(m.entries()).sort((a, b) => a[1].localeCompare(b[1]));
  }, [factsView]);

  const metricOptions = useMemo(() => {
    const s = new Set<string>();
    factsView.forEach((r) => { if (r.metric_key) s.add(r.metric_key); });
    return Array.from(s).sort();
  }, [factsView]);

  const factsColumns = useMemo<Array<SortableColumn<typeof factsView[0]>>>(
    () => [
      { key: "ym_sort", label: "Year-Month", isVisible: false,
        sortValue: (r) => (r.year && r.month) ? r.year * 100 + r.month : null,
        defaultSortDirection: "desc" },
      { key: "hazard_display", label: "Hazard", headerClassName: "text-left",
        cellClassName: "text-left", sortValue: (r) => r.hazard_display,
        render: (r) => r.hazard_display, defaultSortDirection: "asc" },
      { key: "source_id", label: "Source", headerClassName: "text-left",
        cellClassName: "text-left", sortValue: (r) => r.source_id ?? "",
        render: (r) => r.source_id ?? "", defaultSortDirection: "asc" },
      { key: "year", label: "Year", headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (r) => r.year, render: (r) => r.year?.toString() ?? "",
        defaultSortDirection: "desc" },
      { key: "month", label: "Month", headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (r) => r.month, render: (r) => r.month?.toString() ?? "",
        defaultSortDirection: "desc" },
      { key: "metric", label: "Metric", headerClassName: "text-left",
        cellClassName: "text-left", sortValue: (r) => r.metric_key,
        render: (r) => r.metric_key, defaultSortDirection: "asc" },
      { key: "value", label: "Value", headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (r) => r.value, render: (r) => formatValue(r.value),
        defaultSortDirection: "desc" },
    ],
    []
  );

  // --- generic columns for deltas / fatalities ---
  const genericColumns = (rows: Record<string, unknown>[]) => {
    if (rows.length === 0) return [];
    const keys = Object.keys(rows[0]);
    return keys.map((k): SortableColumn<Record<string, unknown>> => ({
      key: k, label: k.replace(/_/g, " "),
      headerClassName: "text-left", cellClassName: "text-left",
      sortValue: (r) => {
        const v = r[k];
        if (v === null || v === undefined) return null;
        return typeof v === "number" ? v : String(v);
      },
      render: (r) => formatValue(r[k]),
    }));
  };

  const subTabs: { key: FactsSubTab; label: string }[] = [
    { key: "facts_resolved", label: "Facts Resolved" },
    { key: "facts_deltas", label: "Facts Deltas" },
    { key: "acled_fatalities", label: "ACLED Fatalities" },
  ];

  const activeRows =
    subTab === "facts_resolved" ? filteredFacts :
    subTab === "facts_deltas" ? deltaRows : fatalityRows;

  return (
    <div className="space-y-4">
      <div className="flex gap-2 border-b border-fred-secondary">
        {subTabs.map((t) => (
          <button
            key={t.key}
            className={`px-3 py-1.5 text-sm font-medium border-b-2 transition-colors ${
              subTab === t.key
                ? "border-fred-primary text-fred-primary"
                : "border-transparent text-fred-muted hover:text-fred-text"
            }`}
            onClick={() => setSubTab(t.key)}
          >
            {t.label}
          </button>
        ))}
      </div>

      {!selectedIso3 && (
        <p className="text-sm text-fred-muted">Select a country to view facts data.</p>
      )}

      {selectedIso3 && subTab === "facts_resolved" && (
        <div className="flex flex-wrap gap-3">
          <select className="rounded-md border border-fred-secondary bg-fred-surface px-2 py-1 text-sm text-fred-text"
            value={sourceFilter} onChange={(e) => setSourceFilter(e.target.value)}>
            <option value="">All Sources</option>
            {sourceOptions.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
          <select className="rounded-md border border-fred-secondary bg-fred-surface px-2 py-1 text-sm text-fred-text"
            value={hazardFilter} onChange={(e) => setHazardFilter(e.target.value)}>
            <option value="">All Hazards</option>
            {hazardOptions.map(([k, v]) => <option key={k} value={k}>{v}</option>)}
          </select>
          <select className="rounded-md border border-fred-secondary bg-fred-surface px-2 py-1 text-sm text-fred-text"
            value={metricFilter} onChange={(e) => setMetricFilter(e.target.value)}>
            <option value="">All Metrics</option>
            {metricOptions.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>
      )}

      {loading && <p className="text-sm text-fred-muted">Loading...</p>}
      {error && <p className="text-sm text-red-400">{error}</p>}

      {selectedIso3 && (
        <>
          <div className="text-sm text-fred-muted">
            {countryName ?? selectedIso3} — {activeRows.length.toLocaleString()} rows
          </div>
          <div className="overflow-x-auto rounded-lg border border-fred-secondary">
            {subTab === "facts_resolved" ? (
              <SortableTable
                columns={factsColumns}
                rows={filteredFacts}
                rowKey={(r) => `${r.hazard_key}-${r.source_key}-${r.metric_key}-${r.year}-${r.month}`}
                initialSortKey="ym_sort"
                initialSortDirection="desc"
                emptyMessage="No facts found."
                tableLayout="auto"
                dense
              />
            ) : (
              <SortableTable
                columns={genericColumns(activeRows)}
                rows={activeRows}
                rowKey={(r) => JSON.stringify(r)}
                emptyMessage="No data found."
                tableLayout="auto"
                dense
              />
            )}
          </div>
        </>
      )}
    </div>
  );
}
