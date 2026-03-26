"use client";

import { useEffect, useMemo, useState } from "react";

import SortableTable, { SortableColumn } from "../../../components/SortableTable";
import { apiGet } from "../../../lib/api";

type ConflictRow = {
  source: string;
  iso3: string;
  hazard_code: string;
  metric: string;
  lead_months: number | null;
  forecast_issue_date: string | null;
  value: number | null;
  target_month: string | null;
  model_version: string | null;
};

type Props = { selectedIso3: string | null; countryName: string | null };

const formatValue = (v: unknown) => {
  if (v === null || v === undefined) return "";
  const num = Number(v);
  if (Number.isNaN(num)) return String(v);
  return num.toLocaleString(undefined, { maximumFractionDigits: 4 });
};

export default function ConflictForecastsTab({ selectedIso3, countryName }: Props) {
  const [rows, setRows] = useState<ConflictRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sourceFilter, setSourceFilter] = useState("");

  useEffect(() => {
    if (!selectedIso3) { setRows([]); return; }
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await apiGet<{ rows: ConflictRow[] }>(
          "/resolver/conflict_forecasts",
          { iso3: selectedIso3, limit: 2000 }
        );
        if (!cancelled) setRows(res.rows ?? []);
      } catch {
        if (!cancelled) setError("Failed to load conflict forecasts.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => { cancelled = true; };
  }, [selectedIso3]);

  const sourceOptions = useMemo(() => {
    const s = new Set<string>();
    rows.forEach((r) => { if (r.source) s.add(r.source); });
    return Array.from(s).sort();
  }, [rows]);

  const filtered = useMemo(() => {
    if (!sourceFilter) return rows;
    return rows.filter((r) => r.source === sourceFilter);
  }, [rows, sourceFilter]);

  const columns = useMemo<Array<SortableColumn<ConflictRow>>>(
    () => [
      { key: "source", label: "Source", headerClassName: "text-left",
        cellClassName: "text-left", sortValue: (r) => r.source,
        render: (r) => r.source, defaultSortDirection: "asc" },
      { key: "metric", label: "Metric", headerClassName: "text-left",
        cellClassName: "text-left", sortValue: (r) => r.metric,
        render: (r) => r.metric, defaultSortDirection: "asc" },
      { key: "lead_months", label: "Lead (mo)", headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (r) => r.lead_months,
        render: (r) => r.lead_months?.toString() ?? "", defaultSortDirection: "asc" },
      { key: "value", label: "Value", headerClassName: "text-right",
        cellClassName: "text-right tabular-nums",
        sortValue: (r) => r.value, render: (r) => formatValue(r.value),
        defaultSortDirection: "desc" },
      { key: "forecast_issue_date", label: "Issue Date", headerClassName: "text-left",
        cellClassName: "text-left", sortValue: (r) => r.forecast_issue_date ?? "",
        render: (r) => r.forecast_issue_date ?? "", defaultSortDirection: "desc" },
      { key: "target_month", label: "Target Month", headerClassName: "text-left",
        cellClassName: "text-left", sortValue: (r) => r.target_month ?? "",
        render: (r) => r.target_month ?? "", defaultSortDirection: "desc" },
    ],
    []
  );

  if (!selectedIso3) {
    return <p className="text-sm text-fred-muted">Select a country to view conflict forecasts.</p>;
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <select className="rounded-md border border-fred-secondary bg-fred-surface px-2 py-1 text-sm text-fred-text"
          value={sourceFilter} onChange={(e) => setSourceFilter(e.target.value)}>
          <option value="">All Sources</option>
          {sourceOptions.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
        <span className="text-sm text-fred-muted">
          {countryName ?? selectedIso3} — {filtered.length.toLocaleString()} rows
        </span>
      </div>
      {loading && <p className="text-sm text-fred-muted">Loading...</p>}
      {error && <p className="text-sm text-red-400">{error}</p>}
      <div className="overflow-x-auto rounded-lg border border-fred-secondary">
        <SortableTable
          columns={columns}
          rows={filtered}
          rowKey={(r) => `${r.source}-${r.metric}-${r.lead_months}-${r.forecast_issue_date}`}
          initialSortKey="forecast_issue_date"
          initialSortDirection="desc"
          emptyMessage="No conflict forecasts found."
          tableLayout="auto"
          dense
        />
      </div>
    </div>
  );
}
