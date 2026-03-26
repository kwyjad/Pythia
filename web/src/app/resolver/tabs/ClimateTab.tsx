"use client";

import { useEffect, useMemo, useState } from "react";

import SortableTable, { SortableColumn } from "../../../components/SortableTable";
import { apiGet } from "../../../lib/api";

type GenericRow = Record<string, unknown>;

const formatCell = (v: unknown) => {
  if (v === null || v === undefined) return "";
  if (typeof v === "number") return v.toLocaleString(undefined, { maximumFractionDigits: 4 });
  return String(v);
};

function genericColumns(rows: GenericRow[]): Array<SortableColumn<GenericRow>> {
  if (rows.length === 0) return [];
  return Object.keys(rows[0]).map((k) => ({
    key: k,
    label: k.replace(/_/g, " "),
    headerClassName: "text-left",
    cellClassName: "text-left",
    sortValue: (r: GenericRow) => {
      const v = r[k];
      if (v === null || v === undefined) return null;
      return typeof v === "number" ? v : String(v);
    },
    render: (r: GenericRow) => formatCell(r[k]),
  }));
}

type Props = { selectedIso3: string | null; countryName: string | null };

export default function ClimateTab({ selectedIso3, countryName }: Props) {
  const [rows, setRows] = useState<GenericRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedIso3) { setRows([]); return; }
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await apiGet<{ rows: GenericRow[] }>(
          "/resolver/seasonal_forecasts",
          { iso3: selectedIso3, limit: 2000 }
        );
        if (!cancelled) setRows(res.rows ?? []);
      } catch {
        if (!cancelled) setError("Failed to load seasonal forecasts.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => { cancelled = true; };
  }, [selectedIso3]);

  const cols = useMemo(() => genericColumns(rows), [rows]);

  if (!selectedIso3) {
    return <p className="text-sm text-fred-muted">Select a country to view climate data.</p>;
  }

  return (
    <div className="space-y-4">
      <div className="text-sm text-fred-muted">
        NMME Seasonal Forecasts — {countryName ?? selectedIso3} — {rows.length.toLocaleString()} rows
      </div>
      {loading && <p className="text-sm text-fred-muted">Loading...</p>}
      {error && <p className="text-sm text-red-400">{error}</p>}
      <div className="overflow-x-auto rounded-lg border border-fred-secondary">
        <SortableTable
          columns={cols}
          rows={rows}
          rowKey={(r) => JSON.stringify(r)}
          emptyMessage="No seasonal forecast data found."
          tableLayout="auto"
          dense
        />
      </div>
    </div>
  );
}
