"use client";

import { useEffect, useMemo, useState } from "react";

import SortableTable, { SortableColumn } from "../../../components/SortableTable";
import { apiGet } from "../../../lib/api";

type ContextSubTab = "hdx_signals" | "crisiswatch";
type GenericRow = Record<string, unknown>;

const formatCell = (v: unknown) => {
  if (v === null || v === undefined) return "";
  if (typeof v === "number") return v.toLocaleString(undefined, { maximumFractionDigits: 2 });
  const s = String(v);
  return s.length > 150 ? s.slice(0, 150) + "..." : s;
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

export default function ContextSignalsTab({ selectedIso3, countryName }: Props) {
  const [subTab, setSubTab] = useState<ContextSubTab>("hdx_signals");
  const [hdxRows, setHdxRows] = useState<GenericRow[]>([]);
  const [cwRows, setCwRows] = useState<GenericRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedIso3) { setHdxRows([]); setCwRows([]); return; }
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const endpoint = subTab === "hdx_signals"
          ? "/resolver/hdx_signals"
          : "/resolver/crisiswatch";
        const res = await apiGet<{ rows: GenericRow[] }>(endpoint, {
          iso3: selectedIso3, limit: 1000,
        });
        if (!cancelled) {
          if (subTab === "hdx_signals") setHdxRows(res.rows ?? []);
          else setCwRows(res.rows ?? []);
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

  const activeRows = subTab === "hdx_signals" ? hdxRows : cwRows;
  const cols = useMemo(() => genericColumns(activeRows), [activeRows]);

  const subTabs: { key: ContextSubTab; label: string }[] = [
    { key: "hdx_signals", label: "HDX Signals" },
    { key: "crisiswatch", label: "CrisisWatch" },
  ];

  if (!selectedIso3) {
    return <p className="text-sm text-fred-muted">Select a country to view context signals.</p>;
  }

  return (
    <div className="space-y-4">
      <div className="flex gap-2 border-b border-fred-secondary">
        {subTabs.map((t) => (
          <button key={t.key}
            className={`px-3 py-1.5 text-sm font-medium border-b-2 transition-colors ${
              subTab === t.key
                ? "border-fred-primary text-fred-primary"
                : "border-transparent text-fred-muted hover:text-fred-text"
            }`}
            onClick={() => setSubTab(t.key)}>
            {t.label}
          </button>
        ))}
      </div>

      <div className="text-sm text-fred-muted">
        {countryName ?? selectedIso3} — {activeRows.length.toLocaleString()} rows
      </div>

      {loading && <p className="text-sm text-fred-muted">Loading...</p>}
      {error && <p className="text-sm text-red-400">{error}</p>}

      <div className="overflow-x-auto rounded-lg border border-fred-secondary">
        <SortableTable
          columns={cols}
          rows={activeRows}
          rowKey={(r) => JSON.stringify(r)}
          emptyMessage="No data found."
          tableLayout="auto"
          dense
        />
      </div>
    </div>
  );
}
