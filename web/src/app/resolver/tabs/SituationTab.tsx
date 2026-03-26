"use client";

import { useEffect, useMemo, useState } from "react";

import SortableTable, { SortableColumn } from "../../../components/SortableTable";
import { apiGet } from "../../../lib/api";

type SituationSubTab = "reliefweb" | "acled_political" | "acaps";
type AcapsDataset = "inform_severity" | "risk_radar" | "daily_monitoring" | "humanitarian_access";

type GenericRow = Record<string, unknown>;

const formatCell = (v: unknown) => {
  if (v === null || v === undefined) return "";
  if (typeof v === "number") return v.toLocaleString(undefined, { maximumFractionDigits: 2 });
  const s = String(v);
  return s.length > 120 ? s.slice(0, 120) + "..." : s;
};

function useGenericFetch(endpoint: string, iso3: string | null, params?: Record<string, string | number | boolean>) {
  const [rows, setRows] = useState<GenericRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!iso3) { setRows([]); return; }
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await apiGet<{ rows: GenericRow[] }>(endpoint, { iso3, ...params });
        if (!cancelled) setRows(res.rows ?? []);
      } catch {
        if (!cancelled) setError("Failed to load data.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [endpoint, iso3, JSON.stringify(params)]);

  return { rows, loading, error };
}

function genericColumns(rows: GenericRow[], skipCols?: Set<string>): Array<SortableColumn<GenericRow>> {
  if (rows.length === 0) return [];
  const keys = Object.keys(rows[0]).filter((k) => !skipCols?.has(k));
  return keys.map((k) => ({
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

export default function SituationTab({ selectedIso3, countryName }: Props) {
  const [subTab, setSubTab] = useState<SituationSubTab>("reliefweb");
  const [acapsDataset, setAcapsDataset] = useState<AcapsDataset>("inform_severity");

  const reliefweb = useGenericFetch("/resolver/reliefweb_reports", subTab === "reliefweb" ? selectedIso3 : null);
  const acledPol = useGenericFetch("/resolver/acled_political_events", subTab === "acled_political" ? selectedIso3 : null);
  const acaps = useGenericFetch("/resolver/acaps", subTab === "acaps" ? selectedIso3 : null, { dataset: acapsDataset });

  const activeData = subTab === "reliefweb" ? reliefweb : subTab === "acled_political" ? acledPol : acaps;

  const cols = useMemo(() => {
    const skip = subTab === "reliefweb" ? new Set(["body"]) : undefined;
    return genericColumns(activeData.rows, skip);
  }, [activeData.rows, subTab]);

  const subTabs: { key: SituationSubTab; label: string }[] = [
    { key: "reliefweb", label: "ReliefWeb" },
    { key: "acled_political", label: "ACLED Political" },
    { key: "acaps", label: "ACAPS" },
  ];

  const acapsOptions: { key: AcapsDataset; label: string }[] = [
    { key: "inform_severity", label: "INFORM Severity" },
    { key: "risk_radar", label: "Risk Radar" },
    { key: "daily_monitoring", label: "Daily Monitoring" },
    { key: "humanitarian_access", label: "Humanitarian Access" },
  ];

  if (!selectedIso3) {
    return <p className="text-sm text-fred-muted">Select a country to view situation reports.</p>;
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4 border-b border-fred-secondary">
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

      {subTab === "acaps" && (
        <div className="flex gap-2">
          {acapsOptions.map((opt) => (
            <button key={opt.key}
              className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                acapsDataset === opt.key
                  ? "bg-fred-primary text-white"
                  : "bg-fred-surface text-fred-muted border border-fred-secondary hover:text-fred-text"
              }`}
              onClick={() => setAcapsDataset(opt.key)}>
              {opt.label}
            </button>
          ))}
        </div>
      )}

      <div className="text-sm text-fred-muted">
        {countryName ?? selectedIso3} — {activeData.rows.length.toLocaleString()} rows
      </div>

      {activeData.loading && <p className="text-sm text-fred-muted">Loading...</p>}
      {activeData.error && <p className="text-sm text-red-400">{activeData.error}</p>}

      <div className="overflow-x-auto rounded-lg border border-fred-secondary">
        <SortableTable
          columns={cols}
          rows={activeData.rows}
          rowKey={(r) => JSON.stringify(r)}
          emptyMessage="No data found."
          tableLayout="auto"
          dense
        />
      </div>
    </div>
  );
}
