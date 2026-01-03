"use client";

import { useEffect, useMemo, useState } from "react";

import { apiGet } from "../../lib/api";

type HsRunRow = {
  run_id: string;
  triage_date?: string | null;
  countries_triaged?: number | null;
};

type HsTriageRow = {
  triage_date: string | null;
  run_id: string;
  iso3: string;
  country: string | null;
  triage_tier: string | null;
  triage_model: string | null;
  triage_score_1: number | null;
  triage_score_2: number | null;
  triage_score_avg: number | null;
};

type HsTriageResponse = {
  rows: HsTriageRow[];
  diagnostics?: {
    parsed_scores?: number;
    null_scores?: number;
    total_calls?: number;
  };
};

type HsTriageClientProps = {
  initialRuns: HsRunRow[];
  initialRunId: string;
  initialError: string | null;
};

const formatScore = (value: number | null) =>
  value == null ? "" : value.toFixed(2);

export default function HsTriageClient({
  initialRuns,
  initialRunId,
  initialError,
}: HsTriageClientProps) {
  const [selectedRun, setSelectedRun] = useState(initialRunId);
  const [iso3Filter, setIso3Filter] = useState("");
  const [hazardFilter, setHazardFilter] = useState("");
  const [rows, setRows] = useState<HsTriageRow[]>([]);
  const [diagnostics, setDiagnostics] = useState<HsTriageResponse["diagnostics"]>(
    undefined
  );
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(initialError);
  const [sortKey, setSortKey] = useState<"iso3" | "avg">("iso3");

  useEffect(() => {
    if (!selectedRun) return;
    let active = true;
    const fetchRows = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const data = await apiGet<HsTriageResponse>("/hs_triage/all", {
          run_id: selectedRun,
          iso3: iso3Filter.trim() || null,
          hazard_code: hazardFilter.trim() || null,
          limit: 2000,
        });
        if (!active) return;
        setRows(data.rows ?? []);
        setDiagnostics(data.diagnostics);
      } catch (fetchError) {
        if (!active) return;
        setError(
          fetchError instanceof Error
            ? fetchError.message
            : "Failed to load HS triage rows."
        );
      } finally {
        if (active) {
          setIsLoading(false);
        }
      }
    };
    void fetchRows();
    return () => {
      active = false;
    };
  }, [selectedRun, iso3Filter, hazardFilter]);

  const sortedRows = useMemo(() => {
    const copied = [...rows];
    if (sortKey === "avg") {
      return copied.sort((a, b) => {
        const aScore = a.triage_score_avg ?? -Infinity;
        const bScore = b.triage_score_avg ?? -Infinity;
        if (aScore !== bScore) {
          return bScore - aScore;
        }
        return a.iso3.localeCompare(b.iso3);
      });
    }
    return copied.sort((a, b) => a.iso3.localeCompare(b.iso3));
  }, [rows, sortKey]);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end gap-3">
        <label className="flex flex-col gap-1 text-xs text-fred-text">
          Run
          <select
            className="rounded border border-fred-secondary bg-fred-surface px-2 py-1 text-xs text-fred-text"
            value={selectedRun}
            onChange={(event) => setSelectedRun(event.target.value)}
          >
            {initialRuns.map((run) => (
              <option key={run.run_id} value={run.run_id}>
                {run.run_id}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-fred-text">
          ISO3
          <input
            className="w-[10ch] rounded border border-fred-secondary bg-fred-surface px-2 py-1 text-xs text-fred-text"
            placeholder="e.g. UKR"
            value={iso3Filter}
            onChange={(event) => setIso3Filter(event.target.value)}
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-fred-text">
          Hazard
          <input
            className="w-[10ch] rounded border border-fred-secondary bg-fred-surface px-2 py-1 text-xs text-fred-text"
            placeholder="e.g. FL"
            value={hazardFilter}
            onChange={(event) => setHazardFilter(event.target.value)}
          />
        </label>
        <label className="flex items-center gap-2 text-xs text-fred-text">
          Sort
          <select
            className="rounded border border-fred-secondary bg-fred-surface px-2 py-1 text-xs text-fred-text"
            value={sortKey}
            onChange={(event) => setSortKey(event.target.value as "iso3" | "avg")}
          >
            <option value="iso3">ISO3</option>
            <option value="avg">Avg score</option>
          </select>
        </label>
      </div>

      {diagnostics ? (
        <div className="text-xs text-fred-muted">
          Parsed scores: {diagnostics.parsed_scores ?? 0} • Null scores: {" "}
          {diagnostics.null_scores ?? 0} • Calls scanned: {" "}
          {diagnostics.total_calls ?? 0}
        </div>
      ) : null}

      {error ? (
        <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
          {error}
        </div>
      ) : null}

      <div className="overflow-x-auto rounded-lg border border-fred-secondary bg-fred-surface">
        <table className="w-full table-fixed border-collapse text-sm">
          <colgroup>
            <col style={{ width: "10ch" }} />
            <col style={{ width: "18ch" }} />
            <col style={{ width: "6ch" }} />
            <col style={{ width: "18ch" }} />
            <col style={{ width: "12ch" }} />
            <col style={{ width: "16ch" }} />
            <col style={{ width: "10ch" }} />
            <col style={{ width: "10ch" }} />
            <col style={{ width: "10ch" }} />
          </colgroup>
          <thead className="bg-fred-bg text-fred-primary">
            <tr>
              <th className="px-3 py-2 text-left text-xs uppercase tracking-wide">
                Triage Date
              </th>
              <th className="px-3 py-2 text-left text-xs uppercase tracking-wide">
                Run ID
              </th>
              <th className="px-3 py-2 text-left text-xs uppercase tracking-wide">
                ISO3
              </th>
              <th className="px-3 py-2 text-left text-xs uppercase tracking-wide">
                Country
              </th>
              <th className="px-3 py-2 text-left text-xs uppercase tracking-wide">
                Triage Tier
              </th>
              <th className="px-3 py-2 text-left text-xs uppercase tracking-wide">
                Triage Model
              </th>
              <th className="px-3 py-2 text-right text-xs uppercase tracking-wide">
                Score 1
              </th>
              <th className="px-3 py-2 text-right text-xs uppercase tracking-wide">
                Score 2
              </th>
              <th className="px-3 py-2 text-right text-xs uppercase tracking-wide">
                Score Avg
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-fred-border text-fred-text">
            {sortedRows.map((row) => (
              <tr key={`${row.run_id}-${row.iso3}-${row.triage_date ?? ""}`}>
                <td className="px-3 py-2 text-left text-xs">
                  {row.triage_date ?? ""}
                </td>
                <td className="px-3 py-2 text-left text-xs">{row.run_id}</td>
                <td className="px-3 py-2 text-left text-xs font-semibold">
                  {row.iso3}
                </td>
                <td className="px-3 py-2 text-left text-xs">
                  {row.country ?? ""}
                </td>
                <td className="px-3 py-2 text-left text-xs">
                  {row.triage_tier ?? ""}
                </td>
                <td className="px-3 py-2 text-left text-xs">
                  {row.triage_model ?? ""}
                </td>
                <td className="px-3 py-2 text-right text-xs tabular-nums">
                  {formatScore(row.triage_score_1)}
                </td>
                <td className="px-3 py-2 text-right text-xs tabular-nums">
                  {formatScore(row.triage_score_2)}
                </td>
                <td className="px-3 py-2 text-right text-xs tabular-nums">
                  {formatScore(row.triage_score_avg)}
                </td>
              </tr>
            ))}
            {!isLoading && sortedRows.length === 0 ? (
              <tr>
                <td className="px-3 py-3 text-fred-muted" colSpan={9}>
                  No HS triage rows found for this run.
                </td>
              </tr>
            ) : null}
            {isLoading ? (
              <tr>
                <td className="px-3 py-3 text-fred-muted" colSpan={9}>
                  Loading HS triage rows...
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </div>
  );
}
