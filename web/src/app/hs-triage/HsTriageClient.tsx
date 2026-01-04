"use client";

import { useEffect, useMemo, useState } from "react";

import { apiGet } from "../../lib/api";
import SortableTable, { SortableColumn } from "../../components/SortableTable";

type HsRunRow = {
  run_id: string;
  triage_date?: string | null;
  countries_triaged?: number | null;
};

type HsTriageRow = {
  triage_date: string | null;
  run_id: string;
  iso3: string;
  hazard_code: string | null;
  hazard_label: string | null;
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
    rows_with_avg?: number;
    rows_returned?: number;
    avg_from_hs_triage_score?: number;
    countries_with_two_calls?: number;
    countries_with_one_call?: number;
    score_avg_from_calls?: number;
    score_avg_from_hs_triage?: number;
  };
};

type HsTriageClientProps = {
  initialRuns: HsRunRow[];
  initialRunId: string;
  initialError: string | null;
};

const formatScore = (value: number | null) =>
  value == null ? "—" : value.toFixed(2);

export default function HsTriageClient({
  initialRuns,
  initialRunId,
  initialError,
}: HsTriageClientProps) {
  const [selectedRun, setSelectedRun] = useState(initialRunId);
  const [iso3Filter, setIso3Filter] = useState("");
  const [countryFilter, setCountryFilter] = useState("");
  const [hazardFilter, setHazardFilter] = useState("");
  const [rows, setRows] = useState<HsTriageRow[]>([]);
  const [diagnostics, setDiagnostics] = useState<HsTriageResponse["diagnostics"]>(
    undefined
  );
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(initialError);

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

  const filteredRows = useMemo(() => {
    const normalizedCountry = countryFilter.trim().toLowerCase();
    const visibleRows = rows.filter((row) =>
      normalizedCountry
        ? row.country?.toLowerCase().includes(normalizedCountry)
        : true
    );
    return [...visibleRows].sort((a, b) => {
      const isoCompare = a.iso3.localeCompare(b.iso3);
      if (isoCompare !== 0) {
        return isoCompare;
      }
      return (a.hazard_code ?? "").localeCompare(b.hazard_code ?? "");
    });
  }, [countryFilter, rows]);

  const columns = useMemo<Array<SortableColumn<HsTriageRow>>>(
    () => [
      {
        key: "triage_date",
        label: "Triage Date",
        headerClassName: "text-xs uppercase tracking-wide",
        cellClassName: "text-xs",
        render: (row) => row.triage_date ?? "",
        sortValue: (row) => row.triage_date ?? "",
      },
      {
        key: "run_id",
        label: "Run ID",
        headerClassName: "text-xs uppercase tracking-wide",
        cellClassName: "text-xs",
        render: (row) => row.run_id,
      },
      {
        key: "iso3",
        label: "ISO3",
        headerClassName: "text-xs uppercase tracking-wide",
        cellClassName: "text-xs font-semibold",
        render: (row) => row.iso3,
      },
      {
        key: "country",
        label: "Country",
        headerClassName: "text-xs uppercase tracking-wide",
        cellClassName: "text-xs",
        render: (row) => row.country ?? "",
      },
      {
        key: "hazard_code",
        label: "Hazard Code",
        headerClassName: "text-xs uppercase tracking-wide",
        cellClassName: "text-xs",
        render: (row) => row.hazard_code ?? "",
      },
      {
        key: "hazard_label",
        label: "Hazard Label",
        headerClassName: "text-xs uppercase tracking-wide",
        cellClassName: "text-xs",
        render: (row) => row.hazard_label ?? "",
      },
      {
        key: "triage_tier",
        label: "Triage Tier",
        headerClassName: "text-xs uppercase tracking-wide",
        cellClassName: "text-xs",
        render: (row) => row.triage_tier ?? "",
      },
      {
        key: "triage_model",
        label: "Triage Model",
        headerClassName: "text-xs uppercase tracking-wide",
        cellClassName: "text-xs",
        render: (row) => row.triage_model ?? "",
      },
      {
        key: "triage_score_1",
        label: "Score 1",
        headerClassName: "text-xs uppercase tracking-wide text-right",
        cellClassName: "text-xs tabular-nums text-right",
        render: (row) => formatScore(row.triage_score_1),
        sortValue: (row) => row.triage_score_1,
      },
      {
        key: "triage_score_2",
        label: "Score 2",
        headerClassName: "text-xs uppercase tracking-wide text-right",
        cellClassName: "text-xs tabular-nums text-right",
        render: (row) => formatScore(row.triage_score_2),
        sortValue: (row) => row.triage_score_2,
      },
      {
        key: "triage_score_avg",
        label: "Score Avg",
        headerClassName: "text-xs uppercase tracking-wide text-right",
        cellClassName: "text-xs tabular-nums text-right",
        render: (row) => formatScore(row.triage_score_avg),
        sortValue: (row) => row.triage_score_avg,
        defaultSortDirection: "desc",
      },
    ],
    []
  );

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
          Country
          <input
            className="w-[18ch] rounded border border-fred-secondary bg-fred-surface px-2 py-1 text-xs text-fred-text"
            placeholder="e.g. Ukraine"
            value={countryFilter}
            onChange={(event) => setCountryFilter(event.target.value)}
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
      </div>

      {diagnostics ? (
        <div className="text-xs text-fred-muted">
          Parsed scores: {diagnostics.parsed_scores ?? 0} • Null scores: {" "}
          {diagnostics.null_scores ?? 0} • Calls scanned: {" "}
          {diagnostics.total_calls ?? 0}
          <div>
            Rows with avg: {diagnostics.rows_with_avg ?? 0} /{" "}
            {diagnostics.rows_returned ?? rows.length} • Avg from hs_triage:{" "}
            {diagnostics.avg_from_hs_triage_score ?? 0}
          </div>
          <div>
            Countries with two calls: {diagnostics.countries_with_two_calls ?? 0} •{" "}
            one call: {diagnostics.countries_with_one_call ?? 0}
          </div>
          <div>
            Avg from calls: {diagnostics.score_avg_from_calls ?? 0} • avg from
            hs_triage: {diagnostics.score_avg_from_hs_triage ?? 0}
          </div>
        </div>
      ) : null}

      {error ? (
        <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
          {error}
        </div>
      ) : null}

      <div className="overflow-x-auto rounded-lg border border-fred-secondary bg-fred-surface">
        <SortableTable
          columns={columns}
          rows={isLoading ? [] : filteredRows}
          rowKey={(row) =>
            `${row.run_id}-${row.iso3}-${row.hazard_code ?? "unknown"}-${
              row.triage_date ?? ""
            }`
          }
          initialSortKey="triage_score_avg"
          initialSortDirection="desc"
          emptyMessage={
            isLoading
              ? "Loading HS triage rows..."
              : "No HS triage rows found for this run."
          }
          colGroup={
            <>
              <col style={{ width: "10ch" }} />
              <col style={{ width: "18ch" }} />
              <col style={{ width: "6ch" }} />
              <col style={{ width: "18ch" }} />
              <col style={{ width: "8ch" }} />
              <col style={{ width: "16ch" }} />
              <col style={{ width: "12ch" }} />
              <col style={{ width: "16ch" }} />
              <col style={{ width: "10ch" }} />
              <col style={{ width: "10ch" }} />
              <col style={{ width: "10ch" }} />
            </>
          }
        />
      </div>
    </div>
  );
}
