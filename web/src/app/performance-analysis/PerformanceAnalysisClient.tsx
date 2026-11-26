"use client";
// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
// See the LICENSE file in the project root for details.

import { usePathname, useRouter } from "next/navigation";
import { useMemo, useState } from "react";

import type {
  AnalysisReportsResponse,
  ReportFilters,
  ReportType,
} from "../../lib/analysis_reports";
import {
  REPORT_TYPES,
  TYPE_LABELS,
  applyFilters,
  dateLabel,
  filtersToQuery,
  periodLabel,
} from "../../lib/analysis_reports";
import ReportRow from "./ReportRow";

export default function PerformanceAnalysisClient({
  data,
  initialFilters,
}: {
  data: AnalysisReportsResponse;
  initialFilters: ReportFilters;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const [filters, setFilters] = useState<ReportFilters>(initialFilters);

  const cycles = useMemo(() => applyFilters(data.cycles, filters), [data.cycles, filters]);

  const update = (next: ReportFilters) => {
    setFilters(next);
    const qs = filtersToQuery(next);
    router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
  };

  const toggleType = (t: ReportType) => {
    const types = filters.types.includes(t)
      ? filters.types.filter((x) => x !== t)
      : [...filters.types, t];
    update({ ...filters, types });
  };

  return (
    <div className="space-y-4">
      <div
        className="flex flex-wrap items-center gap-x-6 gap-y-3 rounded-md border border-fred-border bg-fred-surface p-3 shadow-fredCard"
        role="group"
        aria-label="Filter reports"
      >
        <div className="flex flex-wrap gap-2">
          {REPORT_TYPES.map((t) => {
            const on = filters.types.includes(t);
            return (
              <button
                key={t}
                type="button"
                aria-pressed={on}
                onClick={() => toggleType(t)}
                className={`rounded-full border px-3 py-1 text-xs font-medium focus:outline-none focus-visible:ring-2 focus-visible:ring-fred-primary ${
                  on
                    ? "border-fred-primary bg-fred-primary text-white"
                    : "border-fred-border bg-fred-surface text-fred-muted hover:text-fred-text"
                }`}
              >
                {TYPE_LABELS[t]}
              </button>
            );
          })}
        </div>
        <label className="flex items-center gap-2 text-xs text-fred-text">
          Year
          <select
            value={filters.year ?? ""}
            onChange={(e) => update({ ...filters, year: e.target.value || null })}
            className="rounded-md border border-fred-border bg-fred-surface px-2 py-1 text-xs focus:outline-none focus-visible:ring-2 focus-visible:ring-fred-primary"
          >
            <option value="">All years</option>
            {data.years.map((y) => (
              <option key={y} value={y}>
                {y}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2 text-xs text-fred-text">
          <input
            type="checkbox"
            checked={filters.showDryRuns}
            onChange={(e) => update({ ...filters, showDryRuns: e.target.checked })}
            className="h-4 w-4 accent-fred-primary"
          />
          Show dry runs
        </label>
      </div>

      {data.stale && (
        <p className="text-xs text-fred-muted">
          The report list could not be refreshed just now; this is the last copy the
          dashboard loaded.
        </p>
      )}

      {cycles.length === 0 ? (
        <p className="rounded-md border border-fred-border bg-fred-surface p-4 text-sm text-fred-text">
          No reports match these filters.
        </p>
      ) : (
        cycles.map((cycle, idx) => (
          <details
            key={cycle.period}
            open={idx === 0}
            className="group rounded-md border border-fred-border bg-fred-surface shadow-fredCard"
            data-testid="report-cycle"
          >
            <summary className="flex cursor-pointer items-center justify-between gap-2 px-4 py-3 text-lg font-semibold focus:outline-none focus-visible:ring-2 focus-visible:ring-fred-primary">
              <span>{periodLabel(cycle.period)}</span>
              <span aria-hidden className="text-sm text-fred-muted transition group-open:rotate-90">
                ›
              </span>
            </summary>
            <div className="space-y-4 border-t border-fred-border px-4 py-3">
              {cycle.scoring_rounds.map((round) => (
                <section key={round.publish_run_id ?? "unknown"}>
                  <h3 className="text-sm font-semibold text-fred-secondary">
                    Scoring round{round.date ? ` · ${dateLabel(round.date)}` : ""}
                    {round.publish_run_id && (
                      <span className="ml-2 font-mono text-xs font-normal text-fred-muted">
                        publish run {round.publish_run_id}
                      </span>
                    )}
                  </h3>
                  <ul className="divide-y divide-fred-border">
                    {round.reports.map((entry) => (
                      <ReportRow key={entry.report_id ?? `${entry.type}-placeholder`} entry={entry} />
                    ))}
                  </ul>
                </section>
              ))}
              {cycle.forecast_runs.length > 0 && (
                <section>
                  <h3 className="text-sm font-semibold text-fred-secondary">Forecast runs</h3>
                  {cycle.forecast_runs.map((run) => (
                    <div key={run.fc_run_id} className="mt-2 border-l-2 border-fred-border pl-3">
                      <p className="text-xs text-fred-muted">
                        {dateLabel(run.date) ?? "Date unknown"}
                        <span className="ml-2 font-mono">{run.fc_run_id}</span>
                      </p>
                      <ul className="divide-y divide-fred-border">
                        {run.reports.map((entry) => (
                          <ReportRow
                            key={entry.report_id ?? `${run.fc_run_id}-placeholder`}
                            entry={entry}
                          />
                        ))}
                      </ul>
                    </div>
                  ))}
                </section>
              )}
            </div>
          </details>
        ))
      )}
    </div>
  );
}
