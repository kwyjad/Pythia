// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
// See the LICENSE file in the project root for details.

import { apiGet } from "../../lib/api";
import type { AnalysisReportsResponse } from "../../lib/analysis_reports";
import { filtersFromParams } from "../../lib/analysis_reports";
import PerformanceAnalysisClient from "./PerformanceAnalysisClient";

export const dynamic = "force-dynamic";
export const revalidate = 0;

export default async function PerformanceAnalysisPage({
  searchParams,
}: {
  searchParams: { [key: string]: string | string[] | undefined };
}) {
  let data: AnalysisReportsResponse | null = null;
  let loadError: string | null = null;
  try {
    data = await apiGet<AnalysisReportsResponse>("/analysis_reports");
  } catch (error) {
    // A degraded API must never look like "no reports": say so.
    loadError = "The report list could not be loaded right now.";
    console.warn("Failed to load analysis reports:", error);
  }

  return (
    <div className="space-y-6">
      <section className="space-y-3">
        <h1 className="text-3xl font-semibold">Performance Analysis</h1>
        <p className="max-w-3xl text-sm leading-relaxed text-fred-text">
          Three kinds of report look back at how Fred did. The{" "}
          <strong>skill analysis</strong> asks how good the forecasts were once the
          outcomes came in. The <strong>resolver pipeline review</strong> asks whether
          the data used to score them was sound. The{" "}
          <strong>forecast pipeline review</strong> asks whether the forecasting
          machinery worked on a given run. The first two follow each monthly scoring
          round; the third follows each forecast run.
        </p>
        <p className="max-w-3xl text-sm text-fred-muted">
          Reports marked &ldquo;Dry run&rdquo; were made by hand while the report
          templates were being tested.
        </p>
      </section>
      {loadError ? (
        <div
          role="alert"
          className="rounded-md border border-amber-400 bg-amber-50 px-4 py-3 text-sm text-amber-800"
        >
          {loadError} Check <span className="font-mono">/v1/health</span> for API status.
        </div>
      ) : data && !data.available ? (
        <p
          data-testid="reports-unavailable"
          className="rounded-md border border-fred-border bg-fred-surface p-4 text-sm text-fred-text shadow-fredCard"
        >
          Reports have not been published yet.
        </p>
      ) : data ? (
        <PerformanceAnalysisClient
          data={data}
          initialFilters={filtersFromParams(searchParams ?? {})}
        />
      ) : null}
    </div>
  );
}
