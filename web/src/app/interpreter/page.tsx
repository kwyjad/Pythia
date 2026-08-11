// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
// See the LICENSE file in the project root for details.

import { apiGet } from "../../lib/api";
import type {
  InterpreterAttentionMapResponse,
  InterpreterLatestResponse,
  InterpreterReport,
  InterpreterVersionsResponse,
  VersionResponse,
} from "../../lib/types";
import InterpreterClient from "./InterpreterClient";

export const dynamic = "force-dynamic";
export const revalidate = 0;

export default async function InterpreterPage({
  searchParams,
}: {
  searchParams: { [key: string]: string | string[] | undefined };
}) {
  const includeTest = searchParams?.include_test === "true";
  const interpretationId =
    typeof searchParams?.interpretation_id === "string"
      ? searchParams.interpretation_id
      : undefined;

  let report: InterpreterReport | null = null;
  let versions: InterpreterVersionsResponse = { rows: [] };
  let attentionMap: InterpreterAttentionMapResponse = {
    has_data: false,
    has_movement: false,
    run_id: null,
    rows: [],
  };
  let loadError: string | null = null;

  // Every fetch is independent and null-safe — a degraded API renders an
  // explicit empty state, never a blank page (the standing soft-fail rule).
  const [latestResult, versionsResult, mapResult, versionResult] = await Promise.allSettled([
    interpretationId
      ? apiGet<{ interpretation: InterpreterReport }>(
          `/interpreter/${encodeURIComponent(interpretationId)}`
        ).then((body) => ({
          has_interpretation: true,
          kind: body.interpretation.kind,
          interpretation: body.interpretation,
        }) as InterpreterLatestResponse)
      : apiGet<InterpreterLatestResponse>("/interpreter/latest", {
          kind: "combined",
          include_test: includeTest || undefined,
        }),
    apiGet<InterpreterVersionsResponse>("/interpreter/versions", {
      include_test: includeTest || undefined,
    }),
    apiGet<InterpreterAttentionMapResponse>("/interpreter/attention_map", {
      include_test: includeTest || undefined,
    }),
    // The release-asset PDF URL rides in /v1/version (manifest passthrough).
    apiGet<VersionResponse>("/version"),
  ]);

  if (latestResult.status === "fulfilled") {
    report = latestResult.value.has_interpretation
      ? latestResult.value.interpretation ?? null
      : null;
  } else {
    loadError = "Live report data unavailable right now.";
    console.warn("Failed to load interpretation:", latestResult.reason);
  }
  if (versionsResult.status === "fulfilled") {
    versions = versionsResult.value;
  }
  if (mapResult.status === "fulfilled") {
    attentionMap = mapResult.value;
  }
  const reportPdfUrl =
    versionResult.status === "fulfilled"
      ? versionResult.value.interpreter_report_url ?? null
      : null;
  // The versioned filename (report__2026-08__v10.pdf) is the only way the page
  // can tell that the Download PDF button is offering a NEWER report than
  // anything the API can serve — the release and the API's database are two
  // stores that catch up at different speeds.
  const reportPdfAsset =
    versionResult.status === "fulfilled"
      ? versionResult.value.interpreter_report_versioned_asset ?? null
      : null;
  // The release advertising a report PDF while the API serves no report means
  // the API's DB is behind the release it just read the manifest from — a
  // soft-fail, which must never be dressed as "no data".
  const reportPublishedButMissing = Boolean(reportPdfUrl) && report === null;

  return (
    <div className="space-y-6">
      <section className="space-y-2">
        <h1 className="text-3xl font-semibold">This month, in plain language</h1>
        <p className="text-sm text-fred-text">
          One short report per forecast cycle, written for readers with no
          forecasting background: what deserves attention and why, what
          changed since last month, and — once outcomes exist — how well the
          system performed. Every figure in the report is computed by the
          system itself; the language model only explains.
        </p>
      </section>
      {loadError ? (
        <div className="rounded-md border border-amber-400 bg-amber-50 px-4 py-3 text-sm text-amber-800">
          {loadError} Check <span className="font-mono">/v1/health</span> for
          API status.
        </div>
      ) : null}
      <InterpreterClient
        report={report}
        versions={versions.rows}
        attentionMap={attentionMap}
        reportPdfUrl={reportPdfUrl}
        reportPdfAsset={reportPdfAsset}
        reportPublishedButMissing={reportPublishedButMissing}
      />
    </div>
  );
}
