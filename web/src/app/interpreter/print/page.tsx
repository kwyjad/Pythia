// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
// See the LICENSE file in the project root for details.

// Print-ready report view: no navigation, page breaks between sections, the
// lexicon and glossary as an appendix. This is the always-available PDF
// fallback (window.print()) and the page the Phase 6 CI PDF renderer loads.

import { apiGet } from "../../../lib/api";
import type {
  InterpreterLatestResponse,
  InterpreterReport,
} from "../../../lib/types";
import ReportView from "../ReportView";

export const dynamic = "force-dynamic";
export const revalidate = 0;

export default async function InterpreterPrintPage({
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
  try {
    if (interpretationId) {
      const body = await apiGet<{ interpretation: InterpreterReport }>(
        `/interpreter/${encodeURIComponent(interpretationId)}`
      );
      report = body.interpretation;
    } else {
      const latest = await apiGet<InterpreterLatestResponse>(
        "/interpreter/latest",
        { kind: "combined", include_test: includeTest || undefined }
      );
      report = latest.has_interpretation ? latest.interpretation ?? null : null;
    }
  } catch (error) {
    console.warn("Failed to load interpretation for print:", error);
  }

  return (
    <div className="mx-auto max-w-3xl bg-white px-8 py-6 text-slate-900">
      {/* Print styling: hide the app chrome (nav + test banner) on this
          route entirely, force light surfaces, and let ReportView's
          break-before-page classes paginate. */}
      <style
        dangerouslySetInnerHTML={{
          __html: `
            nav { display: none !important; }
            @media print {
              body { background: #ffffff !important; }
              a { color: inherit !important; text-decoration: none !important; }
            }
            @page { margin: 18mm; }
          `,
        }}
      />
      {report ? (
        <>
          {report.status !== "ok" ? (
            <div className="mb-4 rounded border border-amber-400 bg-amber-50 px-3 py-2 text-sm text-amber-800 print:border print:border-amber-400">
              This report failed automated validation and is shown for
              inspection only.
            </div>
          ) : null}
          <ReportView report={report} print />
        </>
      ) : (
        <p className="text-sm text-slate-600">
          No interpretation is available to print yet.
        </p>
      )}
    </div>
  );
}
