// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
// See the LICENSE file in the project root for details.

import Link from "next/link";
import { notFound } from "next/navigation";

import { apiGet } from "../../../lib/api";
import { downloadUrl } from "../../../lib/analysis_reports";
import { extractHeadings, renderSimpleMarkdown } from "../../../lib/simple_markdown";

export const dynamic = "force-dynamic";
export const revalidate = 0;

const ACTION =
  "inline-flex items-center rounded-md border border-fred-border bg-fred-surface px-3 py-1.5 text-sm font-medium text-fred-primary hover:border-fred-primary focus:outline-none focus-visible:ring-2 focus-visible:ring-fred-primary";

export default async function AnalysisReportPage({ params }: { params: { reportId: string } }) {
  const reportId = decodeURIComponent(params.reportId);
  let body: { report_id: string; title: string | null; markdown: string } | null = null;
  let error: string | null = null;
  try {
    body = await apiGet(`/analysis_reports/${encodeURIComponent(reportId)}/content`);
  } catch (err) {
    if (String(err).includes("(404)")) notFound();
    error = "This report could not be loaded right now. The downloads below may still work.";
    console.warn("Failed to load analysis report:", err);
  }

  const headings = body ? extractHeadings(body.markdown) : [];

  return (
    <div className="space-y-4">
      <Link href="/performance-analysis" className="text-sm text-fred-primary underline underline-offset-2">
        ← All reports
      </Link>
      <div className="flex flex-wrap gap-2">
        <a href={downloadUrl(reportId, "pdf")} className={ACTION} rel="noopener noreferrer">
          Download PDF
        </a>
        <a href={downloadUrl(reportId, "md")} className={ACTION} rel="noopener noreferrer">
          Download Markdown
        </a>
      </div>
      {error && (
        <div role="alert" className="rounded-md border border-amber-400 bg-amber-50 px-4 py-3 text-sm text-amber-800">
          {error}
        </div>
      )}
      {body && (
        <div className="lg:grid lg:grid-cols-[minmax(0,1fr)_16rem] lg:gap-8">
          {headings.length > 2 && (
            <nav
              aria-label="Contents"
              className="mb-4 rounded-md border border-fred-border bg-fred-surface p-3 text-sm shadow-fredCard lg:order-2 lg:sticky lg:top-4 lg:mb-0 lg:max-h-[calc(100vh-2rem)] lg:self-start lg:overflow-y-auto"
            >
              <p className="mb-1 font-semibold text-fred-text">Contents</p>
              <ol className="space-y-1">
                {headings.map((h) => (
                  <li key={h.id}>
                    <a href={`#${h.id}`} className="text-fred-primary hover:underline">
                      {h.text}
                    </a>
                  </li>
                ))}
              </ol>
            </nav>
          )}
          <article className="min-w-0 rounded-md border border-fred-border bg-fred-surface px-4 py-2 shadow-fredCard sm:px-6 lg:order-1 [&_h1]:mt-4">
            {renderSimpleMarkdown(body.markdown, { headingIds: true })}
          </article>
        </div>
      )}
    </div>
  );
}
