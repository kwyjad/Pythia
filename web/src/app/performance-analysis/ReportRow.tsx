// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
// See the LICENSE file in the project root for details.

import Link from "next/link";

import type { ReportEntry } from "../../lib/analysis_reports";
import {
  TYPE_BADGE_CLASSES,
  TYPE_LABELS,
  dateLabel,
  downloadUrl,
  earlierVersions,
  placeholderText,
  verdictChip,
} from "../../lib/analysis_reports";

const ACTION =
  "inline-flex items-center rounded-md border border-fred-border bg-fred-surface px-2.5 py-1 text-xs font-medium text-fred-primary hover:border-fred-primary focus:outline-none focus-visible:ring-2 focus-visible:ring-fred-primary";
const SMALL_LINK =
  "text-fred-primary underline underline-offset-2 hover:text-fred-text focus:outline-none focus-visible:ring-2 focus-visible:ring-fred-primary";

export function TypeBadge({ type }: { type: ReportEntry["type"] }) {
  return (
    <span
      className={`inline-flex shrink-0 items-center rounded border px-2 py-0.5 text-xs font-medium ${TYPE_BADGE_CLASSES[type]}`}
    >
      {TYPE_LABELS[type]}
    </span>
  );
}

export default function ReportRow({ entry }: { entry: ReportEntry }) {
  if (!entry.available || !entry.report_id) {
    return (
      <li
        className="flex flex-wrap items-center gap-2 py-3 text-sm text-fred-muted"
        data-testid="report-placeholder"
      >
        <span className="opacity-70">
          <TypeBadge type={entry.type} />
        </span>
        <span className="italic">{placeholderText(entry)}</span>
      </li>
    );
  }

  const chip = verdictChip(entry.verdict);
  const older = earlierVersions(entry);
  const prepared = dateLabel(entry.prepared);

  return (
    <li className="py-3" data-testid="report-row">
      <div className="flex flex-wrap items-center gap-2">
        <TypeBadge type={entry.type} />
        {chip && (
          <span
            className={`inline-flex items-center rounded border px-2 py-0.5 text-xs font-medium ${chip.className}`}
            data-testid="verdict-chip"
          >
            {chip.label}
          </span>
        )}
        {entry.dry_run && (
          <span className="inline-flex items-center rounded border border-fred-border bg-fred-bg px-2 py-0.5 text-xs font-medium text-fred-muted">
            Dry run
          </span>
        )}
        {entry.rerun_date && (
          <span className="text-xs text-fred-muted">Rerun {dateLabel(entry.rerun_date)}</span>
        )}
      </div>
      <h4 className="mt-1.5 text-sm font-semibold text-fred-text">{entry.title}</h4>
      {prepared && <p className="text-xs text-fred-muted">Prepared {prepared}</p>}
      {entry.summary && (
        <p className="mt-1 max-w-3xl text-sm leading-relaxed text-fred-text">{entry.summary}</p>
      )}
      <div className="mt-2 flex flex-wrap gap-2">
        <Link
          href={`/performance-analysis/${encodeURIComponent(entry.report_id)}`}
          className={ACTION}
        >
          View
        </Link>
        {entry.pdf_asset && (
          <a href={downloadUrl(entry.report_id, "pdf")} className={ACTION} rel="noopener noreferrer">
            Download PDF
          </a>
        )}
        {entry.md_asset && (
          <a href={downloadUrl(entry.report_id, "md")} className={ACTION} rel="noopener noreferrer">
            Download Markdown
          </a>
        )}
      </div>
      {older.length > 0 && (
        <details className="mt-2 text-xs">
          <summary className="cursor-pointer text-fred-muted hover:text-fred-text">
            Earlier versions ({older.length})
          </summary>
          <ul className="mt-1 space-y-1 pl-4">
            {older.map((v) => (
              <li key={v.report_id} className="flex flex-wrap items-center gap-x-3 gap-y-1">
                <span className="text-fred-text">
                  {v.rerun_date ? `Rerun ${dateLabel(v.rerun_date)}` : "Original"}
                  {v.prepared ? `, prepared ${dateLabel(v.prepared)}` : ""}
                </span>
                <Link href={`/performance-analysis/${encodeURIComponent(v.report_id)}`} className={SMALL_LINK}>
                  View
                </Link>
                {v.pdf_asset && (
                  <a href={downloadUrl(v.report_id, "pdf")} className={SMALL_LINK}>
                    PDF
                  </a>
                )}
                {v.md_asset && (
                  <a href={downloadUrl(v.report_id, "md")} className={SMALL_LINK}>
                    Markdown
                  </a>
                )}
              </li>
            ))}
          </ul>
        </details>
      )}
    </li>
  );
}
