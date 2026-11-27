"use client";

import React, { useEffect, useState } from "react";

import CollapsiblePanel from "../../components/CollapsiblePanel";
import type { OverviewVersionEntry } from "../../lib/overview_loader";
import { splitOverview } from "../../lib/overview_split";
import { extractHeadings, renderSimpleMarkdown } from "../../lib/simple_markdown";

const CURRENT_KEY = "__current__";
const CHANGES_ANCHOR = "changes-in-this-version";

interface OverviewSectionProps {
  currentOverview: string;
  versions: OverviewVersionEntry[];
}

/** The overview body: a contents list, the main text with heading anchors,
 *  and the change log last, in a closed panel. Exported for tests. */
export function OverviewBody({ markdown }: { markdown: string }) {
  const { main, changelog } = splitOverview(markdown);
  const headings = extractHeadings(main);
  return (
    <>
      {headings.length > 1 && (
        <nav
          aria-label="Contents"
          className="rounded-md border border-fred-secondary bg-fred-surface p-3 text-sm"
        >
          <p className="mb-1 font-semibold text-fred-text">Contents</p>
          <ol className="space-y-1">
            {headings.map((h) => (
              <li key={h.id}>
                <a className="text-fred-primary underline underline-offset-2" href={`#${h.id}`}>
                  {h.text}
                </a>
              </li>
            ))}
            {changelog !== null && (
              <li>
                <a
                  className="text-fred-primary underline underline-offset-2"
                  href={`#${CHANGES_ANCHOR}`}
                >
                  Changes in this version
                </a>
              </li>
            )}
          </ol>
        </nav>
      )}
      <div data-testid="overview-main">
        {renderSimpleMarkdown(main, { headingIds: true })}
      </div>
      {changelog !== null && (
        <div id={CHANGES_ANCHOR} className="mt-8 scroll-mt-20" data-testid="overview-changes">
          <CollapsiblePanel title="Changes in this version">
            {renderSimpleMarkdown(changelog)}
          </CollapsiblePanel>
        </div>
      )}
    </>
  );
}

export default function OverviewSection({
  currentOverview,
  versions,
}: OverviewSectionProps) {
  const [selectedVersion, setSelectedVersion] = useState(CURRENT_KEY);
  const [archived, setArchived] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Archived versions are fetched only when chosen, never shipped with the page.
  useEffect(() => {
    if (selectedVersion === CURRENT_KEY || archived[selectedVersion]) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetch(`/about/overview/${encodeURIComponent(selectedVersion)}`)
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const text = await res.text();
        if (!cancelled) setArchived((prev) => ({ ...prev, [selectedVersion]: text }));
      })
      .catch(() => {
        if (!cancelled) setError(`The ${selectedVersion} version could not be loaded.`);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedVersion, archived]);

  const activeOverview =
    selectedVersion === CURRENT_KEY
      ? currentOverview
      : (archived[selectedVersion] ?? null);

  const hasVersions = versions.length > 0;

  return (
    <section className="mt-6 space-y-4">
      <div className="flex flex-wrap items-center gap-4">
        <h2 className="text-2xl font-semibold">System Overview</h2>
        {hasVersions && (
          <div className="flex items-center gap-2">
            <label
              htmlFor="overview-version"
              className="text-sm font-medium text-fred-secondary"
            >
              Version:
            </label>
            <select
              id="overview-version"
              value={selectedVersion}
              onChange={(e) => setSelectedVersion(e.target.value)}
              className="rounded-md border border-fred-secondary bg-fred-surface px-3 py-1.5 text-sm text-fred-text"
            >
              <option value={CURRENT_KEY}>Current (live)</option>
              {versions.map((v) => (
                <option key={v.date} value={v.date}>
                  {v.date}: {v.label}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>
      <p className="text-sm text-fred-text">
        {selectedVersion === CURRENT_KEY
          ? "Showing the live system overview."
          : `Showing the archived overview from ${selectedVersion}.`}
      </p>
      {loading && (
        <p role="status" className="text-sm italic text-fred-text/70">
          Loading the {selectedVersion} version…
        </p>
      )}
      {error && (
        <p role="alert" className="text-sm text-red-700">
          {error}
        </p>
      )}
      {activeOverview !== null && <OverviewBody markdown={activeOverview} />}
    </section>
  );
}
