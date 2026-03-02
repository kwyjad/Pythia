"use client";

import React, { useState } from "react";

import type { OverviewVersionEntry } from "../../lib/overview_loader";
import { renderSimpleMarkdown } from "../../lib/simple_markdown";

const CURRENT_KEY = "__current__";

interface OverviewSectionProps {
  currentOverview: string;
  versions: OverviewVersionEntry[];
  versionedOverviews: Record<string, string>;
}

export default function OverviewSection({
  currentOverview,
  versions,
  versionedOverviews,
}: OverviewSectionProps) {
  const [selectedVersion, setSelectedVersion] = useState(CURRENT_KEY);

  const activeOverview =
    selectedVersion === CURRENT_KEY
      ? currentOverview
      : (versionedOverviews[selectedVersion] ?? currentOverview);

  const hasVersions = versions.length > 0;

  return (
    <section className="mt-6 space-y-4">
      {hasVersions && (
        <div className="flex flex-wrap items-center gap-4">
          <h2 className="text-2xl font-semibold">System Overview</h2>
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
                  {v.date} — {v.label}
                </option>
              ))}
            </select>
          </div>
        </div>
      )}
      <p className="text-sm text-fred-text">
        {selectedVersion === CURRENT_KEY
          ? "Showing the live system overview loaded at build time."
          : `Showing archived overview from ${selectedVersion}.`}
      </p>
      {renderSimpleMarkdown(activeOverview)}
    </section>
  );
}
