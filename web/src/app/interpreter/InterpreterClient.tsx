"use client";
// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
// See the LICENSE file in the project root for details.

// The interpreter page: attention map (click a country to jump to its
// section), status banner for non-ok reports, version selector, the report
// body, and the shared score glossary.

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useMemo } from "react";

import InfoTooltip from "../../components/InfoTooltip";
import RiskIndexMap from "../../components/RiskIndexMap";
import { SCORE_GLOSSARY } from "../../lib/score_glossary";
import type {
  InterpreterAttentionMapResponse,
  InterpreterReport,
  InterpreterVersionRow,
} from "../../lib/types";
import ReportView from "./ReportView";
import {
  movementColor,
  movementLabel,
  movementLegend,
  countryAnchorId,
  groupVersionsByRun,
  statusBanner,
} from "./lib";

export default function InterpreterClient({
  report,
  versions,
  attentionMap,
  reportPdfUrl = null,
  reportPublishedButMissing = false,
}: {
  report: InterpreterReport | null;
  versions: InterpreterVersionRow[];
  attentionMap: InterpreterAttentionMapResponse;
  reportPdfUrl?: string | null;
  reportPublishedButMissing?: boolean;
}) {
  const router = useRouter();
  const searchParams = useSearchParams();

  const valuesByIso3 = useMemo(() => {
    const out: Record<string, number> = {};
    attentionMap.rows.forEach((row) => {
      // Signed movement, not undirected divergence. A country whose
      // forecast moved DOWN must not be shaded like one that moved up.
      if (row.movement != null) {
        out[row.iso3] = row.movement;
      }
    });
    return out;
  }, [attentionMap.rows]);

  const reportIso3 = useMemo(() => {
    const set = new Set<string>();
    (report?.content_resolved?.attention ?? report?.content?.attention ?? []).forEach(
      (entry) => set.add((entry.iso3 || "").toUpperCase())
    );
    return set;
  }, [report]);

  const handleCountryClick = useCallback(
    (iso3: string) => {
      if (!reportIso3.has(iso3.toUpperCase())) return;
      document
        .getElementById(countryAnchorId(iso3))
        ?.scrollIntoView({ behavior: "smooth", block: "start" });
    },
    [reportIso3]
  );

  const legendItems = useMemo(() => movementLegend(), []);

  // One option per run, newest first (the API already orders rows
  // created_at DESC, so the first row per run is that run's newest version).
  const runOptions = useMemo(() => groupVersionsByRun(versions), [versions]);

  // The selected run defaults to the most recent, because with no
  // interpretation_id in the URL the page served the newest report.
  const currentRunKey = useMemo(() => {
    if (report) return report.run_id || report.scored_run_id || "";
    return runOptions[0]?.runKey ?? "";
  }, [report, runOptions]);

  const currentRun = useMemo(
    () => runOptions.find((r) => r.runKey === currentRunKey),
    [runOptions, currentRunKey]
  );

  const selectVersion = (interpretationId: string) => {
    const params = new URLSearchParams(searchParams.toString());
    if (interpretationId) {
      params.set("interpretation_id", interpretationId);
    } else {
      params.delete("interpretation_id");
    }
    const qs = params.toString();
    router.push(qs ? `/interpreter?${qs}` : "/interpreter");
  };

  // Picking a run jumps to that run's NEWEST version — the same thing the
  // default view shows for the newest run.
  const selectRun = (runKey: string) => {
    const option = runOptions.find((r) => r.runKey === runKey);
    if (option) selectVersion(option.latest.interpretation_id);
  };

  const banner = report ? statusBanner(report.status) : null;

  return (
    <div className="space-y-6">
      {attentionMap.has_data ? (
        <section className="space-y-2">
          <div className="flex items-center gap-2">
            <h2 className="text-xl font-semibold text-fred-primary">
              Where forecasts sit against their usual level
            </h2>
            <InfoTooltip
              text={
                "Warm shading means the country expects MORE than its history would " +
                "suggest; cool shading means less. The depth is how far it has " +
                "moved, as a multiple of the size worth mobilising against. " +
                "Where a country carries several hazards, the largest movement " +
                "is shown. Click a highlighted country to jump to its section."
              }
            />
          </div>
          <RiskIndexMap
            riskRows={[]}
            countriesRows={[]}
            view="PA_EIV"
            valuesByIso3={valuesByIso3}
            colorFor={movementColor}
            valueLabelFor={movementLabel}
            legendItems={legendItems}
            onCountryClick={handleCountryClick}
            showRcOverlay={false}
            heightClassName="h-[420px]"
          />
        </section>
      ) : null}

      <section className="flex flex-wrap items-center gap-3">
        {runOptions.length > 0 ? (
          <label className="flex items-center gap-2 text-sm text-fred-text">
            Run
            <select
              className="rounded-md border border-fred-secondary bg-fred-surface px-2 py-1 text-sm"
              value={currentRunKey}
              onChange={(event) => selectRun(event.target.value)}
            >
              {runOptions.map((option, index) => (
                <option key={option.runKey} value={option.runKey}>
                  {option.label}
                  {index === 0 ? " (most recent)" : ""}
                </option>
              ))}
            </select>
          </label>
        ) : null}
        {currentRun && currentRun.versions.length > 1 ? (
          <label className="flex items-center gap-2 text-sm text-fred-text">
            Version
            <select
              className="rounded-md border border-fred-secondary bg-fred-surface px-2 py-1 text-sm"
              value={report?.interpretation_id ?? ""}
              onChange={(event) => selectVersion(event.target.value)}
            >
              {currentRun.versions.map((v) => (
                <option key={v.interpretation_id} value={v.interpretation_id}>
                  v{v.version} — {String(v.created_at ?? "").slice(0, 10)}
                  {v.status !== "ok" ? ` (${v.status})` : ""}
                </option>
              ))}
            </select>
          </label>
        ) : null}
        {reportPdfUrl ? (
          <a
            className="rounded-md border border-fred-primary bg-fred-primary px-3 py-1 text-sm text-white hover:opacity-90"
            href={reportPdfUrl}
            target="_blank"
            rel="noopener noreferrer"
          >
            Download PDF
          </a>
        ) : null}
        {report ? (
          <Link
            className="rounded-md border border-fred-secondary px-3 py-1 text-sm text-fred-primary hover:text-fred-secondary"
            href={`/interpreter/print?interpretation_id=${encodeURIComponent(
              report.interpretation_id
            )}`}
          >
            Print / PDF view
          </Link>
        ) : null}
      </section>

      {banner ? (
        <div
          className={`rounded-md border px-4 py-3 text-sm ${
            banner.tone === "error"
              ? "border-red-400 bg-red-50 text-red-800"
              : "border-amber-400 bg-amber-50 text-amber-800"
          }`}
        >
          {banner.text}
        </div>
      ) : null}

      {report ? (
        <ReportView report={report} />
      ) : reportPublishedButMissing ? (
        // A report IS published (the release carries its PDF) but this API
        // instance has not served it yet — almost always its database is
        // still catching up with the release it just fetched the manifest
        // from. Saying "no interpretation exists" here would be a soft-fail
        // wearing the empty state's clothes, which is precisely what the
        // dashboard is not allowed to do.
        <div className="rounded-md border border-amber-400 bg-amber-50 px-4 py-6 text-sm text-amber-800">
          A report has been published for this cycle, but this API instance is
          still syncing its database and cannot serve it yet. The PDF above is
          available now; the on-page report should appear within a minute or
          two — reload to check. If it persists, see{" "}
          <span className="font-mono">/v1/health</span> for sync status.
        </div>
      ) : (
        <div className="rounded-md border border-fred-secondary bg-fred-surface/60 px-4 py-6 text-sm text-fred-muted">
          No interpretation exists for this run yet. The report is generated
          automatically at the end of each forecast cycle — check back after
          the next run, or toggle Test ON if you are looking for a test-mode
          report.
        </div>
      )}

      <section className="space-y-2">
        <h2 className="text-xl font-semibold text-fred-primary">Glossary</h2>
        <div className="flex flex-wrap gap-x-6 gap-y-2 text-sm text-fred-text">
          {SCORE_GLOSSARY.map((item) => (
            <span key={item.term} className="inline-flex items-center gap-1">
              {item.term}
              <InfoTooltip text={item.text} />
            </span>
          ))}
        </div>
      </section>
    </div>
  );
}
