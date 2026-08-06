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
  attentionColor,
  attentionLabel,
  attentionLegend,
  countryAnchorId,
  statusBanner,
} from "./lib";

export default function InterpreterClient({
  report,
  versions,
  attentionMap,
  reportPdfUrl = null,
}: {
  report: InterpreterReport | null;
  versions: InterpreterVersionRow[];
  attentionMap: InterpreterAttentionMapResponse;
  reportPdfUrl?: string | null;
}) {
  const router = useRouter();
  const searchParams = useSearchParams();

  const valuesByIso3 = useMemo(() => {
    const out: Record<string, number> = {};
    attentionMap.rows.forEach((row) => {
      if (row.attention != null) {
        out[row.iso3] = row.attention;
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

  const legendItems = useMemo(() => attentionLegend(), []);

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

  const banner = report ? statusBanner(report.status) : null;

  return (
    <div className="space-y-6">
      {attentionMap.has_data ? (
        <section className="space-y-2">
          <div className="flex items-center gap-2">
            <h2 className="text-xl font-semibold text-fred-primary">
              Attention map
            </h2>
            <InfoTooltip
              text={
                "Coloured by how far each country's strongest forecast moved " +
                "from its historical base rate — attention, not raw risk (the " +
                "risk-index map already shows risk). Click a highlighted " +
                "country to jump to its section in the report."
              }
            />
          </div>
          <RiskIndexMap
            riskRows={[]}
            countriesRows={[]}
            view="PA_EIV"
            valuesByIso3={valuesByIso3}
            colorFor={attentionColor}
            valueLabelFor={attentionLabel}
            legendItems={legendItems}
            onCountryClick={handleCountryClick}
            showRcOverlay={false}
            heightClassName="h-[420px]"
          />
        </section>
      ) : null}

      <section className="flex flex-wrap items-center gap-3">
        {versions.length > 1 ? (
          <label className="flex items-center gap-2 text-sm text-fred-text">
            Version
            <select
              className="rounded-md border border-fred-secondary bg-fred-surface px-2 py-1 text-sm"
              value={report?.interpretation_id ?? ""}
              onChange={(event) => selectVersion(event.target.value)}
            >
              {versions.map((v) => (
                <option key={v.interpretation_id} value={v.interpretation_id}>
                  {v.kind} v{v.version} — {String(v.created_at ?? "").slice(0, 10)}
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
