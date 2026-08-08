// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
// See the LICENSE file in the project root for details.

// Pure helpers for the interpreter page (unit-tested in __tests__/lib.test.ts).

import type {
  InterpreterReasonCode,
  InterpreterVersionRow,
} from "../../lib/types";

// A run's month is the month the report is ABOUT, taken from the HS run id
// (hs_YYYYMMDDTHHMMSS) rather than when it was generated — the same rule
// interpreter/pdf.py::month_label uses, so a backfilled July report reads as
// July here and in its filename. Falls back to created_at.
const HS_RUN_MONTH = /^hs_(\d{4})(\d{2})\d{2}T/;
const MONTHS = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

export const runMonthLabel = (
  hsRunId?: string | null,
  createdAt?: string | null
): string => {
  const match = HS_RUN_MONTH.exec(hsRunId ?? "");
  let year: number | null = null;
  let monthIdx: number | null = null;
  if (match) {
    year = Number(match[1]);
    monthIdx = Number(match[2]) - 1;
  } else if (createdAt) {
    const parts = /^(\d{4})-(\d{2})/.exec(String(createdAt));
    if (parts) {
      year = Number(parts[1]);
      monthIdx = Number(parts[2]) - 1;
    }
  }
  if (year === null || monthIdx === null || monthIdx < 0 || monthIdx > 11) {
    return "Unknown month";
  }
  return `${MONTHS[monthIdx]} ${year}`;
};

export type InterpreterRunOption = {
  runKey: string;
  label: string;
  latest: InterpreterVersionRow;
  versions: InterpreterVersionRow[];
};

/** Group version rows into one option per run, newest run first.
 *
 * The API returns rows already ordered created_at DESC, version DESC, so the
 * FIRST row seen for a run is that run's newest version — which is what the
 * selector selects. Rows with no run key at all are grouped under "" rather
 * than dropped, so a report can never become unreachable through the picker.
 */
export const groupVersionsByRun = (
  rows: InterpreterVersionRow[]
): InterpreterRunOption[] => {
  const byRun = new Map<string, InterpreterRunOption>();
  for (const row of rows ?? []) {
    const runKey = row.run_id || row.scored_run_id || "";
    const existing = byRun.get(runKey);
    if (existing) {
      existing.versions.push(row);
      continue;
    }
    byRun.set(runKey, {
      runKey,
      label: runMonthLabel(row.hs_run_id, row.created_at),
      latest: row,
      versions: [row],
    });
  }
  return Array.from(byRun.values());
};

// Plain-word reason labels — mirrors interpreter/render.py's _REASON_LABELS.
export const REASON_LABELS: Record<InterpreterReasonCode, string> = {
  base_rate_deviation: "far from its base rate",
  large_impact_nominal: "large expected impact",
  large_impact_per_capita: "large expected impact per capita",
  rc_deviation_disagreement: "the system disagrees with itself here",
};

// Attention choropleth: a fixed sequential scale over the [0, 1] attention
// value (js_vs_baserate / ln 2). Fixed thresholds, not Jenks — the map is
// coloured by ATTENTION, deliberately not repeating the risk index's look.
export const ATTENTION_BANDS: Array<{ min: number; label: string; color: string }> = [
  { min: 0.7, label: "Extreme deviation (≥70%)", color: "#4c1d95" },
  { min: 0.45, label: "High (45–70%)", color: "#6d28d9" },
  { min: 0.25, label: "Elevated (25–45%)", color: "#8b5cf6" },
  { min: 0.1, label: "Mild (10–25%)", color: "#c4b5fd" },
  { min: 0, label: "At base rate (<10%)", color: "#ede9fe" },
];

export const attentionColor = (value: number): string => {
  const v = Math.max(0, Math.min(value, 1));
  for (const band of ATTENTION_BANDS) {
    if (v >= band.min) return band.color;
  }
  return ATTENTION_BANDS[ATTENTION_BANDS.length - 1].color;
};

export const attentionLegend = (): Array<{ label: string; color: string }> => [
  ...ATTENTION_BANDS.map((b) => ({ label: b.label, color: b.color })),
  { label: "No deviation data", color: "var(--risk-map-no-questions)" },
];

// Words, not "% of maximum deviation": the maximum is ln 2, which no reader
// can picture. Bands mirror interpreter/render.py's format_figure.
export const attentionLabel = (value: number): string => {
  const v = Math.max(0, Math.min(value, 1));
  if (v >= 0.5) return "a long way from its usual pattern";
  if (v >= 0.25) return "well away from its usual pattern";
  if (v >= 0.1) return "a little away from its usual pattern";
  return "close to its usual pattern";
};

// The report's four boxes. Order and labels mirror interpreter/selection.py.
export const CATEGORY_LABELS: Record<string, string> = {
  worsening: "Potentially worsening situations",
  stable_major: "Major impact but roughly stable",
};

export const FAMILY_LABELS: Record<string, string> = {
  climate: "Climate hazards",
  conflict: "Conflict",
  other: "Other hazards",
};

export const SECTION_ORDER: Array<{ category: string; family: string }> = [
  { category: "worsening", family: "climate" },
  { category: "worsening", family: "conflict" },
  { category: "stable_major", family: "climate" },
  { category: "stable_major", family: "conflict" },
];

export const sectionHeading = (category: string, family: string): string =>
  `${CATEGORY_LABELS[category] ?? category}: ${(
    FAMILY_LABELS[family] ?? family
  ).toLowerCase()}`;

// One entry's heading, in words. The API stamps the names on each entry from
// interpreter/names.py; the codes are the fallback, never the first choice.
export const entryHeading = (entry: {
  country_name?: string;
  iso3?: string;
  hazard_name?: string;
  hazard_code?: string;
  metric_name?: string;
  metric?: string;
}): string => {
  const country = entry.country_name || entry.iso3 || "";
  const hazard = (entry.hazard_name || entry.hazard_code || "").toLowerCase();
  const metric = entry.metric_name || entry.metric || "";
  return `${country}, ${hazard}: ${metric}`;
};

// Entries grouped into the four boxes, in report order, with anything the
// model failed to place kept in a trailing group rather than dropped.
export const groupAttention = <T extends { category?: string; hazard_family?: string; rank?: number }>(
  entries: T[]
): Array<{ key: string; heading: string; entries: T[] }> => {
  const out: Array<{ key: string; heading: string; entries: T[] }> = [];
  const placed = new Set<T>();
  const byRank = (a: T, b: T) => (a.rank ?? 99) - (b.rank ?? 99);
  for (const { category, family } of SECTION_ORDER) {
    const group = entries.filter(
      (e) => e.category === category && e.hazard_family === family
    );
    group.forEach((e) => placed.add(e));
    if (group.length) {
      out.push({
        key: `${category}-${family}`,
        heading: sectionHeading(category, family),
        entries: group.slice().sort(byRank),
      });
    }
  }
  const stragglers = entries.filter((e) => !placed.has(e));
  if (stragglers.length) {
    out.push({
      key: "other",
      heading: "Other situations of note",
      entries: stragglers.slice().sort(byRank),
    });
  }
  return out;
};

// The fixed probability lexicon — FIXED BY DESIGN (interpreter/lexicon.py);
// printed in the report appendix so the reader can check the writer.
export const LEXICON_TABLE: Array<{ word: string; band: string }> = [
  { word: "virtually certain", band: "≥ 99%" },
  { word: "very likely", band: "90–99%" },
  { word: "likely", band: "66–90%" },
  { word: "about as likely as not", band: "33–66%" },
  { word: "unlikely", band: "10–33%" },
  { word: "very unlikely", band: "1–10%" },
  { word: "exceptionally unlikely", band: "< 1%" },
];

export const countryAnchorId = (iso3: string): string =>
  `country-${(iso3 || "").toUpperCase()}`;

export const statusBanner = (
  status: string
): { tone: "ok" | "warn" | "error"; text: string } | null => {
  if (status === "ok") return null;
  if (status === "failed_validation") {
    return {
      tone: "warn",
      text:
        "This report FAILED automated validation (schema, referential, " +
        "numeric or prose checks) and is shown for inspection only — treat " +
        "its statements with caution.",
    };
  }
  if (status === "failed_generation") {
    return {
      tone: "error",
      text:
        "Report generation failed for this cycle — no readable content was " +
        "produced.",
    };
  }
  return { tone: "warn", text: `Unexpected report status: ${status}` };
};
