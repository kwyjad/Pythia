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

/** "2026-08" as "August 2026", for a decision deadline. */
export const monthLabel = (ym?: string | null): string => {
  const parts = /^(\d{4})-(\d{2})$/.exec(String(ym ?? ""));
  if (!parts) return "no dated deadline";
  const monthIdx = Number(parts[2]) - 1;
  if (monthIdx < 0 || monthIdx > 11) return "no dated deadline";
  return `${MONTHS[monthIdx]} ${parts[1]}`;
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

// Movement choropleth: a DIVERGING scale over the signed movement, in
// multiples of the size worth mobilising against. Mirrors
// interpreter/mapviz.py's SCALE_ABOVE / SCALE_BELOW / SCALE_BREAKS, so the
// printed map in the PDF and this one cannot disagree.
//
// The scale used to be sequential over an UNDIRECTED divergence, which shaded
// a country whose forecast had fallen exactly like one whose forecast had
// risen: Uganda appeared among the countries furthest from usual on the same
// page as text saying its ensemble had moved down.
export const MOVEMENT_BREAKS = [0.1, 0.4, 1.0, 2.0];

export const MOVEMENT_ABOVE = ["#F6D6C4", "#EFAF8C", "#DE7F53", "#B85527", "#8A3410"];
export const MOVEMENT_BELOW = ["#D6E3EE", "#A9C4DA", "#7398B8", "#456F92", "#274C68"];
export const MOVEMENT_NEUTRAL = "#F2F0EC";

export const movementColor = (value: number): string => {
  const v = Math.max(-3, Math.min(value, 3));
  const magnitude = Math.abs(v);
  if (magnitude < MOVEMENT_BREAKS[0]) return MOVEMENT_NEUTRAL;
  const ramp = v > 0 ? MOVEMENT_ABOVE : MOVEMENT_BELOW;
  for (let i = 0; i < MOVEMENT_BREAKS.length; i += 1) {
    if (magnitude < MOVEMENT_BREAKS[i]) return ramp[i];
  }
  return ramp[ramp.length - 1];
};

export const movementLegend = (): Array<{ label: string; color: string }> => [
  { label: "Far above its usual level", color: MOVEMENT_ABOVE[4] },
  { label: "Above its usual level", color: MOVEMENT_ABOVE[2] },
  { label: "Near its usual level", color: MOVEMENT_NEUTRAL },
  { label: "Below its usual level", color: MOVEMENT_BELOW[2] },
  { label: "Far below its usual level", color: MOVEMENT_BELOW[4] },
  { label: "No forecast this month", color: "var(--risk-map-no-questions)" },
];

// Words, not a multiple: the reader needs the direction first and the size
// second, and "1.4x the action threshold" is not a phrase anyone thinks in.
export const movementLabel = (value: number): string => {
  const v = Math.max(-3, Math.min(value, 3));
  const magnitude = Math.abs(v);
  if (magnitude < MOVEMENT_BREAKS[0]) return "near its usual level";
  const direction = v > 0 ? "above" : "below";
  if (magnitude >= MOVEMENT_BREAKS[3]) return `far ${direction} its usual level`;
  if (magnitude >= MOVEMENT_BREAKS[2]) return `well ${direction} its usual level`;
  return `${direction} its usual level`;
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

// The version number inside a published PDF's filename, e.g.
// "report__2026-08__v10.pdf" -> 10. The release and the API's database are
// two different stores that catch up at different speeds, and the page has
// no other way to tell that the Download PDF button is offering something
// newer than anything it can show on screen.
export const pdfVersionFromAsset = (asset?: string | null): number | null => {
  const match = /__v(\d+)\.pdf$/i.exec(String(asset ?? ""));
  if (!match) return null;
  const n = Number(match[1]);
  return Number.isFinite(n) ? n : null;
};

// The newest version the API actually served, across every run.
export const newestServedVersion = (
  rows: Array<{ version?: number | null }>
): number | null => {
  const versions = (rows ?? [])
    .map((r) => (typeof r.version === "number" ? r.version : null))
    .filter((v): v is number => v != null);
  return versions.length ? Math.max(...versions) : null;
};
