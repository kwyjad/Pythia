// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
// See the LICENSE file in the project root for details.

/**
 * The Performance Analysis page's data: the API's grouped index of analysis
 * reports (`/v1/analysis_reports`), the page's filters, and small display
 * helpers. The reports live in the analysis-reports GitHub release, copied
 * from Drive by sync_analysis_reports.yml; the API reads the release.
 */

export type ReportType = "skill" | "resolver" | "forecast";

export const REPORT_TYPES: ReportType[] = ["skill", "resolver", "forecast"];

export const TYPE_LABELS: Record<ReportType, string> = {
  skill: "Skill analysis",
  resolver: "Resolver pipeline review",
  forecast: "Forecast pipeline review",
};

/** One fixed colour per type; the text label always sits beside it. */
export const TYPE_BADGE_CLASSES: Record<ReportType, string> = {
  skill: "border-sky-300 bg-sky-50 text-sky-900",
  resolver: "border-violet-300 bg-violet-50 text-violet-900",
  forecast: "border-teal-300 bg-teal-50 text-teal-900",
};

export interface ReportVersion {
  report_id: string;
  rerun_date: string | null;
  prepared: string | null;
  md_asset: string | null;
  pdf_asset: string | null;
  title?: string | null;
  verdict?: string | null;
  dry_run?: boolean;
}

export interface ReportEntry {
  type: ReportType;
  display_name: string;
  period: string;
  available: boolean;
  expected: boolean;
  pending?: boolean;
  report_id?: string;
  title?: string;
  summary?: string;
  prepared?: string | null;
  verdict?: string | null;
  dry_run?: boolean;
  md_asset?: string | null;
  pdf_asset?: string | null;
  rerun_date?: string | null;
  versions: ReportVersion[];
}

export interface ScoringRound {
  publish_run_id: string | null;
  date: string | null;
  folder_name?: string | null;
  reports: ReportEntry[];
}

export interface ForecastRun {
  fc_run_id: string;
  date: string | null;
  publish_run_id: string | null;
  reports: ReportEntry[];
}

export interface ReportCycle {
  period: string;
  scoring_rounds: ScoringRound[];
  forecast_runs: ForecastRun[];
}

export interface AnalysisReportsResponse {
  available: boolean;
  generated_at: string | null;
  stale?: boolean;
  years: string[];
  cycles: ReportCycle[];
}

export interface ReportFilters {
  types: ReportType[];
  year: string | null;
  showDryRuns: boolean;
}

export const DEFAULT_FILTERS: ReportFilters = {
  types: [...REPORT_TYPES],
  year: null,
  showDryRuns: true,
};

type Params = { get(name: string): string | null } | Record<string, string | string[] | undefined>;

function param(params: Params, name: string): string | null {
  if (typeof (params as { get?: unknown }).get === "function") {
    return (params as { get(name: string): string | null }).get(name);
  }
  const value = (params as Record<string, string | string[] | undefined>)[name];
  return Array.isArray(value) ? value[0] ?? null : value ?? null;
}

/** Filters from URL query params: `types=skill,forecast`, `year=2026`, `dry=0`. */
export function filtersFromParams(params: Params): ReportFilters {
  const rawTypes = param(params, "types");
  const types = rawTypes
    ? (rawTypes.split(",").filter((t) => (REPORT_TYPES as string[]).includes(t)) as ReportType[])
    : [...REPORT_TYPES];
  const year = param(params, "year");
  return {
    types,
    year: year && /^\d{4}$/.test(year) ? year : null,
    showDryRuns: param(params, "dry") !== "0",
  };
}

/** Query string for filters; defaults are left out so the plain URL is the default view. */
export function filtersToQuery(filters: ReportFilters): string {
  const qs = new URLSearchParams();
  if (filters.types.length !== REPORT_TYPES.length) {
    qs.set("types", REPORT_TYPES.filter((t) => filters.types.includes(t)).join(","));
  }
  if (filters.year) qs.set("year", filters.year);
  if (!filters.showDryRuns) qs.set("dry", "0");
  return qs.toString();
}

function keep(entry: ReportEntry, filters: ReportFilters): boolean {
  if (!filters.types.includes(entry.type)) return false;
  if (!filters.showDryRuns && entry.available && entry.dry_run) return false;
  return true;
}

/** The cycles the page shows under these filters; empty blocks and months drop out. */
export function applyFilters(cycles: ReportCycle[], filters: ReportFilters): ReportCycle[] {
  const out: ReportCycle[] = [];
  for (const cycle of cycles) {
    if (filters.year && !cycle.period.startsWith(filters.year)) continue;
    const scoring_rounds = cycle.scoring_rounds
      .map((r) => ({ ...r, reports: r.reports.filter((e) => keep(e, filters)) }))
      .filter((r) => r.reports.length > 0);
    const forecast_runs = cycle.forecast_runs
      .map((r) => ({ ...r, reports: r.reports.filter((e) => keep(e, filters)) }))
      .filter((r) => r.reports.length > 0);
    if (scoring_rounds.length || forecast_runs.length) {
      out.push({ period: cycle.period, scoring_rounds, forecast_runs });
    }
  }
  return out;
}

const MONTHS = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

/** "2026-11" -> "November 2026". */
export function periodLabel(period: string): string {
  const m = /^(\d{4})-(\d{2})$/.exec(period);
  if (!m) return period;
  const idx = Number(m[2]) - 1;
  return idx >= 0 && idx < 12 ? `${MONTHS[idx]} ${m[1]}` : period;
}

/** "2026-11-11" -> "11 November 2026". */
export function dateLabel(date: string | null | undefined): string | null {
  if (!date) return null;
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(date);
  if (!m) return date;
  return `${Number(m[3])} ${MONTHS[Number(m[2]) - 1]} ${m[1]}`;
}

export interface VerdictChip {
  label: string;
  className: string;
}

/** The forecast review's verdict as a chip: the word always shown, colour as a second cue. */
export function verdictChip(verdict: string | null | undefined): VerdictChip | null {
  switch ((verdict || "").toLowerCase()) {
    case "worked":
      return { label: "Worked", className: "border-green-300 bg-green-50 text-green-900" };
    case "worked with faults":
      return {
        label: "Worked with faults",
        className: "border-amber-300 bg-amber-50 text-amber-900",
      };
    case "failed":
      return { label: "Failed", className: "border-red-300 bg-red-50 text-red-900" };
    default:
      return null;
  }
}

/** What a placeholder row says. */
export function placeholderText(entry: ReportEntry): string {
  return entry.pending ? "Review pending" : "Not yet written";
}

const API_BASE =
  process.env.NEXT_PUBLIC_PYTHIA_API_BASE ?? "http://localhost:8000/v1";

/** The API's download route for a report; it redirects to the release asset. */
export function downloadUrl(reportId: string, format: "pdf" | "md"): string {
  return `${API_BASE}/analysis_reports/${encodeURIComponent(reportId)}/download?format=${format}`;
}

/** Earlier versions of an entry (every version but the one shown). */
export function earlierVersions(entry: ReportEntry): ReportVersion[] {
  return entry.versions.filter((v) => v.report_id !== entry.report_id);
}
