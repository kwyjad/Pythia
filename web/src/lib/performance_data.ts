// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
// See the LICENSE file in the project root for details.

// Pure data shaping for the Performance page (Nov 2026 overhaul): the
// headline cards, the scores-over-time series and the skill bars. No React,
// so every rule here is pinned by web/src/lib/__tests__/performance_data.test.ts.

import type {
  PairedScoreRow,
  PerformanceSummaryRow,
  PerformanceTimeseriesRow,
  ScoreFamily,
} from "./types";
import {
  NA_BINARY_ENSEMBLE,
  NA_BINARY_SCORE_TYPE,
  NA_NO_SCORES,
} from "./score_glossary";

/** Below this many scored questions a month's point is drawn hollow. */
export const LOW_N_THRESHOLD = 10;
/** Below this many paired questions a skill bar is hatched and marked "low n". */
export const SKILL_LOW_N = 30;
/** A reference needs this many paired questions to be called the best one,
 *  when any reference reaches it. */
export const REFERENCE_MIN_QUESTIONS = 10;

export type Stat = "avg" | "median";
export type RangeKey = "all" | "12" | "6" | "3";
export type ScoreType = "brier" | "log" | "crps";
export const SCORE_TYPES: ScoreType[] = ["brier", "log", "crps"];
export const SCORE_TYPE_LABELS: Record<ScoreType, string> = {
  brier: "Brier",
  log: "Log Loss",
  crps: "CRPS",
};
export const RANGE_LABELS: Record<RangeKey, string> = {
  all: "All time",
  "12": "Last 12 months",
  "6": "Last 6 months",
  "3": "Last 3 months",
};

// ---------------------------------------------------------------------------
// Model classification
// ---------------------------------------------------------------------------

export const isEnsembleAggregate = (name: string | null | undefined): boolean =>
  (name ?? "").toLowerCase().startsWith("ensemble");

export const isExternalReference = (name: string | null | undefined): boolean => {
  const n = name ?? "";
  return n.startsWith("__ext_") && !n.startsWith("__ext_sibyl");
};

/** An individual LLM forecaster: not an aggregate, Sibyl, a reference or a
 *  recalibration copy. What the "Models" KPI counts. */
export function isIndividualForecaster(name: string | null | undefined): boolean {
  if (!name) return false;
  const n = name.toLowerCase();
  if (n.startsWith("ensemble") || n.startsWith("track2_")) return false;
  if (n === "sibyl" || n.startsWith("__ext_")) return false;
  if (n.endsWith("__raw") || n.endsWith("__recal")) return false;
  return true;
}

/** Short label for a reference model in a one-line comparison. */
export function referenceLabel(name: string): string {
  if (name === "__ext_views") return "ViEWS";
  if (name === "__ext_ref_pool") return "ref pool";
  if (name === "__ext_conflictology12") return "12-month conflictology";
  return name.replace(/^__ext_/, "").replace(/_/g, " ");
}

// ---------------------------------------------------------------------------
// Months and ranges
// ---------------------------------------------------------------------------

const monthIdx = (ym: string): number => {
  const [y, m] = ym.split("-").map(Number);
  return y * 12 + (m - 1);
};

const ymOf = (d: Date): string =>
  `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, "0")}`;

/** The first month kept by a range ending in the current month, or null. */
export function rangeCutoff(range: RangeKey, today: Date = new Date()): string | null {
  if (range === "all") return null;
  const idx = monthIdx(ymOf(today)) - (Number(range) - 1);
  return `${Math.floor(idx / 12)}-${String((idx % 12) + 1).padStart(2, "0")}`;
}

export function cutToRange<T extends { month?: string }>(
  rows: T[],
  range: RangeKey,
  today: Date = new Date(),
): T[] {
  const cut = rangeCutoff(range, today);
  if (!cut) return rows;
  return rows.filter((r) => (r.month ?? "") >= cut);
}

// ---------------------------------------------------------------------------
// Scores over time
// ---------------------------------------------------------------------------

/** Score types the chosen family and ensemble have any data for; the metric
 *  selector disables the others. */
export function availableScoreTypes(
  rows: PerformanceTimeseriesRow[],
  family: ScoreFamily,
  ensemble: string | null,
): Set<ScoreType> {
  const out = new Set<ScoreType>();
  for (const r of rows) {
    if (r.score_family !== family || r.model_name !== ensemble) continue;
    if ((SCORE_TYPES as string[]).includes(r.score_type) && r.n_questions > 0) {
      out.add(r.score_type as ScoreType);
    }
  }
  return out;
}

export type SeriesPoint = {
  month: string;
  value: number;
  n: number;
  lowN: boolean;
  /** The benchmark on the same paired questions; null when none is shown. */
  bench: number | null;
};

export type SeriesOptions = {
  family: ScoreFamily;
  ensemble: string | null;
  metric: ScoreType;
  stat: Stat;
  range: RangeKey;
  /** External reference to overlay, paired; null for the ensemble alone. */
  reference: string | null;
  today?: Date;
};

/**
 * The points the line chart draws. With no benchmark the ensemble line covers
 * every scored question of each month. With one, BOTH lines are read from the
 * paired rows, so they describe the same questions month by month.
 */
export function buildSeries(
  rows: PerformanceTimeseriesRow[],
  paired: PairedScoreRow[],
  opt: SeriesOptions,
): SeriesPoint[] {
  const today = opt.today ?? new Date();
  const pts: SeriesPoint[] = [];
  if (opt.reference) {
    for (const p of cutToRange(paired, opt.range, today)) {
      if (
        p.model_name !== opt.ensemble ||
        p.reference !== opt.reference ||
        p.score_family !== opt.family ||
        p.score_type !== opt.metric ||
        !p.month
      ) {
        continue;
      }
      const v = opt.stat === "avg" ? p.model_avg : p.model_median;
      const b = opt.stat === "avg" ? p.reference_avg : p.reference_median;
      if (v == null) continue;
      pts.push({ month: p.month, value: v, n: p.n_questions, lowN: isLowN(p.n_questions), bench: b });
    }
  } else {
    for (const r of cutToRange(rows, opt.range, today)) {
      if (
        r.model_name !== opt.ensemble ||
        r.score_family !== opt.family ||
        r.score_type !== opt.metric
      ) {
        continue;
      }
      const v = opt.stat === "avg" ? r.avg_value : r.median_value;
      if (v == null) continue;
      pts.push({ month: r.month, value: v, n: r.n_questions, lowN: isLowN(r.n_questions), bench: null });
    }
  }
  return pts.sort((a, b) => a.month.localeCompare(b.month));
}

export const isLowN = (n: number): boolean => n < LOW_N_THRESHOLD;

/** A segment is dashed when it runs INTO a low-n point. */
export const segmentIsDashed = (points: SeriesPoint[], i: number): boolean =>
  i > 0 && i < points.length && points[i].lowN;

/** A y domain that starts at a sensible floor, not always at zero. */
export function yDomain(values: number[]): [number, number] {
  const vs = values.filter((v) => Number.isFinite(v));
  if (!vs.length) return [0, 1];
  let lo = Math.min(...vs);
  let hi = Math.max(...vs);
  if (hi - lo < 1e-9) {
    const pad = Math.max(Math.abs(hi) * 0.1, 0.01);
    lo -= pad;
    hi += pad;
  } else {
    const pad = (hi - lo) * 0.15;
    lo -= pad;
    hi += pad;
  }
  return [Math.max(0, lo), hi];
}

/** Round tick values covering a domain. */
export function niceTicks(lo: number, hi: number, count = 4): number[] {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) ?? 10 * mag;
  const out: number[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) {
    out.push(Number(t.toPrecision(12)));
  }
  return out;
}

// ---------------------------------------------------------------------------
// Headline cards
// ---------------------------------------------------------------------------

export type HeadlineCell =
  | { kind: "value"; value: number; n: number }
  | { kind: "na"; reason: string };

/** One family's value for a headline card, or the reason it has none. */
export function headlineCell(
  overall: PerformanceSummaryRow[],
  family: ScoreFamily,
  ensemble: string | null,
  scoreType: ScoreType,
  stat: Stat,
): HeadlineCell {
  const row = overall.find(
    (r) =>
      (r.score_family ?? "spd") === family &&
      r.model_name === ensemble &&
      r.score_type === scoreType,
  );
  const v = row ? (stat === "avg" ? row.avg_value : row.median_value) : null;
  if (row && v != null) return { kind: "value", value: v, n: row.n_questions };
  if (family === "binary") {
    if (scoreType !== "brier") return { kind: "na", reason: NA_BINARY_SCORE_TYPE };
    const anyBinary = overall.some(
      (r) => r.score_family === "binary" && isEnsembleAggregate(r.model_name),
    );
    if (anyBinary) return { kind: "na", reason: NA_BINARY_ENSEMBLE };
  }
  return { kind: "na", reason: NA_NO_SCORES };
}

/**
 * The toughest external reference for an ensemble on one family and score
 * type: the lowest reference-to-ensemble ratio on paired questions, among the
 * references with at least REFERENCE_MIN_QUESTIONS paired questions when any
 * has that many. A ratio, because each reference is paired on a different
 * question set and raw values across sets do not compare.
 */
export function pickBestReference(
  referenceRows: PairedScoreRow[],
  family: ScoreFamily,
  ensemble: string | null,
  scoreType: ScoreType,
  stat: Stat,
): PairedScoreRow | null {
  const cands = referenceRows.filter(
    (r) =>
      r.model_name === ensemble &&
      r.score_family === family &&
      r.score_type === scoreType &&
      pairedValue(r, "model", stat) != null &&
      pairedValue(r, "reference", stat) != null,
  );
  if (!cands.length) return null;
  const wide = cands.filter((r) => r.n_questions >= REFERENCE_MIN_QUESTIONS);
  const pool = wide.length ? wide : cands;
  const ratio = (r: PairedScoreRow) => {
    const m = pairedValue(r, "model", stat) as number;
    const ref = pairedValue(r, "reference", stat) as number;
    return m > 0 ? ref / m : ref;
  };
  return [...pool].sort((a, b) => ratio(a) - ratio(b) || b.n_questions - a.n_questions)[0];
}

export function pairedValue(
  r: PairedScoreRow,
  side: "model" | "reference",
  stat: Stat,
): number | null {
  if (side === "model") return stat === "avg" ? r.model_avg : r.model_median;
  return stat === "avg" ? r.reference_avg : r.reference_median;
}

// ---------------------------------------------------------------------------
// Skill bars
// ---------------------------------------------------------------------------

/** 1 − model / reference. Null when the reference score is not positive. */
export function skillScore(model: number | null, reference: number | null): number | null {
  if (model == null || reference == null || !(reference > 0)) return null;
  return 1 - model / reference;
}

export type SkillInfo = {
  skill: number;
  n: number;
  family: ScoreFamily;
  lowN: boolean;
};

/** A model's paired skill vs climatology on one score type: the SPD family
 *  where it has SPD pairs, else the binary family. */
export function modelSkill(
  skillRows: PairedScoreRow[],
  modelName: string,
  scoreType: ScoreType,
  stat: Stat = "avg",
): SkillInfo | null {
  for (const family of ["spd", "binary"] as ScoreFamily[]) {
    const r = skillRows.find(
      (x) => x.model_name === modelName && x.score_family === family && x.score_type === scoreType,
    );
    if (!r) continue;
    const s = skillScore(pairedValue(r, "model", stat), pairedValue(r, "reference", stat));
    if (s == null) continue;
    return { skill: s, n: r.n_questions, family, lowN: r.n_questions < SKILL_LOW_N };
  }
  return null;
}

/** The bar track's half-width in skill units: the largest |skill| shown,
 *  rounded up to a quarter, never below ±0.25 and never above ±1. */
export function skillScale(skills: Array<number | null | undefined>): number {
  const m = Math.max(0, ...skills.filter((x): x is number => x != null && Number.isFinite(x)).map(Math.abs));
  return Math.min(1, Math.max(0.25, Math.ceil(m * 4) / 4));
}

/** Bar geometry centred on zero: the bar runs right for a positive skill and
 *  left for a negative one, as a percent of HALF the track, clamped at the
 *  scale (a skill beyond it fills its half and is marked clipped). */
export function skillBarGeometry(
  skill: number,
  scale = 1,
): { side: "pos" | "neg" | "zero"; widthPct: number; clipped: boolean } {
  const sc = scale > 0 ? scale : 1;
  const clipped = Math.abs(skill) > sc;
  const c = Math.max(-sc, Math.min(sc, skill));
  if (Math.abs(c) < 1e-9) return { side: "zero", widthPct: 0, clipped: false };
  return { side: c > 0 ? "pos" : "neg", widthPct: (Math.abs(c) / sc) * 50, clipped };
}

/** Score types written for an ACE/PA outcome that is only a LOWER BOUND
 *  (the IOM DTM rung). Their own scale; never pooled with brier/log/crps,
 *  never in a headline, shown as their own line. */
export const CENSORED_SCORE_TYPES = ["brier_censored", "log_censored", "crps_censored"] as const;
export const CENSORED_SCORE_LABELS: Record<(typeof CENSORED_SCORE_TYPES)[number], string> = {
  brier_censored: "Brier (at least)",
  log_censored: "Log Loss (at least)",
  crps_censored: "CRPS (at least)",
};

export type CensoredCell = {
  scoreType: (typeof CENSORED_SCORE_TYPES)[number];
  value: number;
  nQuestions: number;
};

/** The censored scores of one model, from the overall rows; [] when none. */
export function censoredCells(
  rows: PerformanceSummaryRow[] | undefined,
  model: string | null | undefined,
  stat: Stat,
): CensoredCell[] {
  const out: CensoredCell[] = [];
  for (const st of CENSORED_SCORE_TYPES) {
    const r = (rows ?? []).find((x) => x.score_type === st && x.model_name === model);
    const v = r ? (stat === "median" ? r.median_value ?? r.avg_value : r.avg_value) : null;
    if (r && v != null && r.n_questions > 0) {
      out.push({ scoreType: st, value: v, nQuestions: r.n_questions });
    }
  }
  return out;
}
