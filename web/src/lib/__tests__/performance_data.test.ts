import { describe, expect, it } from "vitest";

import {
  LOW_N_THRESHOLD,
  SKILL_LOW_N,
  availableScoreTypes,
  buildSeries,
  cutToRange,
  headlineCell,
  isIndividualForecaster,
  modelSkill,
  niceTicks,
  pickBestReference,
  rangeCutoff,
  segmentIsDashed,
  skillBarGeometry,
  skillScore,
  yDomain,
} from "../performance_data";
import { NA_BINARY_ENSEMBLE, NA_BINARY_SCORE_TYPE, NA_NO_SCORES } from "../score_glossary";
import type { PairedScoreRow, PerformanceSummaryRow, PerformanceTimeseriesRow } from "../types";

const TODAY = new Date(Date.UTC(2026, 10, 24)); // 2026-11-24

const ts = (month: string, model: string, value: number, n = 20, extra: Partial<PerformanceTimeseriesRow> = {}): PerformanceTimeseriesRow => ({
  month,
  score_family: "spd",
  score_type: "brier",
  model_name: model,
  n_samples: n,
  n_questions: n,
  avg_value: value,
  median_value: value / 2,
  ...extra,
});

const paired = (month: string | undefined, model: string, ref: string, m: number, r: number, n = 20, extra: Partial<PairedScoreRow> = {}): PairedScoreRow => ({
  month,
  model_name: model,
  reference: ref,
  score_family: "spd",
  score_type: "brier",
  n_samples: n,
  n_questions: n,
  model_avg: m,
  model_median: m,
  reference_avg: r,
  reference_median: r,
  ...extra,
});

describe("range cut", () => {
  it("keeps the last N months including this one", () => {
    expect(rangeCutoff("3", TODAY)).toBe("2026-09");
    expect(rangeCutoff("12", TODAY)).toBe("2025-12");
    expect(rangeCutoff("all", TODAY)).toBeNull();
    const rows = ["2025-11", "2025-12", "2026-08", "2026-09", "2026-11"].map((m) => ({ month: m }));
    expect(cutToRange(rows, "3", TODAY).map((r) => r.month)).toEqual(["2026-09", "2026-11"]);
    expect(cutToRange(rows, "12", TODAY)).toHaveLength(4);
    expect(cutToRange(rows, "all", TODAY)).toHaveLength(5);
  });

  it("crosses a year boundary", () => {
    expect(rangeCutoff("6", new Date(Date.UTC(2026, 1, 3)))).toBe("2025-09");
  });
});

describe("series", () => {
  const rows = [
    ts("2026-09", "ensemble_mean_v2", 0.5, 30),
    ts("2026-08", "ensemble_mean_v2", 0.4, 30),
    ts("2026-10", "ensemble_mean_v2", 0.6, 4),
    ts("2026-10", "ensemble_bayesmc_v2", 0.9, 30),
    ts("2026-10", "ensemble_mean_v2", 0.02, 30, { score_family: "binary" }),
  ];

  it("follows the chosen ensemble, family and statistic, in month order", () => {
    const pts = buildSeries(rows, [], {
      family: "spd", ensemble: "ensemble_mean_v2", metric: "brier", stat: "avg", range: "all", reference: null, today: TODAY,
    });
    expect(pts.map((p) => [p.month, p.value])).toEqual([["2026-08", 0.4], ["2026-09", 0.5], ["2026-10", 0.6]]);
    const med = buildSeries(rows, [], {
      family: "spd", ensemble: "ensemble_mean_v2", metric: "brier", stat: "median", range: "all", reference: null, today: TODAY,
    });
    expect(med[0].value).toBeCloseTo(0.2);
  });

  it("marks months under the low-n threshold hollow and dashes the segment into them", () => {
    const pts = buildSeries(rows, [], {
      family: "spd", ensemble: "ensemble_mean_v2", metric: "brier", stat: "avg", range: "all", reference: null, today: TODAY,
    });
    expect(LOW_N_THRESHOLD).toBe(10);
    expect(pts.map((p) => p.lowN)).toEqual([false, false, true]);
    expect(segmentIsDashed(pts, 0)).toBe(false); // no segment into the first point
    expect(segmentIsDashed(pts, 1)).toBe(false);
    expect(segmentIsDashed(pts, 2)).toBe(true);
  });

  it("reads BOTH lines from the paired rows when a benchmark is on", () => {
    const pr = [
      paired("2026-08", "ensemble_mean_v2", "__ext_climatology", 0.3, 0.35, 12),
      paired("2026-09", "ensemble_mean_v2", "__ext_climatology", 0.45, 0.4, 8),
      paired("2026-09", "ensemble_mean_v2", "__ext_uniform", 0.45, 0.9, 8),
    ];
    const pts = buildSeries(rows, pr, {
      family: "spd", ensemble: "ensemble_mean_v2", metric: "brier", stat: "avg", range: "all",
      reference: "__ext_climatology", today: TODAY,
    });
    expect(pts.map((p) => [p.month, p.value, p.bench, p.n])).toEqual([
      ["2026-08", 0.3, 0.35, 12],
      ["2026-09", 0.45, 0.4, 8],
    ]);
    expect(pts[1].lowN).toBe(true);
  });

  it("disables metrics the family has no data for", () => {
    const binaryOnlyBrier = [
      ts("2026-09", "ensemble_mean_v2", 0.02, 10, { score_family: "binary" }),
      ts("2026-09", "ensemble_mean_v2", 0.5, 10, { score_type: "crps" }),
      ts("2026-09", "ensemble_mean_v2", 0.4, 10),
    ];
    expect([...availableScoreTypes(binaryOnlyBrier, "binary", "ensemble_mean_v2")]).toEqual(["brier"]);
    expect(availableScoreTypes(binaryOnlyBrier, "spd", "ensemble_mean_v2")).toEqual(new Set(["crps", "brier"]));
    expect(availableScoreTypes(binaryOnlyBrier, "binary", "ensemble_bayesmc_v2").size).toBe(0);
  });

  it("starts the y axis at a floor above zero when the data sits high", () => {
    const [lo, hi] = yDomain([0.5, 0.6, 0.7]);
    expect(lo).toBeGreaterThan(0.4);
    expect(lo).toBeLessThan(0.5);
    expect(hi).toBeGreaterThan(0.7);
    expect(yDomain([0.01, 0.4])[0]).toBe(0);
    const [a, b] = yDomain([0.3, 0.3]);
    expect(a).toBeLessThan(0.3);
    expect(b).toBeGreaterThan(0.3);
    const ticks = niceTicks(0.47, 0.73, 4);
    expect(ticks.every((t) => t >= 0.47 && t <= 0.73)).toBe(true);
    expect(ticks.length).toBeGreaterThanOrEqual(2);
  });
});

describe("skill bar maths", () => {
  it("is one minus the ratio, null without a positive reference", () => {
    expect(skillScore(0.3, 0.4)).toBeCloseTo(0.25);
    expect(skillScore(0.5, 0.4)).toBeCloseTo(-0.25);
    expect(skillScore(0.3, 0)).toBeNull();
    expect(skillScore(null, 0.4)).toBeNull();
  });

  it("draws right for a gain and left for a loss, from the centre, clamped", () => {
    expect(skillBarGeometry(0.25)).toEqual({ side: "pos", widthPct: 12.5 });
    expect(skillBarGeometry(-0.5)).toEqual({ side: "neg", widthPct: 25 });
    expect(skillBarGeometry(-3)).toEqual({ side: "neg", widthPct: 50 });
    expect(skillBarGeometry(0)).toEqual({ side: "zero", widthPct: 0 });
  });

  it("is paired per model, prefers SPD, and flags low n", () => {
    const rows = [
      paired(undefined, "gpt-6-sol", "__ext_climatology", 0.3, 0.4, 26),
      paired(undefined, "gemini-3.1-pro-preview", "__ext_climatology", 0.45, 0.4, 126),
      paired(undefined, "track2_flash", "__ext_climatology", 0.03, 0.06, 50, { score_family: "binary" }),
    ];
    const gpt = modelSkill(rows, "gpt-6-sol", "brier");
    expect(gpt?.skill).toBeCloseTo(0.25);
    expect(gpt?.lowN).toBe(true);
    expect(SKILL_LOW_N).toBe(30);
    const pro = modelSkill(rows, "gemini-3.1-pro-preview", "brier");
    expect(pro?.skill).toBeCloseTo(-0.125);
    expect(pro?.lowN).toBe(false);
    expect(modelSkill(rows, "track2_flash", "brier")?.family).toBe("binary");
    expect(modelSkill(rows, "nobody", "brier")).toBeNull();
  });
});

describe("headline", () => {
  const row = (family: "spd" | "binary", model: string, st: string, v: number): PerformanceSummaryRow => ({
    hazard_code: "*", metric: "*", score_family: family, score_type: st, model_name: model,
    n_samples: 5, n_questions: 5, avg_value: v, median_value: v,
  });
  const overall = [
    row("spd", "ensemble_mean_v2", "brier", 0.5),
    row("spd", "ensemble_bayesmc_v2", "brier", 0.45),
    row("binary", "ensemble_mean_v2", "brier", 0.03),
  ];

  it("says why a cell has no value", () => {
    expect(headlineCell(overall, "spd", "ensemble_mean_v2", "brier", "avg")).toEqual({ kind: "value", value: 0.5, n: 5 });
    expect(headlineCell(overall, "binary", "ensemble_mean_v2", "log", "avg")).toEqual({ kind: "na", reason: NA_BINARY_SCORE_TYPE });
    expect(headlineCell(overall, "binary", "ensemble_bayesmc_v2", "brier", "avg")).toEqual({ kind: "na", reason: NA_BINARY_ENSEMBLE });
    expect(headlineCell(overall, "spd", "ensemble_mean_v2", "crps", "avg")).toEqual({ kind: "na", reason: NA_NO_SCORES });
  });

  it("calls the toughest paired reference the best, on enough questions", () => {
    const refs = [
      paired(undefined, "ensemble_mean_v2", "__ext_uniform", 0.5, 0.9, 100),
      paired(undefined, "ensemble_mean_v2", "__ext_ref_pool", 0.4, 0.38, 40),
      paired(undefined, "ensemble_mean_v2", "__ext_persistence", 0.5, 0.1, 3),
    ];
    expect(pickBestReference(refs, "spd", "ensemble_mean_v2", "brier", "avg")?.reference).toBe("__ext_ref_pool");
    expect(pickBestReference(refs.slice(2), "spd", "ensemble_mean_v2", "brier", "avg")?.reference).toBe("__ext_persistence");
    expect(pickBestReference(refs, "binary", "ensemble_mean_v2", "brier", "avg")).toBeNull();
  });

  it("counts individual forecasters only", () => {
    const names = [
      "gpt-6-sol", "claude-opus-5-5", "ensemble_mean_v2", "ensemble_bayesmc_v2", "track2_flash",
      "sibyl", "__ext_climatology", "__ext_views", "gpt-6-sol__raw", "gemini-3.6-flash__recal", "",
    ];
    expect(names.filter(isIndividualForecaster)).toEqual(["gpt-6-sol", "claude-opus-5-5"]);
  });
});
