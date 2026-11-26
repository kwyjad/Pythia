// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
// See the LICENSE file in the project root for details.

import { describe, expect, it } from "vitest";

import type { ReportCycle, ReportEntry } from "../analysis_reports";
import {
  DEFAULT_FILTERS,
  applyFilters,
  dateLabel,
  downloadUrl,
  earlierVersions,
  filtersFromParams,
  filtersToQuery,
  periodLabel,
  placeholderText,
  verdictChip,
} from "../analysis_reports";

const entry = (over: Partial<ReportEntry>): ReportEntry => ({
  type: "skill",
  display_name: "Skill analysis",
  period: "2026-11",
  available: true,
  expected: true,
  versions: [],
  ...over,
});

const CYCLES: ReportCycle[] = [
  {
    period: "2026-11",
    scoring_rounds: [
      {
        publish_run_id: "43009722469",
        date: "2026-11-11",
        reports: [
          entry({ report_id: "fred_performance_2026-11", dry_run: true }),
          entry({ type: "resolver", report_id: "fred_resolver_review_2026-11", dry_run: true }),
        ],
      },
    ],
    forecast_runs: [
      {
        fc_run_id: "fc_1794543371",
        date: "2026-11-13",
        publish_run_id: null,
        reports: [entry({ type: "forecast", available: false, pending: true })],
      },
    ],
  },
  {
    period: "2025-12",
    scoring_rounds: [],
    forecast_runs: [
      {
        fc_run_id: "fc_1",
        date: "2025-12-13",
        publish_run_id: "1",
        reports: [entry({ type: "forecast", report_id: "f", verdict: "worked" })],
      },
    ],
  },
];

describe("filters", () => {
  it("default to every type, every year and dry runs shown", () => {
    expect(filtersFromParams({})).toEqual(DEFAULT_FILTERS);
    expect(filtersToQuery(DEFAULT_FILTERS)).toBe("");
  });

  it("round-trip through the URL", () => {
    const f = filtersFromParams(new URLSearchParams("types=forecast,skill,bogus&year=2026&dry=0"));
    expect(f).toEqual({ types: ["forecast", "skill"], year: "2026", showDryRuns: false });
    expect(filtersToQuery(f)).toBe("types=skill%2Cforecast&year=2026&dry=0");
    expect(filtersFromParams({ year: "26" }).year).toBeNull();
  });

  it("drop types, years and dry runs, and empty blocks with them", () => {
    const onlyForecast = applyFilters(CYCLES, { ...DEFAULT_FILTERS, types: ["forecast"] });
    expect(onlyForecast.map((c) => c.scoring_rounds.length)).toEqual([0, 0]);
    expect(onlyForecast.map((c) => c.forecast_runs.length)).toEqual([1, 1]);

    const year = applyFilters(CYCLES, { ...DEFAULT_FILTERS, year: "2025" });
    expect(year.map((c) => c.period)).toEqual(["2025-12"]);

    const noDry = applyFilters(CYCLES, { ...DEFAULT_FILTERS, showDryRuns: false });
    // The dry-run scoring round disappears; the pending placeholder stays.
    expect(noDry[0].scoring_rounds).toEqual([]);
    expect(noDry[0].forecast_runs[0].reports[0].pending).toBe(true);

    expect(applyFilters(CYCLES, { ...DEFAULT_FILTERS, types: [] })).toEqual([]);
  });
});

describe("verdict chips", () => {
  it("name the verdict in words, coloured as a second cue", () => {
    expect(verdictChip("worked")?.label).toBe("Worked");
    expect(verdictChip("Worked with faults")?.label).toBe("Worked with faults");
    expect(verdictChip("worked with faults")?.className).toContain("amber");
    expect(verdictChip("failed")?.label).toBe("Failed");
    expect(verdictChip("failed")?.className).toContain("red");
    expect(verdictChip(null)).toBeNull();
    expect(verdictChip("partial")).toBeNull();
  });
});

describe("placeholders and labels", () => {
  it("say pending for a forecast run and not yet written for a scoring report", () => {
    expect(placeholderText(entry({ type: "forecast", available: false, pending: true }))).toBe(
      "Review pending"
    );
    expect(placeholderText(entry({ type: "resolver", available: false }))).toBe("Not yet written");
  });

  it("format months, dates and download links", () => {
    expect(periodLabel("2026-11")).toBe("November 2026");
    expect(dateLabel("2026-11-11")).toBe("11 November 2026");
    expect(dateLabel(null)).toBeNull();
    expect(downloadUrl("fred_performance_2026-11", "pdf")).toMatch(
      /\/analysis_reports\/fred_performance_2026-11\/download\?format=pdf$/
    );
  });

  it("list earlier versions without the one shown", () => {
    const e = entry({
      report_id: "r__rerun_2026-11-28",
      versions: [
        { report_id: "r__rerun_2026-11-28", rerun_date: "2026-11-28", prepared: null, md_asset: null, pdf_asset: null },
        { report_id: "r", rerun_date: null, prepared: null, md_asset: null, pdf_asset: null },
      ],
    });
    expect(earlierVersions(e).map((v) => v.report_id)).toEqual(["r"]);
  });
});
