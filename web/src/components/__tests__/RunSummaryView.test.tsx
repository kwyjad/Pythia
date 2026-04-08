import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import RunSummaryView from "../RunSummaryView";
import type { RunSummaryResponse } from "../../lib/types";

vi.mock("../InfoTooltip", () => ({
  default: ({ text }: { text: string }) => (
    <span data-testid="info-tooltip" title={text}>
      ?
    </span>
  ),
}));

vi.mock("../KpiCard", () => ({
  default: ({
    label,
    value,
  }: {
    label: React.ReactNode;
    value: React.ReactNode;
  }) => (
    <div data-testid="kpi-card">
      <span data-testid="kpi-label">{label}</span>
      <span data-testid="kpi-value">{value}</span>
    </div>
  ),
}));

const MOCK_DATA: RunSummaryResponse = {
  run_id: "fc_test_run",
  hs_run_id: "hs_test_run",
  updated_at: "2026-04-01T03:16:34Z",
  coverage: {
    countries_scanned: 122,
    hazard_pairs_assessed: 365,
    seasonal_screenouts: 123,
    pairs_with_questions: 120,
    total_questions: 229,
    countries_with_forecasts: 73,
    countries_no_questions: 49,
    triaged_quiet: 245,
  },
  metrics: [
    {
      metric: "FATALITIES",
      label: "Fatalities",
      questions: 47,
      countries: 47,
      hazards: [{ hazard_code: "ACE", count: 47 }],
    },
    {
      metric: "PA",
      label: "People affected",
      questions: 89,
      countries: 64,
      hazards: [
        { hazard_code: "ACE", count: 47 },
        { hazard_code: "FL", count: 36 },
        { hazard_code: "TC", count: 6 },
      ],
    },
    {
      metric: "EVENT_OCCURRENCE",
      label: "Event occurrence",
      questions: 73,
      countries: 60,
      hazards: [
        { hazard_code: "DR", count: 31 },
        { hazard_code: "FL", count: 36 },
        { hazard_code: "TC", count: 6 },
      ],
    },
    {
      metric: "PHASE3PLUS_IN_NEED",
      label: "Phase 3+ population",
      questions: 20,
      countries: 20,
      hazards: [{ hazard_code: "DR", count: 20 }],
    },
  ],
  rc_assessment: {
    total_assessed: 365,
    levels: { L0: 309, L1: 35, L2: 15, L3: 6 },
    l1_plus_rate: 0.153,
    by_hazard: [
      { hazard_code: "ACE", L0: 92, L1: 17, L2: 7, L3: 6 },
      { hazard_code: "DR", L0: 82, L1: 7, L2: 6, L3: 0 },
      { hazard_code: "FL", L0: 97, L1: 8, L2: 2, L3: 0 },
      { hazard_code: "TC", L0: 41, L1: 0, L2: 0, L3: 0 },
    ],
    countries_by_level: { L1: 30, L2: 9, L3: 6 },
  },
  tracks: {
    track1: { questions: 103, countries: 45, models: 6 },
    track2: { questions: 126, countries: 51 },
  },
  ensemble: { expected: 7, ok: 6 },
  cost: {
    total_usd: 37.93,
    total_tokens: 18325326,
    by_phase: [
      { phase: "hs_triage", label: "HS triage", cost_usd: 18.4 },
      { phase: "spd_v2", label: "SPD ensemble", cost_usd: 17.23 },
      { phase: "binary_v2", label: "Binary forecasts", cost_usd: 2.02 },
      { phase: "scenario_v2", label: "Scenarios", cost_usd: 0.28 },
    ],
  },
  llm_health: {
    total_calls: 2923,
    errors: 10,
    error_rate: 0.003,
  },
};

describe("RunSummaryView", () => {
  it("renders all six sections", () => {
    render(<RunSummaryView data={MOCK_DATA} />);

    // Section 1: KPI row has KPI cards
    const kpiCards = screen.getAllByTestId("kpi-card");
    expect(kpiCards.length).toBeGreaterThanOrEqual(5);

    // Section 2: Coverage funnel
    expect(screen.getByText(/Coverage funnel/i)).toBeDefined();
    expect(screen.getByText(/122 countries scanned/)).toBeDefined();
    expect(screen.getByText(/229 forecast questions/)).toBeDefined();

    // Section 3: Forecasts by metric
    expect(screen.getByText(/Forecasts by metric/i)).toBeDefined();
    expect(screen.getByText("Fatalities")).toBeDefined();
    expect(screen.getByText("People affected")).toBeDefined();
    expect(screen.getByText("Event occurrence")).toBeDefined();
    expect(screen.getByText("Phase 3+ population")).toBeDefined();

    // Section 4: RC assessment
    expect(screen.getByText(/Regime change assessment/i)).toBeDefined();
    // Hazard table should have ACE row
    expect(
      screen.getByText((_content, element) => {
        return (
          element?.tagName === "TD" &&
          /Armed Conflict/.test(element.textContent ?? "")
        );
      })
    ).toBeDefined();

    // Section 5: Track split
    expect(screen.getByText(/Track split/i)).toBeDefined();
    expect(screen.getByText(/103 questions/)).toBeDefined();
    expect(screen.getByText(/126 questions/)).toBeDefined();

    // Section 6: Cost breakdown
    expect(screen.getByText(/Cost breakdown/i)).toBeDefined();
    expect(screen.getByText("HS triage")).toBeDefined();
    expect(screen.getByText("SPD ensemble")).toBeDefined();
  });

  it("shows metric question counts", () => {
    render(<RunSummaryView data={MOCK_DATA} />);

    // The metric grid should show correct question counts
    expect(screen.getByText("47")).toBeDefined(); // Fatalities
    expect(screen.getByText("89")).toBeDefined(); // PA
    expect(screen.getByText("73")).toBeDefined(); // EVENT_OCCURRENCE
    expect(screen.getByText("20")).toBeDefined(); // PHASE3PLUS_IN_NEED
  });

  it("shows hazard pills", () => {
    render(<RunSummaryView data={MOCK_DATA} />);

    // Should show hazard pills like "Armed Conflict (47)"
    expect(screen.getByText("Armed Conflict (47)")).toBeDefined();
    expect(screen.getByText("Flood (36)")).toBeDefined();
  });

  it("shows cost per phase", () => {
    render(<RunSummaryView data={MOCK_DATA} />);

    expect(screen.getByText("$18.40")).toBeDefined();
    expect(screen.getByText("$17.23")).toBeDefined();
    expect(screen.getByText("$2.02")).toBeDefined();
    expect(screen.getByText("$0.28")).toBeDefined();
  });
});
