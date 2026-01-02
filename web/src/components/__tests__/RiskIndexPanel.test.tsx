import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import RiskIndexPanel from "../RiskIndexPanel";

const kpiScopes = vi.hoisted(() => ({
  available_months: [
    { year_month: "2024-01", label: "Jan 2024", is_latest: true },
  ],
  selected_month: "2024-01",
  scopes: {
    selected_run: {
      label: "Selected run",
      questions: 0,
      forecasts: 0,
      resolved_questions: 0,
      forecasts_by_hazard: {},
    },
  },
}));

const riskIndexResponse = vi.hoisted(() => ({
  metric: "FATALITIES",
  target_month: "2024-02",
  rows: [],
}));

const apiGetMock = vi.hoisted(() =>
  vi.fn(async (path: string) => {
    if (path === "/risk_index") {
      return riskIndexResponse;
    }
    if (path === "/diagnostics/kpi_scopes") {
      return kpiScopes;
    }
    throw new Error(`Unhandled path: ${path}`);
  })
);

vi.mock("next/navigation", () => ({
  useSearchParams: () => ({ get: () => null }),
}));

vi.mock("../../lib/api", () => ({
  apiGet: apiGetMock,
}));

vi.mock("../RiskIndexMap", () => ({
  default: () => <div data-testid="risk-index-map" />,
}));

vi.mock("../RiskIndexTable", () => ({
  default: () => <div data-testid="risk-index-table" />,
}));

describe("RiskIndexPanel", () => {
  it("keeps per-capita options enabled after switching to fatalities EIV", async () => {
    render(
      <RiskIndexPanel
        initialResponse={{
          metric: "PA",
          target_month: "2024-01",
          rows: [],
        }}
        countriesRows={[]}
        kpiScopes={kpiScopes}
      />
    );

    const viewSelect = screen.getByLabelText("View");
    fireEvent.change(viewSelect, { target: { value: "FATALITIES_EIV" } });

    await waitFor(() => expect(apiGetMock).toHaveBeenCalled());

    expect(
      screen.getByRole("option", {
        name: "People Affected (PA) per capita EIV",
      })
    ).not.toBeDisabled();
    expect(
      screen.getByRole("option", {
        name: "Armed Conflict (ACE) fatalities per capita EIV",
      })
    ).not.toBeDisabled();
  });
});
