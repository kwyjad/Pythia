// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
// See the LICENSE file in the project root for details.

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import ReportRow from "../ReportRow";

const base = { display_name: "x", period: "2026-11", expected: true, versions: [] };

describe("ReportRow", () => {
  it("renders a placeholder with no actions", () => {
    render(
      <ul>
        <ReportRow entry={{ ...base, type: "forecast", available: false, pending: true }} />
      </ul>
    );
    expect(screen.getByTestId("report-placeholder").textContent).toContain("Review pending");
    expect(screen.queryByText("View")).toBeNull();
    expect(screen.queryByText("Download PDF")).toBeNull();
  });

  it("renders a report with its verdict, dry-run badge and three actions", () => {
    render(
      <ul>
        <ReportRow
          entry={{
            ...base,
            type: "forecast",
            available: true,
            report_id: "fred_forecast_review_2026-10-13__fc_1791866322",
            title: "Fred forecast pipeline review",
            summary: "The run worked with faults.",
            verdict: "worked with faults",
            dry_run: true,
            prepared: "2026-11-25",
            md_asset: "a.md",
            pdf_asset: "a.pdf",
          }}
        />
      </ul>
    );
    expect(screen.getByTestId("verdict-chip").textContent).toBe("Worked with faults");
    expect(screen.getByText("Dry run")).toBeTruthy();
    expect(screen.getByText("Forecast pipeline review")).toBeTruthy();
    expect(screen.getByText("View").getAttribute("href")).toBe(
      "/performance-analysis/fred_forecast_review_2026-10-13__fc_1791866322"
    );
    expect(screen.getByText("Download PDF").getAttribute("href")).toMatch(/format=pdf$/);
    expect(screen.getByText("Download Markdown").getAttribute("href")).toMatch(/format=md$/);
    expect(screen.getByText("Prepared 25 November 2026")).toBeTruthy();
  });

  it("hides the PDF action when the sync could not build one", () => {
    render(
      <ul>
        <ReportRow
          entry={{ ...base, type: "skill", available: true, report_id: "r", title: "t",
                   md_asset: "r.md", pdf_asset: null }}
        />
      </ul>
    );
    expect(screen.queryByText("Download PDF")).toBeNull();
    expect(screen.getByText("Download Markdown")).toBeTruthy();
  });
});
