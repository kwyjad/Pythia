// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
// See the LICENSE file in the project root for details.

import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { splitOverview } from "../../../lib/overview_split";
import { extractHeadings } from "../../../lib/simple_markdown";
import { OverviewBody } from "../OverviewSection";

const CURRENT = `## Fred in brief

Brief text.

## 1. Purpose and key terms

Purpose text.

## 2. Data

Data text.

## Changes in this Version

### November 2026

- **A change.** Detail.
`;

// Archived versions put the change log first.
const ARCHIVED = `## Changes in this Version

- **Old change.** Detail.

## System at a Glance

Glance text.
`;

describe("splitOverview", () => {
  it("takes the change log out wherever it sits", () => {
    const cur = splitOverview(CURRENT);
    expect(cur.changelog).toContain("A change.");
    expect(cur.main).not.toContain("Changes in this Version");
    expect(cur.main).toContain("2. Data");

    const old = splitOverview(ARCHIVED);
    expect(old.changelog).toContain("Old change.");
    expect(old.changelog).not.toContain("Glance text");
    expect(old.main).toContain("System at a Glance");
  });

  it("returns a document with no change log whole", () => {
    const md = "## Only\n\nText.\n";
    expect(splitOverview(md)).toEqual({ main: md, changelog: null });
  });
});

describe("OverviewBody", () => {
  it("puts the change log last, in a closed panel", () => {
    const { container } = render(<OverviewBody markdown={CURRENT} />);
    const main = screen.getByTestId("overview-main");
    const changes = screen.getByTestId("overview-changes");
    // The change log follows the main text in the document.
    expect(
      main.compareDocumentPosition(changes) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    const details = changes.querySelector("details");
    expect(details).not.toBeNull();
    expect(details?.open).toBe(false);
    expect(within(changes).getByText("Changes in this version")).toBeTruthy();
    expect(main.textContent).not.toContain("A change.");
    expect(container.textContent).toContain("A change.");
  });

  it("renders a document with no change log whole, with no panel", () => {
    const md = "## One\n\nFirst.\n\n## Two\n\nSecond.\n";
    render(<OverviewBody markdown={md} />);
    expect(screen.queryByTestId("overview-changes")).toBeNull();
    expect(screen.getByTestId("overview-main").textContent).toContain("Second.");
  });

  it("lists every ## heading of the main text, linked to its anchor", () => {
    render(<OverviewBody markdown={CURRENT} />);
    const nav = screen.getByRole("navigation", { name: "Contents" });
    const links = within(nav).getAllByRole("link");
    const expected = extractHeadings(splitOverview(CURRENT).main);
    expect(expected.map((h) => h.text)).toEqual([
      "Fred in brief",
      "1. Purpose and key terms",
      "2. Data",
    ]);
    expected.forEach((h, i) => {
      expect(links[i].textContent).toBe(h.text);
      expect(links[i].getAttribute("href")).toBe(`#${h.id}`);
      expect(document.getElementById(h.id)).not.toBeNull();
    });
    // The last entry points at the change log panel.
    expect(links[links.length - 1].getAttribute("href")).toBe("#changes-in-this-version");
  });
});
