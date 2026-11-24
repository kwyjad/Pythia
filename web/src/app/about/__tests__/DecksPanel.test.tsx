// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
// See the LICENSE file in the project root for details.

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { formatSize, parseDecks } from "../../../lib/decks";
import DecksPanel from "../DecksPanel";

const JSON_FIXTURE = {
  drive: { used: true },
  decks: [
    {
      number: 1,
      title: "Resolver",
      file: "1_resolver.pptx",
      view_url: "https://drive.google.com/file/d/ID1/view?usp=sharing",
      download_url: "https://drive.google.com/uc?export=download&id=ID1",
      last_checked: "2026-11-24",
      size_bytes: 809883,
    },
    {
      number: 0,
      title: "Introductory System Overview",
      file: "0_overview.pptx",
      view_url: null,
      download_url: "https://github.com/kwyjad/pythia/raw/abc/docs/decks/0_overview.pptx",
      last_checked: "2026-11-24",
      size_bytes: 2260852,
    },
    { number: 2, title: "Bad link", view_url: "javascript:alert(1)", download_url: null },
  ],
};

describe("parseDecks", () => {
  it("orders decks and drops ones with no safe link", () => {
    const decks = parseDecks(JSON_FIXTURE);
    expect(decks.map((d) => d.number)).toEqual([0, 1]);
    expect(decks[0].viewUrl).toBeNull();
  });

  it("gives nothing for missing or malformed input", () => {
    expect(parseDecks(null)).toEqual([]);
    expect(parseDecks({})).toEqual([]);
    expect(parseDecks({ decks: "x" })).toEqual([]);
  });

  it("formats sizes", () => {
    expect(formatSize(809883)).toBe("810 KB");
    expect(formatSize(2260852)).toBe("2.3 MB");
    expect(formatSize(null)).toBeNull();
  });
});

describe("DecksPanel", () => {
  it("says the decks are not available yet when there are none", () => {
    render(<DecksPanel decks={[]} />);
    expect(screen.getByText("Slide decks")).toBeInTheDocument();
    expect(screen.getByTestId("decks-unavailable")).toHaveTextContent(
      "not available yet",
    );
    expect(screen.queryAllByRole("link")).toHaveLength(0);
  });

  it("renders a row per deck with view and download links", () => {
    render(<DecksPanel decks={parseDecks(JSON_FIXTURE)} />);
    const items = screen.getAllByRole("listitem");
    expect(items).toHaveLength(2);
    expect(items[0]).toHaveTextContent("Introductory System Overview");
    const view = screen.getByRole("link", { name: "View" });
    expect(view).toHaveAttribute("href", "https://drive.google.com/file/d/ID1/view?usp=sharing");
    expect(view).toHaveAttribute("target", "_blank");
    expect(screen.getAllByRole("link", { name: /Download/ })).toHaveLength(2);
    expect(screen.getByRole("link", { name: "Download (810 KB)" })).toHaveAttribute(
      "href",
      "https://drive.google.com/uc?export=download&id=ID1",
    );
    expect(screen.getAllByText("Checked 2026-11-24")).toHaveLength(2);
  });
});
