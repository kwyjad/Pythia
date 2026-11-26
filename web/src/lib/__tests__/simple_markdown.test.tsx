// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
// See the LICENSE file in the project root for details.

import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { extractHeadings, renderSimpleMarkdown } from "../simple_markdown";

const html = (md: string, opts?: { headingIds?: boolean }) =>
  render(<div>{renderSimpleMarkdown(md, opts)}</div>).container.innerHTML;

describe("renderSimpleMarkdown extensions", () => {
  it("renders italics with stars and word-boundary underscores", () => {
    const out = html("Read as *worsening* and _quiet_ here.");
    expect(out).toContain("<em>worsening</em>");
    expect(out).toContain("<em>quiet</em>");
  });

  it("leaves snake_case ids and bold alone", () => {
    const out = html("The id SOM_ACE_PA_2026-08 is **bold**.");
    expect(out).toContain("SOM_ACE_PA_2026-08");
    expect(out).not.toContain("<em>");
    expect(out).toContain("<strong>bold</strong>");
  });

  it("renders inline code without reading emphasis inside it", () => {
    const out = html("See `forecast_review_inputs.json` and `*not em*`.");
    expect(out).toContain("<code");
    expect(out).toContain("forecast_review_inputs.json</code>");
    expect(out).toContain("*not em*</code>");
  });

  it("nests indented list items under the item above", () => {
    const out = html("- **Brief 1.**\n  - Goal: x\n  - Read first: y\n- Brief 2\n\n1. one\n2. two");
    expect(out).toMatch(/<li[^>]*><strong>Brief 1\.<\/strong><ul[^>]*><li[^>]*>Goal: x<\/li><li[^>]*>Read first: y<\/li><\/ul><\/li><li[^>]*>Brief 2<\/li>/);
    expect(out).toMatch(/<ol[^>]*><li[^>]*>one<\/li><li[^>]*>two<\/li><\/ol>/);
  });

  it("gives h2 headings ids only when asked, matching extractHeadings", () => {
    const md = "# T\n\n## 1. Summary\n\ntext\n\n## 2. `Chain` health\n";
    expect(html(md)).not.toContain('id="sec-1"');
    const out = html(md, { headingIds: true });
    expect(out).toContain('id="sec-1"');
    expect(out).toContain('id="sec-2"');
    expect(extractHeadings(md)).toEqual([
      { id: "sec-1", text: "1. Summary" },
      { id: "sec-2", text: "2. Chain health" },
    ]);
  });

  it("keeps tables scrollable in their own container", () => {
    const out = html("| Field | Value |\n|---|---|\n| a | b |");
    expect(out).toContain('class="my-4 overflow-x-auto"');
    // A two-column separator has an inner pipe and is never a body row.
    expect(out).toContain("<thead>");
    expect(out).not.toContain("---");
  });
});
