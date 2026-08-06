// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
// See the LICENSE file in the project root for details.

import { describe, expect, it } from "vitest";

import { SCORE_GLOSSARY } from "../../../lib/score_glossary";
import {
  ATTENTION_BANDS,
  LEXICON_TABLE,
  REASON_LABELS,
  attentionColor,
  attentionLegend,
  countryAnchorId,
  statusBanner,
} from "../lib";

describe("attention scale", () => {
  it("maps values into the fixed bands, clamped to [0, 1]", () => {
    expect(attentionColor(0)).toBe("#ede9fe");
    expect(attentionColor(0.05)).toBe("#ede9fe");
    expect(attentionColor(0.1)).toBe("#c4b5fd");
    expect(attentionColor(0.3)).toBe("#8b5cf6");
    expect(attentionColor(0.5)).toBe("#6d28d9");
    expect(attentionColor(0.9)).toBe("#4c1d95");
    expect(attentionColor(-1)).toBe(attentionColor(0));
    expect(attentionColor(5)).toBe(attentionColor(1));
  });

  it("legend covers every band plus the no-data swatch, colors unique", () => {
    const legend = attentionLegend();
    expect(legend).toHaveLength(ATTENTION_BANDS.length + 1);
    const colors = legend.map((l) => l.color);
    expect(new Set(colors).size).toBe(colors.length);
  });
});

describe("reason labels", () => {
  it("covers exactly the schema's reason codes (interpreter/schema.py)", () => {
    expect(Object.keys(REASON_LABELS).sort()).toEqual([
      "base_rate_deviation",
      "large_impact_nominal",
      "large_impact_per_capita",
      "rc_deviation_disagreement",
    ]);
  });
});

describe("lexicon table", () => {
  it("mirrors interpreter/lexicon.py's fixed bands, highest first", () => {
    expect(LEXICON_TABLE.map((r) => r.word)).toEqual([
      "virtually certain",
      "very likely",
      "likely",
      "about as likely as not",
      "unlikely",
      "very unlikely",
      "exceptionally unlikely",
    ]);
  });
});

describe("status banner", () => {
  it("ok renders nothing; failures degrade visibly, never silently", () => {
    expect(statusBanner("ok")).toBeNull();
    expect(statusBanner("failed_validation")?.tone).toBe("warn");
    expect(statusBanner("failed_generation")?.tone).toBe("error");
    expect(statusBanner("weird")?.tone).toBe("warn");
  });
});

describe("glossary", () => {
  it("every entry has a term and non-trivial copy", () => {
    expect(SCORE_GLOSSARY.length).toBeGreaterThanOrEqual(8);
    for (const item of SCORE_GLOSSARY) {
      expect(item.term).toBeTruthy();
      expect(item.text.length).toBeGreaterThan(40);
    }
  });
});

describe("country anchors", () => {
  it("uppercase, stable ids", () => {
    expect(countryAnchorId("eth")).toBe("country-ETH");
  });
});
