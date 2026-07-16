import { describe, expect, it } from "vitest";

import { PHASE_COLORS, PHASE_LABELS, PHASE_ORDER } from "../CostsClient";

// Guards against backend/frontend drift: every canonical phase emitted by
// resolver/query/costs.py::phase_group (CANONICAL_PHASES) must have a friendly
// label and a fixed color, or the Costs page renders a raw slug / falls back to
// the "other" color for a real phase.
const CANONICAL_PHASES = [
  "web_search",
  "hs",
  "research",
  "forecast",
  "scenario",
  "prediction_markets",
  "sibyl",
  "other",
];

describe("costs phase metadata", () => {
  it("PHASE_ORDER matches the backend CANONICAL_PHASES set", () => {
    expect([...PHASE_ORDER]).toEqual(CANONICAL_PHASES);
  });

  it("every phase has a label and a color", () => {
    for (const phase of PHASE_ORDER) {
      expect(PHASE_LABELS[phase]).toBeTruthy();
      expect(PHASE_COLORS[phase]).toMatch(/^#[0-9a-f]{6}$/i);
    }
  });

  it("colors are unique per phase (no accidental collisions)", () => {
    const colors = PHASE_ORDER.map((p) => PHASE_COLORS[p]);
    expect(new Set(colors).size).toBe(colors.length);
  });
});
