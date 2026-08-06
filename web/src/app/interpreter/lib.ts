// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
// See the LICENSE file in the project root for details.

// Pure helpers for the interpreter page (unit-tested in __tests__/lib.test.ts).

import type { InterpreterReasonCode } from "../../lib/types";

// Plain-word reason labels — mirrors interpreter/render.py's _REASON_LABELS.
export const REASON_LABELS: Record<InterpreterReasonCode, string> = {
  base_rate_deviation: "far from its base rate",
  large_impact_nominal: "large expected impact",
  large_impact_per_capita: "large expected impact per capita",
  rc_deviation_disagreement: "the system disagrees with itself here",
};

// Attention choropleth: a fixed sequential scale over the [0, 1] attention
// value (js_vs_baserate / ln 2). Fixed thresholds, not Jenks — the map is
// coloured by ATTENTION, deliberately not repeating the risk index's look.
export const ATTENTION_BANDS: Array<{ min: number; label: string; color: string }> = [
  { min: 0.7, label: "Extreme deviation (≥70%)", color: "#4c1d95" },
  { min: 0.45, label: "High (45–70%)", color: "#6d28d9" },
  { min: 0.25, label: "Elevated (25–45%)", color: "#8b5cf6" },
  { min: 0.1, label: "Mild (10–25%)", color: "#c4b5fd" },
  { min: 0, label: "At base rate (<10%)", color: "#ede9fe" },
];

export const attentionColor = (value: number): string => {
  const v = Math.max(0, Math.min(value, 1));
  for (const band of ATTENTION_BANDS) {
    if (v >= band.min) return band.color;
  }
  return ATTENTION_BANDS[ATTENTION_BANDS.length - 1].color;
};

export const attentionLegend = (): Array<{ label: string; color: string }> => [
  ...ATTENTION_BANDS.map((b) => ({ label: b.label, color: b.color })),
  { label: "No deviation data", color: "var(--risk-map-no-questions)" },
];

export const attentionLabel = (value: number): string =>
  `${Math.round(Math.max(0, Math.min(value, 1)) * 100)}% of maximum deviation`;

// The fixed probability lexicon — FIXED BY DESIGN (interpreter/lexicon.py);
// printed in the report appendix so the reader can check the writer.
export const LEXICON_TABLE: Array<{ word: string; band: string }> = [
  { word: "virtually certain", band: "≥ 99%" },
  { word: "very likely", band: "90–99%" },
  { word: "likely", band: "66–90%" },
  { word: "about as likely as not", band: "33–66%" },
  { word: "unlikely", band: "10–33%" },
  { word: "very unlikely", band: "1–10%" },
  { word: "exceptionally unlikely", band: "< 1%" },
];

export const countryAnchorId = (iso3: string): string =>
  `country-${(iso3 || "").toUpperCase()}`;

export const statusBanner = (
  status: string
): { tone: "ok" | "warn" | "error"; text: string } | null => {
  if (status === "ok") return null;
  if (status === "failed_validation") {
    return {
      tone: "warn",
      text:
        "This report FAILED automated validation (schema, referential, " +
        "numeric or prose checks) and is shown for inspection only — treat " +
        "its statements with caution.",
    };
  }
  if (status === "failed_generation") {
    return {
      tone: "error",
      text:
        "Report generation failed for this cycle — no readable content was " +
        "produced.",
    };
  }
  return { tone: "warn", text: `Unexpected report status: ${status}` };
};
