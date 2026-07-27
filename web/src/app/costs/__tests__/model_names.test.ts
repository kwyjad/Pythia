import { describe, expect, it } from "vitest";

import { formatModelName } from "../../../lib/model_names";

// Guards against backend/frontend drift: the grounding sentinels written by
// horizon_scanner/llm_logging.py::resolve_grounding_model_id are not models, and
// must render as an explicit failure rather than falling through formatModelName's
// `return raw` branch as a bare slug. Keep in sync with the GROUNDING_*_MODEL_ID
// constants in horizon_scanner/llm_logging.py.
const GROUNDING_SENTINELS = [
  "grounding-failed",
  "grounding-breaker-tripped",
  "grounding-unavailable",
];

describe("formatModelName", () => {
  it("labels every grounding sentinel as a grounding failure", () => {
    for (const sentinel of GROUNDING_SENTINELS) {
      const label = formatModelName(sentinel);
      expect(label).not.toBe(sentinel);
      expect(label).toMatch(/^Grounding /);
    }
  });

  it("distinguishes the sentinels from one another", () => {
    const labels = GROUNDING_SENTINELS.map(formatModelName);
    expect(new Set(labels).size).toBe(GROUNDING_SENTINELS.length);
  });

  it("prettifies the current lineup", () => {
    expect(formatModelName("gpt-5.6-sol")).toBe("GPT-5.6 Sol");
    expect(formatModelName("gpt-5.6-luna")).toBe("GPT-5.6 Luna");
    expect(formatModelName("claude-opus-5")).toBe("Claude Opus 5");
    expect(formatModelName("gemini-3.1-pro-preview")).toBe("Gemini 3.1 Pro");
    expect(formatModelName("gemini-3.5-flash")).toBe("Gemini 3.5 Flash");
  });

  it("still prettifies superseded models and aggregates", () => {
    // Historical forecasts_raw / scores rows keep their original model_name, so
    // these must keep rendering after a lineup swap.
    expect(formatModelName("gpt-5.4")).toBe("GPT-5.4");
    expect(formatModelName("gpt-5.4-mini")).toBe("GPT-5.4 Mini");
    expect(formatModelName("claude-opus-4-8")).toBe("Claude Opus 4.8");
    expect(formatModelName("sibyl")).toBe("Sibyl (deep research)");
  });

  it("leaves genuinely unmapped values untouched", () => {
    // "unknown" still reaches the frontend for rows where the DB has no model at
    // all (resolver/query/costs.py fillna) — that case is legitimately unknown.
    expect(formatModelName("unknown")).toBe("unknown");
    expect(formatModelName("brave-web-search")).toBe("brave-web-search");
  });
});
