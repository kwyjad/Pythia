import { describe, expect, it } from "vitest";

import {
  buildStatusContext,
  missingActiveModels,
  placeholderScoreRow,
  statusOf,
  statusRank,
} from "../model_status";
import type { ActiveModel } from "../types";

const ACTIVE: ActiveModel[] = [
  { model_id: "gpt-6-sol", model_name: "gpt-6-sol", role: "ensemble", family: "gpt", predecessors: ["gpt-5.6-sol", "gpt-5.4"] },
  { model_id: "claude-opus-5-5", model_name: "claude-opus-5-5", role: "ensemble", family: "claude", predecessors: ["claude-opus-5", "claude-opus-4-8"] },
  { model_id: "gemini-3.1-pro-preview", model_name: "gemini-3.1-pro-preview", role: "ensemble", family: "gemini_pro", predecessors: ["gemini-3-pro-preview"] },
  { model_id: "gemini-3.6-flash", model_name: "gemini-3.6-flash", role: "ensemble", family: "gemini_flash", predecessors: ["gemini-3.5-flash", "gemini-3-flash-preview"] },
  { model_id: "gpt-6-luna", model_name: "gpt-6-luna", role: "ensemble", family: "gpt_mini", predecessors: ["gpt-5.6-luna", "gpt-5.4-mini"] },
  { model_id: "gemini-3.6-flash", model_name: "track2_flash", role: "track2", family: "gemini_flash", predecessors: ["gemini-3.5-flash", "gemini-3-flash-preview"] },
];

// What the live summary_rows held on 26 Nov 2026.
const SCORED = [
  "gpt-5.4", "gpt-5.4-mini", "gpt-5.6-sol", "gpt-5.6-luna", "claude-opus-4-8",
  "claude-opus-5", "gemini-3.5-flash", "gemini-3.1-pro-preview", "track2_flash",
  "ensemble_mean_v2", "ensemble_bayesmc_v2", "sibyl", "__ext_climatology",
];

describe("model status", () => {
  const ctx = buildStatusContext(ACTIVE);

  it("gives every active model with no scores a placeholder row", () => {
    const missing = missingActiveModels(ACTIVE, SCORED, null).map((m) => m.model_name);
    expect(missing).toEqual(["gpt-6-sol", "claude-opus-5-5", "gemini-3.6-flash", "gpt-6-luna"]);
    const row = placeholderScoreRow(ACTIVE[1], "Claude Opus 5.5");
    expect(row.status).toBe("active");
    expect(row.placeholder).toBe(true);
    expect(row.n_samples).toBe(0);
    expect(row.avg_brier).toBeNull();
  });

  it("follows the track filter", () => {
    expect(missingActiveModels(ACTIVE, [], 2).map((m) => m.model_name)).toEqual(["track2_flash"]);
    expect(missingActiveModels(ACTIVE, [], 1)).toHaveLength(5);
  });

  it("marks the immediate predecessor superseded and older versions retired", () => {
    expect(statusOf("claude-opus-5", ctx)).toEqual({ status: "superseded", supersededBy: "claude-opus-5-5" });
    expect(statusOf("gpt-5.6-sol", ctx).status).toBe("superseded");
    expect(statusOf("claude-opus-4-8", ctx).status).toBe("retired");
    expect(statusOf("gpt-5.4", ctx).status).toBe("retired");
    // The predecessor of both the ensemble member and Track 2 is labelled by the member.
    expect(statusOf("gemini-3.5-flash", ctx).supersededBy).toBe("gemini-3.6-flash");
  });

  it("classifies active, reference and aggregate rows", () => {
    expect(statusOf("gemini-3.1-pro-preview", ctx).status).toBe("active");
    expect(statusOf("track2_flash", ctx).status).toBe("active");
    expect(statusOf("__ext_climatology", ctx).status).toBe("reference");
    expect(statusOf("ensemble_mean_v2", ctx).status).toBe("aggregate");
    expect(statusOf("ensemble_bayesmc_v2", ctx).status).toBe("aggregate");
    expect(statusOf("sibyl", ctx).status).toBe("aggregate");
  });

  it("leaves individual models as other when the lineup is unknown", () => {
    const empty = buildStatusContext([]);
    expect(statusOf("claude-opus-5", empty).status).toBe("other");
    expect(statusOf("gpt-5.4", empty).status).toBe("other");
    expect(missingActiveModels([], SCORED, null)).toEqual([]);
  });

  it("pins superseded rows after the aggregates and retired rows last", () => {
    expect(statusRank("active")).toBeLessThan(statusRank("aggregate"));
    expect(statusRank("aggregate")).toBeLessThan(statusRank("superseded"));
    expect(statusRank("superseded")).toBeLessThan(statusRank("reference"));
    expect(statusRank("reference")).toBeLessThan(statusRank("retired"));
  });

  it("tolerates an older API with no predecessors", () => {
    const old = buildStatusContext([{ model_id: "x", model_name: "x", role: "ensemble" }]);
    expect(statusOf("claude-opus-5", old).status).toBe("retired");
  });
});
