// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
// See the LICENSE file in the project root for details.

/**
 * What a Detailed Scores row is, for the Performance page.
 *
 * A model swap leaves the new member unscored until its first forecasts
 * resolve, so the table can hold no row for an active model and many rows for
 * models the lineup has moved past. This module decides, from the API's
 * `active_models`, which rows are active, which are the immediate predecessor
 * of an active model ("superseded": their record still seeds the successor's
 * calibration prior), and which are retired; and which active models need a
 * placeholder row because nothing has been scored under their name.
 */

import type { ActiveModel } from "./types";
import { isIndividualForecaster } from "./performance_data";

export type RowStatus =
  | "active"
  | "superseded"
  | "retired"
  | "aggregate"
  | "reference"
  | "other";

export type StatusInfo = {
  status: RowStatus;
  /** For a superseded row: the stored model_name of the active successor. */
  supersededBy?: string;
};

export type StatusContext = {
  active: Set<string>;
  /** Immediate predecessor model_name -> active successor model_name. */
  supersededBy: Map<string, string>;
  knowsActive: boolean;
};

export function buildStatusContext(activeModels: ActiveModel[] | null | undefined): StatusContext {
  const list = activeModels ?? [];
  const active = new Set(list.map((m) => m.model_name));
  const supersededBy = new Map<string, string>();
  // Ensemble members come before Track 2 in the API list, so a model that is
  // the predecessor of both (gemini-3.5-flash) is labelled by the member.
  for (const m of list) {
    // Only the IMMEDIATE predecessor counts as superseded; anything older in
    // the family is retired. It is the immediate predecessor whose record the
    // calibration prior carries forward most directly.
    const prev = m.predecessors?.[0];
    if (!prev || active.has(prev) || supersededBy.has(prev)) continue;
    supersededBy.set(prev, m.model_name);
  }
  return { active, supersededBy, knowsActive: list.length > 0 };
}

const isEnsembleName = (key: string): boolean => key.toLowerCase().startsWith("ensemble");

export function statusOf(key: string, ctx: StatusContext): StatusInfo {
  if (ctx.active.has(key)) return { status: "active" };
  if (key.startsWith("__ext_")) return { status: "reference" };
  if (isEnsembleName(key) || key.startsWith("track2_") || key === "sibyl") {
    return { status: "aggregate" };
  }
  if (ctx.knowsActive && isIndividualForecaster(key)) {
    const succ = ctx.supersededBy.get(key);
    return succ ? { status: "superseded", supersededBy: succ } : { status: "retired" };
  }
  return { status: "other" };
}

/** Order the table pins rows in before the selected column sorts them. */
export function statusRank(status: RowStatus | undefined): number {
  switch (status) {
    case "active":
      return 0;
    case "aggregate":
      return 1;
    case "superseded":
      return 2;
    case "retired":
      return 4;
    default:
      return 3;
  }
}

/**
 * Active models with no row among `presentKeys`, under the track filter in
 * force: Track 1 shows only ensemble members, Track 2 only the Track 2 model.
 */
export function missingActiveModels(
  activeModels: ActiveModel[] | null | undefined,
  presentKeys: Iterable<string>,
  track: number | null,
): ActiveModel[] {
  const present = new Set(presentKeys);
  return (activeModels ?? []).filter((m) => {
    if (present.has(m.model_name)) return false;
    if (track === 1 && m.role !== "ensemble") return false;
    if (track === 2 && m.role !== "track2") return false;
    return true;
  });
}

/** The row a model with no scores yet gets: dashes and zero counts. */
export function placeholderScoreRow(model: ActiveModel, label: string) {
  return {
    key: model.model_name,
    label,
    avg_brier: null,
    median_brier: null,
    avg_brier_binary: null,
    median_brier_binary: null,
    avg_log: null,
    median_log: null,
    avg_crps: null,
    median_crps: null,
    n_questions: 0,
    n_samples: 0,
    n_questions_binary: 0,
    n_samples_binary: 0,
    status: "active" as RowStatus,
    skill: null,
    placeholder: true,
  };
}
