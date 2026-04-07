"use client";

import { useMemo } from "react";

import type { QuestionBundleResponse } from "../../../lib/types";

type BinaryPanelProps = {
  bundle: QuestionBundleResponse;
};

type ForecastRow = Record<string, unknown>;

const percentFormatter = new Intl.NumberFormat(undefined, {
  style: "percent",
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

const brierFormatter = new Intl.NumberFormat(undefined, {
  minimumFractionDigits: 3,
  maximumFractionDigits: 3,
});

const addMonths = (ym: string | null | undefined, months: number): string | null => {
  if (!ym) return null;
  const [year, month] = ym.split("-").map(Number);
  if (!year || !month) return null;
  const total = year * 12 + (month - 1) + months;
  const nextYear = Math.floor(total / 12);
  const nextMonth = (total % 12) + 1;
  return `${nextYear}-${String(nextMonth).padStart(2, "0")}`;
};

const formatMonth = (ym: string | null): string => {
  if (!ym) return "—";
  const [year, month] = ym.split("-").map(Number);
  if (!year || !month) return ym;
  const date = new Date(Date.UTC(year, month - 1, 1));
  return date.toLocaleString("en-US", {
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  });
};

/** Color based on calibration quality: how close P(event) matches outcome. */
const brierColor = (brier: number): string => {
  if (brier <= 0.1) return "text-green-600";
  if (brier <= 0.25) return "text-yellow-600";
  return "text-red-600";
};

export default function BinaryPanel({ bundle }: BinaryPanelProps) {
  const question = (bundle.question ?? {}) as Record<string, unknown>;
  const forecast = (bundle.forecast ?? {}) as Record<string, unknown>;
  const context = (bundle.context ?? {}) as Record<string, unknown>;
  const metric = ((question.metric as string) ?? "").toUpperCase();
  const isBinary = metric === "EVENT_OCCURRENCE";
  const targetMonth = (question.target_month as string | undefined) ?? null;
  const windowStart = question.window_start_date
    ? String(question.window_start_date).slice(0, 7)
    : addMonths(targetMonth, -5);
  const ensembleRows = useMemo(
    () => (forecast.ensemble_spd ?? []) as ForecastRow[],
    [forecast.ensemble_spd],
  );
  const resolutions = useMemo(
    () => (context.resolutions ?? []) as ForecastRow[],
    [context.resolutions],
  );
  const scores = useMemo(
    () => (context.scores ?? []) as ForecastRow[],
    [context.scores],
  );

  // Extract per-month probabilities from ensemble (bucket_1 = P(yes))
  const monthlyProbs = useMemo(() => {
    const map = new Map<number, number>();
    // Prefer ensemble model
    const preferred = ["ensemble_bayesmc_v2", "ensemble_mean_v2"];
    let chosen: ForecastRow[] = [];
    for (const pref of preferred) {
      const filtered = ensembleRows.filter(
        (row) => (row.model_name as string)?.toLowerCase() === pref
      );
      if (filtered.length > 0) {
        chosen = filtered;
        break;
      }
    }
    if (!chosen.length) chosen = ensembleRows;

    chosen.forEach((row) => {
      const horizon =
        (row.horizon_m as number) ??
        (row.month_index as number) ??
        (row.month as number);
      const bucketRaw = row.bucket_index ?? row.class_bin;
      // bucket_1 = P(yes)
      const isBucket1 =
        bucketRaw === 1 ||
        String(bucketRaw) === "1" ||
        String(bucketRaw) === "bucket_1" ||
        String(bucketRaw) === "yes";
      if (typeof horizon === "number" && isBucket1) {
        const prob = (row.probability as number) ?? (row.p as number);
        if (typeof prob === "number") {
          map.set(horizon, prob);
        }
      }
    });
    return map;
  }, [ensembleRows]);

  // Resolution outcomes by horizon
  const resolutionByHorizon = useMemo(() => {
    const map = new Map<number, { value: number; source?: string }>();
    resolutions.forEach((row) => {
      const horizon = row.horizon_m as number | undefined;
      const value = row.value as number | undefined;
      const source = row.source as string | undefined;
      if (typeof horizon === "number" && typeof value === "number") {
        map.set(horizon, { value, source });
      }
    });
    return map;
  }, [resolutions]);

  // Brier scores by horizon
  const brierByHorizon = useMemo(() => {
    const map = new Map<number, number>();
    scores.forEach((row) => {
      const scoreType = (row.score_type as string) ?? "";
      const modelName = row.model_name as string | undefined;
      const horizon = row.horizon_m as number | undefined;
      const value = row.value as number | undefined;
      if (
        scoreType === "brier" &&
        !modelName &&
        typeof horizon === "number" &&
        typeof value === "number"
      ) {
        map.set(horizon, value);
      }
    });
    return map;
  }, [scores]);

  if (!isBinary) return null;

  return (
    <div className="rounded-lg border border-fred-secondary bg-fred-surface p-4 text-fred-text">
      <h2 className="text-lg font-semibold">Binary Event Forecast</h2>
      <p className="mt-1 text-sm text-fred-muted">
        Binary questions scored with Brier score only.
      </p>

      <div className="mt-4 grid gap-3 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-6">
        {[1, 2, 3, 4, 5, 6].map((horizon) => {
          const prob = monthlyProbs.get(horizon);
          const resolution = resolutionByHorizon.get(horizon);
          const brier = brierByHorizon.get(horizon);
          const monthLabel = formatMonth(addMonths(windowStart, horizon - 1));

          return (
            <div
              key={`binary-m${horizon}`}
              className="rounded border border-fred-secondary bg-fred-surface p-3"
            >
              <div className="text-[11px] font-semibold uppercase tracking-wide text-fred-muted">
                {monthLabel}
              </div>

              {/* Probability bar */}
              <div className="mt-2">
                <div className="text-sm font-medium">
                  {prob !== undefined ? percentFormatter.format(prob) : "—"}
                </div>
                {prob !== undefined && (
                  <div className="mt-1 h-2 w-full overflow-hidden rounded bg-gray-200">
                    <div
                      className="h-full rounded bg-blue-500"
                      style={{ width: `${Math.min(prob * 100, 100)}%` }}
                    />
                  </div>
                )}
              </div>

              {/* Resolution */}
              <div className="mt-2 text-xs">
                {resolution !== undefined ? (
                  <span
                    className={
                      resolution.value >= 1
                        ? "font-semibold text-red-600"
                        : "text-green-600"
                    }
                  >
                    {resolution.value >= 1 ? "Event occurred" : "No event"}
                  </span>
                ) : (
                  <span className="text-fred-muted">Unresolved</span>
                )}
              </div>

              {/* Brier */}
              {brier !== undefined && (
                <div className={`mt-1 text-xs ${brierColor(brier)}`}>
                  Brier: {brierFormatter.format(brier)}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
