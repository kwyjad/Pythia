"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import SpdBarChart from "../../../components/SpdBarChart";
import type { QuestionBundleResponse } from "../../../lib/types";

type SpdPanelProps = {
  bundle: QuestionBundleResponse;
};

type SpdRow = Record<string, unknown>;

type SpdSource = {
  key: string;
  label: string;
  rows: SpdRow[];
};

const numberFormatter = new Intl.NumberFormat(undefined, {
  maximumFractionDigits: 0,
});
const percentFormatter = new Intl.NumberFormat(undefined, {
  style: "percent",
  minimumFractionDigits: 1,
  maximumFractionDigits: 2,
});

export const computeTotalEiv = (metric: string, monthlyEivs: number[]) => {
  const metricUpper = metric.toUpperCase();
  if (metricUpper === "FATALITIES") {
    return monthlyEivs.reduce((sum, value) => sum + value, 0);
  }
  return monthlyEivs.length ? Math.max(...monthlyEivs) : 0;
};

const addMonths = (ym: string | null | undefined, months: number): string | null => {
  if (!ym) return null;
  const [year, month] = ym.split("-").map(Number);
  if (!year || !month) return null;
  const total = year * 12 + (month - 1) + months;
  const nextYear = Math.floor(total / 12);
  const nextMonth = (total % 12) + 1;
  return `${nextYear}-${String(nextMonth).padStart(2, "0")}`;
};

const getSourceLabel = (key: string) => {
  const lowered = key.toLowerCase();
  if (lowered.includes("bayes")) {
    return "Ensemble (BayesMC)";
  }
  if (lowered.includes("mean")) {
    return "Ensemble (Mean)";
  }
  if (lowered === "ensemble") {
    return "Ensemble";
  }
  return `Model: ${key}`;
};

const collectSpdSources = (bundle: QuestionBundleResponse): SpdSource[] => {
  const forecast = (bundle.forecast ?? {}) as Record<string, unknown>;
  const ensembleRows = (forecast.ensemble_spd ?? []) as SpdRow[];
  const rawRows = (forecast.raw_spd ?? []) as SpdRow[];
  const map = new Map<string, SpdSource>();

  ensembleRows.forEach((row) => {
    const key =
      (row.model_name as string | undefined) ??
      (row.aggregator as string | undefined) ??
      "ensemble";
    const existing = map.get(key);
    if (existing) {
      existing.rows.push(row);
    } else {
      map.set(key, { key, label: getSourceLabel(key), rows: [row] });
    }
  });

  rawRows.forEach((row) => {
    const modelName =
      typeof row.model_name === "string" ? row.model_name.trim().toLowerCase() : "";
    if (modelName.startsWith("ensemble_")) {
      return;
    }
    const key =
      (row.model_name as string | undefined) ??
      (row.model as string | undefined) ??
      (row.model_id as string | undefined) ??
      "model";
    const existing = map.get(key);
    if (existing) {
      existing.rows.push(row);
    } else {
      map.set(key, { key, label: getSourceLabel(key), rows: [row] });
    }
  });

  return Array.from(map.values());
};

const resolveMonthIndex = (row: SpdRow): number | null => {
  const monthIndex = row.month_index ?? row.horizon_m;
  if (typeof monthIndex === "number" && Number.isFinite(monthIndex)) {
    return monthIndex;
  }
  if (typeof monthIndex === "string") {
    const parsed = Number(monthIndex);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
};

const buildProbVector = (
  rows: SpdRow[],
  monthIndex: number,
  labels: string[],
  sourceKey?: string
): number[] => {
  const probs = Array.from({ length: labels.length }, () => 0);
  const seen = Array.from({ length: labels.length }, () => false);
  const labelIndex = new Map(labels.map((label, index) => [label, index]));
  let duplicateFound = false;

  rows.forEach((row) => {
    const rowMonth = resolveMonthIndex(row);
    if (rowMonth !== monthIndex) return;

    const bucketIndex = row.bucket_index ?? row.class_bin;
    let index: number | null = null;

    if (typeof bucketIndex === "number" && Number.isFinite(bucketIndex)) {
      index = Math.round(bucketIndex) - 1;
    } else if (typeof bucketIndex === "string") {
      const trimmed = bucketIndex.trim();
      index = labelIndex.get(trimmed) ?? null;
    }

    if (index === null || index < 0 || index >= probs.length) return;

    if (seen[index]) {
      duplicateFound = true;
      return;
    }

    const probValue = row.probability ?? row.prob ?? row.p;
    let parsed: number | null = null;
    if (typeof probValue === "number" && Number.isFinite(probValue)) {
      parsed = probValue;
    } else if (typeof probValue === "string") {
      const numeric = Number.parseFloat(probValue);
      if (Number.isFinite(numeric)) {
        parsed = numeric;
      }
    }

    if (parsed !== null) {
      probs[index] = parsed;
      seen[index] = true;
    }
  });

  if (duplicateFound) {
    console.warn(
      `[SPD] Duplicate bucket rows deduped for source "${sourceKey ?? "unknown"}"`,
      { monthIndex }
    );
  }

  const sum = probs.reduce((total, value) => total + value, 0);
  if (sum > 1.01 || sum < 0.99) {
    console.warn(`[SPD] Probability sum out of range`, {
      source: sourceKey ?? "unknown",
      monthIndex,
      sum,
    });
  }

  return probs;
};

const SpdPanel = ({ bundle }: SpdPanelProps) => {
  const question = (bundle.question ?? {}) as Record<string, unknown>;
  const metric = (question.metric as string | undefined) ?? "";
  const targetMonth = (question.target_month as string | undefined) ?? null;
  const forecast = (bundle.forecast ?? {}) as Record<string, unknown>;

  const labels = useMemo(() => {
    const raw = forecast.bucket_labels;
    if (!Array.isArray(raw)) return [];
    return raw.filter((label): label is string => typeof label === "string");
  }, [forecast.bucket_labels]);

  const centroids = useMemo(() => {
    const raw = forecast.bucket_centroids;
    if (!Array.isArray(raw)) {
      return Array.from({ length: labels.length }, () => 0);
    }
    const parsed = raw
      .map((centroid) => (typeof centroid === "number" ? centroid : Number(centroid)))
      .map((centroid) => (Number.isFinite(centroid) ? centroid : 0));
    if (parsed.length < labels.length) {
      return [
        ...parsed,
        ...Array.from({ length: labels.length - parsed.length }, () => 0),
      ];
    }
    return parsed.slice(0, labels.length);
  }, [forecast.bucket_centroids, labels.length]);

  const sources = useMemo(() => collectSpdSources(bundle), [bundle]);

  const defaultSourceKey = useMemo(() => {
    if (!sources.length) return "";
    const bayesSource = sources.find((source) =>
      source.key.toLowerCase().includes("bayes")
    );
    return bayesSource?.key ?? sources[0].key;
  }, [sources]);

  const [selectedSourceKey, setSelectedSourceKey] = useState(defaultSourceKey);
  const [selectedMonthIndex, setSelectedMonthIndex] = useState(1);
  const [showTotalEiv, setShowTotalEiv] = useState(false);
  const headerCellRef = useRef<HTMLTableCellElement | null>(null);
  const headerLogSent = useRef(false);

  const debugEnabled = useMemo(() => {
    if (typeof window === "undefined") return false;
    const params = new URLSearchParams(window.location.search);
    return params.get("debug_question") === "1";
  }, []);

  useEffect(() => {
    setSelectedSourceKey(defaultSourceKey);
  }, [defaultSourceKey]);

  useEffect(() => {
    if (!debugEnabled || headerLogSent.current) return;
    const headerCell = headerCellRef.current;
    if (!headerCell) return;
    const styles = window.getComputedStyle(headerCell);
    console.log("[SPD header]", {
      color: styles.color,
      background: styles.backgroundColor,
    });
    headerLogSent.current = true;
  }, [debugEnabled]);

  const selectedSource = sources.find((source) => source.key === selectedSourceKey);

  const probs = useMemo(() => {
    if (!selectedSource) return Array.from({ length: labels.length }, () => 0);
    return buildProbVector(
      selectedSource.rows,
      selectedMonthIndex,
      labels,
      selectedSource.key
    );
  }, [labels, selectedMonthIndex, selectedSource]);

  const eivMonth = useMemo(() => {
    return probs.reduce((sum, prob, index) => sum + prob * (centroids[index] ?? 0), 0);
  }, [centroids, probs]);

  const probSum = useMemo(() => {
    return probs.reduce((sum, prob) => sum + prob, 0);
  }, [probs]);

  const eivTotal = useMemo(() => {
    if (!selectedSource) return 0;
    const monthlyEivs = [1, 2, 3, 4, 5, 6].map((monthIndex) => {
      const monthProbs = buildProbVector(
        selectedSource.rows,
        monthIndex,
        labels,
        selectedSource.key
      );
      return monthProbs.reduce(
        (subtotal, prob, index) => subtotal + prob * (centroids[index] ?? 0),
        0
      );
    });
    return computeTotalEiv(metric, monthlyEivs);
  }, [centroids, labels, metric, selectedSource]);

  const monthOptions = useMemo(() => {
    return [1, 2, 3, 4, 5, 6].map((monthIndex) => {
      const label = addMonths(targetMonth, monthIndex - 1);
      return {
        value: monthIndex,
        label: label ? `Month ${monthIndex} (${label})` : `Month ${monthIndex}`,
      };
    });
  }, [targetMonth]);

  if (!sources.length) {
    return (
      <section className="rounded-lg border border-fred-secondary bg-fred-surface p-4 text-fred-text">
        <h2 className="text-lg font-semibold">Forecast SPD</h2>
        <p className="mt-3 text-sm text-fred-text">No forecasts available.</p>
      </section>
    );
  }

  if (!labels.length) {
    return (
      <section className="rounded-lg border border-fred-secondary bg-fred-surface p-4 text-fred-text">
        <h2 className="text-lg font-semibold">Forecast SPD</h2>
        <p className="mt-3 text-sm text-fred-text">Bucket definitions unavailable.</p>
      </section>
    );
  }

  return (
    <section className="rounded-lg border border-fred-secondary bg-fred-surface p-4 text-fred-text">
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h2 className="text-lg font-semibold">Forecast SPD</h2>
          <p className="text-sm text-fred-text">
            Probability distribution by bucket for the selected forecast.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <label className="text-sm text-fred-text">
            Source
            <select
              className="ml-2 rounded border border-fred-secondary bg-fred-surface px-3 py-2 text-sm text-fred-text"
              value={selectedSourceKey}
              onChange={(event) => setSelectedSourceKey(event.target.value)}
            >
              {sources.map((source) => (
                <option key={source.key} value={source.key}>
                  {source.label}
                </option>
              ))}
            </select>
          </label>
          <label className="text-sm text-fred-text">
            Month
            <select
              className="ml-2 rounded border border-fred-secondary bg-fred-surface px-3 py-2 text-sm text-fred-text"
              value={selectedMonthIndex}
              onChange={(event) => setSelectedMonthIndex(Number(event.target.value))}
            >
              {monthOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2 text-sm text-fred-text">
            <input
              type="checkbox"
              className="h-4 w-4 rounded border-fred-secondary bg-fred-surface"
              checked={showTotalEiv}
              onChange={(event) => setShowTotalEiv(event.target.checked)}
            />
            {metric.toUpperCase() === "FATALITIES"
              ? "Show 6-Month cumulative expected deaths"
              : "Show 6-Month EIV (peak month)"}
          </label>
          <div className="text-sm text-fred-text">
            EIV: {numberFormatter.format(Math.round(eivMonth))}
            {showTotalEiv
              ? ` • ${
                  metric.toUpperCase() === "FATALITIES"
                    ? "Cumulative"
                    : "Peak Month"
                }: ${numberFormatter.format(Math.round(eivTotal))}`
              : ""}
          </div>
        </div>
      </div>

      <div className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="rounded border border-fred-secondary bg-fred-surface p-3">
          <SpdBarChart labels={labels} probs={probs} />
          {process.env.NODE_ENV !== "production" ? (
            <p className="mt-2 text-xs text-fred-text">
              SPD sum: {probSum.toFixed(3)}
            </p>
          ) : null}
        </div>

        <div className="rounded border border-fred-secondary bg-fred-surface p-3">
          <div className="overflow-x-auto">
            <table className="w-full table-auto text-left text-sm text-fred-text">
              <thead className="bg-fred-primary text-xs uppercase tracking-wide">
                <tr>
                  <th ref={headerCellRef} className="pb-2 pr-4 !text-white">
                    Bucket
                  </th>
                  <th className="pb-2 pr-4 !text-white">Centroid</th>
                  <th className="pb-2 pr-4 !text-white">Probability</th>
                  <th className="pb-2 !text-white">p × centroid</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-fred-secondary/20">
                {labels.map((label, index) => {
                  const prob = probs[index] ?? 0;
                  const centroid = centroids[index] ?? 0;
                  const contribution = prob * centroid;
                  return (
                    <tr key={label}>
                      <td className="py-2 pr-4 text-fred-text">{label}</td>
                      <td className="py-2 pr-4 text-fred-text">
                        {numberFormatter.format(centroid)}
                      </td>
                      <td className="py-2 pr-4 text-fred-text">
                        {percentFormatter.format(prob)}
                      </td>
                      <td className="py-2 text-fred-text">
                        {numberFormatter.format(Math.round(contribution))}
                      </td>
                    </tr>
                  );
                })}
                <tr className="font-semibold text-fred-secondary">
                  <td className="py-2 pr-4">EIV</td>
                  <td className="py-2 pr-4">—</td>
                  <td className="py-2 pr-4">—</td>
                  <td className="py-2">
                    {numberFormatter.format(Math.round(eivMonth))}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
  );
};

export default SpdPanel;
