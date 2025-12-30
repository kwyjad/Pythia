"use client";

import { useEffect, useMemo, useState } from "react";

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

const PA_BINS = ["<10k", "10k-<50k", "50k-<250k", "250k-<500k", ">=500k"];
const FATALITIES_BINS = ["<5", "5-<25", "25-<100", "100-<500", ">=500"];
const PA_CENTROIDS = [0, 30000, 150000, 375000, 700000];
const FATALITIES_CENTROIDS = [0, 15, 62, 300, 700];

const numberFormatter = new Intl.NumberFormat(undefined, {
  maximumFractionDigits: 0,
});
const percentFormatter = new Intl.NumberFormat(undefined, {
  style: "percent",
  minimumFractionDigits: 1,
  maximumFractionDigits: 2,
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

  const labels = metric === "FATALITIES" ? FATALITIES_BINS : PA_BINS;
  const centroids = metric === "FATALITIES" ? FATALITIES_CENTROIDS : PA_CENTROIDS;

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

  useEffect(() => {
    setSelectedSourceKey(defaultSourceKey);
  }, [defaultSourceKey]);

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
    return [1, 2, 3, 4, 5, 6].reduce((sum, monthIndex) => {
      const monthProbs = buildProbVector(
        selectedSource.rows,
        monthIndex,
        labels,
        selectedSource.key
      );
      const monthEiv = monthProbs.reduce(
        (subtotal, prob, index) => subtotal + prob * (centroids[index] ?? 0),
        0
      );
      return sum + monthEiv;
    }, 0);
  }, [centroids, labels, selectedSource]);

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
      <section className="rounded-lg border border-slate-800 bg-slate-900/60 p-4">
        <h2 className="text-lg font-semibold text-white">Forecast SPD</h2>
        <p className="mt-3 text-sm text-slate-400">No forecasts available.</p>
      </section>
    );
  }

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900/60 p-4">
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h2 className="text-lg font-semibold text-white">Forecast SPD</h2>
          <p className="text-sm text-slate-400">
            Probability distribution by bucket for the selected forecast.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <label className="text-sm text-slate-400">
            Source
            <select
              className="ml-2 rounded border border-slate-800 bg-slate-950 px-3 py-2 text-sm text-slate-200"
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
          <label className="text-sm text-slate-400">
            Month
            <select
              className="ml-2 rounded border border-slate-800 bg-slate-950 px-3 py-2 text-sm text-slate-200"
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
          <label className="flex items-center gap-2 text-sm text-slate-400">
            <input
              type="checkbox"
              className="h-4 w-4 rounded border-slate-700 bg-slate-950"
              checked={showTotalEiv}
              onChange={(event) => setShowTotalEiv(event.target.checked)}
            />
            Show total EIV
          </label>
          <div className="text-sm text-slate-200">
            EIV: {numberFormatter.format(Math.round(eivMonth))}
            {showTotalEiv
              ? ` • Total: ${numberFormatter.format(Math.round(eivTotal))}`
              : ""}
          </div>
        </div>
      </div>

      <div className="mt-6">
        <SpdBarChart labels={labels} probs={probs} />
        {process.env.NODE_ENV !== "production" ? (
          <p className="mt-2 text-xs text-slate-400">
            SPD sum: {probSum.toFixed(3)}
          </p>
        ) : null}
      </div>

      <div className="mt-6 overflow-x-auto">
        <table className="w-full table-auto text-left text-sm text-slate-200">
          <thead className="text-xs uppercase tracking-wide text-slate-400">
            <tr>
              <th className="pb-2 pr-4">Bucket</th>
              <th className="pb-2 pr-4">Centroid</th>
              <th className="pb-2 pr-4">Probability</th>
              <th className="pb-2">p × centroid</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {labels.map((label, index) => {
              const prob = probs[index] ?? 0;
              const centroid = centroids[index] ?? 0;
              const contribution = prob * centroid;
              return (
                <tr key={label}>
                  <td className="py-2 pr-4 text-slate-200">{label}</td>
                  <td className="py-2 pr-4 text-slate-200">
                    {numberFormatter.format(centroid)}
                  </td>
                  <td className="py-2 pr-4 text-slate-200">
                    {percentFormatter.format(prob)}
                  </td>
                  <td className="py-2 text-slate-200">
                    {numberFormatter.format(Math.round(contribution))}
                  </td>
                </tr>
              );
            })}
            <tr className="font-semibold text-white">
              <td className="py-2 pr-4">EIV</td>
              <td className="py-2 pr-4">—</td>
              <td className="py-2 pr-4">—</td>
              <td className="py-2">{numberFormatter.format(Math.round(eivMonth))}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  );
};

export default SpdPanel;
