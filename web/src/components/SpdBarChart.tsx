import React from "react";

type SpdBarChartProps = {
  labels: string[];
  probs: number[];
};

const percentFormatter = new Intl.NumberFormat(undefined, {
  style: "percent",
  minimumFractionDigits: 1,
  maximumFractionDigits: 2,
});

const SpdBarChart = ({ labels, probs }: SpdBarChartProps) => {
  const safeProbs = labels.map((_, index) => probs[index] ?? 0);
  const maxProb = Math.max(0, ...safeProbs);
  const maxLabel = maxProb > 0 ? maxProb.toFixed(2) : "0";

  return (
    <div className="space-y-2">
      <div className="rounded-lg border border-fred-secondary bg-fred-surface px-4 py-4">
        <div className="flex h-[440px] items-end gap-3">
          {labels.map((label, index) => {
            const prob = safeProbs[index] ?? 0;
            const height = maxProb > 0 ? (prob / maxProb) * 100 : 0;
            return (
              <div key={label} className="flex flex-1 flex-col items-center gap-2">
                <div className="flex h-[280px] w-full items-end">
                  <div
                    className="w-full rounded-md bg-fred-primary transition-all"
                    style={{ height: `${height}%` }}
                  />
                </div>
                <div className="text-xs text-fred-text">{label}</div>
                <div className="text-[11px] text-fred-secondary">
                  {percentFormatter.format(prob)}
                </div>
              </div>
            );
          })}
        </div>
      </div>
      <div className="flex justify-between text-xs text-fred-text">
        <span>0</span>
        <span>{maxLabel}</span>
      </div>
    </div>
  );
};

export default SpdBarChart;
