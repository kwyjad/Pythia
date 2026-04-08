"use client";

import type { ForecastRun } from "../lib/types";

type RunSelectorProps = {
  availableRuns: ForecastRun[];
  selectedRunId: string | null;
  onChange: (runId: string) => void;
};

const formatRunLabel = (run: ForecastRun, index: number, total: number) => {
  const label = `Run ${total - index}`;
  const testSuffix = run.is_test ? " [test]" : "";
  if (run.started_at) {
    try {
      const date = new Date(run.started_at);
      const day = date.toLocaleDateString(undefined, {
        month: "short",
        day: "numeric",
      });
      return `${label} (${day})${testSuffix}`;
    } catch {
      return `${label}${testSuffix}`;
    }
  }
  return `${label}${testSuffix}`;
};

const RunSelector = ({
  availableRuns,
  selectedRunId,
  onChange,
}: RunSelectorProps) => {
  if (!availableRuns || availableRuns.length === 0) return null;

  return (
    <label className="flex items-center gap-2 text-sm text-fred-text">
      <span>Forecast run</span>
      <select
        className="rounded-md border border-fred-secondary bg-fred-surface px-3 py-2 text-sm text-fred-text focus:outline-none focus:ring-2 focus:ring-fred-primary/30"
        onChange={(event) => onChange(event.target.value)}
        value={selectedRunId ?? ""}
      >
        {availableRuns.map((run, index) => (
          <option key={run.run_id} value={run.run_id}>
            {formatRunLabel(run, index, availableRuns.length)}
          </option>
        ))}
      </select>
    </label>
  );
};

export default RunSelector;
