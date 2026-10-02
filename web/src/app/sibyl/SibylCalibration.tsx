"use client";

import { useEffect, useState } from "react";

import { apiGet } from "../../lib/api";
import type {
  SibylArmComparison,
  SibylCalibrationResponse,
  SibylCalibrationRow,
  SibylCalibrationStat,
} from "../../lib/types";

// Sibyl's own calibration record (sibyl/advice.py) and the advice its prompt
// carries. Read-only; fetched when the tab opens so the page load is unchanged.

const HAZARD_NAMES: Record<string, string> = {
  ACE: "Armed conflict",
  DR: "Drought",
  FL: "Flood",
  TC: "Tropical cyclone",
  "*": "All classes (pooled)",
};
const METRIC_NAMES: Record<string, string> = {
  FATALITIES: "deaths",
  PA: "people affected",
  PHASE3PLUS_IN_NEED: "people in IPC Phase 3+",
};

const pct = (v: number | null | undefined) =>
  v === null || v === undefined ? "—" : `${(v * 100).toFixed(0)}%`;
const num = (v: number | null | undefined, d = 2) =>
  v === null || v === undefined ? "—" : v.toFixed(d);

function Interval({
  stat,
  format,
}: {
  stat: SibylCalibrationStat | undefined;
  format: (v: number | null | undefined) => string;
}) {
  if (!stat || stat.value === null || stat.value === undefined) {
    return <span className="text-fred-muted">—</span>;
  }
  return (
    <span>
      {format(stat.value)}{" "}
      <span className="text-xs text-fred-muted">
        ({format(stat.lo)} to {format(stat.hi)})
      </span>
    </span>
  );
}

function classLabel(row: SibylCalibrationRow) {
  if (row.hazard_code === "*") return HAZARD_NAMES["*"];
  const hz = HAZARD_NAMES[row.hazard_code] ?? row.hazard_code;
  const m = METRIC_NAMES[row.metric] ?? row.metric;
  return `${hz}, ${m}`;
}

function ArmComparison({ arms }: { arms: SibylArmComparison | null }) {
  if (!arms) return null;
  const n = arms.n_questions ?? {};
  if (arms.status !== "ok") {
    return (
      <p className="text-sm text-fred-muted">
        Advice experiment: not yet. The comparison needs{" "}
        {arms.min_questions_per_arm} scored questions in each arm (advice{" "}
        {n.advice ?? 0}, no advice {n.no_advice ?? 0}).
      </p>
    );
  }
  return (
    <table className="min-w-full text-sm">
      <thead>
        <tr className="text-left text-xs text-fred-muted">
          <th className="px-3 py-2">Arm</th>
          <th className="px-3 py-2">Questions</th>
          <th className="px-3 py-2">Mean Brier (90% interval)</th>
          <th className="px-3 py-2">Mean CRPS (90% interval)</th>
        </tr>
      </thead>
      <tbody>
        {(["advice", "no_advice"] as const).map((arm) => (
          <tr key={arm} className="border-t border-fred-secondary/40">
            <td className="px-3 py-2">{arm === "advice" ? "Shown advice" : "No advice"}</td>
            <td className="px-3 py-2">{n[arm] ?? 0}</td>
            <td className="px-3 py-2">
              <Interval stat={arms.brier?.[arm]} format={(v) => num(v, 3)} />
            </td>
            <td className="px-3 py-2">
              <Interval stat={arms.crps?.[arm]} format={(v) => num(v, 3)} />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function SibylCalibration() {
  const [data, setData] = useState<SibylCalibrationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    apiGet<SibylCalibrationResponse>("/sibyl/calibration")
      .then((d) => live && setData(d))
      .catch((e) => {
        console.warn("Failed to load Sibyl calibration:", e);
        if (live) setError("Unable to load Sibyl's calibration record right now.");
      });
    return () => {
      live = false;
    };
  }, []);

  if (error) {
    return (
      <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm">
        {error}
      </div>
    );
  }
  if (!data) {
    return <div className="text-sm text-fred-muted">Loading calibration record…</div>;
  }
  if (!data.has_advice_table || !data.rows.length) {
    return (
      <div className="rounded-lg border border-fred-secondary bg-fred-surface px-4 py-6 text-sm text-fred-muted">
        No calibration record yet. It is written each month after scoring, once
        Sibyl has resolved forecasts.
      </div>
    );
  }
  const minQ = data.min_questions ?? 20;

  return (
    <div className="space-y-6">
      <p className="text-sm text-fred-text">
        How Sibyl&apos;s own resolved forecasts compared with what happened, measured
        on {data.as_of_month}. Each resolved question counts once, however many of
        its months have resolved; intervals are 90% and come from resampling whole
        questions. A class gets its own advice at {minQ} resolved questions; below
        that the pooled row stands in. A finding becomes advice only when its
        interval excludes the calibrated value.
      </p>

      <div className="overflow-x-auto rounded-lg border border-fred-secondary bg-fred-surface">
        <table className="min-w-full text-sm">
          <thead>
            <tr className="text-left text-xs text-fred-muted">
              <th className="px-3 py-2">Class</th>
              <th className="px-3 py-2">Resolved questions</th>
              <th className="px-3 py-2">Inside 10–90% range (target 80%)</th>
              <th className="px-3 py-2">Below q0.1 (target 10%)</th>
              <th className="px-3 py-2">Above q0.9 (target 10%)</th>
              <th className="px-3 py-2">Centre bias, log (target 0)</th>
            </tr>
          </thead>
          <tbody>
            {data.rows.map((row) => (
              <tr
                key={`${row.hazard_code}/${row.metric}`}
                className="border-t border-fred-secondary/40 align-top"
              >
                <td className="px-3 py-2 font-medium">{classLabel(row)}</td>
                <td className="px-3 py-2">
                  {row.n_questions} of {minQ}
                </td>
                <td className="px-3 py-2">
                  <Interval stat={row.diagnostics.coverage_10_90} format={pct} />
                </td>
                <td className="px-3 py-2">
                  <Interval stat={row.diagnostics.below_q10} format={pct} />
                </td>
                <td className="px-3 py-2">
                  <Interval stat={row.diagnostics.above_q90} format={pct} />
                </td>
                <td className="px-3 py-2">
                  <Interval stat={row.diagnostics.centre_bias_log} format={(v) => num(v, 2)} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Advice in force</h2>
        {data.rows.map((row) => (
          <div
            key={`advice-${row.hazard_code}/${row.metric}`}
            className="rounded-lg border border-fred-secondary bg-fred-surface p-4"
          >
            <div className="text-sm font-medium">{classLabel(row)}</div>
            {row.advice ? (
              <pre className="mt-2 whitespace-pre-wrap font-sans text-sm text-fred-text">
                {row.advice}
              </pre>
            ) : row.n_questions < minQ ? (
              <p className="mt-1 text-sm text-fred-muted">
                Not enough resolved questions yet ({row.n_questions} of {minQ}).
              </p>
            ) : (
              <p className="mt-1 text-sm text-fred-muted">
                No advice: {row.gate ?? "no finding clears its interval"}.
              </p>
            )}
          </div>
        ))}
      </section>

      <section className="space-y-2">
        <h2 className="text-lg font-semibold">Does the advice help?</h2>
        <p className="text-sm text-fred-muted">
          Half of Sibyl&apos;s questions are forecast without the advice, chosen by a
          hash of the question id, so the two arms can be compared on the same
          scores.
        </p>
        <ArmComparison arms={data.arm_comparison} />
      </section>
    </div>
  );
}
