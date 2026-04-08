"use client";

import type { RunSummaryResponse, RunSummaryRcByHazard } from "../lib/types";
import InfoTooltip from "./InfoTooltip";
import KpiCard from "./KpiCard";

const HAZARD_LABELS: Record<string, string> = {
  ACE: "Armed Conflict",
  FL: "Flood",
  DR: "Drought",
  TC: "Tropical Cyclone",
  DI: "Displacement Inflow",
};

const RC_COLORS: Record<string, { bg: string; text: string; light: string }> = {
  L0: { bg: "bg-teal-600", text: "text-white", light: "bg-teal-50" },
  L1: { bg: "bg-amber-500", text: "text-white", light: "bg-amber-50" },
  L2: { bg: "bg-orange-500", text: "text-white", light: "bg-orange-50" },
  L3: { bg: "bg-red-600", text: "text-white", light: "bg-red-50" },
};

const RC_LEVEL_LABELS: Record<string, string> = {
  L0: "baseline",
  L1: "watch",
  L2: "elevated",
  L3: "critical",
};

type Props = {
  data: RunSummaryResponse;
};

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-fred-secondary bg-fred-surface p-5 shadow-fredCard">
      <h3 className="mb-4 text-sm font-semibold uppercase tracking-wide text-fred-muted">
        {title}
      </h3>
      {children}
    </div>
  );
}

// ---- 1. Run summary KPI row ----
function KpiRow({ data }: Props) {
  const errPct =
    data.llm_health.total_calls > 0
      ? ((data.llm_health.errors / data.llm_health.total_calls) * 100).toFixed(1)
      : "0.0";

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
      <KpiCard
        label="Total forecasts"
        value={data.coverage.total_questions.toLocaleString()}
      />
      <KpiCard
        label="Countries forecasted"
        value={
          <span>
            {data.coverage.countries_with_forecasts}
            <span className="ml-1 text-sm font-normal text-fred-muted">
              of {data.coverage.countries_scanned} scanned
            </span>
          </span>
        }
      />
      <KpiCard
        label={
          <span className="inline-flex items-center gap-1">
            Ensemble models
            <InfoTooltip text="Distinct LLM models that produced forecasts for Track 1 (full ensemble) questions. Expected count is from the ensemble configuration." />
          </span>
        }
        value={`${data.ensemble.ok} / ${data.ensemble.expected}`}
      />
      <KpiCard
        label="Run cost"
        value={
          <span>
            ${data.cost.total_usd.toFixed(2)}
            <span className="ml-1 text-sm font-normal text-fred-muted">
              {data.cost.total_tokens.toLocaleString()} tokens
            </span>
          </span>
        }
      />
      <KpiCard
        label="LLM errors"
        value={
          <span>
            {data.llm_health.errors}
            <span className="ml-1 text-sm font-normal text-fred-muted">
              {errPct}% of {data.llm_health.total_calls.toLocaleString()} calls
            </span>
          </span>
        }
      />
    </div>
  );
}

// ---- 2. Coverage funnel ----
function CoverageFunnel({ data }: Props) {
  const c = data.coverage;
  const steps = [
    { label: "countries scanned", value: c.countries_scanned, color: "bg-fred-primary" },
    { label: "country/hazard pairs assessed", value: c.hazard_pairs_assessed, color: "bg-fred-primary/80" },
    { label: "country/hazard pairs with questions", value: c.pairs_with_questions, color: "bg-fred-primary/60" },
    { label: "forecast questions", value: c.total_questions, color: "bg-fred-primary/40" },
  ];
  const maxVal = Math.max(...steps.map((s) => s.value), 1);

  return (
    <Section title="Coverage funnel">
      <div className="space-y-2">
        {steps.map((step, i) => (
          <div key={i} className="flex items-center gap-3">
            <div
              className={`${step.color} flex items-center rounded px-3 py-1.5 text-sm font-semibold text-white`}
              style={{ width: `${Math.max((step.value / maxVal) * 100, 20)}%` }}
            >
              {step.value.toLocaleString()} {step.label}
            </div>
            {i < steps.length - 1 && (
              <span className="text-fred-muted">&rarr;</span>
            )}
          </div>
        ))}
      </div>
      <div className="mt-3 space-y-1.5 text-xs text-fred-muted">
        <p>
          {c.seasonal_screenouts.toLocaleString()} country/hazard pairs seasonally screened out
          {" · "}
          {c.acled_low_activity.toLocaleString()} country/hazard pairs with quiet conflict (ACLED low activity)
          {" · "}
          {c.triaged_quiet.toLocaleString()} country/hazard pairs triaged quiet
          {" · "}
          {c.countries_no_questions.toLocaleString()} countries with no questions generated
        </p>
        <p>
          Each country/hazard pair can produce multiple forecast questions because
          different metrics apply to each hazard type. For example, an Armed
          Conflict (ACE) pair generates both a Fatalities and a People Affected
          question. Flood and Tropical Cyclone pairs each produce a People
          Affected question and an Event Occurrence question. Drought pairs
          produce a Phase 3+ Population question (for FEWS NET countries) or an
          Event Occurrence question.
        </p>
      </div>
    </Section>
  );
}

// ---- 3. Forecasts by metric grid ----
function MetricGrid({ data }: Props) {
  const METRIC_INFO: Record<string, string | undefined> = {
    PA: "Drought forecasts use the Phase 3+ population metric instead of people affected.",
  };

  return (
    <Section title="Forecasts by metric">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {data.metrics.map((m) => (
          <div
            key={m.metric}
            className="rounded-lg border border-fred-secondary/70 bg-fred-bg p-4"
          >
            <div className="flex items-center gap-1 text-xs uppercase tracking-wide text-fred-muted">
              {m.label}
              {METRIC_INFO[m.metric] && (
                <InfoTooltip text={METRIC_INFO[m.metric]!} />
              )}
            </div>
            <div className="mt-1 text-3xl font-bold text-fred-primary">
              {m.questions}
            </div>
            <div className="mt-0.5 text-xs text-fred-muted">
              {m.countries} {m.countries === 1 ? "country" : "countries"}
            </div>
            <div className="mt-2 flex flex-wrap gap-1">
              {m.hazards.map((h) => (
                <span
                  key={h.hazard_code}
                  className="inline-block rounded-full bg-fred-primary/10 px-2 py-0.5 text-xs font-medium text-fred-primary"
                >
                  {HAZARD_LABELS[h.hazard_code] ?? h.hazard_code} ({h.count})
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </Section>
  );
}

// ---- 4. RC assessment ----
function RcBar({ levels }: { levels: Record<string, number> }) {
  const total = Object.values(levels).reduce((a, b) => a + b, 0);
  if (total === 0) return null;

  return (
    <div className="flex h-8 w-full overflow-hidden rounded">
      {(["L0", "L1", "L2", "L3"] as const).map((key) => {
        const val = levels[key] ?? 0;
        if (val === 0) return null;
        const pct = (val / total) * 100;
        return (
          <div
            key={key}
            className={`${RC_COLORS[key].bg} ${RC_COLORS[key].text} flex items-center justify-center text-xs font-semibold`}
            style={{ width: `${pct}%`, minWidth: val > 0 ? "28px" : 0 }}
            title={`${key} ${RC_LEVEL_LABELS[key]}: ${val}`}
          >
            {pct > 5 ? val : ""}
          </div>
        );
      })}
    </div>
  );
}

function RcHazardTable({ rows }: { rows: RunSummaryRcByHazard[] }) {
  if (rows.length === 0) return null;

  return (
    <table className="mt-4 w-full text-sm">
      <thead>
        <tr className="border-b border-fred-border text-left text-xs uppercase tracking-wide text-fred-muted">
          <th className="pb-2 pr-4">Hazard</th>
          <th className="pb-2 px-2 text-right">L0</th>
          <th className="pb-2 px-2 text-right">L1</th>
          <th className="pb-2 px-2 text-right">L2</th>
          <th className="pb-2 px-2 text-right">L3</th>
          <th className="pb-2 pl-4 text-right">L1+ rate</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => {
          const rowTotal = row.L0 + row.L1 + row.L2 + row.L3;
          const l1PlusRate =
            rowTotal > 0
              ? (((row.L1 + row.L2 + row.L3) / rowTotal) * 100).toFixed(1)
              : "0.0";
          return (
            <tr key={row.hazard_code} className="border-b border-fred-border/50">
              <td className="py-1.5 pr-4 font-medium text-fred-text">
                {HAZARD_LABELS[row.hazard_code] ?? row.hazard_code} ({row.hazard_code})
              </td>
              {(["L0", "L1", "L2", "L3"] as const).map((key) => (
                <td
                  key={key}
                  className={`py-1.5 px-2 text-right ${row[key] > 0 ? RC_COLORS[key].light : ""}`}
                >
                  {row[key]}
                </td>
              ))}
              <td className="py-1.5 pl-4 text-right font-medium">{l1PlusRate}%</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function RcAssessment({ data }: Props) {
  const rc = data.rc_assessment;
  const l1Plus = rc.levels.L1 + rc.levels.L2 + rc.levels.L3;

  return (
    <Section title="Regime change assessment">
      <p className="mb-3 text-xs text-fred-muted">
        Regime Change (RC) measures how much a country/hazard pair is expected to
        deviate from its historical base rate. Each pair is scored on likelihood
        and magnitude to produce an RC level. L1+ pairs are promoted to Track 1
        (full LLM ensemble forecast); L0 pairs go to Track 2 (single model
        forecast). Higher RC levels indicate greater expected deviation from
        normal conditions.
      </p>
      <RcBar levels={rc.levels} />
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-fred-muted">
        {(["L0", "L1", "L2", "L3"] as const).map((key) => (
          <span key={key} className="inline-flex items-center gap-1">
            <span className={`inline-block h-2.5 w-2.5 rounded-sm ${RC_COLORS[key].bg}`} />
            {key} {RC_LEVEL_LABELS[key]} ({rc.levels[key]} pairs)
            {key !== "L0" && rc.countries_by_level[key] > 0 && (
              <span className="text-fred-muted">
                · {rc.countries_by_level[key]}{" "}
                <span className="inline-flex items-center gap-0.5">
                  countries
                  <InfoTooltip text={`Countries whose highest RC level across all hazards is ${key}.`} />
                </span>
              </span>
            )}
          </span>
        ))}
      </div>
      <p className="mt-2 text-xs text-fred-muted">
        {rc.total_assessed.toLocaleString()} country/hazard pairs assessed
        {" · "}
        {l1Plus} RC-promoted to Track 1
        {" · "}
        L1+ rate: {(rc.l1_plus_rate * 100).toFixed(1)}%
      </p>
      <RcHazardTable rows={rc.by_hazard} />
    </Section>
  );
}

// ---- 5. Track split ----
function TrackSplit({ data }: Props) {
  const t = data.tracks;

  return (
    <Section title="Track split">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div className="rounded-lg border border-fred-primary/30 bg-fred-primary/5 p-4">
          <div className="text-xs font-semibold uppercase tracking-wide text-fred-primary">
            Track 1 — full ensemble forecast
          </div>
          <div className="mt-1 text-2xl font-bold text-fred-primary">
            {t.track1.questions} questions
          </div>
          <div className="mt-0.5 text-xs text-fred-muted">
            {t.track1.models} models · {t.track1.countries} countries · scenarios generated
          </div>
        </div>
        <div className="rounded-lg border border-fred-secondary/30 bg-fred-secondary/5 p-4">
          <div className="text-xs font-semibold uppercase tracking-wide text-fred-secondary">
            Track 2 — single model forecast
          </div>
          <div className="mt-1 text-2xl font-bold text-fred-secondary">
            {t.track2.questions} questions
          </div>
          <div className="mt-0.5 text-xs text-fred-muted">
            Gemini Flash · {t.track2.countries} countries
          </div>
        </div>
      </div>
    </Section>
  );
}

// ---- 6. Cost breakdown ----
function CostBreakdown({ data }: Props) {
  return (
    <Section title="Cost breakdown">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {data.cost.by_phase.map((p) => (
          <div
            key={p.phase}
            className="rounded-lg border border-fred-secondary/70 bg-fred-bg p-3"
          >
            <div className="text-xs uppercase tracking-wide text-fred-muted">
              {p.label}
            </div>
            <div className="mt-1 text-xl font-bold text-fred-primary">
              ${p.cost_usd.toFixed(2)}
            </div>
          </div>
        ))}
      </div>
    </Section>
  );
}

// ---- Main component ----
export default function RunSummaryView({ data }: Props) {
  return (
    <div className="space-y-4">
      <KpiRow data={data} />
      <CoverageFunnel data={data} />
      <MetricGrid data={data} />
      <RcAssessment data={data} />
      <TrackSplit data={data} />
      <CostBreakdown data={data} />
    </div>
  );
}
