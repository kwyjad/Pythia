"use client";

import { useMemo, useState } from "react";

import InfoTooltip from "../../components/InfoTooltip";
import KpiCard from "../../components/KpiCard";
import { apiGet } from "../../lib/api";
import { formatModelName } from "../../lib/model_names";
import { CHART_COLORS, CardHeader, ChartFrame, ChartLegend } from "./chart_kit";
import type {
  SibylComparisonPair,
  SibylComparisonResponse,
  SibylVariantComparison,
  SibylComparisonStat,
} from "../../lib/types";

// Colours come from the shared chart kit (validated palette, light surface).
// Green = Sibyl better, red = Sibyl worse (all three scores are lower-is-better).
const SIBYL_COLOR = CHART_COLORS.sibyl;
const BASELINE_COLOR = CHART_COLORS.ensemble;
const BETTER = CHART_COLORS.better;
const WORSE = CHART_COLORS.worse;
const AXIS = CHART_COLORS.axis;
const GRID = CHART_COLORS.grid;

type Props = {
  data: SibylComparisonResponse;
  includeTest?: boolean;
};

// Per-question, Brier-only aggregation (mean across horizons) used by every chart.
type QuestionPoint = {
  question_id: string;
  iso3: string | null;
  hazard_code: string;
  metric: string;
  sibyl: number;
  standard: number;
  delta: number; // sibyl - standard; negative => Sibyl better
  jsd: number | null;
  volatility: number | null;
  cost: number | null;
};

const fmt4 = (v: number | null | undefined) =>
  v == null || !Number.isFinite(v) ? "—" : v.toFixed(4);
const fmt2 = (v: number | null | undefined) =>
  v == null || !Number.isFinite(v) ? "—" : v.toFixed(2);
const fmtPct = (v: number | null | undefined) =>
  v == null || !Number.isFinite(v) ? "—" : `${(v * 100).toFixed(0)}%`;
const fmtUsd = (v: number | null | undefined) =>
  v == null || !Number.isFinite(v) ? "—" : `$${v.toFixed(2)}`;

const mean = (xs: number[]) =>
  xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN;

function buildQuestionPoints(pairs: SibylComparisonPair[]): QuestionPoint[] {
  const byQ = new Map<string, SibylComparisonPair[]>();
  for (const p of pairs) {
    if (p.score_type !== "brier") continue;
    if (p.sibyl_value == null || p.standard_value == null) continue;
    const arr = byQ.get(p.question_id) ?? [];
    arr.push(p);
    byQ.set(p.question_id, arr);
  }
  const out: QuestionPoint[] = [];
  for (const [qid, rows] of byQ) {
    const sibyl = mean(rows.map((r) => r.sibyl_value as number));
    const standard = mean(rows.map((r) => r.standard_value as number));
    const first = rows[0];
    out.push({
      question_id: qid,
      iso3: first.iso3,
      hazard_code: first.hazard_code,
      metric: first.metric,
      sibyl,
      standard,
      delta: sibyl - standard,
      jsd: first.js_divergence_vs_standard,
      volatility: first.volatility_score,
      cost: first.cost_usd,
    });
  }
  return out.sort((a, b) => a.delta - b.delta);
}

const deltaColor = (d: number | null | undefined) =>
  d == null ? "text-fred-muted" : d < 0 ? "text-emerald-700" : d > 0 ? "text-red-700" : "text-fred-muted";

// ---------------------------------------------------------------------------
// Chart 1 — Skill scatter: Sibyl Brier (y) vs Baseline Brier (x), 45° diagonal.
// Below the line = Sibyl better (green). Dot area ∝ volatility.
// ---------------------------------------------------------------------------
function SkillScatter({ points }: { points: QuestionPoint[] }) {
  const W = 340;
  const H = 300;
  const M = { t: 12, r: 12, b: 40, l: 44 };
  const iw = W - M.l - M.r;
  const ih = H - M.t - M.b;
  const maxV = Math.max(
    0.1,
    ...points.map((p) => Math.max(p.sibyl, p.standard)),
  );
  const dom = Math.ceil(maxV * 10) / 10;
  const sx = (v: number) => M.l + (v / dom) * iw;
  const sy = (v: number) => M.t + ih - (v / dom) * ih;
  const ticks = [0, dom / 2, dom].map((t) => Number(t.toFixed(2)));
  const volMax = Math.max(1, ...points.map((p) => p.volatility ?? 0));

  return (
    <ChartFrame width={W} height={H}
         label={`Scatter of ${points.length} questions: Sibyl Brier against baseline Brier. Points below the diagonal are questions Sibyl scored better.`}>
      {/* grid + ticks */}
      {ticks.map((t) => (
        <g key={`gx-${t}`}>
          <line x1={sx(t)} y1={M.t} x2={sx(t)} y2={M.t + ih} stroke={GRID} strokeWidth={1} />
          <text x={sx(t)} y={M.t + ih + 14} textAnchor="middle" fontSize={9} fill="#8a8a82">{t}</text>
        </g>
      ))}
      {ticks.map((t) => (
        <g key={`gy-${t}`}>
          <line x1={M.l} y1={sy(t)} x2={M.l + iw} y2={sy(t)} stroke={GRID} strokeWidth={1} />
          <text x={M.l - 6} y={sy(t) + 3} textAnchor="end" fontSize={9} fill="#8a8a82">{t}</text>
        </g>
      ))}
      {/* 45° diagonal — points below it are wins */}
      <line x1={sx(0)} y1={sy(0)} x2={sx(dom)} y2={sy(dom)} stroke={AXIS} strokeWidth={1.5} strokeDasharray="4 3" />
      {/* points */}
      {points.map((p) => {
        const r = 4 + 5 * Math.sqrt((p.volatility ?? 0) / volMax);
        const fill = p.delta < 0 ? BETTER : p.delta > 0 ? WORSE : "#8a8a82";
        return (
          <circle key={p.question_id} cx={sx(p.standard)} cy={sy(p.sibyl)} r={r}
                  fill={fill} fillOpacity={0.55} stroke={fill} strokeWidth={1}>
            <title>{`${p.question_id}\nSibyl ${fmt4(p.sibyl)} vs Baseline ${fmt4(p.standard)}\nΔ ${fmt4(p.delta)} (${p.delta < 0 ? "Sibyl better" : p.delta > 0 ? "Sibyl worse" : "tie"})`}</title>
          </circle>
        );
      })}
      <text x={M.l + iw / 2} y={H - 4} textAnchor="middle" fontSize={10} fill="#6a6a62">Baseline Brier →</text>
      <text x={12} y={M.t + ih / 2} textAnchor="middle" fontSize={10} fill="#6a6a62"
            transform={`rotate(-90 12 ${M.t + ih / 2})`}>Sibyl Brier →</text>
    </ChartFrame>
  );
}

// ---------------------------------------------------------------------------
// Chart 2 — Per-question dumbbell, sorted by Δ Brier so the biggest wins sit at
// the top and the biggest losses at the bottom. Baseline vs Sibyl dot joined by
// a segment coloured by the winner. The rows scroll inside a capped box; the
// axis stays pinned above them.
// ---------------------------------------------------------------------------
const DUMBBELL_MAX_HEIGHT = 420;

function Dumbbell({ points }: { points: QuestionPoint[] }) {
  const rowH = 22;
  const W = 400;
  const M = { r: 14, l: 96 };
  const iw = W - M.l - M.r;
  const dom = Math.ceil(Math.max(0.1, ...points.map((p) => Math.max(p.sibyl, p.standard))) * 10) / 10;
  const sx = (v: number) => M.l + (v / dom) * iw;
  const ticks = [0, dom / 2, dom].map((t) => Number(t.toFixed(2)));
  const H = points.length * rowH + 8;
  const sorted = [...points].sort((a, b) => a.delta - b.delta);

  return (
    <div className="mx-auto" style={{ maxWidth: 560 }}>
      <div
        className="overflow-y-auto rounded border border-fred-border"
        style={{ maxHeight: DUMBBELL_MAX_HEIGHT }}
        tabIndex={0}
        aria-label="Per-question comparison, scrollable"
      >
        <div className="sticky top-0 z-10 bg-fred-surface">
          <svg viewBox={`0 0 ${W} 22`} preserveAspectRatio="xMidYMid meet" className="block h-auto w-full"
               aria-hidden="true">
            {ticks.map((t) => (
              <text key={t} x={sx(t)} y={14} textAnchor="middle" fontSize={9} fill="#6a6a62">{t}</text>
            ))}
            <text x={M.l - 6} y={14} textAnchor="end" fontSize={9} fill="#6a6a62">Brier →</text>
            <line x1={M.l} x2={M.l + iw} y1={21} y2={21} stroke={AXIS} strokeWidth={1} />
          </svg>
        </div>
        <ChartFrame width={W} height={H} maxHeight={null}
                    label={`Per-question Brier for ${sorted.length} questions, baseline against Sibyl, sorted from Sibyl's largest win to its largest loss.`}>
          {ticks.map((t) => (
            <line key={t} x1={sx(t)} y1={0} x2={sx(t)} y2={H} stroke={GRID} strokeWidth={1} />
          ))}
          {sorted.map((p, i) => {
            const y = 4 + i * rowH + rowH / 2;
            const win = p.delta < 0 ? BETTER : p.delta > 0 ? WORSE : "#8a8a82";
            return (
              <g key={p.question_id}>
                <text x={M.l - 6} y={y + 3} textAnchor="end" fontSize={9} fill="#4a4a44">
                  {p.iso3 ?? p.question_id.slice(0, 6)} {p.hazard_code}
                </text>
                <line x1={sx(p.standard)} y1={y} x2={sx(p.sibyl)} y2={y} stroke={win} strokeWidth={2.5} />
                <circle cx={sx(p.standard)} cy={y} r={4.5} fill={BASELINE_COLOR}>
                  <title>{`${p.question_id} — Baseline ${fmt4(p.standard)}`}</title>
                </circle>
                <circle cx={sx(p.sibyl)} cy={y} r={4.5} fill={SIBYL_COLOR}>
                  <title>{`${p.question_id} — Sibyl ${fmt4(p.sibyl)} (Δ ${fmt4(p.delta)})`}</title>
                </circle>
              </g>
            );
          })}
        </ChartFrame>
      </div>
      <p className="mt-1 text-[11px] text-fred-muted">
        Top: Sibyl&apos;s largest wins. Bottom: its largest losses. {sorted.length} questions; scroll for more.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Chart 3 — Divergence vs skill: does diverging from the ensemble help?
// x = JS divergence vs standard, y = Δ Brier (sibyl - baseline). y=0 and
// x-median quadrant lines. Points below y=0 = Sibyl won.
// ---------------------------------------------------------------------------
function DivergenceSkill({ points }: { points: QuestionPoint[] }) {
  const pts = points.filter((p) => p.jsd != null && Number.isFinite(p.jsd));
  const W = 340;
  const H = 260;
  const M = { t: 14, r: 14, b: 38, l: 48 };
  const iw = W - M.l - M.r;
  const ih = H - M.t - M.b;
  if (!pts.length) {
    return <p className="p-4 text-xs text-fred-muted">No divergence data for the covered questions yet.</p>;
  }
  const xMax = Math.max(0.05, ...pts.map((p) => p.jsd as number));
  const dMax = Math.max(0.02, ...pts.map((p) => Math.abs(p.delta)));
  const sx = (v: number) => M.l + (v / xMax) * iw;
  const sy = (d: number) => M.t + ih / 2 - (d / dMax) * (ih / 2);
  const sorted = [...pts].map((p) => p.jsd as number).sort((a, b) => a - b);
  const xMed = sorted[Math.floor(sorted.length / 2)];

  return (
    <ChartFrame width={W} height={H}
         label="Scatter of Sibyl's divergence from the standard track against its Brier difference; points below the zero line are Sibyl wins.">
      {/* y=0 line (equal skill) */}
      <line x1={M.l} y1={sy(0)} x2={M.l + iw} y2={sy(0)} stroke={AXIS} strokeWidth={1.5} />
      {/* x median */}
      <line x1={sx(xMed)} y1={M.t} x2={sx(xMed)} y2={M.t + ih} stroke={GRID} strokeWidth={1} strokeDasharray="3 3" />
      <text x={M.l + 2} y={sy(0) - 4} fontSize={8} fill={BETTER}>Sibyl better ↓</text>
      {pts.map((p) => {
        const fill = p.delta < 0 ? BETTER : p.delta > 0 ? WORSE : "#8a8a82";
        return (
          <circle key={p.question_id} cx={sx(p.jsd as number)} cy={sy(p.delta)} r={5}
                  fill={fill} fillOpacity={0.55} stroke={fill} strokeWidth={1}>
            <title>{`${p.question_id}\nJSD ${fmt4(p.jsd)}  ΔBrier ${fmt4(p.delta)}`}</title>
          </circle>
        );
      })}
      <text x={M.l + iw / 2} y={H - 4} textAnchor="middle" fontSize={10} fill="#6a6a62">JS divergence vs standard →</text>
      <text x={12} y={M.t + ih / 2} textAnchor="middle" fontSize={10} fill="#6a6a62"
            transform={`rotate(-90 12 ${M.t + ih / 2})`}>Δ Brier (Sibyl − Base)</text>
    </ChartFrame>
  );
}

// ---------------------------------------------------------------------------
// Chart 4 — Cost-effectiveness: cost/question (x) vs Δ Brier (y).
// ---------------------------------------------------------------------------
function CostEffect({ points }: { points: QuestionPoint[] }) {
  const pts = points.filter((p) => p.cost != null && Number.isFinite(p.cost));
  const W = 340;
  const H = 260;
  const M = { t: 14, r: 14, b: 38, l: 48 };
  const iw = W - M.l - M.r;
  const ih = H - M.t - M.b;
  if (!pts.length) {
    return <p className="p-4 text-xs text-fred-muted">No per-question cost data for the covered questions yet.</p>;
  }
  const xMax = Math.max(0.5, ...pts.map((p) => p.cost as number));
  const dMax = Math.max(0.02, ...pts.map((p) => Math.abs(p.delta)));
  const sx = (v: number) => M.l + (v / xMax) * iw;
  const sy = (d: number) => M.t + ih / 2 - (d / dMax) * (ih / 2);

  return (
    <ChartFrame width={W} height={H}
         label="Scatter of Sibyl's research cost per question against its Brier difference; points below the zero line are Sibyl wins.">
      <line x1={M.l} y1={sy(0)} x2={M.l + iw} y2={sy(0)} stroke={AXIS} strokeWidth={1.5} />
      {pts.map((p) => {
        const fill = p.delta < 0 ? BETTER : p.delta > 0 ? WORSE : "#8a8a82";
        return (
          <circle key={p.question_id} cx={sx(p.cost as number)} cy={sy(p.delta)} r={5}
                  fill={fill} fillOpacity={0.55} stroke={fill} strokeWidth={1}>
            <title>{`${p.question_id}\nCost ${fmtUsd(p.cost)}  ΔBrier ${fmt4(p.delta)}`}</title>
          </circle>
        );
      })}
      <text x={M.l + iw / 2} y={H - 4} textAnchor="middle" fontSize={10} fill="#6a6a62">Sibyl cost / question ($) →</text>
      <text x={12} y={M.t + ih / 2} textAnchor="middle" fontSize={10} fill="#6a6a62"
            transform={`rotate(-90 12 ${M.t + ih / 2})`}>Δ Brier (Sibyl − Base)</text>
    </ChartFrame>
  );
}

// ---------------------------------------------------------------------------
// Chart 5 — Score-type summary bars: Sibyl vs Baseline mean per Brier/Log/CRPS.
// ---------------------------------------------------------------------------
function SummaryBars({ spd }: { spd: Record<string, SibylComparisonStat> }) {
  const order = ["brier", "log", "crps"].filter((k) => spd[k]);
  if (!order.length) return null;
  const label =
    "Mean score for Sibyl and the baseline on each scoring rule: " +
    order
      .map((st) => `${st} Sibyl ${fmt4(spd[st].sibyl_mean)}, baseline ${fmt4(spd[st].standard_mean)}`)
      .join("; ") +
    ". Lower is better.";
  return (
    <div className="mx-auto max-w-[560px] space-y-3" role="img" aria-label={label}>
      {order.map((st) => {
        const s = spd[st];
        const maxV = Math.max(s.sibyl_mean ?? 0, s.standard_mean ?? 0, 0.001);
        const wPct = (v: number | null) => `${((v ?? 0) / maxV) * 100}%`;
        return (
          <div key={st}>
            <div className="mb-1 flex items-center justify-between text-xs">
              <span className="font-medium uppercase tracking-wide text-fred-muted">{st}</span>
              <span className={deltaColor(s.mean_delta)}>
                Δ {fmt4(s.mean_delta)} · win {fmtPct(s.win_rate)}
              </span>
            </div>
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="w-16 text-[10px] text-fred-muted">Sibyl</span>
                <div className="h-3 flex-1 rounded bg-fred-secondary/30">
                  <div className="h-3 rounded" style={{ width: wPct(s.sibyl_mean), backgroundColor: SIBYL_COLOR }} />
                </div>
                <span className="w-14 text-right text-[10px] tabular-nums text-fred-text">{fmt4(s.sibyl_mean)}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-16 text-[10px] text-fred-muted">Baseline</span>
                <div className="h-3 flex-1 rounded bg-fred-secondary/30">
                  <div className="h-3 rounded" style={{ width: wPct(s.standard_mean), backgroundColor: BASELINE_COLOR }} />
                </div>
                <span className="w-14 text-right text-[10px] tabular-nums text-fred-text">{fmt4(s.standard_mean)}</span>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

const ChartCard = ({
  title,
  tip,
  legend,
  children,
}: {
  title: string;
  tip?: string;
  legend?: React.ReactNode;
  children: React.ReactNode;
}) => (
  <div className="rounded-lg border border-fred-border bg-fred-surface p-4 shadow-fredCard">
    <CardHeader title={title} tip={tip} as="h3" />
    {children}
    {legend ? <div className="mx-auto mt-2 max-w-[560px]">{legend}</div> : null}
  </div>
);

// Each chart carries the legend for the marks it draws, inside its own card,
// so a chart, its title and its legend fit in one viewport together.
const WIN_LOSS_LEGEND = (
  <ChartLegend
    items={[
      { label: "Sibyl better (below the line)", color: BETTER },
      { label: "Sibyl worse", color: WORSE },
    ]}
  />
);
const DUMBBELL_LEGEND = (
  <ChartLegend
    items={[
      { label: "Baseline", color: BASELINE_COLOR },
      { label: "Sibyl", color: SIBYL_COLOR },
      { label: "Segment: winner (green Sibyl, red baseline)", color: BETTER, kind: "line" },
    ]}
  />
);

// ---------------------------------------------------------------------------
// Against Sibyl's own reference, the 12-month conflictology and the raw pool
// ---------------------------------------------------------------------------
const VARIANT_LABELS: Record<string, string> = {
  vs_sibyl_ref: "Sibyl vs its reference",
  vs_conflictology12: "Sibyl vs 12-month conflictology",
  vs_raw: "Sibyl vs its raw pool",
  raw_vs_sibyl_ref: "Raw pool vs reference",
  abc_vs_sibyl: "Production trials only (no extra trials) vs Sibyl",
  w25_vs_sibyl: "Reference weight 0.25 vs Sibyl",
  w50_vs_sibyl: "Reference weight 0.5 vs Sibyl",
  w75_vs_sibyl: "Reference weight 0.75 vs Sibyl",
  dw_vs_sibyl: "Weight by trial disagreement vs Sibyl",
  noguard_vs_sibyl: "No outlier guard vs Sibyl",
};

// Variant minus Sibyl on Brier with its 90% interval, or "not yet".
const pairedText = (v: SibylVariantComparison): string | null => {
  const b = v.paired?.all?.brier;
  if (!b) return null;
  if (b.status !== "ok") return `not yet (${b.n_questions} of ${b.min_questions ?? 20})`;
  const f = (x: number | null | undefined) => (x === null || x === undefined ? "—" : x.toFixed(4));
  return `${f(b.mean_diff)} (${f(b.lo)} to ${f(b.hi)})`;
};

export const SibylVariantsTable = ({
  variants,
}: {
  variants: Record<string, SibylVariantComparison> | undefined;
}) => {
  const entries = Object.entries(variants ?? {}).filter(
    ([, v]) => v.n_questions > 0 || v.paired !== undefined,
  );
  if (!entries.length) return null;
  const passes = Array.from(
    new Set(entries.flatMap(([, v]) => Object.keys(v.by_selection_pass ?? {}))),
  ).sort();
  return (
    <div className="overflow-x-auto rounded-lg border border-fred-secondary bg-fred-surface p-4 shadow-fredCard">
      <div className="mb-2 flex items-center gap-1 text-sm font-semibold text-fred-text">
        What the research added
        <InfoTooltip text="Mean Brier difference (left minus right) on the same questions and months. Negative (green) means the left side did better. Split by how each question was chosen: floor, fill, or a no-flag control." />
      </div>
      <table className="min-w-full text-xs">
        <thead>
          <tr className="text-fred-muted">
            <th className="px-2 py-1 text-left">Comparison</th>
            <th className="px-2 py-1 text-right">Questions</th>
            <th className="px-2 py-1 text-right">Mean Δ Brier</th>
            <th className="px-2 py-1 text-right">Paired, 90% interval</th>
            {passes.map((p) => (
              <th key={p} className="px-2 py-1 text-right">Δ {p}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {entries.map(([key, v]) => {
            const b = v.aggregate.spd?.brier as SibylComparisonStat | undefined;
            return (
              <tr key={key} className="border-t border-fred-secondary/40">
                <td className="px-2 py-1 text-left text-fred-text">{VARIANT_LABELS[key] ?? key}</td>
                <td className="px-2 py-1 text-right">{v.n_questions}</td>
                <td className={`px-2 py-1 text-right ${deltaColor(b?.mean_delta)}`}>
                  {fmt4(b?.mean_delta)}
                </td>
                <td className="px-2 py-1 text-right text-fred-muted">{pairedText(v) ?? "—"}</td>
                {passes.map((p) => {
                  const pb = v.by_selection_pass?.[p]?.spd?.brier as SibylComparisonStat | undefined;
                  return (
                    <td key={p} className={`px-2 py-1 text-right ${deltaColor(pb?.mean_delta)}`}>
                      {pb ? `${fmt4(pb.mean_delta)} (${pb.n_questions})` : "—"}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Main section
// ---------------------------------------------------------------------------
export default function SibylComparison({ data: initial, includeTest }: Props) {
  const [open, setOpen] = useState(true);
  const [data, setData] = useState<SibylComparisonResponse>(initial);
  const [baseline, setBaseline] = useState<string>("__best__");
  const [loading, setLoading] = useState(false);

  const points = useMemo(() => buildQuestionPoints(data.pairs), [data.pairs]);
  const spd = data.aggregate.spd ?? {};
  const brier = spd.brier as SibylComparisonStat | undefined;
  const latestRun = data.runs[0];

  async function changeBaseline(next: string) {
    setBaseline(next);
    setLoading(true);
    try {
      const params: Record<string, string | boolean | undefined> = {
        include_test: includeTest || undefined,
      };
      if (next !== "__best__") params.baseline = next;
      const resp = await apiGet<SibylComparisonResponse>(
        "/performance/sibyl_comparison",
        params,
      );
      setData(resp);
    } catch (err) {
      console.warn("Failed to refetch Sibyl comparison:", err);
    } finally {
      setLoading(false);
    }
  }

  // Nothing to show at all (no Sibyl runs and no scores) — hide the section.
  if (!data.has_sibyl && data.runs.length === 0) return null;

  const costPerForecast =
    latestRun && latestRun.n_forecast
      ? ((latestRun.opus_cost_usd ?? 0) + (latestRun.brave_cost_usd ?? 0)) /
        latestRun.n_forecast
      : null;

  return (
    <section className="space-y-4">
      <div className="flex items-center justify-between">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="flex items-center gap-2 text-left"
          aria-expanded={open}
        >
          <span className="text-fred-muted">{open ? "▾" : "▸"}</span>
          <h2 className="text-xl font-semibold">Sibyl — Deep-Research Track</h2>
        </button>
        {data.has_sibyl ? (
          <div className="flex items-center gap-2 text-xs text-fred-muted">
            <label htmlFor="sibyl-baseline">Baseline</label>
            <select
              id="sibyl-baseline"
              value={baseline}
              onChange={(e) => changeBaseline(e.target.value)}
              disabled={loading}
              className="rounded border border-fred-secondary bg-fred-surface px-2 py-1 text-xs"
            >
              <option value="__best__">Best available</option>
              {data.available_baselines.map((m) => (
                <option key={m} value={m}>{formatModelName(m)}</option>
              ))}
            </select>
          </div>
        ) : null}
      </div>

      {open ? (
        <>
          <p className="text-sm text-fred-muted">
            Sibyl independently re-forecasts the highest-volatility questions with
            K deep-research agent trials. This compares it head-to-head against the
            main pipeline <strong>only on the questions Sibyl actually forecast</strong>
            {" "}(a fair, apples-to-apples set) — lower Brier is better, so{" "}
            <span className="text-emerald-700">green means Sibyl won</span>.
          </p>

          {!data.has_sibyl || points.length === 0 ? (
            <div className="rounded-lg border border-fred-secondary bg-fred-surface p-6 text-sm text-fred-muted shadow-fredCard">
              {latestRun ? (
                <>
                  Sibyl forecast <strong>{latestRun.n_forecast ?? 0}</strong> of{" "}
                  <strong>{latestRun.n_selected ?? 0}</strong> selected questions in
                  its latest run
                  {latestRun.n_skipped ? ` (${latestRun.n_skipped} skipped)` : ""}.
                  Head-to-head scores appear here once resolutions land.
                </>
              ) : (
                <>No Sibyl runs yet.</>
              )}
            </div>
          ) : (
            <>
              {/* KPI row */}
              <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-6">
                <KpiCard
                  label={<span className="flex items-center gap-1">Win rate<InfoTooltip text="Share of covered questions where Sibyl's mean Brier beats the baseline's (question-level, averaged over horizons first)." /></span>}
                  value={fmtPct(brier?.win_rate)}
                />
                <KpiCard
                  label="Sibyl vs Base Brier"
                  value={
                    <span className="text-base leading-tight">
                      <span style={{ color: SIBYL_COLOR }}>{fmt4(brier?.sibyl_mean)}</span>
                      {" / "}
                      <span style={{ color: BASELINE_COLOR }}>{fmt4(brier?.standard_mean)}</span>
                    </span>
                  }
                />
                <KpiCard
                  label={<span className="flex items-center gap-1">Mean Δ Brier<InfoTooltip text="Sibyl minus baseline mean Brier on the covered set. Negative (green) = Sibyl better." /></span>}
                  value={<span className={deltaColor(brier?.mean_delta)}>{fmt4(brier?.mean_delta)}</span>}
                />
                <KpiCard
                  label="Covered"
                  value={`${brier?.n_questions ?? 0}${latestRun?.n_selected ? ` / ${latestRun.n_selected}` : ""}`}
                />
                <KpiCard label="Cost / forecast" value={fmtUsd(costPerForecast)} />
                <KpiCard
                  label="Budget"
                  value={
                    latestRun?.budget_capped ? (
                      <span className="text-red-700">Capped</span>
                    ) : latestRun?.time_capped ? (
                      <span className="text-red-700">Time capped</span>
                    ) : (
                      <span className="text-emerald-700">OK</span>
                    )
                  }
                />
              </div>

              <SibylVariantsTable variants={data.variants} />

              {/* Charts */}
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                <ChartCard legend={WIN_LOSS_LEGEND} title="Head-to-head skill (Brier)" tip="Each dot is a covered question. Below the 45° line means Sibyl's Brier is lower (better). Dot size ∝ volatility.">
                  <SkillScatter points={points} />
                </ChartCard>
                <ChartCard legend={DUMBBELL_LEGEND} title="Per-question comparison" tip="Baseline (blue) vs Sibyl (indigo) mean Brier per question, sorted by Δ Brier: Sibyl's biggest wins at the top, its biggest losses at the bottom. Segment coloured by the winner. Scrolls inside the card.">
                  <Dumbbell points={points} />
                </ChartCard>
                <ChartCard title="Score-type summary" tip="Mean Sibyl vs baseline for each scoring rule on the covered set, with win rate.">
                  <SummaryBars spd={spd} />
                </ChartCard>
                <ChartCard legend={WIN_LOSS_LEGEND} title="Does divergence help?" tip="Does diverging more from the standard track (higher JS divergence) correlate with beating it? Points below the equal-skill line are Sibyl wins.">
                  <DivergenceSkill points={points} />
                </ChartCard>
                <ChartCard legend={WIN_LOSS_LEGEND} title="Cost-effectiveness" tip="Sibyl's per-question research cost against how much it beat the baseline. Below the line = a win; far right = expensive.">
                  <CostEffect points={points} />
                </ChartCard>
              </div>
            </>
          )}
        </>
      ) : null}
    </section>
  );
}
