"use client";

import { useMemo, useState } from "react";

import InfoTooltip from "../../components/InfoTooltip";
import KpiCard from "../../components/KpiCard";
import { apiGet } from "../../lib/api";
import { formatModelName } from "../../lib/model_names";
import type {
  SibylComparisonPair,
  SibylComparisonResponse,
  SibylComparisonStat,
} from "../../lib/types";

// ---------------------------------------------------------------------------
// Palette (validated dataviz palette, light surface — matches CostsClient).
// Green = Sibyl better, red = Sibyl worse (all three scores are lower-is-better).
// ---------------------------------------------------------------------------
const SIBYL_COLOR = "#4a3aa7"; // indigo — same hue Sibyl uses on /costs and /sibyl
const BASELINE_COLOR = "#2a78d6"; // blue
const BETTER = "#008300"; // green
const WORSE = "#e34948"; // red
const AXIS = "#c9c9c2";
const GRID = "#e7e7e1";

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
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img"
         aria-label="Sibyl versus baseline Brier scatter">
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
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Chart 2 — Per-question dumbbell (sorted by Δ). Baseline vs Sibyl dot joined
// by a segment colored by winner.
// ---------------------------------------------------------------------------
function Dumbbell({ points }: { points: QuestionPoint[] }) {
  const rowH = 22;
  const W = 360;
  const M = { t: 8, r: 12, b: 24, l: 96 };
  const ih = points.length * rowH;
  const H = M.t + ih + M.b;
  const iw = W - M.l - M.r;
  const dom = Math.ceil(Math.max(0.1, ...points.map((p) => Math.max(p.sibyl, p.standard))) * 10) / 10;
  const sx = (v: number) => M.l + (v / dom) * iw;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Per-question Sibyl vs baseline Brier">
      {[0, dom / 2, dom].map((t) => (
        <g key={t}>
          <line x1={sx(t)} y1={M.t} x2={sx(t)} y2={M.t + ih} stroke={GRID} strokeWidth={1} />
          <text x={sx(t)} y={H - 8} textAnchor="middle" fontSize={9} fill="#8a8a82">{Number(t.toFixed(2))}</text>
        </g>
      ))}
      {points.map((p, i) => {
        const y = M.t + i * rowH + rowH / 2;
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
    </svg>
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
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Divergence versus skill delta">
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
    </svg>
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
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Cost versus skill delta">
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
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Chart 5 — Score-type summary bars: Sibyl vs Baseline mean per Brier/Log/CRPS.
// ---------------------------------------------------------------------------
function SummaryBars({ spd }: { spd: Record<string, SibylComparisonStat> }) {
  const order = ["brier", "log", "crps"].filter((k) => spd[k]);
  if (!order.length) return null;
  return (
    <div className="space-y-3">
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

const ChartCard = ({ title, tip, children }: { title: string; tip?: string; children: React.ReactNode }) => (
  <div className="rounded-lg border border-fred-secondary bg-fred-surface p-4 shadow-fredCard">
    <div className="mb-2 flex items-center gap-1 text-sm font-medium text-fred-text">
      {title}
      {tip ? <InfoTooltip text={tip} /> : null}
    </div>
    {children}
  </div>
);

const Legend = () => (
  <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-fred-muted">
    <span className="flex items-center gap-1.5"><span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: SIBYL_COLOR }} />Sibyl</span>
    <span className="flex items-center gap-1.5"><span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: BASELINE_COLOR }} />Baseline</span>
    <span className="flex items-center gap-1.5"><span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: BETTER }} />Sibyl better</span>
    <span className="flex items-center gap-1.5"><span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: WORSE }} />Sibyl worse</span>
  </div>
);

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
                    ) : (
                      <span className="text-emerald-700">OK</span>
                    )
                  }
                />
              </div>

              <Legend />

              {/* Charts */}
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                <ChartCard title="Head-to-head skill (Brier)" tip="Each dot is a covered question. Below the 45° line means Sibyl's Brier is lower (better). Dot size ∝ volatility.">
                  <SkillScatter points={points} />
                </ChartCard>
                <ChartCard title="Per-question comparison" tip="Baseline (blue) vs Sibyl (indigo) mean Brier per question, sorted by margin. Segment colored by winner.">
                  <Dumbbell points={points} />
                </ChartCard>
                <ChartCard title="Score-type summary" tip="Mean Sibyl vs baseline for each scoring rule on the covered set, with win rate.">
                  <SummaryBars spd={spd} />
                </ChartCard>
                <ChartCard title="Does divergence help?" tip="Does diverging more from the standard track (higher JS divergence) correlate with beating it? Points below the equal-skill line are Sibyl wins.">
                  <DivergenceSkill points={points} />
                </ChartCard>
                <ChartCard title="Cost-effectiveness" tip="Sibyl's per-question research cost against how much it beat the baseline. Below the line = a win; far right = expensive.">
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
