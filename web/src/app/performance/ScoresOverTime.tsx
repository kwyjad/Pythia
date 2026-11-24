"use client";

// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
// See the LICENSE file in the project root for details.

// "Scores over time": the ensemble's score by forecast month, with an
// optional PAIRED benchmark line. Data shaping lives in
// web/src/lib/performance_data.ts; this file only draws.

import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";

import { apiGet } from "../../lib/api";
import { formatModelName } from "../../lib/model_names";
import {
  RANGE_LABELS,
  SCORE_TYPES,
  SCORE_TYPE_LABELS,
  availableScoreTypes,
  buildSeries,
  isExternalReference,
  niceTicks,
  pickBestReference,
  referenceLabel,
  segmentIsDashed,
  yDomain,
  type RangeKey,
  type ScoreType,
  type Stat,
} from "../../lib/performance_data";
import {
  LEGEND_LOW_N,
  TOOLTIP_SCORES_OVER_TIME,
} from "../../lib/score_glossary";
import type {
  PairedScoreRow,
  PerformanceTimeseriesResponse,
  ScoreFamily,
} from "../../lib/types";
import {
  AxisBottom,
  AxisLeft,
  CHART_COLORS,
  CardHeader,
  ChartFrame,
  ChartLegend,
  ChartTooltip,
  HeaderSelect,
  SegmentedToggle,
  Skeleton,
} from "./chart_kit";

const EMPTY: PerformanceTimeseriesResponse = {
  month_basis: "forecast_month",
  rows: [],
  paired_rows: [],
};

const FAMILIES: Array<{ value: ScoreFamily; label: string }> = [
  { value: "spd", label: "SPD (ensemble)" },
  { value: "binary", label: "Binary" },
];

const MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const monthLabel = (ym: string) => {
  const [y, m] = ym.split("-");
  return `${MONTH_NAMES[Number(m) - 1] ?? m} ${y.slice(2)}`;
};
const fmt = (v: number | null | undefined) =>
  v == null || !Number.isFinite(v) ? "—" : v.toFixed(3);

const pick = <T extends string>(raw: string | null, allowed: readonly T[], fallback: T): T =>
  raw && (allowed as readonly string[]).includes(raw) ? (raw as T) : fallback;

type Props = {
  includeTest?: boolean;
  metricFilter: string | null;
  track: number | null;
  ensemble: string | null;
  referenceRows: PairedScoreRow[];
};

export default function ScoresOverTime({
  includeTest,
  metricFilter,
  track,
  ensemble,
  referenceRows,
}: Props) {
  const sp = useSearchParams();
  const [family, setFamily] = useState<ScoreFamily>(
    pick(sp?.get("ts_family") ?? null, ["spd", "binary"] as const, "spd"),
  );
  const [metric, setMetric] = useState<ScoreType>(
    pick(sp?.get("ts_metric") ?? null, SCORE_TYPES, "brier"),
  );
  const [stat, setStat] = useState<Stat>(pick(sp?.get("ts_stat") ?? null, ["avg", "median"] as const, "avg"));
  const [range, setRange] = useState<RangeKey>(
    pick(sp?.get("ts_range") ?? null, ["all", "12", "6", "3"] as const, "all"),
  );
  // "auto" = the best reference for the family; "none" = overlay off.
  const [refChoice, setRefChoice] = useState<string>(sp?.get("ts_ref") ?? "auto");

  const [data, setData] = useState<PerformanceTimeseriesResponse>(EMPTY);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    let live = true;
    setLoading(true);
    setFailed(false);
    apiGet<PerformanceTimeseriesResponse>("/performance/timeseries", {
      include_test: includeTest || undefined,
      metric: metricFilter ?? undefined,
      track: track ?? undefined,
    })
      .then((r) => live && setData(r))
      .catch((e) => {
        console.warn("Failed to load score time series:", e);
        if (live) {
          setFailed(true);
          setData(EMPTY);
        }
      })
      .finally(() => live && setLoading(false));
    return () => {
      live = false;
    };
  }, [includeTest, metricFilter, track]);

  // Selector state goes into the URL so a view can be shared. replaceState
  // keeps the server page from re-rendering on every change.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const url = new URL(window.location.href);
    const set = (k: string, v: string, dflt: string) =>
      v === dflt ? url.searchParams.delete(k) : url.searchParams.set(k, v);
    set("ts_family", family, "spd");
    set("ts_metric", metric, "brier");
    set("ts_stat", stat, "avg");
    set("ts_range", range, "all");
    set("ts_ref", refChoice, "auto");
    window.history.replaceState(window.history.state, "", url.toString());
  }, [family, metric, stat, range, refChoice]);

  const available = useMemo(
    () => availableScoreTypes(data.rows, family, ensemble),
    [data.rows, family, ensemble],
  );
  // Never sit on a metric the family has no data for.
  const effectiveMetric: ScoreType =
    available.has(metric) || available.size === 0
      ? metric
      : (SCORE_TYPES.find((t) => available.has(t)) ?? "brier");

  const referenceOptions = useMemo(() => {
    const names = new Set<string>();
    for (const p of data.paired_rows) {
      if (p.model_name === ensemble && p.score_family === family && isExternalReference(p.reference)) {
        names.add(p.reference);
      }
    }
    return Array.from(names).sort();
  }, [data.paired_rows, ensemble, family]);

  const bestReference = useMemo(() => {
    const best = pickBestReference(referenceRows, family, ensemble, effectiveMetric, stat);
    if (best && referenceOptions.includes(best.reference)) return best.reference;
    return referenceOptions[0] ?? null;
  }, [referenceRows, family, ensemble, effectiveMetric, stat, referenceOptions]);

  const reference =
    refChoice === "none"
      ? null
      : refChoice === "auto"
        ? bestReference
        : referenceOptions.includes(refChoice)
          ? refChoice
          : bestReference;

  const points = useMemo(
    () =>
      buildSeries(data.rows, data.paired_rows, {
        family,
        ensemble,
        metric: effectiveMetric,
        stat,
        range,
        reference,
      }),
    [data, family, ensemble, effectiveMetric, stat, range, reference],
  );

  // ---- Geometry ------------------------------------------------------------
  const W = 560;
  const H = 300;
  const M = { t: 16, r: 16, b: 40, l: 52 };
  const iw = W - M.l - M.r;
  const ih = H - M.t - M.b;
  const values = points.flatMap((p) => (p.bench != null ? [p.value, p.bench] : [p.value]));
  const [lo, hi] = yDomain(values);
  const ticks = niceTicks(lo, hi, 4);
  const sy = (v: number) => M.t + ih - ((v - lo) / (hi - lo || 1)) * ih;
  const sx = (i: number) => (points.length <= 1 ? M.l + iw / 2 : M.l + (i / (points.length - 1)) * iw);
  const xIdx = points.map((_, i) => i);
  const labelEvery = Math.max(1, Math.ceil(points.length / 8));
  const xTicks = xIdx.filter((i) => i % labelEvery === 0 || i === points.length - 1);

  const ensembleLabel = ensemble ? formatModelName(ensemble) : "Ensemble";
  const metricName = SCORE_TYPE_LABELS[effectiveMetric];
  const hovered = hover != null ? points[hover] : null;

  const ariaLabel =
    `${ensembleLabel} ${metricName} (${stat === "avg" ? "average" : "median"}) by forecast month, ` +
    `${family === "spd" ? "SPD" : "binary"} questions, ${points.length} months` +
    (reference ? `, with ${referenceLabel(reference)} paired as a dashed benchmark` : "") +
    ". Lower is better.";

  const controls = (
    <>
      <SegmentedToggle label="Score family" value={family} onChange={setFamily} options={FAMILIES} />
      <SegmentedToggle
        label="Metric"
        value={effectiveMetric}
        onChange={setMetric}
        options={SCORE_TYPES.map((t) => ({
          value: t,
          label: SCORE_TYPE_LABELS[t],
          disabled: available.size > 0 && !available.has(t),
        }))}
      />
      <SegmentedToggle
        label="Statistic"
        value={stat}
        onChange={setStat}
        options={[
          { value: "avg", label: "Average" },
          { value: "median", label: "Median" },
        ]}
      />
      <HeaderSelect
        id="ts-range"
        label="Range"
        value={range}
        onChange={(v) => setRange(v as RangeKey)}
        options={(Object.keys(RANGE_LABELS) as RangeKey[]).map((k) => ({ value: k, label: RANGE_LABELS[k] }))}
      />
      <HeaderSelect
        id="ts-ref"
        label="Benchmark"
        value={refChoice}
        onChange={setRefChoice}
        options={[
          { value: "auto", label: bestReference ? `Best: ${referenceLabel(bestReference)}` : "Best available" },
          ...referenceOptions.map((r) => ({ value: r, label: referenceLabel(r) })),
          { value: "none", label: "Off" },
        ]}
      />
    </>
  );

  let body: React.ReactNode;
  if (loading) {
    body = <Skeleton className="mx-auto h-64 max-w-[560px]" />;
  } else if (failed) {
    body = (
      <p className="py-8 text-center text-sm text-fred-muted">
        The score time series could not be loaded from the API. The rest of the page is unaffected.
      </p>
    );
  } else if (!data.rows.length) {
    body = (
      <p className="py-8 text-center text-sm text-fred-muted">
        No scored forecasts with a dated forecasting run yet. A month appears here once one of its
        forecasts has a resolved horizon.
      </p>
    );
  } else if (points.length < 2) {
    body = (
      <p className="py-8 text-center text-sm text-fred-muted">
        {points.length === 0
          ? `No ${metricName} scores for ${ensembleLabel} on ${family === "spd" ? "SPD" : "binary"} questions in this range`
          : `Only one forecast month (${monthLabel(points[0].month)}: ${fmt(points[0].value)}) has scores in this range`}
        {reference && points.length === 0 ? ` paired with ${referenceLabel(reference)}` : ""}. Widen the range
        {reference ? " or switch the benchmark off" : ""} to see a trend.
      </p>
    );
  } else {
    body = (
      <div className="relative mx-auto" style={{ maxWidth: 560 }} onMouseLeave={() => setHover(null)}>
        <ChartFrame width={W} height={H} label={ariaLabel}>
          <AxisLeft ticks={ticks} scale={sy} margin={M} innerWidth={iw} innerHeight={ih}
                    format={(v) => v.toFixed(2)} title={`${metricName} (lower is better)`} />
          <AxisBottom ticks={xTicks} scale={sx} y={H - 22} labels={xTicks.map((i) => monthLabel(points[i].month))}
                      title="Forecast month" titleY={H - 4} centerX={M.l + iw / 2} />
          {/* benchmark (dashed) */}
          {reference
            ? points.slice(1).map((p, k) =>
                p.bench != null && points[k].bench != null ? (
                  <line key={`b-${p.month}`} x1={sx(k)} y1={sy(points[k].bench as number)} x2={sx(k + 1)}
                        y2={sy(p.bench)} stroke={CHART_COLORS.benchmark} strokeWidth={2} strokeDasharray="6 4" />
                ) : null,
              )
            : null}
          {/* ensemble line; a segment into a low-n point is dashed */}
          {points.slice(1).map((p, k) => (
            <line key={`e-${p.month}`} x1={sx(k)} y1={sy(points[k].value)} x2={sx(k + 1)} y2={sy(p.value)}
                  stroke={CHART_COLORS.ensemble} strokeWidth={2}
                  strokeDasharray={segmentIsDashed(points, k + 1) ? "2 4" : undefined} />
          ))}
          {hovered ? (
            <line x1={sx(hover as number)} x2={sx(hover as number)} y1={M.t} y2={M.t + ih}
                  stroke={CHART_COLORS.axis} strokeWidth={1} />
          ) : null}
          {reference
            ? points.map((p, i) =>
                p.bench != null ? (
                  <rect key={`bm-${p.month}`} x={sx(i) - 3.5} y={sy(p.bench) - 3.5} width={7} height={7}
                        fill={CHART_COLORS.benchmark} stroke="#fff" strokeWidth={2} />
                ) : null,
              )
            : null}
          {points.map((p, i) => (
            <circle key={`em-${p.month}`} cx={sx(i)} cy={sy(p.value)} r={4.5}
                    fill={p.lowN ? "#fff" : CHART_COLORS.ensemble}
                    stroke={p.lowN ? CHART_COLORS.ensemble : "#fff"} strokeWidth={2} />
          ))}
          {/* hit targets, wider than the marks, keyboard-reachable */}
          {points.map((p, i) => {
            const half = points.length > 1 ? iw / (points.length - 1) / 2 : iw / 2;
            return (
              <rect key={`hit-${p.month}`} x={sx(i) - half} y={M.t} width={half * 2} height={ih}
                    fill="transparent" tabIndex={0}
                    aria-label={`${monthLabel(p.month)}: ${fmt(p.value)} on ${p.n} questions`}
                    onMouseEnter={() => setHover(i)} onFocus={() => setHover(i)} onBlur={() => setHover(null)} />
            );
          })}
        </ChartFrame>
        {hovered ? (
          <ChartTooltip xPct={(sx(hover as number) / W) * 100} yPct={(sy(hovered.value) / H) * 100}>
            <div className="font-semibold">{monthLabel(hovered.month)}</div>
            <div>
              {ensembleLabel}: {fmt(hovered.value)}
            </div>
            <div className="text-fred-muted">
              {hovered.n} question{hovered.n === 1 ? "" : "s"}
              {reference ? " (paired)" : ""}
              {hovered.lowN ? " · few, unsettled" : ""}
            </div>
            {reference ? (
              <div>
                {referenceLabel(reference)}: {fmt(hovered.bench)}
              </div>
            ) : null}
          </ChartTooltip>
        ) : null}
      </div>
    );
  }

  return (
    <section className="rounded-lg border border-fred-border bg-fred-surface p-4 shadow-fredCard">
      <CardHeader title="Scores over time" tip={TOOLTIP_SCORES_OVER_TIME}>
        {controls}
      </CardHeader>
      {body}
      {!loading && points.length >= 2 ? (
        <div className="mx-auto mt-2 max-w-[560px] space-y-1">
          <ChartLegend
            items={[
              { label: ensembleLabel, color: CHART_COLORS.ensemble, kind: "line" },
              ...(reference
                ? [{ label: `${referenceLabel(reference)} (paired)`, color: CHART_COLORS.benchmark, kind: "dashed" as const }]
                : []),
              { label: "Fewer than 10 questions", color: CHART_COLORS.ensemble, kind: "hollow" },
            ]}
            note={LEGEND_LOW_N}
          />
          <p className="text-xs text-fred-muted">
            Month = forecast month (when the forecasting run took place).
            {reference
              ? " With a benchmark on, both lines cover only the questions both scored that month."
              : " Each point covers every scored question of that month."}
          </p>
        </div>
      ) : null}
    </section>
  );
}
