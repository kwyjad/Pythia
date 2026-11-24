"use client";

// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
// See the LICENSE file in the project root for details.

// The pieces every Performance-page chart shares: the size-capped frame,
// axes, the hover tooltip, the legend, the card header and the loading
// skeleton. Hand-written SVG; no charting dependency.

import type { ReactNode } from "react";

import InfoTooltip from "../../components/InfoTooltip";

// Validated categorical pair (dataviz validator, light surface): the ensemble
// blue and the benchmark burnt orange. Green/red keep their page-wide meaning
// (better/worse) and always travel with a sign or a word, never colour alone.
export const CHART_COLORS = {
  ensemble: "#2a78d6",
  benchmark: "#b4590f",
  sibyl: "#4a3aa7",
  better: "#008300",
  worse: "#e34948",
  neutral: "#8a8a82",
  axis: "#c9c9c2",
  grid: "#e7e7e1",
  ink: "#4a4a44",
  inkMuted: "#6a6a62",
} as const;

/** Every chart's maximum drawn size, so a wide monitor never blows one up. */
export const CHART_MAX_WIDTH = 560;
export const CHART_MAX_HEIGHT = 380;

/** An SVG that scales down cleanly and never grows past the caps. */
export function ChartFrame({
  width,
  height,
  label,
  children,
  maxHeight = CHART_MAX_HEIGHT,
}: {
  width: number;
  height: number;
  label: string;
  children: ReactNode;
  maxHeight?: number | null;
}) {
  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label={label}
      className="mx-auto block h-auto w-full"
      style={{
        maxWidth: CHART_MAX_WIDTH,
        ...(maxHeight ? { maxHeight } : {}),
      }}
    >
      {children}
    </svg>
  );
}

export type Margin = { t: number; r: number; b: number; l: number };

export function AxisLeft({
  ticks,
  scale,
  margin,
  innerWidth,
  format = (v: number) => String(v),
  title,
  innerHeight,
}: {
  ticks: number[];
  scale: (v: number) => number;
  margin: Margin;
  innerWidth: number;
  innerHeight: number;
  format?: (v: number) => string;
  title?: string;
}) {
  return (
    <g>
      {ticks.map((t) => (
        <g key={`y-${t}`}>
          <line x1={margin.l} x2={margin.l + innerWidth} y1={scale(t)} y2={scale(t)}
                stroke={CHART_COLORS.grid} strokeWidth={1} />
          <text x={margin.l - 6} y={scale(t) + 3} textAnchor="end" fontSize={10}
                fill={CHART_COLORS.inkMuted}>{format(t)}</text>
        </g>
      ))}
      {title ? (
        <text x={12} y={margin.t + innerHeight / 2} textAnchor="middle" fontSize={10}
              fill={CHART_COLORS.inkMuted}
              transform={`rotate(-90 12 ${margin.t + innerHeight / 2})`}>{title}</text>
      ) : null}
    </g>
  );
}

export function AxisBottom({
  ticks,
  scale,
  y,
  format = (v: number) => String(v),
  title,
  titleY,
  centerX,
  labels,
}: {
  ticks: number[];
  scale: (v: number) => number;
  y: number;
  format?: (v: number) => string;
  title?: string;
  titleY?: number;
  centerX?: number;
  labels?: string[];
}) {
  return (
    <g>
      {ticks.map((t, i) => (
        <text key={`x-${t}`} x={scale(t)} y={y} textAnchor="middle" fontSize={10}
              fill={CHART_COLORS.inkMuted}>{labels ? labels[i] : format(t)}</text>
      ))}
      {title && titleY != null && centerX != null ? (
        <text x={centerX} y={titleY} textAnchor="middle" fontSize={10}
              fill={CHART_COLORS.inkMuted}>{title}</text>
      ) : null}
    </g>
  );
}

/** A tooltip positioned inside a `relative` wrapper, as a percent of it. */
export function ChartTooltip({
  xPct,
  yPct,
  children,
}: {
  xPct: number;
  yPct: number;
  children: ReactNode;
}) {
  const flip = xPct > 60;
  return (
    <div
      role="status"
      className="pointer-events-none absolute z-10 rounded border border-fred-border bg-fred-surface px-2 py-1 text-xs text-fred-text shadow-fredCard"
      style={{
        left: `${xPct}%`,
        top: `${yPct}%`,
        transform: `translate(${flip ? "calc(-100% - 10px)" : "10px"}, -50%)`,
        whiteSpace: "nowrap",
      }}
    >
      {children}
    </div>
  );
}

export type LegendItem = {
  label: string;
  color: string;
  kind?: "swatch" | "line" | "dashed" | "hollow";
};

export function ChartLegend({ items, note }: { items: LegendItem[]; note?: string }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-fred-muted">
      {items.map((it) => (
        <span key={it.label} className="flex items-center gap-1.5">
          <LegendKey item={it} />
          {it.label}
        </span>
      ))}
      {note ? <span className="basis-full">{note}</span> : null}
    </div>
  );
}

function LegendKey({ item }: { item: LegendItem }) {
  const kind = item.kind ?? "swatch";
  if (kind === "swatch") {
    return <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: item.color }} />;
  }
  return (
    <svg width={22} height={10} aria-hidden="true">
      {kind === "hollow" ? (
        <circle cx={11} cy={5} r={4} fill="#fff" stroke={item.color} strokeWidth={2} />
      ) : (
        <line x1={1} x2={21} y1={5} y2={5} stroke={item.color} strokeWidth={2}
              strokeDasharray={kind === "dashed" ? "4 3" : undefined} />
      )}
    </svg>
  );
}

/** One card header style: title, info tooltip, controls on the right. */
export function CardHeader({
  title,
  tip,
  children,
  as = "h2",
}: {
  title: ReactNode;
  tip?: string;
  children?: ReactNode;
  as?: "h2" | "h3";
}) {
  const Tag = as;
  return (
    <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
      <Tag className="flex items-center gap-1 text-sm font-semibold uppercase tracking-wide text-fred-text">
        {title}
        {tip ? <InfoTooltip text={tip} /> : null}
      </Tag>
      {children ? <div className="flex flex-wrap items-center gap-2">{children}</div> : null}
    </div>
  );
}

/** A small labelled select in the card-header control style. */
export function HeaderSelect({
  id,
  label,
  value,
  onChange,
  options,
  disabled,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: Array<{ value: string; label: string; disabled?: boolean }>;
  disabled?: boolean;
}) {
  return (
    <label htmlFor={id} className="flex items-center gap-1 text-xs text-fred-muted">
      {label}
      <select
        id={id}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-md border border-fred-border bg-fred-surface px-2 py-1 text-xs text-fred-text focus:outline-none focus:ring-2 focus:ring-fred-primary"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value} disabled={o.disabled}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}

/** A two-or-more-way toggle (radio group semantics). */
export function SegmentedToggle<T extends string>({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: T;
  onChange: (v: T) => void;
  options: Array<{ value: T; label: string; disabled?: boolean }>;
}) {
  return (
    <div role="radiogroup" aria-label={label}
         className="flex gap-0.5 rounded-md border border-fred-border bg-fred-bg p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          disabled={o.disabled}
          onClick={() => onChange(o.value)}
          className={`rounded px-2 py-0.5 text-xs font-medium focus:outline-none focus:ring-2 focus:ring-fred-primary disabled:cursor-not-allowed disabled:opacity-40 ${
            value === o.value ? "bg-fred-primary text-white" : "text-fred-text hover:bg-fred-surface"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** A grey placeholder block shown while data loads. */
export function Skeleton({ className = "h-40" }: { className?: string }) {
  return (
    <div
      aria-hidden="true"
      className={`animate-pulse rounded-md bg-fred-border/60 ${className}`}
    />
  );
}
