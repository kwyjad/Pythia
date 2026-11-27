"use client";

import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";

import InfoTooltip from "../../components/InfoTooltip";
import KpiCard from "../../components/KpiCard";
import SortableTable, { SortableColumn } from "../../components/SortableTable";
import { apiGet } from "../../lib/api";
import { formatModelName } from "../../lib/model_names";
import {
  TOOLTIP_BRIER_BINARY,
  TOOLTIP_BRIER_SPD,
  TOOLTIP_CRPS,
  TOOLTIP_EXTERNAL_BENCHMARK,
  TOOLTIP_LOG,
  TOOLTIP_SAMPLES,
  TOOLTIP_ACTIVE_MODEL,
  TOOLTIP_LOWER_IS_BETTER,
  TOOLTIP_MODELS_KPI,
  TOOLTIP_PAIRED_REFERENCE,
  TOOLTIP_SKILL_BAR,
  TOOLTIP_TRACK2_NAME,
  TOOLTIP_TRACK_SPLIT,
  ACE_PA_INDICATIVE_NOTE,
  isAcePa,
} from "../../lib/score_glossary";
import {
  CENSORED_SCORE_LABELS,
  censoredCells,
  SCORE_TYPES,
  SCORE_TYPE_LABELS,
  headlineCell,
  isIndividualForecaster,
  modelSkill,
  pairedValue,
  pickBestReference,
  referenceLabel,
  skillBarGeometry,
  skillScale,
  type HeadlineCell,
  type ScoreType,
  type SkillInfo,
  type Stat,
} from "../../lib/performance_data";
import {
  buildStatusContext,
  missingActiveModels,
  placeholderScoreRow,
  statusOf as modelStatusOf,
  statusRank,
  type RowStatus,
} from "../../lib/model_status";
import type {
  PairedScoreRow,
  PerformanceRunRow,
  PerformanceScoresResponse,
  PerformanceSummaryRow,
  ResolutionRateRow,
  ResolutionRatesResponse,
  ScoreFamily,
  SibylComparisonResponse,
} from "../../lib/types";
import AboutScores from "./AboutScores";
import ScoresOverTime from "./ScoresOverTime";
import SibylComparison from "./SibylComparison";
import {
  CHART_COLORS,
  CardHeader,
  HeaderSelect,
  SegmentedToggle,
  Skeleton,
} from "./chart_kit";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

// Top-level page tabs: the live scoring dashboard vs. the scoring-methodology
// reference. Kept distinct from `ViewMode` (the Detailed Scores sub-tabs).
type PageTab = "dashboard" | "about";

type ViewMode = "total" | "by_hazard" | "by_run" | "by_model";

type PerformancePanelProps = {
  initialData: PerformanceScoresResponse;
  initialSibyl?: SibylComparisonResponse;
  includeTest?: boolean;
};

// Aggregated row used by Total, By Hazard, and By Model views after pivoting.
// `avg_brier`/`n_questions`/`n_samples` are the SPD family (multiclass, 0-2);
// the `*_binary` fields are the EVENT_OCCURRENCE family (0-1). The two families
// are never blended into a single average.
type PivotedRow = {
  key: string;
  label: string;
  avg_brier: number | null;
  median_brier: number | null;
  avg_brier_binary: number | null;
  median_brier_binary: number | null;
  avg_log: number | null;
  median_log: number | null;
  avg_crps: number | null;
  median_crps: number | null;
  n_questions: number;
  n_samples: number;
  n_questions_binary: number;
  n_samples_binary: number;
  /** Detailed Scores only: what the row is (drives the badge and muting). */
  status?: RowStatus;
  /** Paired skill vs climatology; null where none is drawn. */
  skill?: SkillInfo | null;
  /** Superseded rows: the active successor's stored model_name. */
  supersededBy?: string;
  /** An active model with nothing scored under its name yet. */
  placeholder?: boolean;
};

// Aggregated row used by the By Run view after pivoting.
type RunPivotedRow = {
  key: string;
  forecaster_run_id: string | null;
  hs_run_id: string;
  run_date: string | null;
  avg_brier: number | null;
  median_brier: number | null;
  avg_brier_binary: number | null;
  median_brier_binary: number | null;
  avg_log: number | null;
  median_log: number | null;
  avg_crps: number | null;
  median_crps: number | null;
  n_questions: number;
  n_samples: number;
  n_questions_binary: number;
  n_samples_binary: number;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const VIEW_LABELS: { key: ViewMode; label: string }[] = [
  { key: "total", label: "Total" },
  { key: "by_hazard", label: "By Hazard" },
  { key: "by_run", label: "By Run" },
  { key: "by_model", label: "By Model" },
];

// Shared tooltip texts are single-sourced from web/src/lib/score_glossary.ts
// (the Interpreter page glossary reads the same copy, so the two never drift).

/** Check if a model_name represents an ensemble method. */
const isEnsembleModel = (name: string | null): boolean => {
  if (name == null) return true;
  const lower = name.toLowerCase();
  return lower.startsWith("ensemble");
};

/** Check if a model_name represents an external benchmark (not a Pythia model). */
const isExternalBenchmark = (name: string | null): boolean => {
  if (name == null) return false;
  return name.startsWith("__ext_");
};

/** Sibyl's rearranged variants (sibyl/score_variants.py, Oct 2026): scored
 * measurements of trials already run, shown in the Sibyl section's "What the
 * research added" table and kept out of Detailed Scores. */
export const SIBYL_REARRANGED_MODELS = new Set([
  "__ext_sibyl_abc",
  "__ext_sibyl_w25",
  "__ext_sibyl_w50",
  "__ext_sibyl_w75",
  "__ext_sibyl_dw",
  "__ext_sibyl_noguard",
]);

/** Pick the preferred ensemble model name from available model names. */
const pickDefaultEnsemble = (models: string[]): string | null => {
  const bayesmc = models.find((m) =>
    m.toLowerCase().includes("ensemble_bayesmc"),
  );
  if (bayesmc) return bayesmc;
  const anyEnsemble = models.find((m) =>
    m.toLowerCase().startsWith("ensemble"),
  );
  if (anyEnsemble) return anyEnsemble;
  return models[0] ?? null;
};

/** Friendly display name for a model, including external benchmarks. */
const displayModelName = (name: string | null): string => {
  if (name == null) return "Ensemble";
  if (name === "__ext_views") return "ViEWS (external benchmark)";
  if (name.startsWith("__ext_")) {
    // Future external benchmarks: strip prefix and capitalize
    return name.replace("__ext_", "").replace(/_/g, " ") + " (external)";
  }
  // Delegates to the shared prettifier, which maps sibyl + the ensemble/track2
  // aggregate slugs to friendly labels (and base model ids to product names).
  return formatModelName(name);
};

const formatScore = (value: number | null | undefined) => {
  if (value == null) return "\u2014";
  return value.toLocaleString(undefined, {
    minimumFractionDigits: 4,
    maximumFractionDigits: 4,
  });
};

const formatInt = (value: number | null | undefined) => {
  if (value == null) return "\u2014";
  return value.toLocaleString();
};

/**
 * Resolve a row's Brier family. Prefers the API-provided `score_family`, but
 * falls back to deriving it from the metric so the UI stays correct even if the
 * frontend is deployed ahead of the API that adds the field.
 */
const familyOf = (row: {
  score_family?: ScoreFamily;
  metric?: string;
}): ScoreFamily =>
  row.score_family ??
  ((row.metric ?? "").toUpperCase() === "EVENT_OCCURRENCE" ? "binary" : "spd");

/**
 * Compute weighted average + weighted median of `scoreType` across rows.
 * When `family` is provided, only rows of that Brier family are included, so a
 * single call never blends binary (0-1) and multiclass SPD (0-2) Brier scales.
 */
function aggregateScore<
  T extends {
    score_type: string;
    score_family?: ScoreFamily;
    metric?: string;
    avg_value: number | null;
    median_value: number | null;
    n_samples: number;
    n_questions: number;
  },
>(
  rows: T[],
  scoreType: string,
  family?: ScoreFamily,
): {
  avg: number | null;
  median: number | null;
  nQuestions: number;
  nSamples: number;
} {
  const matching = rows.filter(
    (r) =>
      r.score_type === scoreType &&
      (family === undefined || familyOf(r) === family),
  );
  let totalWeight = 0;
  let weightedSumAvg = 0;
  let weightedSumMedian = 0;
  let medianWeight = 0;
  let nQuestions = 0;
  let nSamples = 0;
  for (const r of matching) {
    if (r.avg_value != null && r.n_samples > 0) {
      weightedSumAvg += r.avg_value * r.n_samples;
      totalWeight += r.n_samples;
    }
    if (r.median_value != null && r.n_samples > 0) {
      weightedSumMedian += r.median_value * r.n_samples;
      medianWeight += r.n_samples;
    }
    nQuestions += r.n_questions;
    nSamples += r.n_samples;
  }
  return {
    avg: totalWeight > 0 ? weightedSumAvg / totalWeight : null,
    median: medianWeight > 0 ? weightedSumMedian / medianWeight : null,
    nQuestions,
    nSamples,
  };
}

// ---------------------------------------------------------------------------
// Column definitions
// ---------------------------------------------------------------------------

const PIVOTED_COLUMNS: Array<SortableColumn<PivotedRow>> = [
  {
    key: "label",
    label: "Name",
    sortValue: (row) => row.label,
    defaultSortDirection: "asc",
    render: (row) => {
      const isExternal = row.key.startsWith("__ext_");
      const isSibyl = row.key === "sibyl";
      const retired = row.status === "retired";
      const superseded = row.status === "superseded";
      return (
        <span
          className={`font-medium ${
            isExternal ? "text-amber-800" : retired || superseded ? "text-fred-muted" : "text-fred-text"
          }`}
        >
          {row.status === "active" ? <ActiveBadge /> : null}
          {row.label}
          {row.key === "track2_flash" ? <InfoTooltip text={TOOLTIP_TRACK2_NAME} /> : null}
          {retired ? <span className="ml-1.5 text-[10px] uppercase tracking-wide">retired</span> : null}
          {superseded && row.supersededBy ? (
            <span className="ml-1.5 text-[10px] tracking-wide" title={TOOLTIP_ACTIVE_MODEL}>
              superseded by {formatModelName(row.supersededBy)}
            </span>
          ) : null}
          {isExternal ? (
            <InfoTooltip text={TOOLTIP_EXTERNAL_BENCHMARK} />
          ) : null}
          {isSibyl ? (
            <span
              className="ml-2 rounded-sm px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-white"
              style={{ backgroundColor: "#4a3aa7" }}
            >
              deep research
            </span>
          ) : null}
        </span>
      );
    },
  },
  {
    key: "avg_brier",
    label: (
      <span className="inline-flex items-center gap-1">
        SPD Brier <InfoTooltip text={TOOLTIP_BRIER_SPD} />
      </span>
    ),
    sortValue: (row) => row.avg_brier,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.avg_brier),
  },
  {
    key: "median_brier",
    label: "Mdn SPD Brier",
    sortValue: (row) => row.median_brier,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.median_brier),
  },
  {
    key: "avg_brier_binary",
    label: (
      <span className="inline-flex items-center gap-1">
        Binary Brier <InfoTooltip text={TOOLTIP_BRIER_BINARY} />
      </span>
    ),
    sortValue: (row) => row.avg_brier_binary,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.avg_brier_binary),
  },
  {
    key: "avg_log",
    label: (
      <span className="inline-flex items-center gap-1">
        Avg Log Loss <InfoTooltip text={TOOLTIP_LOG} />
      </span>
    ),
    sortValue: (row) => row.avg_log,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.avg_log),
  },
  {
    key: "median_log",
    label: "Mdn Log Loss",
    sortValue: (row) => row.median_log,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.median_log),
  },
  {
    key: "avg_crps",
    label: (
      <span className="inline-flex items-center gap-1">
        Avg CRPS <InfoTooltip text={TOOLTIP_CRPS} />
      </span>
    ),
    sortValue: (row) => row.avg_crps,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.avg_crps),
  },
  {
    key: "median_crps",
    label: "Mdn CRPS",
    sortValue: (row) => row.median_crps,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.median_crps),
  },
  {
    key: "n_questions",
    label: "SPD Q",
    sortValue: (row) => row.n_questions,
    defaultSortDirection: "desc",
    render: (row) => formatInt(row.n_questions),
  },
  {
    key: "n_questions_binary",
    label: "Binary Q",
    sortValue: (row) => row.n_questions_binary,
    defaultSortDirection: "desc",
    render: (row) => formatInt(row.n_questions_binary),
  },
  {
    key: "n_samples",
    label: (
      <span className="inline-flex items-center gap-1">
        Samples <InfoTooltip text={TOOLTIP_SAMPLES} />
      </span>
    ),
    sortValue: (row) => row.n_samples,
    defaultSortDirection: "desc",
    render: (row) => formatInt(row.n_samples),
  },
];

/** "Active" badge: a filled dot plus the word, so the meaning never rests on colour. */
function ActiveBadge() {
  return (
    <span
      title={TOOLTIP_ACTIVE_MODEL}
      className="mr-1.5 inline-flex items-center gap-1 rounded-sm border border-fred-primary/40 bg-fred-primary/10 px-1.5 py-0.5 align-middle text-[10px] font-semibold uppercase tracking-wide text-fred-primary"
    >
      <span aria-hidden="true" className="inline-block h-1.5 w-1.5 rounded-full bg-fred-primary" />
      Active
    </span>
  );
}

/** Inline skill bar centred on zero: right and green beats climatology, left
 *  and red loses to it, hatched below SKILL_LOW_N paired questions. */
function SkillBar({ info, scale }: { info: SkillInfo; scale: number }) {
  const g = skillBarGeometry(info.skill, scale);
  const color = g.side === "neg" ? CHART_COLORS.worse : CHART_COLORS.better;
  const pct = `${info.skill >= 0 ? "+" : "−"}${Math.abs(info.skill * 100).toFixed(0)}%`;
  const fill = info.lowN
    ? `repeating-linear-gradient(45deg, ${color} 0 3px, rgba(255,255,255,0.9) 3px 6px)`
    : color;
  return (
    <span
      role="img"
      aria-label={`Skill vs climatology ${pct} on ${info.n} paired questions${info.lowN ? ", low n" : ""}`}
      className="inline-flex items-center gap-2"
    >
      <span className="relative inline-block h-3 w-32 rounded-sm bg-fred-bg">
        <span className="absolute inset-y-0 left-1/2 w-px bg-fred-muted" aria-hidden="true" />
        {g.side !== "zero" ? (
          <span
            aria-hidden="true"
            className="absolute inset-y-0 rounded-sm"
            style={{
              width: `${g.widthPct}%`,
              left: g.side === "pos" ? "50%" : `${50 - g.widthPct}%`,
              background: fill,
              border: info.lowN ? `1px solid ${color}` : undefined,
            }}
          />
        ) : null}
      </span>
      <span className="w-10 text-right text-xs tabular-nums text-fred-text">
        {pct}
        {g.clipped ? "+" : ""}
      </span>
      <span className="text-[10px] text-fred-muted">
        n={info.n}
        {info.lowN ? " · low n" : ""}
      </span>
    </span>
  );
}

function skillColumn(scale: number): SortableColumn<PivotedRow> {
  const pct = `${Math.round(scale * 100)}%`;
  return {
    key: "skill",
    label: (
      <span className="inline-flex items-center gap-1">
        Skill vs climatology
        <span className="text-[10px] font-normal text-fred-muted">(−{pct} … +{pct})</span>
        <InfoTooltip text={TOOLTIP_SKILL_BAR} />
      </span>
    ),
    sortValue: (row) => row.skill?.skill ?? null,
    defaultSortDirection: "desc",
    render: (row) =>
      row.skill ? (
        <SkillBar info={row.skill} scale={scale} />
      ) : (
        <span className="text-xs text-fred-muted">
          {row.placeholder
            ? "awaiting first scores"
            : row.status === "retired"
              ? "retired"
              : "\u2014"}
        </span>
      ),
  };
}

const RUN_COLUMNS: Array<SortableColumn<RunPivotedRow>> = [
  {
    key: "run_date",
    label: "Run Date",
    sortValue: (row) => row.run_date ?? "",
    defaultSortDirection: "desc",
    render: (row) => (
      <span className="font-medium text-fred-text">
        {row.run_date ?? row.hs_run_id}
      </span>
    ),
  },
  {
    key: "avg_brier",
    label: (
      <span className="inline-flex items-center gap-1">
        SPD Brier <InfoTooltip text={TOOLTIP_BRIER_SPD} />
      </span>
    ),
    sortValue: (row) => row.avg_brier,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.avg_brier),
  },
  {
    key: "median_brier",
    label: "Mdn SPD Brier",
    sortValue: (row) => row.median_brier,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.median_brier),
  },
  {
    key: "avg_brier_binary",
    label: (
      <span className="inline-flex items-center gap-1">
        Binary Brier <InfoTooltip text={TOOLTIP_BRIER_BINARY} />
      </span>
    ),
    sortValue: (row) => row.avg_brier_binary,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.avg_brier_binary),
  },
  {
    key: "avg_log",
    label: (
      <span className="inline-flex items-center gap-1">
        Avg Log Loss <InfoTooltip text={TOOLTIP_LOG} />
      </span>
    ),
    sortValue: (row) => row.avg_log,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.avg_log),
  },
  {
    key: "median_log",
    label: "Mdn Log Loss",
    sortValue: (row) => row.median_log,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.median_log),
  },
  {
    key: "avg_crps",
    label: (
      <span className="inline-flex items-center gap-1">
        Avg CRPS <InfoTooltip text={TOOLTIP_CRPS} />
      </span>
    ),
    sortValue: (row) => row.avg_crps,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.avg_crps),
  },
  {
    key: "median_crps",
    label: "Mdn CRPS",
    sortValue: (row) => row.median_crps,
    defaultSortDirection: "asc",
    render: (row) => formatScore(row.median_crps),
  },
  {
    key: "n_questions",
    label: "SPD Q",
    sortValue: (row) => row.n_questions,
    defaultSortDirection: "desc",
    render: (row) => formatInt(row.n_questions),
  },
  {
    key: "n_questions_binary",
    label: "Binary Q",
    sortValue: (row) => row.n_questions_binary,
    defaultSortDirection: "desc",
    render: (row) => formatInt(row.n_questions_binary),
  },
  {
    key: "n_samples",
    label: (
      <span className="inline-flex items-center gap-1">
        Samples <InfoTooltip text={TOOLTIP_SAMPLES} />
      </span>
    ),
    sortValue: (row) => row.n_samples,
    defaultSortDirection: "desc",
    render: (row) => formatInt(row.n_samples),
  },
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/** A headline value with its paired reference line, or "n/a" with the reason. */
function HeadlineValue({
  cell,
  family,
  best,
  stat,
  large,
}: {
  cell: HeadlineCell;
  family: ScoreFamily;
  best: PairedScoreRow | null;
  stat: Stat;
  large: boolean;
}) {
  const famLabel = family === "spd" ? "SPD" : "Binary";
  const valueCls = large ? "text-2xl font-semibold text-fred-primary" : "text-base font-semibold text-fred-text";
  const value =
    cell.kind === "value" ? (
      <span className={valueCls}>{formatScore(cell.value)}</span>
    ) : (
      <span className={`${valueCls} inline-flex items-center gap-1 text-fred-muted`}>
        n/a <InfoTooltip text={cell.reason} />
      </span>
    );
  let refLine: React.ReactNode = null;
  if (cell.kind === "value" && best) {
    const m = pairedValue(best, "model", stat);
    const r = pairedValue(best, "reference", stat);
    if (m != null && r != null) {
      const better = m < r;
      const tie = Math.abs(m - r) < 1e-12;
      refLine = (
        <div
          className={`text-xs ${tie ? "text-fred-muted" : better ? "text-emerald-800" : "text-red-800"}`}
          title={TOOLTIP_PAIRED_REFERENCE}
        >
          {tie ? "=" : better ? "▼ beats" : "▲ trails"} {referenceLabel(best.reference)}{" "}
          {formatScore3(r)}{" "}
          <span className="text-fred-muted">
            (ens {formatScore3(m)}, {best.n_questions} q paired)
          </span>
        </div>
      );
    }
  }
  return (
    <div className="leading-tight">
      <div className="flex items-baseline gap-2">
        <span className="text-[11px] font-medium uppercase tracking-wide text-fred-muted">{famLabel}</span>
        {value}
      </div>
      {refLine}
    </div>
  );
}

const formatScore3 = (v: number | null | undefined) =>
  v == null || !Number.isFinite(v) ? "—" : v.toFixed(3);

const SCORE_TOOLTIPS: Record<ScoreType, string> = {
  brier: `${TOOLTIP_BRIER_SPD}\n\n${TOOLTIP_BRIER_BINARY}`,
  log: TOOLTIP_LOG,
  crps: TOOLTIP_CRPS,
};

/** Pivot rows for the overall (cross-hazard) block, keyed by model. */
function overallAsSummary(
  overall: PerformanceSummaryRow[] | undefined,
  summary: PerformanceSummaryRow[],
): PerformanceSummaryRow[] {
  if (overall && overall.length) return overall;
  // An older API: rebuild sample-weighted means from the per-hazard rows.
  // Medians cannot be rebuilt without a median of medians, so they stay null.
  const groups = new Map<string, PerformanceSummaryRow[]>();
  for (const r of summary) {
    const k = `${familyOf(r)}|${r.score_type}|${r.model_name ?? ""}`;
    groups.set(k, [...(groups.get(k) ?? []), r]);
  }
  const out: PerformanceSummaryRow[] = [];
  for (const [k, rows] of groups) {
    const [fam, st, model] = k.split("|");
    const a = aggregateScore(rows, st, fam as ScoreFamily);
    out.push({
      hazard_code: "*",
      metric: "*",
      score_family: fam as ScoreFamily,
      score_type: st,
      model_name: model || null,
      n_samples: a.nSamples,
      n_questions: a.nQuestions,
      avg_value: a.avg,
      median_value: null,
    });
  }
  return out;
}

export default function PerformancePanel({
  initialData,
  initialSibyl,
  includeTest,
}: PerformancePanelProps) {
  const searchParams = useSearchParams();
  const [pageTab, setPageTab] = useState<PageTab>("dashboard");
  const [view, setView] = useState<ViewMode>("total");
  const [metric, setMetric] = useState<string | null>(null);
  const [track, setTrack] = useState<number | null>(null);
  const [data, setData] = useState<PerformanceScoresResponse>(initialData);
  const [loading, setLoading] = useState(false);
  const [resolutionRates, setResolutionRates] = useState<ResolutionRateRow[] | null>(null);
  const [headlineStat, setHeadlineStat] = useState<Stat>(
    searchParams?.get("stat") === "median" ? "median" : "avg",
  );
  const [skillMetric, setSkillMetric] = useState<ScoreType>("brier");

  // Fetch resolution rates on mount
  useEffect(() => {
    apiGet<ResolutionRatesResponse>("/diagnostics/resolution_rates", {
      include_test: includeTest || undefined,
    })
      .then((res) => setResolutionRates(res.rows ?? []))
      .catch(() => setResolutionRates([]));
  }, [includeTest]);

  // ---- Detect available ensemble models ------------------------------------

  const ensembleModels = useMemo(() => {
    const names = new Set<string>();
    for (const row of data.summary_rows) {
      if (row.model_name && isEnsembleModel(row.model_name)) {
        names.add(row.model_name);
      }
    }
    return Array.from(names).sort();
  }, [data]);

  // The one ensemble selector the page has: the headline cards, the time
  // series and the By Hazard / By Run tables all read it. Kept in the URL.
  const [selectedEnsemble, setSelectedEnsemble] = useState<string | null>(
    searchParams?.get("ensemble") ?? null,
  );

  const activeEnsemble = useMemo(() => {
    if (selectedEnsemble && ensembleModels.includes(selectedEnsemble)) {
      return selectedEnsemble;
    }
    return pickDefaultEnsemble(ensembleModels);
  }, [selectedEnsemble, ensembleModels]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const url = new URL(window.location.href);
    if (selectedEnsemble) url.searchParams.set("ensemble", selectedEnsemble);
    else url.searchParams.delete("ensemble");
    if (headlineStat === "median") url.searchParams.set("stat", "median");
    else url.searchParams.delete("stat");
    window.history.replaceState(window.history.state, "", url.toString());
  }, [selectedEnsemble, headlineStat]);

  // ---- Fetch when metric changes -------------------------------------------

  const refetchScores = async (
    newMetric: string | null,
    newTrack: number | null,
  ) => {
    setLoading(true);
    try {
      const params: Record<string, string | boolean | undefined> = {};
      if (newMetric) params.metric = newMetric;
      if (newTrack != null) params.track = String(newTrack);
      params.include_test = includeTest || undefined;
      const response = await apiGet<PerformanceScoresResponse>(
        "/performance/scores",
        params,
      );
      setData(response);
    } catch (error) {
      console.warn("Failed to fetch performance scores:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleMetricChange = async (newMetric: string | null) => {
    setMetric(newMetric);
    await refetchScores(newMetric, track);
  };

  const handleTrackChange = async (newTrack: number | null) => {
    setTrack(newTrack);
    await refetchScores(metric, newTrack);
  };

  // ---- Coverage KPIs -------------------------------------------------------

  const kpiStats = useMemo(() => {
    const allHazards = new Set(data.summary_rows.map((r) => r.hazard_code));
    const forecasters = new Set(
      data.summary_rows
        .map((r) => r.model_name)
        .filter((m): m is string => isIndividualForecaster(m)),
    );
    return { hazards: allHazards.size, models: forecasters.size };
  }, [data]);

  // ---- Headline skill cards -----------------------------------------------

  const overall = useMemo(
    () => overallAsSummary(data.overall_rows, data.summary_rows),
    [data],
  );
  const referenceRows = useMemo(() => data.reference_rows ?? [], [data]);
  const censored = useMemo(
    () => censoredCells(data.overall_rows, activeEnsemble, headlineStat),
    [data, activeEnsemble, headlineStat],
  );

  const headline = useMemo(
    () =>
      SCORE_TYPES.map((st) => ({
        st,
        spd: headlineCell(overall, "spd", activeEnsemble, st, headlineStat),
        binary: headlineCell(overall, "binary", activeEnsemble, st, headlineStat),
        bestSpd: pickBestReference(referenceRows, "spd", activeEnsemble, st, headlineStat),
        bestBinary: pickBestReference(referenceRows, "binary", activeEnsemble, st, headlineStat),
      })),
    [overall, referenceRows, activeEnsemble, headlineStat],
  );

  // ---- Model status and skill for the Detailed Scores rows -----------------

  const statusCtx = useMemo(() => buildStatusContext(data.active_models), [data]);
  const knowsActive = statusCtx.knowsActive;

  /** Status for each row (lib/model_status), skill where it is drawn, and a
   *  placeholder row for every active model nothing has been scored under
   *  yet. Placeholders go in the model-keyed views only (Total, By Model):
   *  the By Hazard and By Run rows are hazards and runs, not models. */
  const decorate = (rows: PivotedRow[]): PivotedRow[] => {
    const out: PivotedRow[] = rows.map((r) => {
      const info = modelStatusOf(r.key, statusCtx);
      const status: RowStatus = info.status;
      // Superseded rows keep their bar: that record seeds the successor's
      // calibration prior. Retired rows do not.
      const showSkill =
        status === "active" ||
        status === "superseded" ||
        (status === "aggregate" && isEnsembleModel(r.key));
      return {
        ...r,
        status,
        supersededBy: info.supersededBy,
        skill: showSkill ? modelSkill(data.skill_rows ?? [], r.key, skillMetric) : null,
      };
    });
    for (const m of missingActiveModels(data.active_models, rows.map((r) => r.key), track)) {
      out.push(placeholderScoreRow(m, displayModelName(m.model_name)));
    }
    return out;
  };

  /** Replace the cross-hazard medians (a median of medians when built from
   *  per-hazard rows) with the API's true per-model medians. */
  const withTrueMedians = (rows: PivotedRow[]): PivotedRow[] => {
    if (!data.overall_rows?.length) return rows;
    const find = (model: string, fam: ScoreFamily, st: string) =>
      data.overall_rows?.find(
        (o) => o.model_name === model && (o.score_family ?? "spd") === fam && o.score_type === st,
      );
    return rows.map((r) => ({
      ...r,
      median_brier: find(r.key, "spd", "brier")?.median_value ?? r.median_brier,
      median_brier_binary: find(r.key, "binary", "brier")?.median_value ?? r.median_brier_binary,
      median_log: find(r.key, "spd", "log")?.median_value ?? r.median_log,
      median_crps: find(r.key, "spd", "crps")?.median_value ?? r.median_crps,
    }));
  };

  const pinActiveFirst = (row: PivotedRow): number => statusRank(row.status);

  // ---- Helper to build pivoted rows from summary data ----------------------

  function buildPivoted(
    rows: PerformanceSummaryRow[],
    groupBy: (r: PerformanceSummaryRow) => string,
    labelFor: (key: string) => string,
  ): PivotedRow[] {
    const groups = new Map<string, PerformanceSummaryRow[]>();
    for (const row of rows) {
      const gk = groupBy(row);
      const existing = groups.get(gk) ?? [];
      existing.push(row);
      groups.set(gk, existing);
    }
    const result: PivotedRow[] = [];
    for (const [gk, gRows] of groups) {
      const brierSpd = aggregateScore(gRows, "brier", "spd");
      const brierBinary = aggregateScore(gRows, "brier", "binary");
      const log = aggregateScore(gRows, "log", "spd");
      const crps = aggregateScore(gRows, "crps", "spd");
      result.push({
        key: gk,
        label: labelFor(gk),
        avg_brier: brierSpd.avg,
        median_brier: brierSpd.median,
        avg_brier_binary: brierBinary.avg,
        median_brier_binary: brierBinary.median,
        avg_log: log.avg,
        median_log: log.median,
        avg_crps: crps.avg,
        median_crps: crps.median,
        n_questions: brierSpd.nQuestions,
        n_samples: brierSpd.nSamples,
        n_questions_binary: brierBinary.nQuestions,
        n_samples_binary: brierBinary.nSamples,
      });
    }
    return result;
  }

  // ---- Total view ----------------------------------------------------------

  const totalRows = useMemo(
    (): PivotedRow[] =>
      decorate(
        withTrueMedians(
          buildPivoted(
            data.summary_rows,
            (r) => r.model_name ?? "__null__",
            (k) => displayModelName(k === "__null__" ? null : k),
          ),
        ),
      ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [data, skillMetric, track],
  );

  // ---- By Hazard view ------------------------------------------------------

  const hazardRows = useMemo((): PivotedRow[] => {
    const filtered = activeEnsemble
      ? data.summary_rows.filter((r) => r.model_name === activeEnsemble)
      : data.summary_rows;
    return buildPivoted(
      filtered,
      (r) => r.hazard_code,
      (k) => k,
    );
  }, [data, activeEnsemble]);

  // ---- By Run view ---------------------------------------------------------

  const runRows = useMemo((): RunPivotedRow[] => {
    const filtered = activeEnsemble
      ? data.run_rows.filter((r) => r.model_name === activeEnsemble)
      : data.run_rows;
    const groups = new Map<string, PerformanceRunRow[]>();
    for (const row of filtered) {
      // Group by forecaster_run_id (the run that produced forecasts), falling
      // back to hs_run_id for backward compat / external benchmark scores.
      const groupKey = row.forecaster_run_id ?? row.hs_run_id;
      const existing = groups.get(groupKey) ?? [];
      existing.push(row);
      groups.set(groupKey, existing);
    }
    const result: RunPivotedRow[] = [];
    for (const [runId, gRows] of groups) {
      const brierSpd = aggregateScore(gRows, "brier", "spd");
      const brierBinary = aggregateScore(gRows, "brier", "binary");
      const log = aggregateScore(gRows, "log", "spd");
      const crps = aggregateScore(gRows, "crps", "spd");
      const runDate = gRows[0]?.run_date ?? null;
      const forecasterRunId = gRows[0]?.forecaster_run_id ?? null;
      const hsRunId = gRows[0]?.hs_run_id ?? runId;
      result.push({
        key: runId,
        forecaster_run_id: forecasterRunId,
        hs_run_id: hsRunId,
        run_date: runDate,
        avg_brier: brierSpd.avg,
        median_brier: brierSpd.median,
        avg_brier_binary: brierBinary.avg,
        median_brier_binary: brierBinary.median,
        avg_log: log.avg,
        median_log: log.median,
        avg_crps: crps.avg,
        median_crps: crps.median,
        n_questions: brierSpd.nQuestions,
        n_samples: brierSpd.nSamples,
        n_questions_binary: brierBinary.nQuestions,
        n_samples_binary: brierBinary.nSamples,
      });
    }
    return result;
  }, [data, activeEnsemble]);

  // ---- By Model view -------------------------------------------------------

  const modelRows = useMemo((): PivotedRow[] => {
    const pivoted = buildPivoted(
      data.summary_rows.filter((r) => !SIBYL_REARRANGED_MODELS.has(r.model_name ?? "")),
      (r) => r.model_name ?? "__null__",
      (k) => displayModelName(k === "__null__" ? null : k),
    );
    // The table pins active models first (pinActiveFirst) and then sorts by
    // the selected column; this order only breaks ties.
    return decorate(withTrueMedians(pivoted)).sort((a, b) => a.label.localeCompare(b.label));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, skillMetric, track]);

  // One scale for every bar in the table, so lengths compare across rows.
  const skillTrackScale = useMemo(
    () => skillScale([...totalRows, ...modelRows].map((r) => r.skill?.skill)),
    [totalRows, modelRows],
  );
  const modelColumns = useMemo(
    () => [PIVOTED_COLUMNS[0], skillColumn(skillTrackScale), ...PIVOTED_COLUMNS.slice(1)],
    [skillTrackScale],
  );

  // ---- Render --------------------------------------------------------------

  const isEmpty =
    data.summary_rows.length === 0 && data.run_rows.length === 0;

  // Defensive filter: drop blocked hazards (DI, HW, CU, ACO) from the
  // coverage panel even if the API hasn't been updated yet. We don't forecast
  // these anymore, so their tiles would always sit at 0% and create the
  // impression of broken calibration.
  const BLOCKED_HAZARDS = new Set(["DI", "HW", "CU", "ACO"]);
  const ratesLoading = resolutionRates === null;
  const visibleResolutionRates = (resolutionRates ?? []).filter(
    (row) => !BLOCKED_HAZARDS.has(row.hazard_code.toUpperCase()),
  );

  // Summary across all coverage rows: how many questions are eligible
  // for resolution (cutoff has passed), how many of those are scored, how
  // many are pending (too new). Pre-Track questions are included here via
  // the questions table — this is the count the user is looking for.
  const coverageTotals = useMemo(() => {
    let total = 0;
    let resolved = 0;
    let pending = 0;
    for (const row of visibleResolutionRates) {
      total += row.total_questions;
      resolved += row.resolved_questions;
      pending += row.pending_too_new ?? 0;
    }
    const eligible = Math.max(0, total - pending);
    return { total, resolved, pending, eligible };
  }, [visibleResolutionRates]);

  const totalScored =
    data.track_counts?.total ??
    (data.track_counts
      ? (data.track_counts.track1 ?? 0) + (data.track_counts.track2 ?? 0)
      : 0);

  const ensembleOptions = ensembleModels.map((m) => ({ value: m, label: formatModelName(m) }));
  const ensembleSelector =
    ensembleOptions.length > 0 ? (
      <HeaderSelect
        id="perf-ensemble"
        label="Ensemble"
        value={activeEnsemble ?? ""}
        onChange={(v) => setSelectedEnsemble(v || null)}
        options={ensembleOptions}
      />
    ) : null;

  const coverageColor = (rate: number | null, allPending: boolean) =>
    allPending || rate == null
      ? "text-fred-muted"
      : rate >= 0.9
        ? "text-emerald-800"
        : rate >= 0.5
          ? "text-amber-800"
          : "text-red-800";

  return (
    <div className="space-y-6">
      {/* Top-level page tabs: live dashboard vs. scoring methodology */}
      <div role="tablist" aria-label="Performance page"
           className="flex gap-1 rounded-lg border border-fred-border bg-fred-bg p-1">
        {(
          [
            { key: "dashboard", label: "Dashboard" },
            { key: "about", label: "About performance scores" },
          ] as { key: PageTab; label: string }[]
        ).map(({ key, label }) => (
          <button
            key={key}
            type="button"
            role="tab"
            aria-selected={pageTab === key}
            className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-fred-primary ${
              pageTab === key
                ? "bg-fred-primary text-white"
                : "text-fred-text hover:bg-fred-surface"
            }`}
            onClick={() => setPageTab(key)}
          >
            {label}
          </button>
        ))}
      </div>

      {pageTab === "about" ? <AboutScores /> : null}

      {pageTab === "dashboard" ? (
        <>
      {/* Summary: coverage row, then the ensemble skill row */}
      <section className="space-y-4 rounded-lg border border-fred-border bg-fred-surface p-4 shadow-fredCard">
        <CardHeader title="Summary">
          <span className="text-xs text-fred-muted">
            Counts cover every question in the database, including legacy
            questions before the Track 1/2 split.
          </span>
        </CardHeader>
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
          <KpiCard
            label={
              <span className="inline-flex items-center gap-1">
                Resolved Questions{" "}
                <InfoTooltip
                  text={
                    "Questions with at least one resolved horizon (ground " +
                    "truth from the Resolver). Excludes blocked hazards " +
                    "(DI/HW/CU/ACO). Pending questions whose earliest " +
                    "horizon is in the future are not counted here."
                  }
                />
              </span>
            }
            value={
              ratesLoading ? (
                <Skeleton className="mt-1 h-7 w-24" />
              ) : (
                `${formatInt(coverageTotals.resolved)} / ${formatInt(coverageTotals.eligible)}`
              )
            }
          />
          <KpiCard
            label={
              <span className="inline-flex items-center gap-1">
                Scored Questions{" "}
                <InfoTooltip
                  text={
                    "Distinct questions with at least one row in the scores " +
                    "table. Includes legacy pre-Track questions that don't " +
                    "carry a track tag."
                  }
                />
              </span>
            }
            value={formatInt(totalScored)}
          />
          <KpiCard
            label={
              <span className="inline-flex items-center gap-1">
                Track Split{" "}
                <InfoTooltip
                  text={TOOLTIP_TRACK_SPLIT}
                />
              </span>
            }
            value={`T1 ${formatInt(data.track_counts?.track1 ?? 0)} / T2 ${formatInt(data.track_counts?.track2 ?? 0)}`}
          />
          <KpiCard label="Hazards" value={formatInt(kpiStats.hazards)} />
          <KpiCard
            label={
              <span className="inline-flex items-center gap-1">
                Models <InfoTooltip text={TOOLTIP_MODELS_KPI} />
              </span>
            }
            value={
              <span className="inline-flex items-baseline gap-2">
                {formatInt(kpiStats.models)}
                {knowsActive ? (
                  <span className="text-xs font-normal text-fred-muted">
                    {formatInt(data.active_models?.filter((m) => m.role === "ensemble").length ?? 0)} in the current ensemble
                  </span>
                ) : null}
              </span>
            }
          />
        </div>

        <div>
          <CardHeader
            as="h3"
            title={`Ensemble skill${activeEnsemble ? ` · ${formatModelName(activeEnsemble)}` : ""}`}
            tip={`${TOOLTIP_LOWER_IS_BETTER}\n\n${TOOLTIP_PAIRED_REFERENCE}`}
          >
            <span className="text-xs text-fred-muted">Lower is better for all three.</span>
            {ensembleSelector}
            <SegmentedToggle
              label="Statistic"
              value={headlineStat}
              onChange={setHeadlineStat}
              options={[
                { value: "avg", label: "Average" },
                { value: "median", label: "Median" },
              ]}
            />
          </CardHeader>
          {loading ? (
            <div className="grid gap-4 md:grid-cols-3">
              {SCORE_TYPES.map((st) => (
                <Skeleton key={st} className="h-28" />
              ))}
            </div>
          ) : (
            <div className="grid gap-4 md:grid-cols-3">
              {headline.map((h) => (
                <div key={h.st} className="rounded-lg border border-fred-border bg-fred-surface p-4 shadow-fredCard">
                  <div className="mb-2 flex items-center gap-1 text-xs uppercase tracking-wide text-fred-muted">
                    {SCORE_TYPE_LABELS[h.st]} <InfoTooltip text={SCORE_TOOLTIPS[h.st]} />
                  </div>
                  <div className="space-y-2">
                    <HeadlineValue cell={h.spd} family="spd" best={h.bestSpd} stat={headlineStat} large />
                    <HeadlineValue cell={h.binary} family="binary" best={h.bestBinary} stat={headlineStat} large={false} />
                  </div>
                </div>
              ))}
            </div>
          )}
          {!loading && censored.length ? (
            <p className="mt-3 text-xs text-fred-muted" data-testid="censored-scores-line">
              Conflict displacement months known only as a lower bound (IOM DTM), scored on
              &ldquo;at least this many&rdquo;, apart from the cards above:{" "}
              {censored
                .map((c) => `${CENSORED_SCORE_LABELS[c.scoreType]} ${c.value.toFixed(3)}`)
                .join(" · ")}{" "}
              ({censored[0].nQuestions} question{censored[0].nQuestions === 1 ? "" : "s"}).
            </p>
          ) : null}
        </div>
      </section>

      <ScoresOverTime
        includeTest={includeTest}
        metricFilter={metric}
        track={track}
        ensemble={activeEnsemble}
        referenceRows={referenceRows}
      />

      {/* Resolution Coverage Summary */}
      {ratesLoading ? (
        <section className="rounded-lg border border-fred-border bg-fred-surface p-4 shadow-fredCard">
          <CardHeader title="Resolution Coverage" />
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {[0, 1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-16" />
            ))}
          </div>
        </section>
      ) : visibleResolutionRates.length > 0 ? (
        <section className="rounded-lg border border-fred-border bg-fred-surface p-4 shadow-fredCard">
          <CardHeader title="Resolution Coverage">
            <span className="text-xs text-fred-muted">
              Eligible {formatInt(coverageTotals.eligible)} · Resolved{" "}
              {formatInt(coverageTotals.resolved)} · Pending (too new){" "}
              {formatInt(coverageTotals.pending)}
            </span>
          </CardHeader>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {visibleResolutionRates.map((row) => {
              const pending = row.pending_too_new ?? 0;
              const eligible = Math.max(
                0,
                row.total_questions - pending,
              );
              // Eligible rate = resolved / (total - pending). This is the
              // honest accuracy number — denominator excludes questions that
              // can't possibly be resolved yet.
              const eligibleRate =
                eligible > 0 ? row.resolved_questions / eligible : null;
              const isAllPending =
                row.total_questions > 0 && eligible === 0 && pending > 0;
              const colorClass = coverageColor(eligibleRate, isAllPending);
              const metricLabel =
                row.metric === "EVENT_OCCURRENCE"
                  ? "Binary (event)"
                  : row.metric === "PHASE3PLUS_IN_NEED"
                    ? "Phase 3+"
                    : row.metric;
              const pctLabel =
                eligibleRate == null
                  ? "No eligible questions"
                  : `${(eligibleRate * 100).toFixed(1)}% scored`;
              return (
                <div
                  key={`${row.hazard_code}-${row.metric}`}
                  className="rounded border border-fred-border bg-fred-surface px-3 py-2"
                  title={
                    isAllPending
                      ? `${pending} questions in this group are from the latest forecast epoch — their earliest horizon hasn't reached the resolution calendar cutoff yet.`
                      : pending > 0
                        ? `${pending} additional question(s) too new to resolve yet.`
                        : undefined
                  }
                >
                  <div className="text-[11px] font-semibold uppercase tracking-wide text-fred-muted">
                    {row.hazard_code} / {metricLabel}
                  </div>
                  <div className={`mt-1 text-sm font-medium ${colorClass}`}>
                    {isAllPending ? "Awaiting first horizon" : pctLabel}
                  </div>
                  <div className="text-xs text-fred-muted">
                    {isAllPending ? (
                      <>
                        0 of 0 eligible — {pending} too new, will resolve next
                        month
                      </>
                    ) : (
                      <>
                        {row.resolved_questions} of {eligible} eligible
                        {pending > 0 ? ` · ${pending} too new` : null}
                      </>
                    )}
                  </div>
                  {isAcePa(row.hazard_code, row.metric) ? (
                    <div className="mt-1 text-[11px] leading-snug text-fred-muted">
                      {ACE_PA_INDICATIVE_NOTE}
                    </div>
                  ) : null}
                </div>
              );
            })}
          </div>
        </section>
      ) : null}

      {/* Sibyl deep-research track: understand it + head-to-head vs the ensemble */}
      {initialSibyl ? (
        <SibylComparison data={initialSibyl} includeTest={includeTest} />
      ) : null}

      {/* Detailed Scores: filters + table grouped together */}
      <section className="rounded-lg border border-fred-border bg-fred-surface p-4 shadow-fredCard">
        <CardHeader title="Detailed Scores">
          {/* View selector tabs */}
          <SegmentedToggle
            label="Detailed scores view"
            value={view}
            onChange={setView}
            options={VIEW_LABELS.map(({ key, label }) => ({ value: key, label }))}
          />
          <HeaderSelect
            id="perf-metric"
            label="Metric"
            value={metric ?? ""}
            onChange={(v) => handleMetricChange(v || null)}
            options={[
              { value: "", label: "All Metrics" },
              { value: "PA", label: "People Affected (PA)" },
              { value: "FATALITIES", label: "Fatalities" },
              { value: "EVENT_OCCURRENCE", label: "Event Occurrence (binary)" },
              { value: "PHASE3PLUS_IN_NEED", label: "Phase 3+ (IPC)" },
            ]}
          />
          <HeaderSelect
            id="perf-track"
            label="Track"
            value={track != null ? String(track) : ""}
            onChange={(v) => handleTrackChange(v ? Number(v) : null)}
            options={[
              { value: "", label: "All Tracks" },
              { value: "1", label: "Track 1" },
              { value: "2", label: "Track 2" },
            ]}
          />
          {view === "total" || view === "by_model" ? (
            <HeaderSelect
              id="perf-skill-metric"
              label="Skill on"
              value={skillMetric}
              onChange={(v) => setSkillMetric(v as ScoreType)}
              options={SCORE_TYPES.map((t) => ({ value: t, label: SCORE_TYPE_LABELS[t] }))}
            />
          ) : null}
        </CardHeader>
        <p className="mb-3 text-xs text-fred-muted">
          Metric and Track filter the whole page. Ensemble for the By Hazard and By Run views:{" "}
          <strong>{activeEnsemble ? formatModelName(activeEnsemble) : "none"}</strong> (set in the
          Summary above).
        </p>
        {data?.summary_rows?.some((r) => isAcePa(r.hazard_code, r.metric)) ? (
          <p className="mb-3 text-xs text-fred-muted">{ACE_PA_INDICATIVE_NOTE}</p>
        ) : null}

        {/* Table */}
        {loading ? (
          <div className="space-y-2" aria-busy="true" aria-label="Loading scores">
            {[0, 1, 2, 3, 4, 5].map((i) => (
              <Skeleton key={i} className="h-7" />
            ))}
          </div>
        ) : isEmpty ? (
          <div className="rounded-lg border border-fred-border bg-fred-bg px-4 py-6 text-center text-sm text-fred-text">
            No scores yet{metric || track ? " for these filters" : ""}. A question is scored once one
            of its horizons resolves against the Resolver&apos;s ground truth, which happens on the 11th of
            the month after that horizon.
          </div>
        ) : (
          <div className="overflow-x-auto rounded-lg border border-fred-border">
            {view === "total" && (
              <SortableTable
                columns={modelColumns}
                rows={totalRows}
                rowKey={(row) => row.key}
                initialSortKey="avg_brier"
                initialSortDirection="asc"
                tableLayout="auto"
                emptyMessage="No scores available."
                pinnedOrder={pinActiveFirst}
                rowClassName={(row) =>
                  row.key.startsWith("__ext_")
                    ? "bg-amber-500/5 border-l-2 border-l-amber-400/40"
                    : row.status === "retired" || row.status === "superseded"
                      ? "text-fred-muted"
                      : undefined
                }
              />
            )}

            {view === "by_hazard" && (
              <SortableTable
                columns={PIVOTED_COLUMNS}
                rows={hazardRows}
                rowKey={(row) => row.key}
                initialSortKey="avg_brier"
                initialSortDirection="asc"
                tableLayout="auto"
                emptyMessage="No hazard-level scores available."
              />
            )}

            {view === "by_run" && (
              <SortableTable
                columns={RUN_COLUMNS}
                rows={runRows}
                rowKey={(row) => row.key}
                initialSortKey="run_date"
                initialSortDirection="desc"
                tableLayout="auto"
                emptyMessage="No per-run scores available."
              />
            )}

            {view === "by_model" && (
              <SortableTable
                columns={modelColumns}
                rows={modelRows}
                rowKey={(row) => row.key}
                initialSortKey="avg_brier"
                initialSortDirection="asc"
                tableLayout="auto"
                emptyMessage="No per-model scores available."
                pinnedOrder={pinActiveFirst}
                rowClassName={(row) =>
                  row.key.startsWith("__ext_")
                    ? "bg-amber-500/5 border-l-2 border-l-amber-400/40"
                    : row.status === "retired" || row.status === "superseded"
                      ? "text-fred-muted"
                      : undefined
                }
              />
            )}
          </div>
        )}
      </section>
        </>
      ) : null}
    </div>
  );
}
