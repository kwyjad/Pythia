"use client";

import { useMemo, useState } from "react";
import type { ReactNode } from "react";

import CollapsiblePanel from "../../../components/CollapsiblePanel";
import SpdPanel from "./SpdPanel";
import { asArray, asString, pickResearchJson } from "../../../lib/excerpts";
import { extractForecastRationale } from "../../../lib/forecast_rationale";
import { formatModelList } from "../../../lib/model_names";
import { formatScenario } from "../../../lib/scenario_format";
import type { QuestionBundleResponse } from "../../../lib/types";

type QuestionDetailViewProps = {
  bundle: QuestionBundleResponse;
};

type ModelRow = Record<string, unknown>;

const getTextList = (value: unknown): string[] => {
  if (Array.isArray(value)) {
    return value.map((item) => String(item));
  }
  if (value && typeof value === "object") {
    const obj = value as Record<string, unknown>;
    const values = Object.values(obj).flatMap((item) =>
      Array.isArray(item) ? item : []
    );
    if (values.length) {
      return values.map((item) => String(item));
    }
  }
  if (typeof value === "string" && value.trim().length > 0) {
    return [value];
  }
  return [];
};

const addMonths = (ym: string | null | undefined, months: number): string | null => {
  if (!ym) return null;
  const [year, month] = ym.split("-").map(Number);
  if (!year || !month) return null;
  const total = year * 12 + (month - 1) + months;
  const nextYear = Math.floor(total / 12);
  const nextMonth = (total % 12) + 1;
  return `${nextYear}-${String(nextMonth).padStart(2, "0")}`;
};

const resolveLatestDate = (rows: ModelRow[]): string | null => {
  const dates = rows
    .map((row) => row.created_at)
    .filter((value): value is string => typeof value === "string")
    .map((value) => new Date(value))
    .filter((value) => !Number.isNaN(value.getTime()));
  if (!dates.length) return null;
  const latest = dates.reduce((max, current) => (current > max ? current : max));
  return latest.toISOString().slice(0, 10);
};

const getModelNamesForPhase = (
  byPhase: Record<string, ModelRow[]>,
  phase: string
): string => {
  const rows = byPhase?.[phase] ?? [];
  const names: string[] = [];
  rows.forEach((row) => {
    const name =
      (row.model_id as string | undefined) ??
      (row.model_name as string | undefined) ??
      (row.model as string | undefined);
    if (name) {
      names.push(name);
    }
  });
  return formatModelList(names);
};

const getModelNamesForPhases = (
  byPhase: Record<string, ModelRow[]>,
  phases: string[]
): string => {
  const names: string[] = [];
  phases.forEach((phase) => {
    const rows = byPhase?.[phase] ?? [];
    rows.forEach((row) => {
      const name =
        (row.model_id as string | undefined) ??
        (row.model_name as string | undefined) ??
        (row.model as string | undefined);
      if (name) {
        names.push(name);
      }
    });
  });
  return formatModelList(names);
};

const renderScenarioBlocks = (text?: string | null) => {
  if (!text) {
    return <p className="text-sm text-slate-400">No scenario narrative available.</p>;
  }
  const blocks = formatScenario(text);
  if (!blocks.length) {
    return <p className="text-sm text-slate-400">No scenario narrative available.</p>;
  }
  const elements: ReactNode[] = [];
  const bulletBuffer: string[] = [];
  let listIndex = 0;

  const flushBullets = () => {
    if (!bulletBuffer.length) return;
    elements.push(
      <ul
        key={`scenario-list-${listIndex}`}
        className="list-disc space-y-1 pl-5 text-sm text-slate-200/90"
      >
        {bulletBuffer.map((item, index) => (
          <li key={`scenario-li-${listIndex}-${index}`}>{item}</li>
        ))}
      </ul>
    );
    bulletBuffer.length = 0;
    listIndex += 1;
  };

  blocks.forEach((block, index) => {
    if (block.type === "li") {
      bulletBuffer.push(block.text);
      return;
    }
    flushBullets();
    if (block.type === "h2") {
      elements.push(
        <h3 key={`scenario-h2-${index}`} className="text-base font-semibold text-slate-100">
          {block.text}
        </h3>
      );
      return;
    }
    if (block.type === "h3") {
      elements.push(
        <h4 key={`scenario-h3-${index}`} className="text-sm font-semibold text-slate-200">
          {block.text}
        </h4>
      );
      return;
    }
    elements.push(
      <p
        key={`scenario-p-${index}`}
        className="text-sm text-slate-200/90 leading-relaxed"
      >
        {block.text}
      </p>
    );
  });

  flushBullets();

  return <div className="space-y-2">{elements}</div>;
};

const formatTimeframe = (value: unknown): string | null => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return `${value} months`;
  }
  if (typeof value === "string" && value.trim().length > 0) {
    return value;
  }
  return null;
};

const formatConfidence = (value: unknown): string | null => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value.toFixed(2);
  }
  return asString(value);
};

const getSourceLabel = (url: string): string => {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    const cleaned = url.replace(/^https?:\/\//, "");
    return cleaned.split("/")[0] || url;
  }
};

const normalizeSources = (value: unknown): string[] => {
  if (Array.isArray(value)) {
    return value
      .map((entry) => asString(entry))
      .filter((entry): entry is string => Boolean(entry));
  }
  const single = asString(value);
  return single ? [single] : [];
};

const renderSupportText = (support: unknown, fields: { key: string; label: string }[]) => {
  if (!support) {
    return <p className="text-sm text-slate-200/90">—</p>;
  }
  if (typeof support === "string") {
    return <p className="text-sm text-slate-200/90">{support}</p>;
  }
  if (typeof support !== "object") {
    return <p className="text-sm text-slate-200/90">{String(support)}</p>;
  }
  const entries = fields
    .map(({ key, label }) => ({
      label,
      value: asString((support as Record<string, unknown>)[key]),
    }))
    .filter((entry) => entry.value);
  if (!entries.length) {
    return <p className="text-sm text-slate-200/90">—</p>;
  }
  return (
    <ul className="mt-2 space-y-1 text-sm text-slate-200/90">
      {entries.map((entry) => (
        <li key={entry.label}>
          <span className="text-slate-400">{entry.label}:</span> {entry.value}
        </li>
      ))}
    </ul>
  );
};

const renderExternalSupport = (support: unknown) => {
  if (!support) {
    return <p className="text-sm text-slate-200/90">—</p>;
  }
  if (typeof support === "string") {
    return <p className="text-sm text-slate-200/90">{support}</p>;
  }
  if (typeof support !== "object") {
    return <p className="text-sm text-slate-200/90">{String(support)}</p>;
  }
  const supportObj = support as Record<string, unknown>;
  const consensus = asString(supportObj.consensus);
  const dataQuality = asString(supportObj.data_quality);
  const analyses = asArray<unknown>(supportObj.recent_analyses)
    .map((entry) => asString(entry))
    .filter((entry): entry is string => Boolean(entry));
  return (
    <div className="space-y-2 text-sm text-slate-200/90">
      {consensus ? (
        <p>
          <span className="text-slate-400">Consensus:</span> {consensus}
        </p>
      ) : null}
      {dataQuality ? (
        <p>
          <span className="text-slate-400">Data quality:</span> {dataQuality}
        </p>
      ) : null}
      {analyses.length ? (
        <ul className="list-disc space-y-1 pl-5">
          {analyses.map((analysis, index) => (
            <li key={`analysis-${index}`}>{analysis}</li>
          ))}
        </ul>
      ) : null}
    </div>
  );
};

const renderParagraphs = (text: string) => {
  const blocks = text
    .split(/\n\s*\n/)
    .map((block) => block.trim())
    .filter(Boolean);
  return blocks.map((block, index) => (
    <p key={`paragraph-${index}`} className="text-sm text-slate-200/90 leading-relaxed">
      {block.replace(/\s+/g, " ")}
    </p>
  ));
};

const QuestionDetailView = ({ bundle }: QuestionDetailViewProps) => {
  const question = (bundle.question ?? {}) as Record<string, unknown>;
  const hs = (bundle.hs ?? {}) as Record<string, unknown>;
  const forecast = (bundle.forecast ?? {}) as Record<string, unknown>;
  const context = (bundle.context ?? {}) as Record<string, unknown>;
  const llm = (bundle.llm_calls ?? {}) as Record<string, unknown>;

  const ensemble = (forecast.ensemble_spd ?? []) as ModelRow[];
  const rawSpd = (forecast.raw_spd ?? []) as ModelRow[];
  const research = (forecast.research ?? {}) as Record<string, unknown>;
  const scenarioWriter = (forecast.scenario_writer ?? []) as ModelRow[];
  const triage = (hs.triage ?? {}) as Record<string, unknown>;
  const resolutions = (context.resolutions ?? []) as ModelRow[];
  const scores = (context.scores ?? []) as ModelRow[];
  const byPhase = (llm.by_phase ?? {}) as Record<string, ModelRow[]>;
  const researchJson = pickResearchJson(bundle);
  const forecastRationale = extractForecastRationale(bundle);

  const [scenarioIndex, setScenarioIndex] = useState(0);
  const selectedScenario = scenarioWriter[scenarioIndex] ?? scenarioWriter[0];

  const forecastDate =
    resolveLatestDate(ensemble) ?? resolveLatestDate(rawSpd) ?? "—";
  const targetMonth = (question.target_month as string | undefined) ?? null;
  const endMonth = addMonths(targetMonth, 5);
  const status = (question.status as string | undefined) ?? "—";

  const triageScore = triage.triage_score ?? "—";
  const triageTier = triage.tier ?? "—";
  const webSearchModels = getModelNamesForPhases(byPhase, [
    "hs_web_research",
    "research_web_research",
    "forecast_web_research",
  ]);

  const resolutionForTarget = resolutions.find(
    (row) => row.observed_month === targetMonth
  );
  const resolutionSummary = resolutions.length
    ? `${resolutions.length} row(s)${
        resolutionForTarget?.value !== undefined
          ? ` • target month: ${resolutionForTarget.value}`
          : ""
      }`
    : "No resolutions yet";

  const baseRate = (researchJson?.base_rate ?? null) as Record<string, unknown> | null;
  const updateSignals = asArray<Record<string, unknown>>(researchJson?.update_signals);
  const regimeShiftSignals = asArray<Record<string, unknown>>(
    researchJson?.regime_shift_signals
  );
  const dataGaps = asArray<unknown>(researchJson?.data_gaps)
    .map((entry) => asString(entry))
    .filter((entry): entry is string => Boolean(entry));
  const researchSources = asArray<unknown>(researchJson?.sources)
    .map((entry) => asString(entry))
    .filter((entry): entry is string => Boolean(entry));

  const scenarioStubText = asString(triage.scenario_stub);
  const regimeShiftRows = asArray<Record<string, unknown>>(triage.regime_shifts_json);
  const dataQuality = (triage.data_quality_json ?? null) as Record<string, unknown> | null;
  const dataQualityEntries = [
    { label: "Source", value: asString(dataQuality?.resolution_source) },
    { label: "Reliability", value: asString(dataQuality?.reliability) },
    { label: "Notes", value: asString(dataQuality?.notes) },
  ].filter((entry) => entry.value);
  const selectedScenarioText = asString(selectedScenario?.text);

  const brierScores = useMemo(() => {
    return scores.filter((row) => {
      const scoreType = row.score_type as string | undefined;
      const modelName = row.model_name as string | undefined;
      return scoreType === "brier" && !modelName;
    });
  }, [scores]);

  const brierAverage = useMemo(() => {
    if (!brierScores.length) return null;
    const values = brierScores
      .map((row) => row.value)
      .filter((value): value is number => typeof value === "number");
    if (!values.length) return null;
    const total = values.reduce((sum, value) => sum + value, 0);
    return total / values.length;
  }, [brierScores]);

  const brierByHorizon = useMemo(() => {
    const map = new Map<number, number>();
    brierScores.forEach((row) => {
      const horizon = row.horizon_m as number | undefined;
      const value = row.value as number | undefined;
      if (typeof horizon === "number" && typeof value === "number") {
        map.set(horizon, value);
      }
    });
    return map;
  }, [brierScores]);

  return (
    <div className="space-y-8">
      <section className="space-y-2">
        <h1 className="text-2xl font-semibold text-white">
          {(question.wording as string | undefined) ?? "Question detail"}
        </h1>
        <p className="text-sm text-slate-400">
          {(question.iso3 as string | undefined) ?? ""} •{" "}
          {(question.hazard_code as string | undefined) ?? ""} •{" "}
          {(question.metric as string | undefined) ?? ""} •{" "}
          {(question.target_month as string | undefined) ?? ""}
        </p>
      </section>

      <section className="rounded-lg border border-slate-800 bg-slate-900/60 p-4">
        <h2 className="text-lg font-semibold text-white">Summary</h2>
        <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          {[
            { label: "Status", value: status },
            { label: "Forecast date", value: forecastDate },
            {
              label: "Forecast window",
              value: `${targetMonth ?? "—"} → ${endMonth ?? "—"}`,
            },
            { label: "Triage score", value: triageScore },
            { label: "Triage tier", value: triageTier },
            { label: "Web search", value: webSearchModels },
            { label: "HS triage", value: getModelNamesForPhase(byPhase, "hs_triage") },
            { label: "Research", value: getModelNamesForPhase(byPhase, "research_v2") },
            { label: "SPD", value: getModelNamesForPhase(byPhase, "spd_v2") },
            { label: "Scenario", value: getModelNamesForPhase(byPhase, "scenario_v2") },
          ].map((item) => (
            <div
              key={item.label}
              className="rounded border border-slate-800 bg-slate-950 px-3 py-2"
            >
              <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">
                {item.label}
              </div>
              <div className="mt-1 text-sm text-slate-200">{item.value}</div>
            </div>
          ))}
        </div>
        <div className="mt-4 grid gap-4 md:grid-cols-2">
          <div className="rounded border border-slate-800 bg-slate-950 px-3 py-3">
            <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">
              Resolutions
            </div>
            <div className="mt-2 text-sm text-slate-200">{resolutionSummary}</div>
          </div>
          <div className="rounded border border-slate-800 bg-slate-950 px-3 py-3">
            <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">
              Brier
            </div>
            <div className="mt-2 space-y-2">
              {brierAverage !== null ? (
                <div className="text-sm text-slate-200">
                  Avg Brier (resolved months): {brierAverage.toFixed(3)}
                </div>
              ) : (
                <div className="text-sm text-slate-400">No Brier yet</div>
              )}
              <div className="grid grid-cols-3 gap-2 text-xs text-slate-200 md:grid-cols-6">
                {[1, 2, 3, 4, 5, 6].map((horizon) => (
                  <div
                    key={`brier-${horizon}`}
                    className="rounded border border-slate-800 bg-slate-950 px-2 py-1"
                  >
                    <div className="text-[11px] text-slate-400">
                      Month {horizon}
                    </div>
                    <div>
                      {brierByHorizon.has(horizon)
                        ? brierByHorizon.get(horizon)?.toFixed(3)
                        : "—"}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      <SpdPanel bundle={bundle} />

      <section className="space-y-4">
        <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-4">
          <h2 className="text-lg font-semibold text-white">Forecast Rationale</h2>
          <div className="mt-3 space-y-2">
            {forecastRationale ? (
              renderParagraphs(forecastRationale)
            ) : (
              <p className="text-sm text-slate-400">
                No forecast rationale available for this forecast.
              </p>
            )}
          </div>
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-4">
          <h2 className="text-lg font-semibold text-white">Scenario</h2>
          {scenarioWriter.length > 1 ? (
            <div className="mt-3 flex flex-col gap-2 md:flex-row md:items-center">
              <label className="text-sm text-slate-400">Scenario:</label>
              <select
                className="rounded border border-slate-800 bg-slate-950 px-3 py-2 text-sm text-slate-200"
                value={scenarioIndex}
                onChange={(event) => setScenarioIndex(Number(event.target.value))}
              >
                {scenarioWriter.map((row, index) => (
                  <option key={`scenario-${index}`} value={index}>
                    {(row.scenario_type as string | undefined) ??
                      `Scenario ${index + 1}`}
                  </option>
                ))}
              </select>
            </div>
          ) : null}
          <div className="mt-4">{renderScenarioBlocks(selectedScenarioText)}</div>
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-4">
          <h2 className="text-lg font-semibold text-white">Horizon Scan Triage</h2>
          <div className="mt-3 space-y-3">
            <div>
              <h3 className="text-sm font-semibold text-slate-200">Drivers</h3>
              {getTextList(triage.drivers_json).length ? (
                <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-200/90">
                  {getTextList(triage.drivers_json).map((item, index) => (
                    <li key={`driver-${index}`}>{item}</li>
                  ))}
                </ul>
              ) : (
                <p className="mt-2 text-sm text-slate-400">No drivers captured.</p>
              )}
            </div>
            <div>
              <h3 className="text-sm font-semibold text-slate-200">Regime shifts</h3>
              {regimeShiftRows.length ? (
                <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-200/90">
                  {regimeShiftRows.map((item, index) => {
                    const isShiftObject = typeof item === "object" && item !== null;
                    const shift = !isShiftObject
                      ? String(item)
                      : asString(item.shift) ?? asString(item.regime_shift) ?? "—";
                    const likelihood = isShiftObject ? asString(item.likelihood) : null;
                    const timeframe = isShiftObject
                      ? formatTimeframe(item.timeframe) ??
                        formatTimeframe(item.timeframe_months)
                      : null;
                    const meta = [
                      likelihood ? `Likelihood: ${likelihood}` : null,
                      timeframe ? `Timeframe: ${timeframe}` : null,
                    ]
                      .filter(Boolean)
                      .join("; ");
                    return (
                      <li key={`regime-${index}`}>
                        <span className="font-medium text-slate-200">{shift}</span>
                        {meta ? <span className="text-slate-400"> — {meta}</span> : null}
                      </li>
                    );
                  })}
                </ul>
              ) : (
                <p className="mt-2 text-sm text-slate-400">No regime shifts noted.</p>
              )}
            </div>
            <div>
              <h3 className="text-sm font-semibold text-slate-200">Data quality</h3>
              {dataQualityEntries.length ? (
                <div className="mt-2 rounded border border-slate-800 bg-slate-950/30 p-3 text-sm text-slate-200/90">
                  {dataQualityEntries.map((entry) => (
                    <p key={entry.label}>
                      <span className="text-slate-400">{entry.label}:</span>{" "}
                      {entry.value}
                    </p>
                  ))}
                </div>
              ) : (
                <p className="mt-2 text-sm text-slate-400">No data quality notes.</p>
              )}
            </div>
            {scenarioStubText ? (
              <div>
                <h3 className="text-sm font-semibold text-slate-200">Scenario stub</h3>
                <div className="mt-2">{renderScenarioBlocks(scenarioStubText)}</div>
              </div>
            ) : null}
          </div>
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-4">
          <h2 className="text-lg font-semibold text-white">Research</h2>
          {!baseRate ? (
            <p className="mt-3 text-sm text-slate-200/90">
              No research narrative available for this forecast.
            </p>
          ) : (
            <div className="mt-3 space-y-4">
              <div>
                <h3 className="text-sm font-semibold text-slate-200">
                  Qualitative summary
                </h3>
                <p className="mt-2 text-sm text-slate-200/90 leading-relaxed">
                  {asString(baseRate.qualitative_summary) ??
                    "No qualitative summary available."}
                </p>
              </div>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="rounded border border-slate-800 bg-slate-950/30 p-3">
                  <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                    Resolver support
                  </h4>
                  {renderSupportText(baseRate.resolver_support, [
                    { key: "recent_level", label: "Recent level" },
                    { key: "trend", label: "Trend" },
                    { key: "data_quality", label: "Data quality" },
                    { key: "notes", label: "Notes" },
                  ])}
                </div>
                <div className="rounded border border-slate-800 bg-slate-950/30 p-3">
                  <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                    External support
                  </h4>
                  {renderExternalSupport(baseRate.external_support)}
                </div>
              </div>
              <div>
                <h3 className="text-sm font-semibold text-slate-200">
                  Update signals
                </h3>
                {updateSignals.length ? (
                  <ul className="mt-2 list-disc space-y-2 pl-5 text-sm text-slate-200/90">
                    {updateSignals.map((signal, index) => {
                      const isSignalObject = typeof signal === "object" && signal !== null;
                      const description = !isSignalObject
                        ? String(signal)
                        : asString(signal.description) ?? asString(signal.signal) ?? "—";
                      const direction = isSignalObject ? asString(signal.direction) : null;
                      const confidence = isSignalObject
                        ? formatConfidence(signal.confidence)
                        : null;
                      const timeframe = isSignalObject
                        ? formatTimeframe(signal.timeframe) ??
                          formatTimeframe(signal.timeframe_months)
                        : null;
                      const sources = isSignalObject
                        ? normalizeSources(
                            signal.sources ?? signal.source_urls ?? signal.source
                          )
                        : [];
                      const meta = [
                        direction ? `Direction: ${direction}` : null,
                        confidence ? `Confidence: ${confidence}` : null,
                        timeframe ? `Timeframe: ${timeframe}` : null,
                      ]
                        .filter(Boolean)
                        .join(" • ");
                      return (
                        <li key={`update-${index}`} className="space-y-1">
                          <div>{description}</div>
                          {meta ? <div className="text-xs text-slate-400">{meta}</div> : null}
                          {sources.length ? (
                            <div className="flex flex-wrap gap-2 text-xs">
                              {sources.map((source) => (
                                <a
                                  key={source}
                                  href={source}
                                  className="text-slate-300 underline underline-offset-2"
                                  target="_blank"
                                  rel="noreferrer"
                                >
                                  {getSourceLabel(source)}
                                </a>
                              ))}
                            </div>
                          ) : null}
                        </li>
                      );
                    })}
                  </ul>
                ) : (
                  <p className="mt-2 text-sm text-slate-400">No update signals.</p>
                )}
              </div>
              <div>
                <h3 className="text-sm font-semibold text-slate-200">
                  Regime shift signals
                </h3>
                {regimeShiftSignals.length ? (
                  <ul className="mt-2 list-disc space-y-2 pl-5 text-sm text-slate-200/90">
                    {regimeShiftSignals.map((signal, index) => {
                      const isSignalObject = typeof signal === "object" && signal !== null;
                      const description = !isSignalObject
                        ? String(signal)
                        : asString(signal.description) ?? asString(signal.signal) ?? "—";
                      const likelihood = isSignalObject ? asString(signal.likelihood) : null;
                      const timeframe = isSignalObject
                        ? formatTimeframe(signal.timeframe) ??
                          formatTimeframe(signal.timeframe_months)
                        : null;
                      const sources = isSignalObject
                        ? normalizeSources(
                            signal.sources ?? signal.source_urls ?? signal.source
                          )
                        : [];
                      const meta = [
                        likelihood ? `Likelihood: ${likelihood}` : null,
                        timeframe ? `Timeframe: ${timeframe}` : null,
                      ]
                        .filter(Boolean)
                        .join(" • ");
                      return (
                        <li key={`regime-signal-${index}`} className="space-y-1">
                          <div>{description}</div>
                          {meta ? <div className="text-xs text-slate-400">{meta}</div> : null}
                          {sources.length ? (
                            <div className="flex flex-wrap gap-2 text-xs">
                              {sources.map((source) => (
                                <a
                                  key={source}
                                  href={source}
                                  className="text-slate-300 underline underline-offset-2"
                                  target="_blank"
                                  rel="noreferrer"
                                >
                                  {getSourceLabel(source)}
                                </a>
                              ))}
                            </div>
                          ) : null}
                        </li>
                      );
                    })}
                  </ul>
                ) : (
                  <p className="mt-2 text-sm text-slate-400">No regime shift signals.</p>
                )}
              </div>
              <div>
                <h3 className="text-sm font-semibold text-slate-200">Data gaps</h3>
                {dataGaps.length ? (
                  <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-200/90">
                    {dataGaps.map((item, index) => (
                      <li key={`data-gap-${index}`}>{item}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="mt-2 text-sm text-slate-400">No data gaps noted.</p>
                )}
              </div>
              {researchSources.length ? (
                <div>
                  <h3 className="text-sm font-semibold text-slate-200">Sources</h3>
                  <div className="mt-2 flex flex-wrap gap-2 text-xs">
                    {researchSources.map((source) => (
                      <a
                        key={source}
                        href={source}
                        className="text-slate-300 underline underline-offset-2"
                        target="_blank"
                        rel="noreferrer"
                      >
                        {getSourceLabel(source)}
                      </a>
                    ))}
                  </div>
                </div>
              ) : null}
            </div>
          )}
        </div>
      </section>

      <section className="space-y-3">
        <CollapsiblePanel title="Raw HS bundle">
          <pre className="whitespace-pre-wrap text-xs text-slate-200">
            {JSON.stringify(hs ?? null, null, 2)}
          </pre>
        </CollapsiblePanel>
        <CollapsiblePanel title="Raw Research bundle">
          <pre className="whitespace-pre-wrap text-xs text-slate-200">
            {JSON.stringify(research ?? null, null, 2)}
          </pre>
        </CollapsiblePanel>
        <CollapsiblePanel title="Raw Forecast SPD rows">
          <pre className="whitespace-pre-wrap text-xs text-slate-200">
            {JSON.stringify({ ensemble_spd: ensemble, raw_spd: rawSpd }, null, 2)}
          </pre>
        </CollapsiblePanel>
        <CollapsiblePanel title="Raw Scenario writer output">
          <pre className="whitespace-pre-wrap text-xs text-slate-200">
            {JSON.stringify(scenarioWriter ?? null, null, 2)}
          </pre>
        </CollapsiblePanel>
        <CollapsiblePanel title="Raw LLM calls (no transcripts)">
          <pre className="whitespace-pre-wrap text-xs text-slate-200">
            {JSON.stringify(llm ?? null, null, 2)}
          </pre>
        </CollapsiblePanel>
      </section>
    </div>
  );
};

export default QuestionDetailView;
