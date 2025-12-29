"use client";

import { useMemo, useState } from "react";

import CollapsiblePanel from "../../../components/CollapsiblePanel";
import SpdPanel from "./SpdPanel";
import { formatModelList } from "../../../lib/model_names";
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

const renderScenarioText = (text?: string | null) => {
  if (!text) {
    return <p className="text-sm text-slate-400">No scenario narrative available.</p>;
  }
  const blocks = text.split(/\n\s*\n/).map((block) => block.trim()).filter(Boolean);
  return blocks.map((block, index) => {
    const lines = block.split("\n").map((line) => line.trim()).filter(Boolean);
    const bulletLines = lines.filter((line) => line.startsWith("-") || line.startsWith("•"));
    if (bulletLines.length === lines.length) {
      return (
        <ul key={`block-${index}`} className="list-disc space-y-1 pl-5 text-sm text-slate-200">
          {bulletLines.map((line, lineIndex) => (
            <li key={`bullet-${index}-${lineIndex}`}>
              {line.replace(/^[-•]\s?/, "")}
            </li>
          ))}
        </ul>
      );
    }
    return (
      <p key={`block-${index}`} className="text-sm text-slate-200">
        {lines.join(" ")}
      </p>
    );
  });
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

  const baseRate = (research.base_rate ?? {}) as Record<string, unknown>;
  const updateSignals = getTextList(research.update_signals);
  const regimeShiftSignals = getTextList(research.regime_shift_signals);
  const dataGaps = getTextList(research.data_gaps);

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
          <h2 className="text-lg font-semibold text-white">HS triage excerpt</h2>
          <div className="mt-3 space-y-3">
            <div>
              <h3 className="text-sm font-semibold text-slate-300">Drivers</h3>
              {getTextList(triage.drivers_json).length ? (
                <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-200">
                  {getTextList(triage.drivers_json).map((item, index) => (
                    <li key={`driver-${index}`}>{item}</li>
                  ))}
                </ul>
              ) : (
                <p className="mt-2 text-sm text-slate-400">No drivers captured.</p>
              )}
            </div>
            <div>
              <h3 className="text-sm font-semibold text-slate-300">Regime shifts</h3>
              {getTextList(triage.regime_shifts_json).length ? (
                <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-200">
                  {getTextList(triage.regime_shifts_json).map((item, index) => (
                    <li key={`regime-${index}`}>{item}</li>
                  ))}
                </ul>
              ) : (
                <p className="mt-2 text-sm text-slate-400">No regime shifts noted.</p>
              )}
            </div>
            <div>
              <h3 className="text-sm font-semibold text-slate-300">Data quality</h3>
              {getTextList(triage.data_quality_json).length ? (
                <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-200">
                  {getTextList(triage.data_quality_json).map((item, index) => (
                    <li key={`data-quality-${index}`}>{item}</li>
                  ))}
                </ul>
              ) : (
                <p className="mt-2 text-sm text-slate-400">No data quality notes.</p>
              )}
            </div>
            {triage.scenario_stub ? (
              <div>
                <h3 className="text-sm font-semibold text-slate-300">Scenario stub</h3>
                <p className="mt-2 text-sm text-slate-200">
                  {String(triage.scenario_stub)}
                </p>
              </div>
            ) : null}
          </div>
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-4">
          <h2 className="text-lg font-semibold text-white">Research excerpt</h2>
          <div className="mt-3 space-y-4">
            <div>
              <h3 className="text-sm font-semibold text-slate-300">
                Qualitative summary
              </h3>
              <p className="mt-2 text-sm text-slate-200">
                {(baseRate.qualitative_summary as string | undefined) ??
                  "No qualitative summary available."}
              </p>
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="rounded border border-slate-800 bg-slate-950 p-3">
                <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                  Resolver support
                </h4>
                <p className="mt-2 text-sm text-slate-200">
                  {(baseRate.resolver_support as string | undefined) ?? "—"}
                </p>
              </div>
              <div className="rounded border border-slate-800 bg-slate-950 p-3">
                <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                  External support
                </h4>
                <p className="mt-2 text-sm text-slate-200">
                  {(baseRate.external_support as string | undefined) ?? "—"}
                </p>
              </div>
            </div>
            <div>
              <h3 className="text-sm font-semibold text-slate-300">Update signals</h3>
              {updateSignals.length ? (
                <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-200">
                  {updateSignals.map((item, index) => (
                    <li key={`update-${index}`}>{item}</li>
                  ))}
                </ul>
              ) : (
                <p className="mt-2 text-sm text-slate-400">No update signals.</p>
              )}
            </div>
            <div>
              <h3 className="text-sm font-semibold text-slate-300">
                Regime shift signals
              </h3>
              {regimeShiftSignals.length ? (
                <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-200">
                  {regimeShiftSignals.map((item, index) => (
                    <li key={`regime-signal-${index}`}>{item}</li>
                  ))}
                </ul>
              ) : (
                <p className="mt-2 text-sm text-slate-400">No regime shift signals.</p>
              )}
            </div>
            <div>
              <h3 className="text-sm font-semibold text-slate-300">Data gaps</h3>
              {dataGaps.length ? (
                <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-200">
                  {dataGaps.map((item, index) => (
                    <li key={`data-gap-${index}`}>{item}</li>
                  ))}
                </ul>
              ) : (
                <p className="mt-2 text-sm text-slate-400">No data gaps noted.</p>
              )}
            </div>
          </div>
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-4">
          <h2 className="text-lg font-semibold text-white">Scenario excerpt</h2>
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
          <div className="mt-4 space-y-3">
            {renderScenarioText((selectedScenario?.text as string | undefined) ?? null)}
          </div>
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
