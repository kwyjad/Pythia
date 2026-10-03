"use client";

import { Fragment, useCallback, useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import InfoTooltip from "../../components/InfoTooltip";
import { apiGet } from "../../lib/api";
import { safeHref } from "../../lib/safe_href";
import SibylMetricsAbout from "./SibylMetricsAbout";
import SibylCalibration from "./SibylCalibration";
import type {
  SibylQuestionDetailResponse,
  SibylQuestionRow,
  SibylQuestionsResponse,
  SibylRun,
  SibylShadowComparison,
  SibylShadowStat,
  SibylRunsResponse,
  SibylSummaryResponse,
  SibylTrial,
} from "../../lib/types";

const pct = new Intl.NumberFormat(undefined, {
  style: "percent",
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

const usd = new Intl.NumberFormat(undefined, {
  style: "currency",
  currency: "USD",
  minimumFractionDigits: 2,
});

const num = new Intl.NumberFormat();

const fmtJsd = (v: number | null | undefined) =>
  v === null || v === undefined ? "—" : v.toFixed(4);

type SortKey =
  | "js_divergence_vs_standard"
  | "js_divergence_inter_trial"
  | "volatility_score"
  | "cost_usd";

const QUANTILE_ORDER = ["0.1", "0.25", "0.5", "0.75", "0.9", "0.95", "0.99"];

// Concise tooltips for the per-question table headers. The "About these
// metrics" tab carries the full technical + plain-language explanations.
const TOOLTIP_JSD_STANDARD =
  "Jensen–Shannon divergence between Sibyl's pooled distribution and the " +
  "standard pipeline's aggregate SPD (averaged over the 6 window months). " +
  "A disagreement signal, not an accuracy score. 0 = identical; higher " +
  "(toward ~0.69) = the two methods disagree. See the About these metrics tab.";
const TOOLTIP_JSD_INTER_TRIAL =
  "Mean pairwise Jensen–Shannon divergence across Sibyl's K independent " +
  "research trials for this question. Low = trials converged (robust); high " +
  "= trials scattered (ambiguous/thin evidence). 0 → ~0.69.";
const TOOLTIP_VOLATILITY =
  "The Horizon Scanner regime-change score (likelihood × magnitude of a " +
  "departure from base rate) that selected this question for deep research. " +
  "Higher = more turbulent situation. A prioritization value, not a " +
  "performance score. Range ~0–1.";

const QuantileRow = ({
  label,
  quantiles,
}: {
  label: string;
  quantiles: Record<string, number> | null | undefined;
}) => (
  <tr className="border-t border-fred-secondary/40">
    <td className="px-2 py-1 text-left text-xs font-medium text-fred-text">{label}</td>
    {QUANTILE_ORDER.map((q) => (
      <td key={q} className="px-2 py-1 text-right text-xs text-fred-text">
        {quantiles && quantiles[q] !== undefined ? num.format(Math.round(quantiles[q])) : "—"}
      </td>
    ))}
  </tr>
);

const OverlayChart = ({
  labels,
  sibyl,
  standard,
  standardName,
  reference = null,
}: {
  labels: string[];
  sibyl: number[];
  standard: number[] | null;
  standardName: string | null;
  reference?: number[] | null;
}) => {
  const maxProb = Math.max(0.0001, ...sibyl, ...(standard ?? []), ...(reference ?? []));
  return (
    <div className="rounded-lg border border-fred-secondary bg-fred-surface px-4 py-4">
      <div className="mb-2 flex items-center gap-4 text-xs text-fred-text">
        <span className="flex items-center gap-1">
          <span className="inline-block h-3 w-3 rounded-sm bg-indigo-500" /> Sibyl (pooled)
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-3 w-3 rounded-sm bg-fred-primary" />{" "}
          {standardName ? `Standard (${standardName})` : "Standard (none)"}
        </span>
        {reference ? (
          <span className="flex items-center gap-1">
            <span className="inline-block h-3 w-3 rounded-sm border-2 border-amber-500" />{" "}
            Sibyl&apos;s reference (prior)
          </span>
        ) : null}
      </div>
      <div className="flex h-56 items-end gap-3">
        {labels.map((label, i) => {
          const sp = sibyl[i] ?? 0;
          const st = standard ? standard[i] ?? 0 : 0;
          const rf = reference ? reference[i] ?? 0 : null;
          return (
            <div key={label} className="flex flex-1 flex-col items-center gap-1">
              <div className="relative flex h-40 w-full items-end justify-center gap-1">
                {rf !== null ? (
                  <div
                    className="pointer-events-none absolute inset-x-0 border-t-2 border-amber-500"
                    style={{ bottom: `${(rf / maxProb) * 100}%` }}
                    title={`Reference: ${pct.format(rf)}`}
                  />
                ) : null}
                <div
                  className="w-1/2 rounded-t-sm bg-indigo-500"
                  style={{ height: `${(sp / maxProb) * 100}%` }}
                  title={`Sibyl: ${pct.format(sp)}`}
                />
                <div
                  className="w-1/2 rounded-t-sm bg-fred-primary"
                  style={{ height: `${(st / maxProb) * 100}%` }}
                  title={`Standard: ${pct.format(st)}`}
                />
              </div>
              <div className="text-[10px] text-fred-text">{label}</div>
              <div className="text-[10px] text-fred-secondary">
                {pct.format(sp)} / {standard ? pct.format(st) : "—"}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

const TrialCard = ({ trial }: { trial: SibylTrial }) => {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-lg border border-fred-secondary bg-fred-surface p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-sm font-semibold text-fred-primary">
          Trial {trial.trial_index + 1}
          {trial.error ? (
            <span className="ml-2 rounded bg-red-100 px-2 py-0.5 text-xs text-red-700">
              {trial.error}
            </span>
          ) : null}
        </div>
        <div className="text-xs text-fred-muted">
          {trial.steps_used ?? 0} steps · {trial.submitted ? "submitted" : "step limit"} ·{" "}
          confidence: {trial.confidence ?? "?"} ·{" "}
          {trial.cost?.total_usd !== undefined ? usd.format(trial.cost.total_usd) : "—"}
        </div>
      </div>
      {trial.perspective ? (
        <p className="mt-1 text-xs italic text-fred-muted">{trial.perspective}</p>
      ) : null}
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="mt-2 text-xs font-medium text-fred-primary underline"
      >
        {open ? "Hide belief-state trace" : "Show belief-state trace"}
      </button>
      {open ? (
        <div className="mt-2 space-y-2">
          {(trial.belief_trace ?? []).map((step) => (
            <div
              key={step.step}
              className="rounded border border-fred-secondary/40 bg-fred-bg p-2 text-xs"
            >
              <div className="font-medium text-fred-primary">
                Step {step.step}: {step.action}
                {step.action_input ? (
                  <span className="ml-1 break-all font-normal text-fred-muted">
                    ({step.action_input})
                  </span>
                ) : null}
                {step.tool_ok === false ? (
                  <span className="ml-2 text-red-600">tool failed</span>
                ) : null}
                {step.repaired ? (
                  <span className="ml-2 text-amber-600">quantiles repaired</span>
                ) : null}
              </div>
              {step.belief?.step_rationale ? (
                <p className="mt-1 text-fred-text">{step.belief.step_rationale}</p>
              ) : null}
              {step.belief?.baserate_reconciliation ? (
                <p className="mt-1 text-fred-muted">
                  <span className="font-medium">Base-rate reconciliation:</span>{" "}
                  {step.belief.baserate_reconciliation}
                </p>
              ) : null}
              {step.belief?.quantiles ? (
                <p className="mt-1 text-fred-muted">
                  p10 {num.format(Math.round(step.belief.quantiles["0.1"] ?? 0))} · p50{" "}
                  {num.format(Math.round(step.belief.quantiles["0.5"] ?? 0))} · p90{" "}
                  {num.format(Math.round(step.belief.quantiles["0.9"] ?? 0))} · p99{" "}
                  {num.format(Math.round(step.belief.quantiles["0.99"] ?? 0))}
                </p>
              ) : null}
            </div>
          ))}
          {(trial.evidence_higher?.length || trial.evidence_lower?.length) ? (
            <div className="grid gap-2 md:grid-cols-2">
              <div>
                <div className="text-xs font-semibold text-fred-primary">Evidence higher</div>
                <ul className="list-disc pl-4 text-xs text-fred-text">
                  {(trial.evidence_higher ?? []).map((e, i) => (
                    <li key={i}>{e}</li>
                  ))}
                </ul>
              </div>
              <div>
                <div className="text-xs font-semibold text-fred-primary">Evidence lower</div>
                <ul className="list-disc pl-4 text-xs text-fred-text">
                  {(trial.evidence_lower ?? []).map((e, i) => (
                    <li key={i}>{e}</li>
                  ))}
                </ul>
              </div>
            </div>
          ) : null}
          {trial.source_urls?.length ? (
            <div>
              <div className="text-xs font-semibold text-fred-primary">Sources consulted</div>
              <ul className="list-disc pl-4 text-xs">
                {trial.source_urls.slice(0, 12).map((u) => {
                  // Source URLs are written by the model; only a web address
                  // becomes a link.
                  const href = safeHref(u);
                  return (
                    <li key={u}>
                      {href ? (
                        <a
                          className="break-all text-fred-primary underline"
                          href={href}
                          target="_blank"
                          rel="noreferrer noopener"
                        >
                          {u}
                        </a>
                      ) : (
                        <span className="break-all">{u}</span>
                      )}
                    </li>
                  );
                })}
              </ul>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
};

const fmtShare = (v: number | null | undefined) =>
  v === null || v === undefined ? "—" : pct.format(v);

const SHADOW_STATUS_TEXT: Record<string, string> = {
  on: "ran",
  off: "switched off",
  backtest: "not run in backtest",
  expired: "ended (past its last month)",
  no_key: "skipped: the OpenAI key was missing",
  no_shadow_call: "not run (test run)",
  unknown_model: "skipped: the model did not resolve",
  unsupported_provider: "skipped: provider not wired",
};

const fmtDiff = (st: SibylShadowStat | undefined) => {
  if (!st) return "—";
  if (st.status !== "ok" || st.mean_diff === null || st.mean_diff === undefined) {
    return `not yet (${st.n_questions} of ${st.min_questions ?? 20} questions)`;
  }
  const f = (v: number | null | undefined) => (v === null || v === undefined ? "?" : v.toFixed(4));
  return `${f(st.mean_diff)} [${f(st.lo)}, ${f(st.hi)}] · ${st.n_questions} questions`;
};

// The shadow arm (sibyl/shadow.py): a second model family's lane C trial,
// scored apart and never published. A finding; nothing here switches it on.
export const ShadowArm = ({
  run,
  shadow,
}: {
  run: SibylRun;
  shadow: SibylShadowComparison | null | undefined;
}) => {
  if (!run.shadow_status && !shadow?.model) return null;
  const status = run.shadow_status
    ? SHADOW_STATUS_TEXT[run.shadow_status] ?? run.shadow_status
    : "—";
  return (
    <div className="rounded-lg border border-fred-secondary bg-fred-surface p-4">
      <div className="flex items-center gap-1 text-xs uppercase text-fred-muted">
        Shadow arm ({run.shadow_model ?? shadow?.model ?? "second model"})
        <InfoTooltip text="One extra research trial per question on a second model family. Its pool (with Claude's lane C trial swapped for it) is scored apart and never published. Differences are shadow minus Sibyl; negative means the shadow arm scored better. 90% intervals resample questions." />
      </div>
      <div className="mt-1 text-sm text-fred-text">
        This run: {status}
        {run.shadow_status === "on"
          ? ` · ${run.n_shadow_trials ?? 0} trials, ${run.n_shadow_series ?? 0} series, ${
              run.n_shadow_skipped ?? 0
            } skipped near the caps · ${usd.format(run.shadow_cost_usd ?? 0)}`
          : ""}
      </div>
      {shadow ? (
        <div className="mt-2 grid gap-1 text-xs text-fred-text sm:grid-cols-2">
          <div>
            <span className="text-fred-muted">Pool, Δ Brier: </span>
            {fmtDiff(shadow.series?.brier)}
          </div>
          <div>
            <span className="text-fred-muted">Single trial vs Claude lane C, Δ Brier: </span>
            {fmtDiff(shadow.trial?.brier)}
          </div>
        </div>
      ) : null}
    </div>
  );
};

// How Sibyl researched in this run (sibyl/measure.py). Never a score: the
// About tab says what each one is for.
export const ProcessMeasures = ({ run }: { run: SibylRun }) => {
  const items: { label: string; value: string; tip: string }[] = [
    {
      label: "Resolver slot done",
      value: fmtShare(run.share_resolver_done),
      tip: "Share of trials that found how the question resolves before submitting.",
    },
    {
      label: "Documents per trial",
      value:
        run.docs_per_trial === null || run.docs_per_trial === undefined
          ? "—"
          : run.docs_per_trial.toFixed(1),
      tip: "Documents (pages and PDFs) read per research trial.",
    },
    {
      label: "Dated figures in ledger",
      value: fmtShare(run.share_ledger_dated_figure),
      tip: "Share of evidence-ledger items carrying both a date and a figure.",
    },
    {
      label: "Forecasts at the floor",
      value: fmtShare(run.share_forecasts_at_floor),
      tip: "Share of forecasts with a bucket at the minimum probability Sibyl allows.",
    },
    {
      label: "Divergence from reference",
      value: fmtJsd(run.mean_jsd_from_reference),
      tip: "Mean month-1 Jensen–Shannon divergence of the trials' pool from Sibyl's reference.",
    },
    {
      label: "Reference weight",
      value:
        run.reference_weight === null || run.reference_weight === undefined
          ? "—"
          : `${run.reference_weight.toFixed(2)} (${run.reference_weight_source ?? "?"})`,
      tip: "The reference's share of the published forecast: fitted from scores, or fixed.",
    },
  ];
  return (
    <div className="rounded-lg border border-fred-secondary bg-fred-surface p-4">
      <div className="text-xs uppercase text-fred-muted">How this run researched</div>
      <div className="mt-2 grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {items.map((it) => (
          <div key={it.label}>
            <div className="flex items-center gap-1 text-[11px] text-fred-muted">
              {it.label}
              <InfoTooltip text={it.tip} />
            </div>
            <div className="text-sm font-semibold text-fred-text">{it.value}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

const QuestionDetail = ({
  questionId,
  sibylRunId,
  includeTest,
}: {
  questionId: string;
  sibylRunId: string | null;
  includeTest: boolean;
}) => {
  const [detail, setDetail] = useState<SibylQuestionDetailResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [month, setMonth] = useState(1);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiGet<SibylQuestionDetailResponse>("/sibyl/question_detail", {
        question_id: questionId,
        sibyl_run_id: sibylRunId ?? undefined,
        include_test: includeTest || undefined,
      });
      setDetail(data);
    } catch (err) {
      setError("Failed to load question detail.");
      console.warn(err);
    } finally {
      setLoading(false);
    }
  }, [questionId, sibylRunId, includeTest]);

  useEffect(() => {
    void load();
  }, [load]);

  if (loading) {
    return <div className="p-3 text-sm text-fred-muted">Loading detail…</div>;
  }
  if (error) {
    return <div className="p-3 text-sm text-red-600">{error}</div>;
  }
  if (!detail) {
    return null;
  }

  // Since Oct 2026 Sibyl writes a vector per window month (months 1 and 6
  // elicited, 2 to 5 mixed); older rows carry one vector for all six.
  const sibylProbs =
    detail.record.final_by_month?.[String(month)] ?? detail.record.bucket_probs ?? [];
  const referenceProbs = detail.record.reference?.by_month?.[String(month)] ?? null;
  const standardByMonth = detail.standard_spd.by_month ?? {};
  const standardProbs = standardByMonth[String(month)] ?? null;
  const trials = detail.record.trials ?? [];

  return (
    <div className="space-y-4 p-3">
      {detail.question?.wording ? (
        <p className="text-sm text-fred-text">{String(detail.question.wording)}</p>
      ) : null}

      <div className="flex flex-wrap items-center gap-3">
        <span className="text-xs font-medium text-fred-text">
          Window month (Sibyl elicits months 1 and 6; 2 to 5 mix them):
        </span>
        {[1, 2, 3, 4, 5, 6].map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => setMonth(m)}
            className={`rounded px-2 py-1 text-xs ${
              month === m
                ? "bg-fred-primary text-white"
                : "border border-fred-secondary text-fred-text"
            }`}
          >
            M{m}
            {m === 1 || m === 6 ? "*" : ""}
          </button>
        ))}
      </div>

      <OverlayChart
        labels={detail.bucket_labels}
        sibyl={sibylProbs}
        standard={standardProbs}
        standardName={detail.standard_spd.model_name}
        reference={referenceProbs}
      />

      <div className="overflow-x-auto rounded-lg border border-fred-secondary">
        <table className="min-w-full text-xs">
          <thead>
            <tr className="bg-fred-surface text-fred-muted">
              <th className="px-2 py-1 text-left">Distribution</th>
              {QUANTILE_ORDER.map((q) => (
                <th key={q} className="px-2 py-1 text-right">
                  p{Math.round(parseFloat(q) * 100)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <QuantileRow
              label="Pooled (Sibyl)"
              quantiles={detail.record.pooled_quantiles ?? null}
            />
            {trials
              .filter((t) => t.quantiles)
              .map((t) => (
                <QuantileRow
                  key={t.trial_index}
                  label={`Trial ${t.trial_index + 1}`}
                  quantiles={t.quantiles ?? null}
                />
              ))}
          </tbody>
        </table>
      </div>

      <div className="space-y-2">
        {trials.map((t) => (
          <TrialCard key={t.trial_index} trial={t} />
        ))}
      </div>
    </div>
  );
};

const SibylClient = ({
  summary,
  questions,
  runs,
  includeTest,
}: {
  summary: SibylSummaryResponse;
  questions: SibylQuestionsResponse;
  runs: SibylRunsResponse;
  includeTest: boolean;
}) => {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [sortKey, setSortKey] = useState<SortKey>("js_divergence_vs_standard");
  const [sortDesc, setSortDesc] = useState(true);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [tab, setTab] = useState<"questions" | "calibration" | "about">("questions");

  const run = summary.run;
  const rows = questions.rows;

  const sorted = useMemo(() => {
    const copy = [...rows];
    copy.sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (av === null || av === undefined) return 1;
      if (bv === null || bv === undefined) return -1;
      return sortDesc ? Number(bv) - Number(av) : Number(av) - Number(bv);
    });
    return copy;
  }, [rows, sortKey, sortDesc]);

  const toggleSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortDesc((d) => !d);
    } else {
      setSortKey(key);
      setSortDesc(true);
    }
  };

  const selectRun = (id: string) => {
    const params = new URLSearchParams(searchParams.toString());
    params.set("sibyl_run_id", id);
    router.push(`/sibyl?${params.toString()}`);
  };

  const sortHeader = (key: SortKey, label: string, tooltip?: string) => (
    <th className="px-3 py-2 text-right text-xs font-semibold text-fred-muted">
      <span className="inline-flex items-center justify-end gap-1">
        <span
          className="cursor-pointer hover:text-fred-primary"
          onClick={() => toggleSort(key)}
        >
          {label}
          {sortKey === key ? (sortDesc ? " ↓" : " ↑") : ""}
        </span>
        {tooltip ? <InfoTooltip text={tooltip} /> : null}
      </span>
    </th>
  );

  // Top-level tab bar — always rendered so the metric explanations stay
  // reachable even before the first Sibyl run exists.
  const tabBar = (
    <div className="flex gap-1 rounded-lg border border-fred-secondary bg-fred-bg/40 p-1">
      {(
        [
          { key: "questions", label: "Questions" },
          { key: "calibration", label: "Calibration" },
          { key: "about", label: "About these metrics" },
        ] as { key: "questions" | "calibration" | "about"; label: string }[]
      ).map(({ key, label }) => (
        <button
          key={key}
          type="button"
          className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
            tab === key
              ? "bg-fred-primary text-white"
              : "text-fred-text hover:bg-fred-bg/60"
          }`}
          onClick={() => setTab(key)}
        >
          {label}
        </button>
      ))}
    </div>
  );

  if (tab === "calibration") {
    return (
      <div className="space-y-6">
        {tabBar}
        <SibylCalibration />
      </div>
    );
  }

  if (tab === "about") {
    return (
      <div className="space-y-6">
        {tabBar}
        <SibylMetricsAbout />
      </div>
    );
  }

  if (!run) {
    return (
      <div className="space-y-6">
        {tabBar}
        <div className="rounded-lg border border-fred-secondary bg-fred-surface px-4 py-6 text-sm text-fred-muted">
          No Sibyl runs yet. The Sibyl workflow runs after each Horizon Scanner
          pipeline completes.
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {tabBar}

      {runs.rows.length > 1 ? (
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="text-fred-muted">Run:</span>
          {runs.rows.slice(0, 8).map((r) => (
            <button
              key={r.sibyl_run_id}
              type="button"
              onClick={() => selectRun(r.sibyl_run_id)}
              className={`rounded px-2 py-1 ${
                r.sibyl_run_id === run.sibyl_run_id
                  ? "bg-fred-primary text-white"
                  : "border border-fred-secondary text-fred-text"
              }`}
            >
              {r.sibyl_run_id}
            </button>
          ))}
        </div>
      ) : null}

      <div className="grid gap-4 md:grid-cols-3">
        <div className="rounded-lg border border-fred-secondary bg-fred-surface p-4">
          <div className="text-xs uppercase text-fred-muted">Coverage</div>
          <div className="mt-1 text-2xl font-semibold text-fred-primary">
            {run.n_forecast ?? 0} / {run.n_selected ?? 0}
          </div>
          <div className="text-xs text-fred-muted">
            questions forecast · {run.n_skipped ?? 0} skipped
          </div>
          {run.budget_capped ? (
            <div className="mt-2 inline-block rounded bg-red-100 px-2 py-1 text-xs font-semibold text-red-700">
              BUDGET CAPPED — remaining questions skipped
            </div>
          ) : null}
          {run.time_capped ? (
            <div className="mt-2 inline-block rounded bg-red-100 px-2 py-1 text-xs font-semibold text-red-700">
              TIME CAPPED — remaining questions skipped
            </div>
          ) : null}
        </div>
        <div className="rounded-lg border border-fred-secondary bg-fred-surface p-4">
          <div className="text-xs uppercase text-fred-muted">Run cost</div>
          <div className="mt-1 text-2xl font-semibold text-fred-primary">
            {usd.format(run.run_cost_usd ?? 0)}
          </div>
          <div className="text-xs text-fred-muted">
            Opus {usd.format(run.opus_cost_usd ?? 0)} · Brave{" "}
            {usd.format(run.brave_cost_usd ?? 0)} · cap{" "}
            {usd.format(run.run_hard_cap_usd ?? 0)}
          </div>
        </div>
        <div className="rounded-lg border border-fred-secondary bg-fred-surface p-4">
          <div className="text-xs uppercase text-fred-muted">Method</div>
          <div className="mt-1 text-sm text-fred-text">
            {run.model ?? "?"} · K={run.k ?? "?"} · {run.aggregation ?? "?"}
          </div>
          <div className="text-xs text-fred-muted">
            hs_run: {run.hs_run_id ?? "?"} · {run.created_at ?? ""}
          </div>
        </div>
      </div>

      <ProcessMeasures run={run} />
      <ShadowArm run={run} shadow={summary.shadow} />

      <div className="overflow-x-auto rounded-lg border border-fred-secondary">
        <table className="min-w-full text-sm">
          <thead>
            <tr className="bg-fred-surface">
              <th className="px-3 py-2 text-left text-xs font-semibold text-fred-muted">
                Question
              </th>
              <th className="px-3 py-2 text-left text-xs font-semibold text-fred-muted">
                Status
              </th>
              {sortHeader(
                "js_divergence_vs_standard",
                "JSD vs standard",
                TOOLTIP_JSD_STANDARD,
              )}
              {sortHeader(
                "js_divergence_inter_trial",
                "Inter-trial JSD",
                TOOLTIP_JSD_INTER_TRIAL,
              )}
              {sortHeader("volatility_score", "Volatility", TOOLTIP_VOLATILITY)}
              {sortHeader("cost_usd", "Cost")}
            </tr>
          </thead>
          <tbody>
            {sorted.map((row) => (
              <Fragment key={row.question_id}>
                <tr
                  className="cursor-pointer border-t border-fred-secondary/40 hover:bg-fred-surface"
                  onClick={() =>
                    setExpanded((cur) =>
                      cur === row.question_id ? null : row.question_id
                    )
                  }
                >
                  <td className="px-3 py-2">
                    <div className="font-medium text-fred-primary">
                      {row.question_id}
                    </div>
                    <div className="text-xs text-fred-muted">
                      {row.iso3} · {row.hazard_code} · {row.metric}
                    </div>
                  </td>
                  <td className="px-3 py-2 text-xs">
                    {row.status === "ok" && row.evidence_ok === false ? (
                      <span
                        className="rounded bg-red-100 px-2 py-0.5 text-red-700"
                        title="This forecast rested on no evidence: its trials' searches failed. It is kept and scored, and left out of every comparison."
                      >
                        no evidence
                      </span>
                    ) : row.status === "ok" ? (
                      <span className="rounded bg-green-100 px-2 py-0.5 text-green-700">
                        forecast
                      </span>
                    ) : (
                      <span
                        className="rounded bg-amber-100 px-2 py-0.5 text-amber-700"
                        title={row.skip_reason ?? undefined}
                      >
                        {row.status}
                        {row.skip_reason ? `: ${row.skip_reason}` : ""}
                      </span>
                    )}
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-sm font-semibold text-indigo-600">
                    {fmtJsd(row.js_divergence_vs_standard)}
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-xs">
                    {fmtJsd(row.js_divergence_inter_trial)}
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-xs">
                    {row.volatility_score !== null && row.volatility_score !== undefined
                      ? row.volatility_score.toFixed(3)
                      : "—"}
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-xs">
                    {row.cost_usd !== null && row.cost_usd !== undefined
                      ? usd.format(row.cost_usd)
                      : "—"}
                  </td>
                </tr>
                {expanded === row.question_id && row.status === "ok" ? (
                  <tr>
                    <td colSpan={6} className="bg-fred-bg">
                      <QuestionDetail
                        questionId={row.question_id}
                        sibylRunId={questions.sibyl_run_id}
                        includeTest={includeTest}
                      />
                    </td>
                  </tr>
                ) : null}
              </Fragment>
            ))}
            {sorted.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-3 py-4 text-sm text-fred-muted">
                  No questions in this Sibyl run.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default SibylClient;
