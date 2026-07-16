"use client";

import { ReactNode } from "react";

// ---------------------------------------------------------------------------
// "About these metrics" reference for the Sibyl deep-research track.
//
// Explains the three numbers in the per-question table — JSD vs standard,
// Inter-trial JSD, and Volatility — with a technical definition, a
// plain-language description, and guidance on how to interpret the value.
//
// Kept in sync with sibyl/spd.py (track_divergence, inter_trial_divergence),
// sibyl/select_questions.py (volatility_score), and the shared JS divergence
// helper in pythia/tools/generate_calibration_advice.py (natural-log JSD,
// range 0 to ln 2 ≈ 0.693).
// ---------------------------------------------------------------------------

type MetricCardProps = {
  title: string;
  kind: string;
  range: string;
  technical: ReactNode;
  plain: ReactNode;
  interpret: ReactNode;
};

const MetricCard = ({
  title,
  kind,
  range,
  technical,
  plain,
  interpret,
}: MetricCardProps) => (
  <div className="rounded-lg border border-fred-secondary bg-fred-surface p-4 shadow-fredCard">
    <div className="mb-3 flex flex-wrap items-center gap-2">
      <h3 className="text-base font-semibold text-fred-primary">{title}</h3>
      <span className="rounded-full border border-fred-border bg-fred-bg px-2 py-0.5 text-[11px] font-medium uppercase tracking-wide text-fred-muted">
        {kind}
      </span>
      <span className="rounded-full border border-fred-border bg-fred-bg px-2 py-0.5 text-[11px] font-medium text-fred-muted">
        {range}
      </span>
    </div>
    <div className="space-y-3 text-sm leading-relaxed text-fred-text">
      <div>
        <div className="text-[11px] font-semibold uppercase tracking-wide text-fred-secondary">
          How it&apos;s calculated
        </div>
        <div className="mt-1">{technical}</div>
      </div>
      <div>
        <div className="text-[11px] font-semibold uppercase tracking-wide text-fred-secondary">
          In plain terms
        </div>
        <div className="mt-1">{plain}</div>
      </div>
      <div>
        <div className="text-[11px] font-semibold uppercase tracking-wide text-fred-secondary">
          How to read it
        </div>
        <div className="mt-1">{interpret}</div>
      </div>
    </div>
  </div>
);

const Formula = ({ children }: { children: ReactNode }) => (
  <code className="rounded bg-fred-bg px-1 py-0.5 font-mono text-[13px] text-fred-secondary">
    {children}
  </code>
);

export default function SibylMetricsAbout() {
  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-fred-secondary bg-fred-surface p-4">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-fred-text">
          About these metrics
        </h2>
        <p className="mt-2 text-sm leading-relaxed text-fred-text">
          Sibyl re-forecasts each selected question with{" "}
          <strong>K independent agentic research trials</strong> (Claude Opus +
          open-web search), then pools their probability distributions into one
          Sibyl forecast. Two of the columns below are{" "}
          <strong>divergence signals</strong> — they measure disagreement, not
          accuracy, so there is no inherently &ldquo;good&rdquo; or
          &ldquo;bad&rdquo; value. The third, Volatility, is the{" "}
          <strong>selection score</strong> that decided the question was worth
          deep research in the first place. To judge whether Sibyl was actually{" "}
          <em>right</em>, see the head-to-head Brier/Log/CRPS scores on the{" "}
          Performance page&apos;s deep-research section.
        </p>
        <p className="mt-2 text-xs text-fred-muted">
          Both divergences use the Jensen&ndash;Shannon divergence,{" "}
          <Formula>JSD(P,Q) = ½·KL(P‖M) + ½·KL(Q‖M)</Formula> with{" "}
          <Formula>M = ½(P + Q)</Formula>. It is symmetric, always finite, and —
          computed with the natural logarithm as Fred does — ranges from{" "}
          <strong>0</strong> (identical distributions) to{" "}
          <strong>ln 2 ≈ 0.693</strong> (maximally different).
        </p>
      </section>

      <div className="grid gap-4 lg:grid-cols-3">
        <MetricCard
          title="JSD vs standard"
          kind="Divergence"
          range="0 → ≈ 0.693"
          technical={
            <>
              The Jensen&ndash;Shannon divergence between Sibyl&apos;s pooled
              bucket distribution and the standard pipeline&apos;s chosen
              aggregate SPD (preference order{" "}
              <Formula>ensemble_bayesmc → ensemble_mean → track2</Formula>),
              averaged over the six forecast-window months. Sibyl emits one
              distribution that it reuses across all months, so this is the mean
              of the per-month <Formula>JSD(sibyl, standard_m)</Formula>.
            </>
          }
          plain={
            <>
              How much Sibyl&apos;s deep-web-research forecast{" "}
              <strong>disagrees with the main structured-data pipeline</strong>{" "}
              for this question. Zero means the two independent methods landed on
              essentially the same distribution; larger means they see the
              situation differently.
            </>
          }
          interpret={
            <>
              This is a <strong>disagreement flag, not a score</strong>. Values{" "}
              <strong>near 0</strong> are reassuring — two very different methods
              converged. <strong>High</strong> values (toward 0.4&ndash;0.69)
              flag questions worth a human look: Sibyl&apos;s live research may
              have found something the structured data missed, or vice versa. It
              cannot tell you <em>who</em> is right on its own — pair it with the
              resolved head-to-head scores to see whether diverging actually
              helped.
            </>
          }
        />

        <MetricCard
          title="Inter-trial JSD"
          kind="Divergence"
          range="0 → ≈ 0.693"
          technical={
            <>
              The mean pairwise Jensen&ndash;Shannon divergence across the K
              independent trials&apos; bucket distributions for this question
              (average of <Formula>JSD(trialᵢ, trialⱼ)</Formula> over all pairs).
              Defined only when at least two trials produced usable quantiles.
            </>
          }
          plain={
            <>
              How much Sibyl&apos;s <strong>separate research attempts disagreed
              with each other</strong>. It is a window into Sibyl&apos;s own
              internal uncertainty: did three independent investigations of the
              same question reach the same place, or scatter?
            </>
          }
          interpret={
            <>
              <strong>Low</strong> = the trials independently converged →
              a robust, trustworthy signal. <strong>High</strong> = the trials
              scattered → the question is genuinely ambiguous or the evidence is
              thin/conflicting, so treat Sibyl&apos;s pooled answer with more
              caution. Because trials are combined by linear pooling, high
              inter-trial divergence also <strong>widens the pooled
              distribution</strong> — Sibyl expresses that uncertainty rather
              than hiding it behind false confidence.
            </>
          }
        />

        <MetricCard
          title="Volatility"
          kind="Selection score"
          range="≈ 0 → 1"
          technical={
            <>
              The question&apos;s Horizon Scanner{" "}
              <Formula>regime_change_score = likelihood × magnitude</Formula> of
              a departure from the historical base rate (tie-broken by{" "}
              <Formula>triage_score</Formula>). It is <strong>not</strong>{" "}
              computed by Sibyl — it is the criterion used to pick the top-N most
              volatile affected/fatalities questions each run and hand them to
              Sibyl.
            </>
          }
          plain={
            <>
              How <strong>unsettled or fast-changing</strong> the situation is
              judged to be — high when the Horizon Scanner thinks the near future
              could break sharply from the historical norm. It is the reason this
              question was chosen for deep research.
            </>
          }
          interpret={
            <>
              <strong>Higher = more turbulent/uncertain</strong>, which is
              exactly where deep research is most likely to add value — so a high
              volatility question is not &ldquo;bad,&rdquo; it is a priority. This
              is a <strong>context and prioritization value, not a performance
              score</strong>. Under the run&apos;s budget cap the{" "}
              <em>least</em>-volatile selected questions are the first to be
              skipped, so everything you see here cleared that bar.
            </>
          }
        />
      </div>
    </div>
  );
}
