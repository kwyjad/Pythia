"use client";

import { ReactNode } from "react";

// ---------------------------------------------------------------------------
// "About performance scores" reference tab.
//
// A self-contained explainer for every scoring metric shown on this page. Each
// score gets three layers: (A) a precise technical definition of how it is
// computed, (B) a plain-language description, and (C) concrete guidance on how
// to read the value — what is good, what is bad, and the nuances that make a
// raw number misleading on its own.
//
// Definitions here are kept in sync with pythia/tools/compute_scores.py and
// forecaster/scoring.py. If the scoring math changes, update this copy too.
// ---------------------------------------------------------------------------

type ScoreCardProps = {
  title: string;
  family: string;
  range: string;
  technical: ReactNode;
  plain: ReactNode;
  interpret: ReactNode;
};

const ScoreCard = ({
  title,
  family,
  range,
  technical,
  plain,
  interpret,
}: ScoreCardProps) => (
  <div className="rounded-lg border border-fred-secondary bg-fred-surface p-4 shadow-fredCard">
    <div className="mb-3 flex flex-wrap items-center gap-2">
      <h3 className="text-base font-semibold text-fred-primary">{title}</h3>
      <span className="rounded-full border border-fred-border bg-fred-bg px-2 py-0.5 text-[11px] font-medium uppercase tracking-wide text-fred-muted">
        {family}
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

export default function AboutScores() {
  return (
    <div className="space-y-6">
      {/* Intro */}
      <section className="rounded-lg border border-fred-secondary bg-fred-surface p-4">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-fred-text">
          About performance scores
        </h2>
        <p className="mt-2 text-sm leading-relaxed text-fred-text">
          Fred&apos;s forecasts are probability distributions, not single
          numbers, so we grade them with <em>proper scoring rules</em> —
          measures designed so that a forecaster earns the best expected score
          only by reporting its true beliefs honestly. Every score below is{" "}
          <strong>negatively oriented</strong>: <strong>lower is better</strong>{" "}
          and <strong>0 is a perfect forecast</strong>. A score is only
          meaningful when compared like-for-like — same metric family, and
          ideally against a baseline (a naïve uniform or base-rate forecast, or
          an external benchmark such as ViEWS). Absolute values depend heavily on
          how hard the underlying questions are.
        </p>
      </section>

      {/* Quick reference table */}
      <section className="rounded-lg border border-fred-secondary bg-fred-surface p-4">
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-fred-text">
          Quick reference
        </h2>
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="border-b border-fred-secondary text-left text-xs uppercase tracking-wide text-fred-muted">
                <th className="px-3 py-2">Score</th>
                <th className="px-3 py-2">Applies to</th>
                <th className="px-3 py-2">Range</th>
                <th className="px-3 py-2">Perfect</th>
                <th className="px-3 py-2">&ldquo;No-skill&rdquo; baseline</th>
              </tr>
            </thead>
            <tbody className="text-fred-text">
              <tr className="border-b border-fred-border/60">
                <td className="px-3 py-2 font-medium">SPD Brier</td>
                <td className="px-3 py-2">
                  SPD questions (PA, Fatalities, Phase 3+)
                </td>
                <td className="px-3 py-2">0 &ndash; 2</td>
                <td className="px-3 py-2">0</td>
                <td className="px-3 py-2">~1 &minus; 1/K (≈ 0.83&ndash;0.86)</td>
              </tr>
              <tr className="border-b border-fred-border/60">
                <td className="px-3 py-2 font-medium">Binary Brier</td>
                <td className="px-3 py-2">EVENT_OCCURRENCE (yes/no)</td>
                <td className="px-3 py-2">0 &ndash; 1</td>
                <td className="px-3 py-2">0</td>
                <td className="px-3 py-2">0.25 (a 50/50 guess)</td>
              </tr>
              <tr className="border-b border-fred-border/60">
                <td className="px-3 py-2 font-medium">Log Loss</td>
                <td className="px-3 py-2">SPD questions</td>
                <td className="px-3 py-2">0 &ndash; ∞</td>
                <td className="px-3 py-2">0</td>
                <td className="px-3 py-2">ln K (≈ 1.79&ndash;1.95)</td>
              </tr>
              <tr>
                <td className="px-3 py-2 font-medium">CRPS (RPS)</td>
                <td className="px-3 py-2">SPD questions</td>
                <td className="px-3 py-2">0 &ndash; 1</td>
                <td className="px-3 py-2">0</td>
                <td className="px-3 py-2">varies (rewards near-misses)</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-xs text-fred-muted">
          K = number of ordered buckets for the metric (6 for People Affected and
          Phase 3+, 7 for Fatalities). Binary and SPD Brier live on different
          scales (0&ndash;1 vs 0&ndash;2) and are <strong>never</strong> averaged
          together.
        </p>
      </section>

      {/* Per-score detail cards */}
      <div className="grid gap-4 lg:grid-cols-2">
        <ScoreCard
          title="SPD Brier score"
          family="Multiclass"
          range="0 (perfect) → 2 (worst)"
          technical={
            <>
              For distributional (SPD) questions — People Affected, Fatalities,
              IPC Phase 3+ In Need — the forecast is a probability vector over K
              ordered buckets. The multiclass Brier score is the sum of squared
              errors against the one-hot outcome:{" "}
              <Formula>BS = Σₖ (fₖ − oₖ)²</Formula>, where{" "}
              <Formula>oₖ = 1</Formula> for the bucket that contains the resolved
              value and 0 for every other bucket. It is computed for each
              (question, horizon) pair and then averaged.
            </>
          }
          plain={
            <>
              It measures how well the whole spread of predicted probabilities
              matched what actually happened. You are rewarded for placing
              probability on the outcome that occurred, and penalized for
              confidence in outcomes that did not.
            </>
          }
          interpret={
            <>
              <strong>0</strong> is perfect (all probability on the correct
              bucket). A totally uninformed uniform forecast scores about{" "}
              <strong>1 − 1/K</strong> (roughly 0.83&ndash;0.86 here), so anything
              well below that shows skill; a value around{" "}
              <strong>0.5 or lower is strong</strong> for a 6&ndash;7 bucket
              problem. Approaching or exceeding <strong>1.0</strong> signals poor
              calibration or systematically betting on the wrong bucket, and{" "}
              <strong>2</strong> means the forecast put all its confidence on a
              single, completely wrong bucket. Nuance: easy questions (very
              likely bucket obvious in advance) naturally produce low scores, so a
              good average partly reflects question mix — compare hazards and
              models against each other, not against an absolute target.
            </>
          }
        />

        <ScoreCard
          title="Binary Brier score"
          family="Binary"
          range="0 (perfect) → 1 (worst)"
          technical={
            <>
              For binary EVENT_OCCURRENCE questions the forecast is a single
              probability <Formula>p</Formula> that the event happens in a given
              month. The score is <Formula>(p − o)²</Formula>, where{" "}
              <Formula>o</Formula> is 1 if the event occurred and 0 otherwise,
              averaged over the scored month-horizons.
            </>
          }
          plain={
            <>
              The squared distance between the single yes/no probability and what
              actually happened. Say 90% and it happens → small penalty; say 90%
              and it doesn&apos;t → large penalty.
            </>
          }
          interpret={
            <>
              <strong>0</strong> is perfect and <strong>0.25</strong> is what a
              non-committal 50/50 forecast earns, so scores below 0.25 indicate
              real signal. Important nuance: these scores are{" "}
              <strong>structurally near zero for rare events</strong> correctly
              called unlikely — if an event almost never happens and the model
              says ~2%, it will score very low every month simply by being right
              that &ldquo;nothing happens.&rdquo; A tiny binary Brier therefore is
              not automatically proof of skill; judge it against the event&apos;s
              base rate. This scale (0&ndash;1) is not comparable to SPD Brier
              (0&ndash;2).
            </>
          }
        />

        <ScoreCard
          title="Log Loss (logarithmic score)"
          family="SPD"
          range="0 (perfect) → ∞ (worst)"
          technical={
            <>
              The negative natural logarithm of the probability the forecast
              assigned to the bucket that actually occurred:{" "}
              <Formula>−ln(f_outcome)</Formula>. Averaged across (question,
              horizon) pairs. Assigning a near-zero probability to something that
              then happens drives the term toward <Formula>+∞</Formula>.
            </>
          }
          plain={
            <>
              A severity dial for overconfidence. It punishes being confidently
              wrong far more harshly than Brier does — declaring an outcome nearly
              impossible and then watching it occur is the worst thing a
              forecaster can do, and log loss reflects that.
            </>
          }
          interpret={
            <>
              <strong>0</strong> is perfect. A uniform forecast over K buckets
              scores <strong>ln K</strong> (≈ 1.79 for 6 buckets, ≈ 1.95 for 7),
              so lower than that is skill; there is no upper bound. Because it is
              unbounded above, a <strong>single catastrophic overconfident
              miss can dominate the average</strong> — a high log loss paired with
              a respectable Brier usually means a few disastrous calls rather than
              broadly poor forecasting. It is the most sensitive of the four to
              tail miscalibration.
            </>
          }
        />

        <ScoreCard
          title="CRPS (normalized RPS)"
          family="SPD · ordinal"
          range="0 (perfect) → 1 (worst)"
          technical={
            <>
              The discrete form of the Continuous Ranked Probability Score for
              ordered buckets — the normalized Ranked Probability Score:{" "}
              <Formula>RPS = Σₖ₌₁..ₖ₋₁ (Fₖ − Hₖ)² / (K − 1)</Formula>, where{" "}
              <Formula>Fₖ</Formula> is the forecast&apos;s cumulative
              distribution up to bucket k and <Formula>Hₖ</Formula> is the outcome
              step-CDF (0 before the true bucket, 1 from it on). It compares
              cumulative distributions rather than individual buckets.
            </>
          }
          plain={
            <>
              Brier&apos;s distance-aware cousin. Because the buckets are ordered
              (e.g. by magnitude), being off by one bucket is penalized much less
              than being off by five. It credits forecasts that land{" "}
              <em>close</em> to the truth, not only exactly on it.
            </>
          }
          interpret={
            <>
              <strong>0</strong> is perfect; <strong>1</strong> means all
              probability sat in the bucket farthest from the outcome. A forecast
              adjacent to the correct bucket scores much better than a far miss.
              For magnitude questions this is often the fairest single number,
              because it rewards near-misses that plain Brier treats as complete
              failures. (Stored under <Formula>score_type = &apos;crps&apos;</Formula>{" "}
              in the database; the older non-standard variant divided by K and
              deflated scores ~20%.)
            </>
          }
        />
      </div>

      {/* Closing guidance */}
      <section className="rounded-lg border border-fred-secondary bg-fred-surface p-4">
        <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-fred-text">
          Reading the numbers well
        </h2>
        <ul className="list-disc space-y-1 pl-5 text-sm leading-relaxed text-fred-text">
          <li>
            <strong>Lower is always better; 0 is perfect.</strong> All four are
            proper scoring rules that reward honest probabilities.
          </li>
          <li>
            <strong>Compare within a family.</strong> SPD Brier (0&ndash;2) and
            Binary Brier (0&ndash;1) are different scales — the page shows them in
            separate columns and never blends them. Log Loss and CRPS apply to SPD
            questions only.
          </li>
          <li>
            <strong>Judge against a baseline, not an absolute.</strong> A score
            &ldquo;looks&rdquo; good or bad only relative to a no-skill forecast or
            a benchmark. Use the By&nbsp;Hazard, By&nbsp;Model, and By&nbsp;Run
            views, and the ViEWS external benchmark, to see relative skill.
          </li>
          <li>
            <strong>Difficulty is baked in.</strong> Rare, extreme, or
            climate-hazard outcomes with sparse resolution data are genuinely
            harder to forecast, so a higher score there does not necessarily mean
            a worse model.
          </li>
          <li>
            <strong>Use several scores together.</strong> Brier for overall
            calibration, CRPS for &ldquo;how close,&rdquo; and Log Loss to catch
            rare but costly overconfidence. They agree on good forecasts and
            disagree in instructive ways on flawed ones.
          </li>
        </ul>
      </section>
    </div>
  );
}
