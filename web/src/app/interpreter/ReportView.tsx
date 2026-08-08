// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
// See the LICENSE file in the project root for details.

// The report body, rendered from the RESOLVED structured content (every
// {{fig:...}} placeholder already substituted server-side). Shared by the
// interactive page and the print route — one rendering, two stylings.

import Link from "next/link";

import type { InterpreterCallEntry, InterpreterReport } from "../../lib/types";
import {
  LEXICON_TABLE,
  REASON_LABELS,
  countryAnchorId,
  entryHeading,
  groupAttention,
} from "./lib";

const QuestionLinks = ({ ids }: { ids: string[] }) => (
  <span className="inline-flex flex-wrap gap-2">
    {ids.map((qid) => (
      <Link
        key={qid}
        href={`/questions/${encodeURIComponent(qid)}`}
        className="rounded bg-fred-surface px-1.5 py-0.5 font-mono text-[11px] text-fred-primary hover:text-fred-secondary"
      >
        {qid}
      </Link>
    ))}
  </span>
);

const CallList = ({ title, calls }: { title: string; calls: InterpreterCallEntry[] }) => (
  <div className="space-y-2">
    <h3 className="text-lg font-semibold text-fred-primary">{title}</h3>
    <ul className="list-disc space-y-2 pl-5 text-sm text-fred-text">
      {calls.map((call, i) => (
        <li key={i}>
          {call.what_was_right || call.what_went_wrong}{" "}
          <QuestionLinks ids={call.question_ids || []} />
        </li>
      ))}
    </ul>
  </div>
);

const Bullets = ({ items }: { items: string[] }) => (
  <ul className="list-disc space-y-1 pl-5 text-sm text-fred-text">
    {items.map((item, i) => (
      <li key={i}>{item}</li>
    ))}
  </ul>
);

export default function ReportView({
  report,
  print = false,
}: {
  report: InterpreterReport;
  print?: boolean;
}) {
  const content = report.content_resolved ?? report.content;
  if (!content) {
    return (
      <p className="text-sm text-fred-muted">
        No readable report content for this cycle.
      </p>
    );
  }
  const sectionBreak = print ? "break-before-page" : "";
  const attention = content.attention ?? [];
  const performance = content.performance;

  return (
    <article className="space-y-8">
      <p className="text-lg font-semibold text-fred-primary">{content.headline}</p>

      {content.run_summary ? (
        <section className="space-y-2">
          <h2 className="text-2xl font-semibold text-fred-primary">
            The scan this month
          </h2>
          <p className="text-sm text-fred-text">{content.run_summary}</p>
        </section>
      ) : null}

      {groupAttention(attention).map((group) => (
        <section key={group.key} className="space-y-5">
          <h2 className="text-2xl font-semibold text-fred-primary">
            {group.heading}
          </h2>
          {group.entries.map((entry) => (
              <div
                key={`${entry.rank}-${entry.iso3}-${entry.hazard_code}`}
                id={countryAnchorId(entry.iso3)}
                className="scroll-mt-24 rounded-lg border border-fred-secondary bg-fred-surface/60 p-4"
              >
                <h3 className="text-lg font-semibold text-fred-primary">
                  {entry.rank}. {entryHeading(entry)}{" "}
                  <span className="text-sm font-normal text-fred-muted">
                    ({REASON_LABELS[entry.reason_code] ?? entry.reason_code})
                  </span>
                </h3>
                {entry.why_it_stands_out ? (
                  <p className="mt-2 text-sm text-fred-text">{entry.why_it_stands_out}</p>
                ) : null}
                {entry.spd_shape ? (
                  <p className="mt-2 text-sm text-fred-text">
                    <span className="font-medium">The shape of the forecast:</span>{" "}
                    {entry.spd_shape}
                  </p>
                ) : null}
                {entry.how_to_read_the_distribution ? (
                  <p className="mt-2 text-sm text-fred-text">
                    <span className="font-medium">Reading the forecast:</span>{" "}
                    {entry.how_to_read_the_distribution}
                  </p>
                ) : null}
                {entry.what_the_model_was_reacting_to ? (
                  <p className="mt-2 text-sm text-fred-text">
                    <span className="font-medium">What the system was reacting to:</span>{" "}
                    {entry.what_the_model_was_reacting_to}
                  </p>
                ) : null}
                {entry.impacts?.length ? (
                  <div className="mt-2">
                    <span className="text-sm font-medium text-fred-text">Likely impacts:</span>
                    <Bullets items={entry.impacts} />
                  </div>
                ) : null}
                {entry.operational_challenges?.length ? (
                  <div className="mt-2">
                    <span className="text-sm font-medium text-fred-text">
                      Operational challenges:
                    </span>
                    <Bullets items={entry.operational_challenges} />
                  </div>
                ) : null}
                {entry.lead_time_months != null ? (
                  <p className="mt-2 text-sm text-fred-muted">
                    Lead time: about {entry.lead_time_months} month(s).
                  </p>
                ) : null}
                <p className="mt-2 text-xs text-fred-muted">
                  Questions: <QuestionLinks ids={entry.question_ids || []} />
                </p>
              </div>
          ))}
        </section>
      ))}

      {content.changes_since_last_run?.length ? (
        <section className={`space-y-2 ${sectionBreak}`}>
          <h2 className="text-2xl font-semibold text-fred-primary">
            What changed since last month
          </h2>
          <Bullets items={content.changes_since_last_run} />
        </section>
      ) : null}

      {performance ? (
        <section className={`space-y-4 ${sectionBreak}`}>
          <h2 className="text-2xl font-semibold text-fred-primary">How well did we do</h2>
          {performance.plain_summary ? (
            <p className="text-sm text-fred-text">{performance.plain_summary}</p>
          ) : null}
          {performance.skill_statement ? (
            <p className="text-sm text-fred-text">{performance.skill_statement}</p>
          ) : null}
          {performance.best_calls?.length ? (
            <CallList title="Best calls" calls={performance.best_calls} />
          ) : null}
          {performance.worst_calls?.length ? (
            <CallList title="Worst calls" calls={performance.worst_calls} />
          ) : null}
          {performance.track_comparison ? (
            <div>
              <h3 className="text-lg font-semibold text-fred-primary">Track 1 vs Track 2</h3>
              <p className="text-sm text-fred-text">{performance.track_comparison}</p>
            </div>
          ) : null}
          {performance.sibyl_comparison ? (
            <div>
              <h3 className="text-lg font-semibold text-fred-primary">
                Deep research (Sibyl) vs the standard track
              </h3>
              <p className="text-sm text-fred-text">{performance.sibyl_comparison}</p>
            </div>
          ) : null}
          {performance.vs_system_average ? (
            <div>
              <h3 className="text-lg font-semibold text-fred-primary">
                This run vs the system average
              </h3>
              <p className="text-sm text-fred-text">{performance.vs_system_average}</p>
            </div>
          ) : null}
        </section>
      ) : null}

      {content.blind_spots?.length ? (
        <section className="space-y-2">
          <h2 className="text-2xl font-semibold text-fred-primary">What we cannot see</h2>
          <Bullets items={content.blind_spots} />
        </section>
      ) : null}

      {content.confidence_notes?.length ? (
        <section className="space-y-2">
          <h2 className="text-2xl font-semibold text-fred-primary">Confidence notes</h2>
          <Bullets items={content.confidence_notes} />
        </section>
      ) : null}

      <section className={`space-y-3 ${sectionBreak}`}>
        <h2 className="text-2xl font-semibold text-fred-primary">Appendix</h2>
        <div>
          <h3 className="text-lg font-semibold text-fred-primary">Probability words</h3>
          <p className="text-sm text-fred-muted">
            The report uses these words with fixed meanings, so the reader can
            check the writer:
          </p>
          <table className="mt-2 text-sm text-fred-text">
            <thead>
              <tr className="text-left text-fred-muted">
                <th className="pr-6 font-medium">word</th>
                <th className="font-medium">probability band</th>
              </tr>
            </thead>
            <tbody>
              {LEXICON_TABLE.map((row) => (
                <tr key={row.word}>
                  <td className="pr-6">{row.word}</td>
                  <td>{row.band}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div>
          <h3 className="text-lg font-semibold text-fred-primary">Run provenance</h3>
          <ul className="text-xs text-fred-muted">
            {[
              ["Report", report.interpretation_id],
              ["Kind", report.kind],
              ["Version", `v${report.version}`],
              ["Forecaster run", report.run_id],
              ["HS run", report.hs_run_id],
              ["Scored round", report.scored_run_id],
              ["Model", report.model_name],
              ["Thinking", report.thinking_level],
              ["Template", report.template_version],
              ["Generated", report.created_at],
              [
                "Cost",
                report.cost_usd != null ? `$${Number(report.cost_usd).toFixed(2)}` : null,
              ],
            ]
              .filter(([, v]) => v != null && v !== "")
              .map(([k, v]) => (
                <li key={String(k)}>
                  {k}: <span className="font-mono">{String(v)}</span>
                </li>
              ))}
          </ul>
        </div>
        <p className="text-xs text-fred-muted">
          Generated automatically. No numbers in this report were written by
          the language model; every figure is substituted from the
          system&apos;s own computed data.
        </p>
      </section>
    </article>
  );
}
