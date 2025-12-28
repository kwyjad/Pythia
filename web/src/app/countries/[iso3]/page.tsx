import Link from "next/link";

import { apiGet } from "../../../lib/api";
import type { QuestionsResponse } from "../../../lib/types";

type CountryPageProps = {
  params: { iso3: string };
};

const CountryPage = async ({ params }: CountryPageProps) => {
  let questions: QuestionsResponse["rows"] = [];
  let loadError: string | null = null;
  try {
    const response = await apiGet<QuestionsResponse>("/questions", {
      iso3: params.iso3,
      latest_only: true,
    });
    questions = response.rows;
  } catch (error) {
    loadError = "Unable to load questions right now.";
    console.warn("Failed to load questions:", error);
  }

  return (
    <div className="space-y-6">
      <section>
        <Link className="text-sm text-sky-300 underline underline-offset-2 hover:text-sky-200" href="/countries">
          ← Back to Countries
        </Link>
        <h1 className="text-3xl font-semibold text-white">{params.iso3}</h1>
        <p className="text-sm text-slate-400">
          Latest questions for {params.iso3}
        </p>
      </section>

      {loadError || questions.length === 0 ? (
        <div className="rounded-lg border border-slate-800 bg-slate-900/40 px-4 py-6 text-center text-slate-300">
          {loadError ?? `No questions available for ${params.iso3} in this DB snapshot.`}
        </div>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-slate-800">
          <table className="w-full border-collapse text-sm">
            <thead className="bg-slate-900 text-slate-300">
              <tr>
                <th className="px-3 py-2 text-left">Question</th>
                <th className="px-3 py-2 text-left">Hazard</th>
                <th className="px-3 py-2 text-left">Metric</th>
                <th className="px-3 py-2 text-left">Target month</th>
                <th className="px-3 py-2 text-left">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {questions.map((row) => (
                <tr key={row.question_id} className="text-slate-200">
                  <td className="px-3 py-2">
                    <div className="font-medium text-white">{row.wording}</div>
                    <Link
                      className="text-sky-300 underline underline-offset-2 hover:text-sky-200"
                      href={`/questions/${row.question_id}?hs_run_id=${encodeURIComponent(
                        row.hs_run_id ?? ""
                      )}`}
                    >
                      {row.question_id}
                    </Link>
                  </td>
                  <td className="px-3 py-2">{row.hazard_code}</td>
                  <td className="px-3 py-2">{row.metric}</td>
                  <td className="px-3 py-2">{row.target_month}</td>
                  <td className="px-3 py-2">{row.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default CountryPage;
