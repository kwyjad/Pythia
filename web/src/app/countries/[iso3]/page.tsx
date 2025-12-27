import Link from "next/link";

import { apiGet } from "../../../lib/api";
import type { QuestionsResponse } from "../../../lib/types";

type CountryPageProps = {
  params: { iso3: string };
};

const CountryPage = async ({ params }: CountryPageProps) => {
  const response = await apiGet<QuestionsResponse>("/questions", {
    iso3: params.iso3,
    latest_only: true
  });
  const questions = response.rows;

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-3xl font-semibold text-white">{params.iso3}</h1>
        <p className="text-sm text-slate-400">
          Latest questions for {params.iso3}
        </p>
      </section>

      <div className="overflow-x-auto rounded-lg border border-slate-800">
        <table>
          <thead>
            <tr>
              <th>Question</th>
              <th>Hazard</th>
              <th>Metric</th>
              <th>Target month</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {questions.map((row) => (
              <tr key={row.question_id}>
                <td>
                  <div className="font-medium text-white">{row.wording}</div>
                  <Link href={`/questions/${row.question_id}`}>
                    {row.question_id}
                  </Link>
                </td>
                <td>{row.hazard_code}</td>
                <td>{row.metric}</td>
                <td>{row.target_month}</td>
                <td>{row.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default CountryPage;
