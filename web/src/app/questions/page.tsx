import Link from "next/link";

import { apiGet } from "../../lib/api";

type QuestionRow = {
  question_id: string;
  hs_run_id?: string | null;
  iso3: string;
  hazard_code: string;
  metric: string;
  target_month: string;
  status?: string;
  wording?: string;
};

type QuestionsResponse = {
  rows: QuestionRow[];
};

const QuestionsPage = async () => {
  let rows: QuestionRow[] = [];
  try {
    const response = await apiGet<QuestionsResponse>("/questions", {
      latest_only: true
    });
    rows = response.rows;
  } catch (error) {
    console.warn("Failed to load questions:", error);
  }

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-3xl font-semibold text-white">Questions</h1>
        <p className="text-sm text-slate-400">
          Browse the latest questions by concept.
        </p>
      </section>

      <div className="overflow-x-auto rounded-lg border border-slate-800">
        <table>
          <thead>
            <tr>
              <th>Question</th>
              <th>Country</th>
              <th>Hazard</th>
              <th>Metric</th>
              <th>Target month</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.question_id}>
                <td>
                  <div className="font-medium text-white">{row.wording}</div>
                  <Link
                    href={`/questions/${row.question_id}?hs_run_id=${encodeURIComponent(
                      row.hs_run_id ?? ""
                    )}`}
                  >
                    {row.question_id}
                  </Link>
                </td>
                <td>{row.iso3}</td>
                <td>{row.hazard_code}</td>
                <td>{row.metric}</td>
                <td>{row.target_month}</td>
                <td>{row.status}</td>
              </tr>
            ))}
            {rows.length === 0 ? (
              <tr>
                <td colSpan={6} className="text-slate-400">
                  No questions available.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default QuestionsPage;
