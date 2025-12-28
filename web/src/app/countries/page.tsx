import Link from "next/link";

import { apiGet } from "../../lib/api";

type CountriesRow = {
  iso3: string;
  n_questions: number;
  n_forecasted: number;
};

type CountriesResponse = {
  rows: CountriesRow[];
};

const CountriesPage = async () => {
  let rows: CountriesRow[] = [];
  try {
    const response = await apiGet<CountriesResponse>("/countries");
    rows = response.rows;
  } catch (error) {
    console.warn("Failed to load countries:", error);
  }

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-3xl font-semibold text-white">Countries</h1>
        <p className="text-sm text-slate-400">
          Browse available countries in the database.
        </p>
      </section>

      <div className="overflow-x-auto rounded-lg border border-slate-800">
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr className="bg-slate-900 text-slate-300">
              <th className="px-3 py-2 text-left">Country</th>
              <th className="px-3 py-2 text-right">Questions</th>
              <th className="px-3 py-2 text-right">Forecasted</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {rows.map((row) => (
              <tr key={row.iso3} className="hover:bg-slate-900/60">
                <td className="px-3 py-2">
                  <Link
                    href={`/countries/${row.iso3}`}
                    className="block w-full px-0 py-0 text-sky-300 underline underline-offset-2 hover:text-sky-200"
                  >
                    {row.iso3}
                  </Link>
                </td>
                <td className="px-3 py-2 text-right">{row.n_questions}</td>
                <td className="px-3 py-2 text-right">{row.n_forecasted}</td>
              </tr>
            ))}
            {rows.length === 0 ? (
              <tr>
                <td colSpan={3} className="px-3 py-6 text-center text-slate-400">
                  No countries available. No data in DB snapshot.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default CountriesPage;
