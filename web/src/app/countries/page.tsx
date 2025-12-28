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
        <table>
          <thead>
            <tr>
              <th>Country</th>
              <th>Questions</th>
              <th>Forecasted</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.iso3}>
                <td>
                  <Link href={`/countries/${row.iso3}`}>{row.iso3}</Link>
                </td>
                <td>{row.n_questions}</td>
                <td>{row.n_forecasted}</td>
              </tr>
            ))}
            {rows.length === 0 ? (
              <tr>
                <td colSpan={3} className="text-slate-400">
                  No countries available.
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
