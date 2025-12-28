import { apiGet } from "../../lib/api";
import CountriesTable from "./CountriesTable";

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
        <CountriesTable rows={rows} />
      </div>
    </div>
  );
};

export default CountriesPage;
