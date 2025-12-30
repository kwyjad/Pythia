import { readFile } from "node:fs/promises";
import path from "node:path";

import { apiGet } from "../../lib/api";
import type { CountriesResponse, CountriesRow } from "../../lib/types";
import CountriesTable from "./CountriesTable";

function parseCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }
    if (ch === "," && !inQuotes) {
      out.push(cur);
      cur = "";
      continue;
    }
    cur += ch;
  }
  out.push(cur);
  return out.map((v) => v.trim());
}

async function loadCountryNameMap(): Promise<Map<string, string>> {
  const candidates = [
    path.join(process.cwd(), "..", "resolver", "data", "countries.csv"),
    path.join(process.cwd(), "resolver", "data", "countries.csv"),
  ];

  let csvText: string | null = null;
  for (const p of candidates) {
    try {
      csvText = await readFile(p, "utf-8");
      break;
    } catch {
      // keep trying
    }
  }

  const map = new Map<string, string>();
  if (!csvText) return map;

  const lines = csvText.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length < 2) return map;

  const header = parseCsvLine(lines[0]).map((h) => h.toLowerCase());
  const isoIdx = header.indexOf("iso3");
  const nameIdx = header.indexOf("country_name");
  if (isoIdx === -1 || nameIdx === -1) return map;

  for (let i = 1; i < lines.length; i++) {
    const cols = parseCsvLine(lines[i]);
    const iso3 = (cols[isoIdx] ?? "").trim().toUpperCase();
    const name = (cols[nameIdx] ?? "").trim();
    if (iso3) map.set(iso3, name);
  }
  return map;
}

const CountriesPage = async () => {
  let rows: CountriesRow[] = [];
  try {
    const response = await apiGet<CountriesResponse>("/countries");
    rows = response.rows;
  } catch (error) {
    console.warn("Failed to load countries:", error);
  }

  const nameMap = await loadCountryNameMap();

  rows = rows.map((r) => {
    const iso3 = (r.iso3 ?? "").toUpperCase();
    return {
      ...r,
      iso3,
      country_name: nameMap.get(iso3) ?? null,
    };
  });

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
