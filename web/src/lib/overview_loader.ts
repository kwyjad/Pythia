import { readFile } from "node:fs/promises";
import path from "node:path";

export interface OverviewVersionEntry {
  date: string;
  label: string;
}

export interface VersionedOverviewData {
  versions: OverviewVersionEntry[];
  overviews: Record<string, string>;
}

/**
 * Read the versions manifest from docs/overview/versions.json.
 */
async function listOverviewVersions(): Promise<OverviewVersionEntry[]> {
  const candidates = [
    path.join(process.cwd(), "..", "docs", "overview", "versions.json"),
    path.join(process.cwd(), "docs", "overview", "versions.json"),
  ];
  for (const p of candidates) {
    try {
      const raw = await readFile(p, "utf-8");
      return JSON.parse(raw) as OverviewVersionEntry[];
    } catch {
      // keep trying
    }
  }
  return [];
}

/**
 * Read the archived fred_overview.md for a specific version date.
 */
async function loadOverviewForVersion(
  date: string,
): Promise<string | null> {
  const candidates = [
    path.join(process.cwd(), "..", "docs", "overview", date, "fred_overview.md"),
    path.join(process.cwd(), "docs", "overview", date, "fred_overview.md"),
  ];
  for (const p of candidates) {
    try {
      return await readFile(p, "utf-8");
    } catch {
      // keep trying
    }
  }
  return null;
}

/**
 * Load all versioned overview snapshots.
 * Returns the version list and a map of date -> markdown content.
 */
export async function loadAllVersionedOverviews(): Promise<VersionedOverviewData> {
  const versions = await listOverviewVersions();
  const overviews: Record<string, string> = {};
  await Promise.all(
    versions.map(async (v) => {
      const content = await loadOverviewForVersion(v.date);
      if (content) {
        overviews[v.date] = content;
      }
    }),
  );
  return { versions, overviews };
}
