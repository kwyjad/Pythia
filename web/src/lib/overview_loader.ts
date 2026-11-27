import { readFile } from "node:fs/promises";
import path from "node:path";

export interface OverviewVersionEntry {
  date: string;
  label: string;
}

/** An archived version key: a date, optionally followed by -2, -3, ... */
const VERSION_PATTERN = /^\d{4}-\d{2}-\d{2}(-\d{1,3})?$/;

/** True when the string has the shape of a version key. Checked before any
 *  file is read, so a crafted string can never become a path. */
export function isVersionShaped(version: string): boolean {
  return VERSION_PATTERN.test(version);
}

/**
 * Read the versions manifest from docs/overview/versions.json.
 */
export async function listOverviewVersions(): Promise<OverviewVersionEntry[]> {
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
 * Read the archived fred_overview.md for one version, or null when the
 * version is not listed in the manifest. The shape check comes first, then
 * the manifest; only a listed version is ever turned into a path.
 */
export async function loadListedOverview(version: string): Promise<string | null> {
  if (!isVersionShaped(version)) return null;
  const versions = await listOverviewVersions();
  if (!versions.some((v) => v.date === version)) return null;
  const candidates = [
    path.join(process.cwd(), "..", "docs", "overview", version, "fred_overview.md"),
    path.join(process.cwd(), "docs", "overview", version, "fred_overview.md"),
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
