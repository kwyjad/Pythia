// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
// See the LICENSE file in the project root for details.

// @vitest-environment node

import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { GET as getOverview } from "../overview/[version]/route";
import { GET as getPrompts } from "../prompts/[version]/route";

const REPO = path.resolve(__dirname, "../../../../..");

function firstListed(dir: string): string {
  const versions = JSON.parse(
    readFileSync(path.join(REPO, "docs", dir, "versions.json"), "utf-8"),
  ) as { date: string }[];
  return versions[versions.length - 1].date;
}

const req = new Request("http://localhost/about");

describe("archived overview route", () => {
  it("serves a version listed in the manifest", async () => {
    const version = firstListed("overview");
    const res = await getOverview(req, { params: { version } });
    expect(res.status).toBe(200);
    expect((await res.text()).length).toBeGreaterThan(100);
  });

  it("returns 404 for a version not in the manifest", async () => {
    const res = await getOverview(req, { params: { version: "2099-01-01" } });
    expect(res.status).toBe(404);
  });

  it.each(["../../CLAUDE.md", "..", "2026-03-02/../../..", "%2e%2e%2fetc", "/etc/passwd"])(
    "returns 404 for a path-like string %s",
    async (version) => {
      const res = await getOverview(req, { params: { version } });
      expect(res.status).toBe(404);
    },
  );
});

describe("archived prompts route", () => {
  it("serves a version listed in the manifest", async () => {
    const version = firstListed("prompts");
    const res = await getPrompts(req, { params: { version } });
    expect(res.status).toBe(200);
    expect(typeof (await res.json())).toBe("object");
  });

  it.each(["2099-01-01", "../../CLAUDE.md", "2026-03-02/../.."])(
    "returns 404 for %s",
    async (version) => {
      const res = await getPrompts(req, { params: { version } });
      expect(res.status).toBe(404);
    },
  );
});
