// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
// See the LICENSE file in the project root for details.

import { describe, expect, it } from "vitest";

import { SCORE_GLOSSARY } from "../../../lib/score_glossary";
import { NAV_LINKS } from "../../../components/Nav";
import type { InterpreterVersionRow } from "../../../lib/types";
import {
  LEXICON_TABLE,
  MOVEMENT_ABOVE,
  MOVEMENT_BELOW,
  MOVEMENT_NEUTRAL,
  REASON_LABELS,
  movementColor,
  movementLabel,
  movementLegend,
  countryAnchorId,
  entryHeading,
  groupAttention,
  monthLabel,
  groupVersionsByRun,
  newestServedVersion,
  pdfVersionFromAsset,
  runMonthLabel,
  sectionHeading,
  statusBanner,
} from "../lib";

describe("movement scale", () => {
  it("is diverging, so a fall is never shaded as a rise", () => {
    // The regression this replaced: Uganda's ensemble moved DOWN and the
    // undirected scale shaded it among the countries furthest from usual,
    // on the same page as text saying so.
    expect(movementColor(2.5)).toBe(MOVEMENT_ABOVE[4]);
    expect(movementColor(-2.5)).toBe(MOVEMENT_BELOW[4]);
    expect(movementColor(2.5)).not.toBe(movementColor(-2.5));
  });

  it("treats a movement near the anchor as neutral in either direction", () => {
    expect(movementColor(0)).toBe(MOVEMENT_NEUTRAL);
    expect(movementColor(0.05)).toBe(MOVEMENT_NEUTRAL);
    expect(movementColor(-0.05)).toBe(MOVEMENT_NEUTRAL);
  });

  it("clamps rather than running off the ramp", () => {
    expect(movementColor(99)).toBe(movementColor(3));
    expect(movementColor(-99)).toBe(movementColor(-3));
  });

  it("mirrors the printed map's ramps (interpreter/mapviz.py)", () => {
    expect(MOVEMENT_ABOVE).toHaveLength(5);
    expect(MOVEMENT_BELOW).toHaveLength(5);
  });

  it("legend names the direction and colors are unique", () => {
    const legend = movementLegend();
    const labels = legend.map((l) => l.label.toLowerCase());
    expect(labels.some((l) => l.includes("above its usual level"))).toBe(true);
    expect(labels.some((l) => l.includes("below its usual level"))).toBe(true);
    expect(labels.some((l) => l.includes("no forecast"))).toBe(true);
    const colors = legend.map((l) => l.color);
    expect(new Set(colors).size).toBe(colors.length);
  });
});

describe("reason labels", () => {
  it("covers exactly the schema's reason codes (interpreter/schema.py)", () => {
    expect(Object.keys(REASON_LABELS).sort()).toEqual([
      "base_rate_deviation",
      "large_impact_nominal",
      "large_impact_per_capita",
      "rc_deviation_disagreement",
    ]);
  });
});

describe("lexicon table", () => {
  it("mirrors interpreter/lexicon.py's fixed bands, highest first", () => {
    expect(LEXICON_TABLE.map((r) => r.word)).toEqual([
      "virtually certain",
      "very likely",
      "likely",
      "about as likely as not",
      "unlikely",
      "very unlikely",
      "exceptionally unlikely",
    ]);
  });
});

describe("status banner", () => {
  it("ok renders nothing; failures degrade visibly, never silently", () => {
    expect(statusBanner("ok")).toBeNull();
    expect(statusBanner("failed_validation")?.tone).toBe("warn");
    expect(statusBanner("failed_generation")?.tone).toBe("error");
    expect(statusBanner("weird")?.tone).toBe("warn");
  });
});

describe("glossary", () => {
  it("every entry has a term and non-trivial copy", () => {
    expect(SCORE_GLOSSARY.length).toBeGreaterThanOrEqual(8);
    for (const item of SCORE_GLOSSARY) {
      expect(item.term).toBeTruthy();
      expect(item.text.length).toBeGreaterThan(40);
    }
  });
});

describe("country anchors", () => {
  it("uppercase, stable ids", () => {
    expect(countryAnchorId("eth")).toBe("country-ETH");
  });
});

const row = (over: Partial<InterpreterVersionRow>): InterpreterVersionRow => ({
  interpretation_id: "int_x",
  kind: "combined",
  run_id: "fc_1",
  hs_run_id: "hs_20260801T025754",
  scored_run_id: null,
  version: 1,
  status: "ok",
  model_name: "claude-opus-5",
  template_version: "v1",
  created_at: "2026-08-06 18:58:23",
  is_test: false,
  ...over,
});

describe("run month label", () => {
  it("names the month the report is ABOUT, from the HS run id", () => {
    // A July run interpreted in August reads as July — matching
    // interpreter/pdf.py::month_label, so page and filename agree.
    expect(runMonthLabel("hs_20260715T103033", "2026-08-06 18:54:48")).toBe(
      "July 2026"
    );
    expect(runMonthLabel("hs_20260801T025754", "2026-08-06")).toBe("August 2026");
  });

  it("falls back to created_at when the run id is missing or malformed", () => {
    expect(runMonthLabel(null, "2026-08-06 18:58:23")).toBe("August 2026");
    expect(runMonthLabel("not-a-run-id", "2026-12-01")).toBe("December 2026");
    expect(runMonthLabel(null, null)).toBe("Unknown month");
  });
});

describe("run grouping for the selector", () => {
  it("one option per run, newest first, each holding its newest version", () => {
    const options = groupVersionsByRun([
      row({ interpretation_id: "aug_v2", run_id: "fc_aug", version: 2 }),
      row({ interpretation_id: "aug_v1", run_id: "fc_aug", version: 1 }),
      row({
        interpretation_id: "jul_v1",
        run_id: "fc_jul",
        hs_run_id: "hs_20260715T103033",
        created_at: "2026-08-06 18:54:48",
      }),
    ]);
    expect(options.map((o) => o.runKey)).toEqual(["fc_aug", "fc_jul"]);
    expect(options.map((o) => o.label)).toEqual(["August 2026", "July 2026"]);
    // The API orders rows newest-first, so the first row per run is its latest.
    expect(options[0].latest.interpretation_id).toBe("aug_v2");
    expect(options[0].versions).toHaveLength(2);
    expect(options[1].versions).toHaveLength(1);
  });

  it("defaulting to the first option selects the most recent run", () => {
    const options = groupVersionsByRun([
      row({ interpretation_id: "aug", run_id: "fc_aug" }),
      row({ interpretation_id: "jul", run_id: "fc_jul" }),
    ]);
    expect(options[0].runKey).toBe("fc_aug");
  });

  it("a scored row groups under its scored_run_id", () => {
    const options = groupVersionsByRun([
      row({ kind: "scored", run_id: null, scored_run_id: "2026-08" }),
    ]);
    expect(options[0].runKey).toBe("2026-08");
  });

  it("rows with no run key are kept, never dropped", () => {
    // Dropping them would make a stored report unreachable from the picker.
    const options = groupVersionsByRun([
      row({ interpretation_id: "orphan", run_id: null, scored_run_id: null }),
    ]);
    expect(options).toHaveLength(1);
    expect(options[0].latest.interpretation_id).toBe("orphan");
  });

  it("empty input yields no options rather than throwing", () => {
    expect(groupVersionsByRun([])).toEqual([]);
  });
});

describe("navigation", () => {
  it("is a single source of truth (no desktop/mobile duplicate lists)", () => {
    const hrefs = NAV_LINKS.map((l) => l.href);
    expect(new Set(hrefs).size).toBe(hrefs.length);
  });

  it("links to the report under its full name", () => {
    const report = NAV_LINKS.find((l) => l.href === "/interpreter");
    expect(report?.label).toBe("Fred's Monthly Forecast Report");
  });

  it("keeps every pre-existing destination", () => {
    const hrefs = NAV_LINKS.map((l) => l.href);
    for (const href of [
      "/", "/interpreter", "/countries", "/hs-triage", "/questions",
      "/resolver", "/sibyl", "/costs", "/performance", "/downloads", "/about",
    ]) {
      expect(hrefs).toContain(href);
    }
    expect(NAV_LINKS.find((l) => l.label === "Substack")?.external).toBe(true);
  });
});


describe("report sections", () => {
  const entry = (
    category: string | undefined,
    hazard_family: string | undefined,
    rank: number
  ) => ({ category, hazard_family, rank });

  it("orders the four boxes: what is changing before what is merely large", () => {
    const groups = groupAttention([
      entry("stable_major", "conflict", 1),
      entry("worsening", "conflict", 1),
      entry("stable_major", "climate", 1),
      entry("worsening", "climate", 1),
    ]);
    expect(groups.map((g) => g.key)).toEqual([
      "worsening-climate",
      "worsening-conflict",
      "stable_major-climate",
      "stable_major-conflict",
    ]);
  });

  it("never drops an entry the model failed to place", () => {
    const stray = entry(undefined, undefined, 3);
    const groups = groupAttention([entry("worsening", "climate", 1), stray]);
    expect(groups.map((g) => g.key)).toEqual(["worsening-climate", "other"]);
    expect(groups[1].entries).toEqual([stray]);
  });

  it("sorts within a box by rank and skips empty boxes", () => {
    const groups = groupAttention([
      entry("worsening", "climate", 3),
      entry("worsening", "climate", 1),
    ]);
    expect(groups).toHaveLength(1);
    expect(groups[0].entries.map((e) => e.rank)).toEqual([1, 3]);
  });

  it("headings read as sentences", () => {
    expect(sectionHeading("worsening", "climate")).toBe(
      "Potentially worsening situations: climate hazards"
    );
    expect(sectionHeading("stable_major", "conflict")).toBe(
      "Major impact but roughly stable: conflict"
    );
  });
});

describe("entry headings", () => {
  it("prints names, never codes, when the API supplied them", () => {
    expect(
      entryHeading({
        country_name: "Nicaragua",
        iso3: "NIC",
        hazard_name: "Drought",
        hazard_code: "DR",
        metric_name: "a major disaster alert",
        metric: "EVENT_OCCURRENCE",
      })
    ).toBe("Nicaragua, drought: a major disaster alert");
  });

  it("falls back to codes rather than printing nothing", () => {
    expect(
      entryHeading({ iso3: "NIC", hazard_code: "DR", metric: "PA" })
    ).toBe("NIC, dr: PA");
  });
});

describe("movement wording", () => {
  it("gives the direction first and the size second", () => {
    expect(movementLabel(2.5)).toBe("far above its usual level");
    expect(movementLabel(-2.5)).toBe("far below its usual level");
    expect(movementLabel(1.2)).toBe("well above its usual level");
    expect(movementLabel(-1.2)).toBe("well below its usual level");
    expect(movementLabel(0.5)).toBe("above its usual level");
    expect(movementLabel(0.02)).toBe("near its usual level");
  });
});

describe("decision deadlines", () => {
  it("prints the month a decision is due", () => {
    expect(monthLabel("2026-08")).toBe("August 2026");
    expect(monthLabel("2027-01")).toBe("January 2027");
  });

  it("says so plainly when a row carries no derived deadline", () => {
    // The deadline comes from the peak horizon; a row without one has no
    // date, and inventing a plausible month is exactly what the derivation
    // exists to prevent.
    expect(monthLabel(undefined)).toBe("no dated deadline");
    expect(monthLabel("")).toBe("no dated deadline");
    expect(monthLabel("2026-13")).toBe("no dated deadline");
  });
});

describe("the published PDF against the report on screen", () => {
  // The release and the API's database catch up at different speeds. On
  // 2026-08-11 the Download button offered v10 while the page's newest
  // selectable report was v9, and nothing on the page said so: the reader
  // could only tell by opening the PDF. The existing "still syncing" banner
  // did not cover it, because it fires only when the API has NO report.
  it("reads the version out of a published asset name", () => {
    expect(pdfVersionFromAsset("report__2026-08__v10.pdf")).toBe(10);
    expect(pdfVersionFromAsset("report__2026-08__v9.pdf")).toBe(9);
  });

  it("returns nothing rather than guessing", () => {
    // The constant-name asset carries no version, and a missing manifest key
    // must not be read as version zero — either would make the comparison
    // fire on every page load.
    expect(pdfVersionFromAsset("interpreter_report_latest.pdf")).toBeNull();
    expect(pdfVersionFromAsset(null)).toBeNull();
    expect(pdfVersionFromAsset(undefined)).toBeNull();
    expect(pdfVersionFromAsset("")).toBeNull();
  });

  it("takes the newest version served across every run", () => {
    const rows = [
      { version: 9 },
      { version: 2 },
      { version: 7 },
    ];
    expect(newestServedVersion(rows)).toBe(9);
  });

  it("says nothing when the API served no report at all", () => {
    expect(newestServedVersion([])).toBeNull();
    expect(newestServedVersion([{ version: null }])).toBeNull();
  });

  it("catches the live mismatch and stays quiet when they agree", () => {
    const served = newestServedVersion([{ version: 9 }]);
    expect(pdfVersionFromAsset("report__2026-08__v10.pdf")! > served!).toBe(
      true
    );
    expect(pdfVersionFromAsset("report__2026-08__v9.pdf")! > served!).toBe(
      false
    );
  });
});
