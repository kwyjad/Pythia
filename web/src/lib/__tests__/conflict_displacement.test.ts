import { describe, expect, it } from "vitest";

import {
  CONFLICT_DISPLACEMENT_NOT_FORECAST,
  displacementNotForecastWindows,
  notForecastSet,
  withDisplacementNote,
} from "../conflict_displacement";

describe("conflict displacement not forecast", () => {
  it("names the windows with conflict deaths and no displacement question", () => {
    const rows = [
      { iso3: "SDN", hazard_code: "ACE", metric: "FATALITIES", target_month: "2027-05" },
      { iso3: "SDN", hazard_code: "ACE", metric: "FATALITIES", target_month: "2027-04" },
      { iso3: "SDN", hazard_code: "ACE", metric: "PA", target_month: "2027-04" },
      { iso3: "SDN", hazard_code: "FL", metric: "PA", target_month: "2027-05" },
    ];
    expect(displacementNotForecastWindows(rows)).toEqual(["2027-05"]);
  });

  it("appends the note to a map tooltip only for listed countries", () => {
    const set = notForecastSet([{ iso3: "sdn" }]);
    expect(withDisplacementNote("12,000", "SDN", set)).toBe(
      `12,000 · ${CONFLICT_DISPLACEMENT_NOT_FORECAST}`
    );
    expect(withDisplacementNote("12,000", "SOM", set)).toBe("12,000");
    expect(withDisplacementNote("12,000", "SOM", null)).toBe("12,000");
  });
});
