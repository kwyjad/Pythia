import { describe, expect, it } from "vitest";

import { computeTotalEiv } from "../SpdPanel";

describe("computeTotalEiv", () => {
  it("uses the peak month for PA totals", () => {
    expect(computeTotalEiv("PA", [1, 3, 2, 5, 4, 0])).toBe(5);
  });

  it("uses the cumulative sum for fatalities totals", () => {
    expect(computeTotalEiv("FATALITIES", [1, 3, 2, 5, 4, 0])).toBe(15);
  });
});
