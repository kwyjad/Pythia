import { describe, expect, it } from "vitest";

import { safeHref } from "../safe_href";

describe("safeHref", () => {
  it("keeps http and https addresses", () => {
    expect(safeHref("https://reliefweb.int/report/x")).toBe("https://reliefweb.int/report/x");
    expect(safeHref("http://example.org")).toBe("http://example.org");
  });

  it("refuses script and data schemes, however they are written", () => {
    for (const bad of [
      "javascript:alert(1)",
      " JavaScript:alert(1)",
      "java\nscript:alert(1)",
      "data:text/html,<script>alert(1)</script>",
      "vbscript:msgbox(1)",
      "file:///etc/passwd",
    ]) {
      expect(safeHref(bad)).toBeNull();
    }
  });

  it("refuses relative and empty values", () => {
    expect(safeHref("/interpreter")).toBeNull();
    expect(safeHref("")).toBeNull();
    expect(safeHref(null)).toBeNull();
    expect(safeHref(undefined)).toBeNull();
  });
});
