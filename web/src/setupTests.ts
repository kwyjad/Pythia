import "@testing-library/jest-dom/vitest";

import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

// Unmount rendered components after each test. testing-library only
// auto-registers this when `afterEach` is a global (vitest `globals: true`),
// which this project does not enable — without it, each render() stacks in
// document.body and `getByText` finds duplicate matches across tests.
afterEach(() => {
  cleanup();
});
