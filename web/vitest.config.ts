import { defineConfig } from "vitest/config";

export default defineConfig({
  // Use the automatic JSX runtime (react/jsx-runtime), matching how Next builds
  // the app (tsconfig "jsx": "preserve"). Without this, esbuild defaults to the
  // classic transform (React.createElement), which throws "React is not defined"
  // when rendering components that don't import React — the app relies on the
  // automatic runtime, so components intentionally don't.
  esbuild: { jsx: "automatic" },
  test: {
    environment: "jsdom",
    setupFiles: "./src/setupTests.ts",
  },
});
