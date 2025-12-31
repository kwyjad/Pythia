import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        fredBg: "#F5F5F5",
        fredPrimary: "#156082",
        fredSecondary: "#80350E",
        fredText: "#3A3A3A",
        fredSurface: "#FFFFFF",
        fredBorder: "#D6D6D6",
        fredMuted: "#6B7280"
      },
      boxShadow: {
        fredCard: "0 1px 2px rgba(0,0,0,0.06)"
      }
    }
  },
  plugins: []
};

export default config;
