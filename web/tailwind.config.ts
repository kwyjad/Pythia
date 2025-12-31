import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        fred: {
          bg: "#F5F5F5",
          primary: "#156082",
          secondary: "#80350E",
          text: "#3A3A3A",
          surface: "#FFFFFF",
          border: "#D6D6D6",
          muted: "#6B7280"
        }
      },
      boxShadow: {
        fredCard: "0 1px 2px rgba(0,0,0,0.06)"
      }
    }
  },
  plugins: []
};

export default config;
