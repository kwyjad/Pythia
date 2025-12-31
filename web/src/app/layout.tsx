import "../styles/globals.css";

import type { Metadata } from "next";

import Nav from "../components/Nav";

export const metadata: Metadata = {
  title: "Fred: Humanitarian Forecasting System",
  description: "An AI-driven end-to-end humanitarian impact forecasting system"
};

type RootLayoutProps = {
  children: React.ReactNode;
};

const RootLayout = ({ children }: RootLayoutProps) => {
  return (
    <html lang="en">
      <body>
        <Nav />
        <main>{children}</main>
        {process.env.NODE_ENV === "development" ? (
          <script
            dangerouslySetInnerHTML={{
              __html: `(() => {
  if (typeof document === "undefined") return;
  const body = document.body;
  if (body && body.classList.contains("bg-slate-950")) {
    console.warn("Fred theme warning: body still has bg-slate-950.");
  }
})();`,
            }}
          />
        ) : null}
      </body>
    </html>
  );
};

export default RootLayout;
