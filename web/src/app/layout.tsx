import "../styles/globals.css";

import type { Metadata } from "next";

import Nav from "../components/Nav";

export const metadata: Metadata = {
  title: "Pythia Dashboard",
  description: "Pythia public diagnostics dashboard"
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
      </body>
    </html>
  );
};

export default RootLayout;
