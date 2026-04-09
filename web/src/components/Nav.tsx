"use client";

import Link from "next/link";
import { useRouter, useSearchParams, usePathname } from "next/navigation";
import { useState } from "react";

const Nav = () => {
  const showDebug = process.env.NEXT_PUBLIC_FRED_DEBUG_UI === "1";
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const includeTest = searchParams.get("include_test") === "true";

  const toggleTest = () => {
    const params = new URLSearchParams(searchParams.toString());
    if (includeTest) {
      params.delete("include_test");
    } else {
      params.set("include_test", "true");
    }
    const qs = params.toString();
    router.push(qs ? `${pathname}?${qs}` : pathname);
  };

  return (
    <>
      <nav className="border-b border-fred-secondary bg-fred-surface">
        <div className="flex w-full items-center justify-between px-6 py-4">
          <Link href="/" className="flex items-center gap-3">
            <img
              src="/brand/fred_logo.png"
              alt="Fred logo"
              className="h-10 w-10 md:h-24 md:w-24"
            />
            <div className="flex flex-col">
              <span className="text-lg font-semibold text-fred-primary md:text-2xl">
                Fred: Humanitarian Forecasting System
              </span>
              <span className="text-xs text-fred-muted">
                An AI-driven end-to-end humanitarian impact forecasting system
              </span>
            </div>
          </Link>
          <div className="hidden items-center flex-wrap justify-center gap-x-6 gap-y-2 text-sm md:flex">
            <Link
              className="text-fred-primary font-semibold hover:text-fred-secondary"
              href="/"
            >
              Run Results
            </Link>
            <Link
              className="text-fred-primary font-semibold hover:text-fred-secondary"
              href="/countries"
            >
              Countries
            </Link>
            <Link
              className="text-fred-primary font-semibold hover:text-fred-secondary"
              href="/hs-triage"
            >
              HS Triage & RC
            </Link>
            <Link
              className="text-fred-primary font-semibold hover:text-fred-secondary"
              href="/questions"
            >
              Forecasts
            </Link>
            <Link
              className="text-fred-primary font-semibold hover:text-fred-secondary"
              href="/resolver"
            >
              Resolver
            </Link>
            <Link
              className="text-fred-primary font-semibold hover:text-fred-secondary"
              href="/costs"
            >
              Costs
            </Link>
            <Link
              className="text-fred-primary font-semibold hover:text-fred-secondary"
              href="/performance"
            >
              Performance
            </Link>
            <Link
              className="text-fred-primary font-semibold hover:text-fred-secondary"
              href="/downloads"
            >
              Downloads
            </Link>
            <Link
              className="text-fred-primary font-semibold hover:text-fred-secondary"
              href="/about"
            >
              About
            </Link>
            <Link
              className="text-fred-primary font-semibold hover:text-fred-secondary"
              href="https://fredforecaster.substack.com/"
              rel="noreferrer noopener"
              target="_blank"
            >
              Substack
            </Link>
            {showDebug ? (
              <Link
                className="text-fred-primary font-semibold hover:text-fred-secondary"
                href="/debug"
              >
                Debug
              </Link>
            ) : null}
          </div>
          <button
            type="button"
            className="flex items-center text-fred-primary md:hidden"
            aria-label="Toggle navigation menu"
            aria-expanded={isMobileMenuOpen}
            onClick={() => setIsMobileMenuOpen((open) => !open)}
          >
            <svg
              className="h-6 w-6"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <line x1="3" y1="6" x2="21" y2="6" />
              <line x1="3" y1="12" x2="21" y2="12" />
              <line x1="3" y1="18" x2="21" y2="18" />
            </svg>
          </button>
        </div>
        <div className="hidden justify-center pb-2 md:flex">
          <button
            type="button"
            onClick={toggleTest}
            className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
              includeTest
                ? "bg-amber-100 text-amber-800 border border-amber-300"
                : "bg-fred-surface text-fred-muted border border-fred-secondary"
            }`}
            title={includeTest ? "Test data included — click to hide" : "Test data hidden — click to show"}
          >
            {includeTest ? "Test ON" : "Test OFF"}
          </button>
        </div>
        {isMobileMenuOpen ? (
          <div className="border-t border-fred-secondary px-6 pb-4 md:hidden">
            <div className="flex flex-col gap-3 pt-4 text-sm">
              <Link
                className="text-fred-primary font-semibold hover:text-fred-secondary"
                href="/"
              >
                Run Results
              </Link>
              <Link
                className="text-fred-primary font-semibold hover:text-fred-secondary"
                href="/countries"
              >
                Countries
              </Link>
              <Link
                className="text-fred-primary font-semibold hover:text-fred-secondary"
                href="/hs-triage"
              >
                HS Triage & RC
              </Link>
              <Link
                className="text-fred-primary font-semibold hover:text-fred-secondary"
                href="/questions"
              >
                Forecasts
              </Link>
              <Link
                className="text-fred-primary font-semibold hover:text-fred-secondary"
                href="/resolver"
              >
                Resolver
              </Link>
              <Link
                className="text-fred-primary font-semibold hover:text-fred-secondary"
                href="/costs"
              >
                Costs
              </Link>
              <Link
                className="text-fred-primary font-semibold hover:text-fred-secondary"
                href="/performance"
              >
                Performance
              </Link>
              <Link
                className="text-fred-primary font-semibold hover:text-fred-secondary"
                href="/downloads"
              >
                Downloads
              </Link>
              <Link
                className="text-fred-primary font-semibold hover:text-fred-secondary"
                href="/about"
              >
                About
              </Link>
              <Link
                className="text-fred-primary font-semibold hover:text-fred-secondary"
                href="https://fredforecaster.substack.com/"
                rel="noreferrer noopener"
                target="_blank"
              >
                Substack
              </Link>
              <button
                type="button"
                onClick={toggleTest}
                className={`rounded-full px-3 py-1 text-xs font-medium transition-colors self-start ${
                  includeTest
                    ? "bg-amber-100 text-amber-800 border border-amber-300"
                    : "bg-fred-surface text-fred-muted border border-fred-secondary"
                }`}
                title={includeTest ? "Test data included — click to hide" : "Test data hidden — click to show"}
              >
                {includeTest ? "Test ON" : "Test OFF"}
              </button>
              {showDebug ? (
                <Link
                  className="text-fred-primary font-semibold hover:text-fred-secondary"
                  href="/debug"
                >
                  Debug
                </Link>
              ) : null}
            </div>
          </div>
        ) : null}
      </nav>
      {includeTest && (
        <div className="bg-amber-50 border-b border-amber-200 px-6 py-1 text-center text-xs text-amber-700">
          Showing test run data alongside production data
        </div>
      )}
    </>
  );
};

export default Nav;
