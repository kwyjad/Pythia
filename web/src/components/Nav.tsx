"use client";

import Link from "next/link";
import { useState } from "react";

const Nav = () => {
  const showDebug = process.env.NEXT_PUBLIC_FRED_DEBUG_UI === "1";
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  return (
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
        <div className="hidden items-center gap-4 text-sm md:flex">
          <Link
            className="text-fred-primary font-semibold hover:text-fred-secondary"
            href="/"
          >
            Risk Index
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
            HS Triage
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
      {isMobileMenuOpen ? (
        <div className="border-t border-fred-secondary px-6 pb-4 md:hidden">
          <div className="flex flex-col gap-3 pt-4 text-sm">
            <Link
              className="text-fred-primary font-semibold hover:text-fred-secondary"
              href="/"
            >
              Risk Index
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
              HS Triage
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
        </div>
      ) : null}
    </nav>
  );
};

export default Nav;
