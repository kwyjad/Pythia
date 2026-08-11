"use client";

import Link from "next/link";
import { useRouter, useSearchParams, usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";

// One dropdown menu at every breakpoint. It used to be an inline row on
// desktop and a dropdown only on mobile — two hand-maintained copies of the
// same link list, which is exactly how a nav item ends up in one menu and not
// the other. NAV_LINKS below is now the single source of truth.
export const NAV_LINKS: Array<{
  href: string;
  label: string;
  external?: boolean;
}> = [
  { href: "/", label: "Run Results" },
  { href: "/interpreter", label: "Fred's Monthly Forecast Report" },
  { href: "/countries", label: "Countries" },
  { href: "/hs-triage", label: "HS Triage & RC" },
  { href: "/questions", label: "Forecasts" },
  { href: "/resolver", label: "Resolver" },
  { href: "/sibyl", label: "Sibyl" },
  { href: "/costs", label: "Costs" },
  { href: "/performance", label: "Performance" },
  { href: "/downloads", label: "Downloads" },
  { href: "/about", label: "About" },
  {
    href: "https://fredforecaster.substack.com/",
    label: "Substack",
    external: true,
  },
];

const Nav = () => {
  const showDebug = process.env.NEXT_PUBLIC_FRED_DEBUG_UI === "1";
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);

  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const includeTest = searchParams.get("include_test") === "true";

  // A dropdown that covers content must close on outside click and on Escape,
  // or it traps the page underneath it.
  useEffect(() => {
    if (!isMenuOpen) return;
    const onPointerDown = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsMenuOpen(false);
      }
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setIsMenuOpen(false);
    };
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [isMenuOpen]);

  // Close the menu on navigation, otherwise it stays open over the new page.
  useEffect(() => {
    setIsMenuOpen(false);
  }, [pathname]);

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

  const links = showDebug
    ? [...NAV_LINKS, { href: "/debug", label: "Debug" }]
    : NAV_LINKS;

  const isActive = (href: string) =>
    href === "/" ? pathname === "/" : pathname.startsWith(href);

  return (
    <>
      <nav className="border-b border-fred-secondary bg-fred-surface">
        <div className="flex w-full items-center justify-between gap-4 px-6 py-4">
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

          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={toggleTest}
              className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
                includeTest
                  ? "bg-amber-100 text-amber-800 border border-amber-300"
                  : "bg-fred-surface text-fred-muted border border-fred-secondary"
              }`}
              title={
                includeTest
                  ? "Test data included — click to hide"
                  : "Test data hidden — click to show"
              }
            >
              {includeTest ? "Test ON" : "Test OFF"}
            </button>

            <div className="relative" ref={menuRef}>
              <button
                type="button"
                className="flex items-center gap-2 rounded-md border border-fred-secondary px-3 py-2 text-sm font-semibold text-fred-primary hover:text-fred-secondary"
                aria-label="Toggle navigation menu"
                aria-haspopup="true"
                aria-expanded={isMenuOpen}
                onClick={() => setIsMenuOpen((open) => !open)}
              >
                <svg
                  className="h-5 w-5"
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
                <span className="hidden sm:inline">Menu</span>
              </button>

              {isMenuOpen ? (
                <div
                  className="absolute right-0 z-50 mt-2 w-64 rounded-md border border-fred-secondary bg-fred-surface py-2 shadow-lg"
                  role="menu"
                >
                  {links.map((link) =>
                    link.external ? (
                      <a
                        key={link.href}
                        className="block px-4 py-2 text-sm font-semibold text-fred-primary hover:bg-fred-bg hover:text-fred-secondary"
                        href={link.href}
                        rel="noreferrer noopener"
                        target="_blank"
                        role="menuitem"
                      >
                        {link.label}
                      </a>
                    ) : (
                      <Link
                        key={link.href}
                        className={`block px-4 py-2 text-sm font-semibold hover:bg-fred-bg hover:text-fred-secondary ${
                          isActive(link.href)
                            ? "text-fred-secondary"
                            : "text-fred-primary"
                        }`}
                        href={link.href}
                        role="menuitem"
                        onClick={() => setIsMenuOpen(false)}
                      >
                        {link.label}
                      </Link>
                    )
                  )}
                </div>
              ) : null}
            </div>
          </div>
        </div>
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
